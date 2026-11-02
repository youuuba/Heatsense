"""HeatSense model: heat-balance physics + heart-rate observation + unscented Kalman filter.

How it works, once per minute:

1. Physics (time update). Core temperature moves toward an equilibrium set by effort
   and heat stress, with a time constant of ~20 min (~45 min for passive cooling when stopped):

       CT_eq = CT_rest + a_effort * (M - M_rest) + a_env * max(WBGT - WBGT_ref, 0) * M / 10
       CT_t  = CT_eq + (CT_{t-1} - CT_eq) * exp(-1 / tau)

   M is metabolic power (W/kg, from pace and slope). Stopped -> M = rest -> CT cools,
   but only after a short delay (core temperature keeps up for a few minutes after a stop).

2. Heart-rate observation. Heart rate is explained by several causes, not heat alone:

       HR = base + k_effort * M_lagged + k_heat * (CT - 37) + drift * active_minutes

   softly capped at the runner's maximum heart rate.

3. Unscented Kalman filter. Compares predicted with measured heart rate and corrects
   the core-temperature STATE.

Design constraints (see README, "Prior work and patent notice"):
* Personal parameters (base, k_effort, drift, noise, max HR) are fitted OFFLINE from
  completed past runs (``calibrate``) and are FROZEN during a run. Nothing in the
  in-run filter modifies model parameters; only the temperature state is updated.
* No skin-temperature input. Heart rate is never the only input.

All default values are starting points from physiology literature and must be checked
in validation; they are not validated constants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .effort import REST_W_PER_KG


@dataclass(frozen=True)
class PhysicsParams:
    ct_rest: float = 37.0  # degC
    tau_min: float = 20.0  # core temperature time constant while heating, minutes
    # Passive cooling is much slower than heating: ~0.03-0.05 degC/min from ~39.5 degC.
    tau_cool_min: float = 45.0  # used while stopped, when the equilibrium is below the state
    cool_delay_min: int = 3  # after a stop, core temperature doesn't start falling for this long
    a_effort: float = 0.15  # degC per W/kg above rest (temperate conditions)
    a_env: float = 0.057  # extra degC per degC WBGT above ref, per 10 W/kg
    wbgt_ref: float = 15.0  # degC; below this, environment adds no heat strain
    effort_lag_min: float = 1.0  # heart rate responds to effort with ~1 min lag
    # Process noise, degC per minute. Large on purpose: it stands for the physics model's own
    # error (heat gain differs between runners and isn't learned), not true minute-to-minute
    # randomness. Tuned so the ±2 SD band covers ~95% of true values on simulated athletes.
    q_sd: float = 0.11
    ct0: float = 37.0
    ct0_sd: float = 0.15
    indoor_wbgt: float = 18.0  # ASSUMED for treadmill / indoor runs (labelled as an assumption)
    # Fallback when outdoor weather is unknown: deliberately hot, so a failed lookup can't
    # silently make a hot day look mild.
    unknown_wbgt: float = 28.0


@dataclass
class Profile:
    """A runner's personal parameters, fitted offline from past runs."""

    heat_gain: float = 1.0  # how fast this runner heats up vs typical; fitted ONLY from
    # runs with a core-temperature reference (heart rate alone can't determine it)
    hr_base: float = 78.0  # bpm
    hr_per_effort: float = 5.5  # bpm per W/kg
    hr_per_degc: float = 15.0  # bpm per degC core; PROVISIONAL prior, set properly by validation
    drift_per_min: float = 0.05  # bpm per active minute (cardiac drift)
    hr_max: float = 200.0
    hr_noise_sd: float = 8.0  # bpm, minute-level residual SD
    hr_noise_rho: float = 0.85  # lag-1 autocorrelation of minute-level residuals
    n_runs: int = 0
    n_minutes: int = 0
    source: str = "population default"
    notes: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- features


def build_features(minutes: pd.DataFrame, effort: pd.DataFrame, wbgt,
                   physics: PhysicsParams = PhysicsParams()) -> pd.DataFrame:
    """Per-minute model inputs.

    ``wbgt`` is a per-minute Series/array, or a single number (indoor / unknown).
    """
    m = effort["met_power"].to_numpy(dtype=float)
    alpha = 1 - np.exp(-1 / physics.effort_lag_min)
    lagged = np.empty_like(m)
    acc = REST_W_PER_KG
    for i, v in enumerate(m):  # heart rate follows effort with a short lag
        acc += alpha * (v - acc)
        lagged[i] = acc
    active = np.cumsum(~minutes["paused"].to_numpy(dtype=bool)).astype(float)
    w = np.broadcast_to(np.asarray(wbgt, dtype=float), m.shape).copy()
    return pd.DataFrame({
        "met_power": m, "met_lagged": lagged, "active_min": active, "wbgt": w,
        "hr": minutes["hr"].to_numpy(dtype=float),
        "paused": minutes["paused"].to_numpy(dtype=bool),
        "hr_coverage": minutes["hr_coverage"].to_numpy(dtype=float),
    }, index=minutes.index)


# ---------------------------------------------------------------- model pieces


def personal_physics(p: "PhysicsParams", heat_gain: float) -> "PhysicsParams":
    """Physics scaled by a runner's heat gain (fitted offline, fixed during a run)."""
    return replace(p, a_effort=p.a_effort * heat_gain, a_env=p.a_env * heat_gain)


def ct_equilibrium(met, wbgt, p: PhysicsParams):
    heat = np.maximum(np.asarray(wbgt) - p.wbgt_ref, 0.0)
    return p.ct_rest + p.a_effort * (np.asarray(met) - REST_W_PER_KG) + p.a_env * heat * np.asarray(met) / 10


def physics_step(ct, met, wbgt, p: PhysicsParams, stopped: bool = False, hold: bool = False):
    """One minute of physics.

    ``stopped``: passive cooling, slower (``tau_cool_min``). While running, heating and
    cooling share ``tau_min``. ``hold``: just after a stop, core temperature can't fall yet.
    """
    ct = np.asarray(ct, dtype=float)
    eq = ct_equilibrium(met, wbgt, p)
    tau = np.where(stopped & (eq < ct), p.tau_cool_min, p.tau_min)
    new = eq + (ct - eq) * np.exp(-1.0 / tau)
    return np.maximum(new, ct) if hold else new


def cooling_hold(paused) -> np.ndarray:
    """Minute index -> how many consecutive stopped minutes so far (0 while moving)."""
    run, out = 0, np.zeros(len(paused), dtype=int)
    for i, stopped in enumerate(np.asarray(paused, dtype=bool)):
        run = run + 1 if stopped else 0
        out[i] = run
    return out


def effective_hr_noise(prof: Profile) -> float:
    """Observation noise SD (bpm) the filter uses per minute.

    Heart-rate residuals are strongly autocorrelated from minute to minute, so each minute
    is far less than one independent observation. Inflating the SD by
    sqrt((1 + rho) / (1 - rho)) (the AR(1) variance-of-the-mean factor) keeps the filter
    from treating 30 correlated minutes as 30 pieces of evidence. Fixed for the whole run.
    """
    rho = float(np.clip(prof.hr_noise_rho, 0.0, 0.95))
    return prof.hr_noise_sd * np.sqrt((1 + rho) / (1 - rho))


def predict_hr(ct, met_lagged, active_min, prof: Profile, k: float = 4.0):
    lin = (prof.hr_base + prof.hr_per_effort * met_lagged
           + prof.hr_per_degc * (np.asarray(ct) - 37.0) + prof.drift_per_min * active_min)
    # Smooth minimum with HRmax: heart rate saturates instead of growing without bound.
    return -k * np.logaddexp(-lin / k, -prof.hr_max / k)


def physics_only(feat: pd.DataFrame, p: PhysicsParams = PhysicsParams()) -> np.ndarray:
    """Open-loop core temperature from effort and weather alone (no heart rate)."""
    ct, out = p.ct0, np.empty(len(feat))
    stopped_for = cooling_hold(feat["paused"])
    for i, (m, w) in enumerate(zip(feat["met_power"], feat["wbgt"])):
        ct = float(physics_step(ct, m, w, p, stopped=stopped_for[i] > 0,
                                hold=0 < stopped_for[i] <= p.cool_delay_min))
        out[i] = ct
    return out


# ---------------------------------------------------------------- filter


def run_heatsense(feat: pd.DataFrame, prof: Profile = Profile(),
                  p: PhysicsParams = PhysicsParams()) -> pd.DataFrame:
    """One-state unscented Kalman filter. Parameters are fixed for the whole run."""
    p = personal_physics(p, prof.heat_gain)
    n, lam = 1, 2.0  # alpha=1, kappa=2  ->  n + lambda = 3
    wm = np.array([lam / (n + lam), 0.5 / (n + lam), 0.5 / (n + lam)])
    wc = wm.copy()
    spread = np.sqrt(n + lam)
    q, r = p.q_sd**2, effective_hr_noise(prof) ** 2

    x, P = p.ct0, p.ct0_sd**2
    P_open = P  # what the variance would be with no heart rate at all (for the "check" zone)
    rows = []
    cols = ["met_power", "met_lagged", "active_min", "wbgt", "hr", "hr_coverage"]
    stopped_for = cooling_hold(feat["paused"])
    for i, (m, ml, act, w, hr, cov) in enumerate(feat[cols].itertuples(index=False)):
        stop = {"stopped": stopped_for[i] > 0, "hold": 0 < stopped_for[i] <= p.cool_delay_min}
        sig = x + spread * np.sqrt(P_open) * np.array([0.0, 1.0, -1.0])
        prop = physics_step(sig, m, w, p, **stop)
        P_open = float(wc @ (prop - wm @ prop) ** 2) + q

        # Time update: push sigma points through the heat-balance physics.
        sig = x + spread * np.sqrt(P) * np.array([0.0, 1.0, -1.0])
        prop = physics_step(sig, m, w, p, **stop)
        x = float(wm @ prop)
        P = float(wc @ (prop - x) ** 2) + q

        # Observation update: predicted vs measured heart rate.
        sig = x + spread * np.sqrt(P) * np.array([0.0, 1.0, -1.0])
        y = predict_hr(sig, ml, act, prof)
        y_mean = float(wm @ y)
        innovation = np.nan
        if not np.isnan(hr):
            # A minute averaged from fewer good seconds is a noisier observation.
            pyy = float(wc @ (y - y_mean) ** 2) + r / np.clip(cov, 0.25, 1.0)
            pxy = float(wc @ ((sig - x) * (y - y_mean)))
            gain = pxy / pyy
            innovation = hr - y_mean
            x = x + gain * innovation
            P = max(P - gain**2 * pyy, 1e-6)
        rows.append((x, np.sqrt(P), np.sqrt(P_open), y_mean, innovation))

    return pd.DataFrame(rows, columns=["ct", "ct_sd", "ct_sd_open", "hr_pred", "innovation"],
                        index=feat.index)


# ---------------------------------------------------------------- offline calibration


def calibrate(runs: list[pd.DataFrame], p: PhysicsParams = PhysicsParams(),
              base: Profile = Profile(), window: tuple[int, int] = (4, 12),
              min_minutes: int = 20) -> Profile:
    """Fit a personal profile OFFLINE from completed runs (``build_features`` tables).

    Learns the runner's heart-rate-at-effort curve (``hr_base``, ``hr_per_effort``) from
    the early part of each run (minutes 4-12 by default): warmed up, but before heat has
    built up much. Later in a run, heart rate above this personal curve is what the filter
    attributes to heat.

    Least squares on clean running minutes in the window:

        HR - k_heat * (CT_physics - 37) - drift * active_min = base + k_effort * M_lagged

    CT_physics comes from effort and weather only (heart rate is not used twice); its
    contribution in this window is small, so the population k_heat and drift are enough.

    Why not fit more? Evaluated on simulated athletes, heat gain, heat sensitivity and drift
    could not be separated reliably from heart rate alone (consecutive minutes are highly
    correlated, so a few runs hold only tens of independent minutes). Those parameters stay
    at population values until runs with a core-temperature reference are available.
    """
    lo, hi = window
    X, y, hr_all, per_run = [], [], [], []
    for feat in runs:
        hr = feat["hr"].to_numpy()
        hr_all.append(hr[~np.isnan(hr)])
        idx = np.arange(len(feat))
        m = (idx >= lo) & (idx < hi) & ~np.isnan(hr) & ~feat["paused"].to_numpy()
        if m.sum() == 0:
            continue
        ct = physics_only(feat, personal_physics(p, base.heat_gain))
        X.append(np.c_[np.ones(m.sum()), feat["met_lagged"].to_numpy()[m]])
        y.append(hr[m] - base.hr_per_degc * (ct[m] - 37.0) - base.drift_per_min * feat["active_min"].to_numpy()[m])
        per_run.append(len(y) - 1)
    n = int(sum(len(v) for v in y))
    if n < min_minutes or len(y) < 2:
        return Profile(**{**asdict(base), "notes": [f"not enough data ({len(runs)} runs, {n} usable early minutes): population default"]})
    X, y = np.vstack(X), np.concatenate(y)

    notes = []
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pace_spread = float(np.std(X[:, 1]))
    if coef[1] <= 1.0 or pace_spread < 0.5:
        # Too little pace variety to learn the slope: keep the population slope, fit the offset.
        notes.append(f"pace too uniform to learn effort slope (spread {pace_spread:.2f} W/kg): kept population slope")
        coef = np.array([float(np.mean(y - base.hr_per_effort * X[:, 1])), base.hr_per_effort])
    all_hr = np.concatenate(hr_all)
    fitted = Profile(**{**asdict(base), "hr_base": float(coef[0]), "hr_per_effort": float(coef[1])})
    sd, rho = residual_noise(runs, fitted, p)
    notes.append(f"heart-rate residuals: SD {sd:.1f} bpm, minute-to-minute correlation {rho:.2f}")
    notes.append("heat gain, heat sensitivity and drift: population values (need core-temperature reference to personalize)")
    return Profile(
        **{**asdict(base),
           "hr_base": float(coef[0]), "hr_per_effort": float(coef[1]),
           "hr_max": float(max(np.percentile(all_hr, 99.5) + 5, 185.0)),
           "hr_noise_sd": sd, "hr_noise_rho": rho,
           "n_runs": len(runs), "n_minutes": n, "source": "calibrated offline from early minutes of past runs",
           "notes": notes},
    )


def residual_noise(runs: list[pd.DataFrame], prof: Profile,
                   p: PhysicsParams = PhysicsParams()) -> tuple[float, float]:
    """SD and lag-1 autocorrelation of heart-rate residuals, OFFLINE, over all clean minutes.

    Residual = measured HR - HR predicted from the fitted curve with physics-only core
    temperature, over every clean running minute of the calibration runs (not just the
    fitting window). Correlation uses pairs of consecutive minutes within a run.
    """
    resid, pairs = [], []
    for feat in runs:
        ct = physics_only(feat, personal_physics(p, prof.heat_gain))
        pred = predict_hr(ct, feat["met_lagged"].to_numpy(), feat["active_min"].to_numpy(), prof)
        e = feat["hr"].to_numpy() - pred
        e[feat["paused"].to_numpy()] = np.nan
        resid.append(e[~np.isnan(e)])
        both = ~np.isnan(e[:-1]) & ~np.isnan(e[1:])
        pairs.append(np.c_[e[:-1][both], e[1:][both]])
    resid, pairs = np.concatenate(resid), np.vstack(pairs)
    sd = float(np.std(resid))
    rho = float(np.corrcoef(pairs.T)[0, 1]) if len(pairs) > 10 else Profile.hr_noise_rho
    return float(np.clip(sd, 3.0, 15.0)), float(np.clip(rho, 0.0, 0.95))


# ---------------------------------------------------------------- strain zones

ZONES = ("green", "amber", "red", "check")


@dataclass(frozen=True)
class ZoneThresholds:
    """PROVISIONAL strain-zone thresholds: placeholders until validation sets them.

    All core-temperature levels are compared with the UPPER bound ct + upper_z * ct_sd
    (one-sided 95%), not the mean, so an uncertain estimate leans toward warning.
    """

    upper_z: float = 1.64
    amber_ct: float = 39.0  # amber at this upper bound...
    amber_rising_ct: float = 38.5  # ...or this one while rising
    amber_hot_ct: float = 38.5  # ...or this one in hot weather
    red_ct: float = 39.7  # red at this upper bound...
    red_rising_ct: float = 39.2  # ...or this one while rising
    hot_wbgt: float = 28.0  # degC
    rising_c_per_min: float = 0.03  # ~1 degC per 30 min, on the mean estimate
    trend_min: int = 10
    # Low confidence ("check") when heart rate is missing in >= check_missing of the last
    # check_window running minutes, or when the estimate has become essentially
    # physics-only: SD within 1% of the SD the same run would have with no heart rate at
    # all (``ct_sd_open`` from the filter). With usable heart rate the SD stays clearly
    # below that, so the SD rule catches long or repeated gaps the window can miss.
    check_missing: int = 3
    check_window: int = 5
    check_sd_fraction: float = 0.99
    check_sd_floor: float = 0.9  # ...and SD at least this share of the heating open-loop SD
    # Zone hold: warnings rise at once but ease off slowly (e.g. after a stop, the estimate
    # falls faster than the risk does). At most one level down per hold_step_min, and no
    # green until hold_green_min after the last amber/red level.
    hold_step_min: int = 5
    hold_green_min: int = 10


LEVELS = ("green", "amber", "red")


def hold_zones(level: np.ndarray, th: ZoneThresholds = ZoneThresholds()) -> np.ndarray:
    """Apply the zone hold to per-minute levels (0 green, 1 amber, 2 red)."""
    shown, out = 0, np.zeros(len(level), dtype=int)
    changed, last_warn = -10**9, -10**9
    for i, raw in enumerate(level):
        if raw >= 1:
            last_warn = i
        if raw >= shown:  # (re)confirms the shown level: the hold restarts
            shown, changed = raw, i
        elif raw < shown and i - changed >= th.hold_step_min:
            if shown - 1 >= 1 or i - last_warn >= th.hold_green_min:
                shown, changed = shown - 1, i
        out[i] = shown
    return out


def open_loop_sd(p: PhysicsParams = PhysicsParams()) -> float:
    """Steady-state SD with no heart rate while heating (fallback when ``ct_sd_open`` is absent)."""
    return p.q_sd / np.sqrt(1 - np.exp(-2.0 / p.tau_min))


def strain_zones(est: pd.DataFrame, feat: pd.DataFrame, p: PhysicsParams = PhysicsParams(),
                 th: ZoneThresholds = ZoneThresholds()) -> pd.DataFrame:
    """PROVISIONAL zones from the upper bound, trend, heat stress and data quality.

    Returns columns ``zone``, ``ct_upper`` and ``low_confidence``. When confidence is low
    the estimate is still reported, but the zone is never green: a green level becomes
    "check", while amber and red stay visible (bad data must not hide a warning).
    Levels then pass through the zone hold (``hold_zones``). Symptoms always override.
    """
    ct = est["ct"].to_numpy()
    upper = ct + th.upper_z * est["ct_sd"].to_numpy()
    n = th.trend_min
    slope = (ct - np.r_[np.full(n, ct[0]), ct[:-n]]) / n  # degC/min
    rising = slope >= th.rising_c_per_min
    hot = feat["wbgt"].to_numpy() >= th.hot_wbgt
    missing = pd.Series(np.isnan(feat["hr"].to_numpy()) & ~feat["paused"].to_numpy())
    sd = est["ct_sd"].to_numpy()
    sd_open = est["ct_sd_open"].to_numpy() if "ct_sd_open" in est else open_loop_sd(p)
    # Physics-only AND no longer small: at the start both SDs are still near the trusted
    # resting prior, which is not a reason to doubt the estimate.
    physics_only_sd = (sd >= th.check_sd_fraction * sd_open) & (sd >= th.check_sd_floor * open_loop_sd(p))
    low_conf = ((missing.rolling(th.check_window, min_periods=1).sum().to_numpy() >= th.check_missing)
                | physics_only_sd)

    level = np.zeros(len(ct), dtype=int)
    level[(upper >= th.amber_ct) | ((upper >= th.amber_rising_ct) & rising)
          | (hot & (upper >= th.amber_hot_ct))] = 1
    level[(upper >= th.red_ct) | ((upper >= th.red_rising_ct) & rising)] = 2
    zone = np.array(LEVELS, dtype=object)[hold_zones(level, th)]
    zone[low_conf & (zone == "green")] = "check"
    return pd.DataFrame({"zone": zone, "ct_upper": upper, "low_confidence": low_conf}, index=est.index)
