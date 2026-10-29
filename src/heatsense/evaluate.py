"""Compare HeatSense against the reference baseline where the truth is known.

Synthetic athlete runs (``synthetic.make_athlete_run``) are split into calibration runs
and test runs. Every method is scored on the test runs against the true core
temperature, minute by minute. This checks the method end to end; it is NOT a
substitute for validation against a real core-temperature reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import run_baseline
from .effort import minute_effort
from .io_fit import to_minutes
from .metrics import compare
from .model import Profile, build_features, calibrate, physics_only, run_heatsense
from .quality import check_hr
from .synthetic import make_athlete_run


def prepare(seed: int, heat_gain: float = 1.0, hr_offset: float = 0.0):
    seconds, info = make_athlete_run(seed, heat_gain=heat_gain, hr_offset=hr_offset)
    checked = check_hr(seconds)
    minutes = to_minutes(checked)
    feat = build_features(minutes, minute_effort(checked, len(minutes)), info["wbgt"])
    elapsed = ((checked.index - checked.index[0]).total_seconds() // 60).astype(int).to_numpy()
    truth = checked["true_ct"].groupby(elapsed).mean().reindex(minutes.index).interpolate()
    return {"seed": seed, "info": info, "minutes": minutes, "feat": feat, "truth": truth.to_numpy()}


def estimates(run: dict, profile: Profile) -> dict[str, np.ndarray]:
    return {
        "Army baseline (HR only)": run_baseline(run["minutes"]["hr"])["ct"].to_numpy(),
        "Physics only (no HR)": physics_only(run["feat"]),
        "HeatSense, population default": run_heatsense(run["feat"], Profile())["ct"].to_numpy(),
        "HeatSense, calibrated": run_heatsense(run["feat"], profile)["ct"].to_numpy(),
    }


def evaluate_synthetic(n_calibration: int = 8, n_test: int = 30, skip_minutes: int = 10,
                       hot_threshold: float = 38.8, seed0: int = 1000,
                       heat_gain: float = 1.0, hr_offset: float = 0.0):
    """One simulated athlete: calibrate on their past runs, score on their other runs."""
    cal = [prepare(seed0 + i, heat_gain, hr_offset) for i in range(n_calibration)]
    test = [prepare(seed0 + 500 + i, heat_gain, hr_offset) for i in range(n_test)]
    profile = calibrate([r["feat"] for r in cal])

    rows, per_run = [], []
    for run in test:
        est = estimates(run, profile)
        for name, ct in est.items():
            m = compare(ct, run["truth"], skip_minutes=skip_minutes)
            per_run.append({"seed": run["seed"], "method": name, "wbgt": run["info"]["wbgt"],
                            "indoor": run["info"]["indoor"], **m})
            truth = run["truth"][skip_minutes:]
            e = ct[skip_minutes:]
            rows.append(pd.DataFrame({"method": name, "err": e - truth,
                                      "hot": truth >= hot_threshold, "flag": e >= hot_threshold}))
    allm = pd.concat(rows)
    summary = allm.groupby("method").apply(lambda g: pd.Series({
        "minutes": len(g),
        "bias_C": g["err"].mean(),
        "rmse_C": np.sqrt((g["err"] ** 2).mean()),
        "mae_C": g["err"].abs().mean(),
        f"hot_minutes_caught_(>={hot_threshold})": (g["flag"] & g["hot"]).sum() / max(g["hot"].sum(), 1),
        "false_hot_alarms": (g["flag"] & ~g["hot"]).sum() / max((~g["hot"]).sum(), 1),
    }), include_groups=False)
    return summary, pd.DataFrame(per_run), profile, test


def evaluate_athletes(athletes=((0.7, -10.0), (1.0, 0.0), (1.4, 10.0), (1.2, -8.0), (0.85, 12.0)),
                      **kw) -> pd.DataFrame:
    """Several simulated athletes who differ in heat physiology and heart rate."""
    out = []
    for k, (gain, offset) in enumerate(athletes):
        summary, *_ = evaluate_synthetic(heat_gain=gain, hr_offset=offset, seed0=2000 + 100 * k, **kw)
        out.append(summary.assign(athlete=f"heat_gain={gain}, hr_offset={offset:+.0f}"))
    return pd.concat(out).reset_index().set_index(["athlete", "method"])
