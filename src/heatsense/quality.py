"""Heart-rate quality checks for wearable data.

Wrist optical sensors fail in predictable ways while running. Each check returns a boolean
Series (True = problem) aligned to the per-second input, so failures can be plotted and
counted, and the combined ``hr_ok`` mask feeds the minute aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HR_MIN, HR_MAX = 30, 220


def flag_missing(hr: pd.Series) -> pd.Series:
    """No reading, or a zero (some devices write 0 when contact is lost)."""
    return hr.isna() | (hr <= 0)


def flag_out_of_range(hr: pd.Series) -> pd.Series:
    return hr.notna() & ((hr < HR_MIN) | (hr > HR_MAX))


def flag_spikes(hr: pd.Series, max_jump_bpm_per_s: float = 15.0) -> pd.Series:
    """Physiologically implausible jumps between consecutive seconds."""
    dt = hr.index.to_series().diff().dt.total_seconds().clip(lower=1)
    rate = hr.diff().abs() / dt
    return (rate > max_jump_bpm_per_s).fillna(False)


def flag_flatline(hr: pd.Series, min_seconds: int = 60, moving: pd.Series | None = None):
    """The exact same value for a long stretch while moving: the sensor stopped updating."""
    run_id = (hr != hr.shift()).cumsum()
    run_len = hr.groupby(run_id).transform("size")
    flat = hr.notna() & (run_len >= min_seconds)
    if moving is not None:
        flat &= moving
    return flat


def flag_cadence_lock(
    hr: pd.Series,
    cadence_spm: pd.Series,
    tolerance_bpm: float = 4.0,
    min_seconds: int = 30,
    window_s: int = 120,
    min_corr: float = 0.5,
) -> pd.Series:
    """Optical sensor reading your step rate instead of your heart rate.

    Being *close* to cadence is not enough: many runners genuinely run with heart rate
    near their cadence (e.g. both ~163). A locked sensor *follows* cadence, so a stretch
    is flagged only when, for at least ``min_seconds``:

    * heart rate is within ``tolerance_bpm`` of cadence (or half cadence, a harmonic), and
    * over a ``window_s`` window, heart rate is correlated with cadence (r >= ``min_corr``).

    If cadence is too steady to tell (window SD < 1 spm), the stretch is NOT flagged:
    we don't discard data we can't prove is bad. ``flag_hr_near_cadence`` reports those
    stretches separately for inspection.
    """
    running = cadence_spm > 120
    near_full = (hr - cadence_spm).abs() <= tolerance_bpm
    near_half = (hr - cadence_spm / 2).abs() <= tolerance_bpm
    near = (near_full | near_half) & running & hr.notna()

    target = cadence_spm.where(near_full, cadence_spm / 2)
    roll = dict(window=window_s, center=True, min_periods=window_s // 2)
    corr = hr.rolling(**roll).corr(target)
    moving_cadence = target.rolling(**roll).std() >= 1.0
    tracking = (corr >= min_corr) & moving_cadence

    # Hysteresis: a contiguous "near cadence" stretch is flagged as a whole if it holds at
    # least min_seconds of proven tracking. The rolling window blurs a lock's edges, so the
    # proof comes from its middle and the flag extends to the stretch's true boundaries.
    proven = (near & tracking.fillna(False)).astype(int)
    stretch = (near != near.shift()).cumsum()
    proven_in_stretch = proven.groupby(stretch).transform("sum")
    return near & (proven_in_stretch >= min_seconds)


def flag_soft_lock(
    hr: pd.Series,
    cadence_spm: pd.Series,
    speed: pd.Series | None = None,
    altitude: pd.Series | None = None,
    distance: pd.Series | None = None,
    step_bpm: float = 6.0,
    step_window_s: int = 15,
    near_bpm: float = 8.0,
    max_minutes: float = 15.0,
) -> pd.Series:
    """Smoothed cadence lock: heart rate *snaps* toward cadence and sits there.

    Garmin smooths heart rate on the watch, so a partial lock does not wiggle along with
    cadence (``flag_cadence_lock`` misses it). Its signature, seen in real runs, is:

    1. an abrupt drop of >= ``step_bpm`` within ``step_window_s`` seconds, with no change
       in effort (pace steady, and no downhill when altitude is available);
    2. heart rate then sitting within ``near_bpm`` of cadence, usually 10+ bpm closer
       than it was before;
    3. an abrupt jump back up, again with no change in effort.

    The stretch from the drop until the jump back (or until heart rate leaves the
    near-cadence band, or ``max_minutes``) is flagged.
    """
    hr_s = hr.rolling(5, center=True, min_periods=1).median()
    half = step_window_s // 2
    step = hr_s.shift(-half) - hr_s.shift(half)

    steady = pd.Series(True, index=hr.index)
    if speed is not None:
        # Heart rate answers an effort change 30-60 s later, so compare the pace a minute
        # before the step with the pace just after it.
        pace = (1000 / 60) / speed.where(speed > 0.5)
        pace_before = pace.rolling(30, min_periods=10).mean().shift(45)
        pace_after = pace.rolling(30, min_periods=10).mean().shift(-30)
        steady &= (pace_after - pace_before).abs().fillna(0) < 0.3
    if altitude is not None and distance is not None and altitude.notna().any():
        d_alt = altitude.shift(-15) - altitude.shift(15)
        d_dist = (distance.shift(-15) - distance.shift(15)).where(lambda x: x > 20)
        grade = (d_alt / d_dist * 100).fillna(0)
        steady &= grade.abs() < 2.0

    raw_gap = hr_s - cadence_spm
    gap_before = raw_gap.rolling(30, min_periods=10).mean().shift(half)
    gap_after = raw_gap.rolling(30, min_periods=10).mean().shift(-30)
    near_now = raw_gap.rolling(30, center=True, min_periods=10).mean().abs() <= near_bpm

    # Only while actually running: stops and walk breaks are real effort changes.
    running = cadence_spm.rolling(30, center=True, min_periods=10).min() > 140
    if speed is not None:
        running &= speed.rolling(30, center=True, min_periods=10).min() > 1.5
    near_now &= running

    down = (
        (step <= -step_bpm) & steady & running
        & (gap_after.abs() <= near_bpm)        # lands near cadence ...
        & (gap_before > near_bpm)              # ... from clearly above it ...
        & (gap_before - gap_after >= step_bpm)  # ... by a real margin
    )
    up = (step >= step_bpm) & steady

    flags = np.zeros(len(hr), dtype=bool)
    down_pos = np.flatnonzero(down.to_numpy())
    up_arr, near_arr = up.to_numpy(), near_now.to_numpy()
    max_len = int(max_minutes * 60)
    i_end_prev = -1
    for start in down_pos:
        if start <= i_end_prev:
            continue
        start = max(start - half, 0)
        end = start
        grace = 45  # seconds for the 30-s averages to settle after the step
        while end < len(hr) - 1 and end - start < max_len:
            end += 1
            if end - start <= grace:
                continue
            if up_arr[end] or not near_arr[end]:
                break
        flags[start:end + half] = True
        i_end_prev = end + half
    return pd.Series(flags, index=hr.index)


def flag_hr_near_cadence(hr: pd.Series, cadence_spm: pd.Series, tolerance_bpm: float = 4.0):
    """Informational only: heart rate close to cadence, whether or not it is locked."""
    return ((hr - cadence_spm).abs() <= tolerance_bpm) & (cadence_spm > 120) & hr.notna()


def check_hr(df: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per check plus a combined ``hr_ok`` column."""
    out = df.copy()
    hr = out["heart_rate"]
    moving = out["speed"] > 0.5 if "speed" in out else None

    out["q_missing"] = flag_missing(hr)
    out["q_range"] = flag_out_of_range(hr)
    out["q_spike"] = flag_spikes(hr)
    out["q_flat"] = flag_flatline(hr, moving=moving)
    if "cadence_spm" in out and out["cadence_spm"].notna().any():
        out["q_cadence_lock"] = flag_cadence_lock(hr, out["cadence_spm"])
        out["q_soft_lock"] = flag_soft_lock(
            hr, out["cadence_spm"], out.get("speed"), out.get("altitude"), out.get("distance"))
    else:
        out["q_cadence_lock"] = False
        out["q_soft_lock"] = False

    q_cols = [c for c in out.columns if c.startswith("q_")]
    out["hr_ok"] = ~out[q_cols].any(axis=1)
    return out


def summarize(checked: pd.DataFrame) -> dict[str, float]:
    """Share of samples flagged by each check, plus overall usable share."""
    q_cols = [c for c in checked.columns if c.startswith("q_")]
    summary = {c.removeprefix("q_"): float(checked[c].mean()) for c in q_cols}
    summary["usable"] = float(checked["hr_ok"].mean())
    return summary
