"""How hard the runner is working, minute by minute.

Effort is expressed as metabolic power per kilogram (W/kg), from speed and slope using
the energy cost of running on gradients from Minetti et al. (2002), J Appl Physiol
93(3):1039-1046. On the flat it gives ~3.6 J/kg/m, so running at 3 m/s is ~11 W/kg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REST_W_PER_KG = 1.5  # standing / stopped (about 1.3 MET)


def minetti_cost(grade) -> np.ndarray:
    """Energy cost of running, J/kg/m, for grade as a fraction (0.05 = 5% uphill)."""
    i = np.clip(np.asarray(grade, dtype=float), -0.45, 0.45)
    return 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6


def grade_per_second(df: pd.DataFrame, half_window_s: int = 15) -> pd.Series:
    """Slope over a ~30 s window from altitude and distance; 0 when unavailable (treadmill)."""
    if "altitude" not in df or df["altitude"].notna().sum() < 10 or "distance" not in df:
        return pd.Series(0.0, index=df.index)
    alt = df["altitude"].interpolate(limit_direction="both")
    dist = df["distance"].interpolate(limit_direction="both")
    d_alt = alt.shift(-half_window_s) - alt.shift(half_window_s)
    d_dist = dist.shift(-half_window_s) - dist.shift(half_window_s)
    grade = (d_alt / d_dist.where(d_dist > 10)).clip(-0.3, 0.3)
    return grade.fillna(0.0)


def metabolic_power(df: pd.DataFrame) -> pd.Series:
    """Per-second metabolic power, W/kg. Stopped seconds get the resting value."""
    speed = df["speed"].fillna(0.0)
    grade = grade_per_second(df)
    power = minetti_cost(grade) * speed
    return power.where(speed > 0.5, REST_W_PER_KG).clip(lower=REST_W_PER_KG)


def minute_effort(df: pd.DataFrame, n_minutes: int) -> pd.DataFrame:
    """Per-minute mean metabolic power and grade, aligned with ``io_fit.to_minutes``.

    Seconds with no record (watch timer paused) count as rest, so a minute that was half
    paused gets half the effort.
    """
    elapsed = (df.index - df.index[0]).total_seconds()
    minute = (elapsed // 60).astype(int).to_numpy()
    power = metabolic_power(df)
    grade = grade_per_second(df)
    idx = pd.RangeIndex(0, n_minutes, name="minute")
    recorded_work = pd.Series(power.to_numpy() - REST_W_PER_KG).groupby(minute).sum()
    met = REST_W_PER_KG + recorded_work.reindex(idx, fill_value=0.0) / 60.0
    return pd.DataFrame({
        "met_power": met.to_numpy(),
        "grade": pd.Series(grade.to_numpy()).groupby(minute).mean().reindex(idx).fillna(0.0).to_numpy(),
    }, index=idx)
