"""Synthetic hot runs with a known 'true' core temperature, for tests and demos.

The ground truth here comes from a simple heat-balance model, NOT from the baseline
filter, so comparing the filter against it is a fair (if idealised) test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_hot_run(
    minutes: int = 75,
    air_temp_c: float = 30.0,
    pace_min_per_km: float = 5.5,
    seed: int = 7,
    cadence_lock: tuple[int, int] | None = (40, 43),
    dropout: tuple[int, int] | None = (55, 56),
) -> pd.DataFrame:
    """Per-second synthetic run.

    Returns columns heart_rate, speed, cadence_spm, true_ct. ``cadence_lock`` and
    ``dropout`` are (start_minute, end_minute) windows of injected sensor faults.
    """
    rng = np.random.default_rng(seed)
    n = minutes * 60
    t = np.arange(n)
    idx = pd.date_range("2026-07-12 11:00", periods=n, freq="s", tz="UTC")

    speed = np.full(n, 1000 / (pace_min_per_km * 60)) + rng.normal(0, 0.05, n)
    speed[:180] *= np.linspace(0.6, 1.0, 180)  # gentle start
    cadence = 166 + 6 * (speed - speed.mean()) + rng.normal(0, 1.5, n)

    # Heat balance: core temp rises toward an equilibrium set by heat load, ~20 min lag,
    # plus slow uncompensable drift in hot air.
    heat_load = (speed / 3.0) * (1 + max(air_temp_c - 15, 0) / 30)
    equilibrium = 37.0 + 1.2 * heat_load + 0.003 * max(air_temp_c - 25, 0) * (t / 60)
    true_ct = np.empty(n)
    ct = 37.0
    for i in range(n):
        ct += (equilibrium[i] - ct) / (20 * 60)
        true_ct[i] = ct

    # Heart rate: driven by core temp, plus effort and cardiac drift, plus sensor noise.
    hr = 60 + 22 * (true_ct - 36.8) + 18 * speed + 0.12 * (t / 60) + rng.normal(0, 2.0, n)

    if cadence_lock:
        a, b = (min(m * 60, n) for m in cadence_lock)
        hr[a:b] = cadence[a:b] + rng.normal(0, 1.0, b - a)
    if dropout:
        a, b = (min(m * 60, n) for m in dropout)
        hr[a:b] = np.nan

    return pd.DataFrame(
        {"heart_rate": np.round(hr), "speed": speed, "cadence_spm": cadence, "true_ct": true_ct},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


# ----------------------------------------------------------------------------------------
# Synthetic athlete for model evaluation.
#
# Deliberately built with DIFFERENT equations from heatsense.model (multiplicative heat
# stress, asymmetric heating/cooling time constants, uncompensable creep in hot weather,
# non-linear heart-rate response, dehydration drift, AR(1) noise, sensor artifacts), so
# the model can't score well just by matching its own assumptions. Still an idealisation:
# real validation needs a core-temperature reference.
# ----------------------------------------------------------------------------------------


def make_athlete_run(seed: int, minutes: int | None = None,
                     heat_gain: float = 1.0, hr_offset: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """One per-second run plus its weather. Returns (seconds, info).

    ``heat_gain`` scales how strongly this athlete heats up (1 = typical; 0.7 = heat
    tolerant; 1.4 = heats up fast). ``hr_offset`` shifts their heart rate (bpm).
    """
    from .effort import REST_W_PER_KG, minetti_cost
    from .weather import wbgt_bom_c

    rng = np.random.default_rng(seed)
    minutes = minutes or int(rng.integers(45, 100))
    n = minutes * 60
    air = float(rng.uniform(8, 33))
    rh = float(rng.uniform(30, 85))
    wbgt = float(wbgt_bom_c(air, rh))
    indoor = bool(rng.random() < 0.2)
    if indoor:
        air, rh, wbgt = 20.0, 45.0, 18.0

    # Pace: base speed with slow wander, a gentle start, optional faster finish.
    speed = float(rng.uniform(2.4, 3.4)) + np.cumsum(rng.normal(0, 0.004, n))
    speed = np.clip(speed, 1.8, 4.5)
    speed[:180] *= np.linspace(0.6, 1.0, 180)
    if rng.random() < 0.4:
        speed[-8 * 60:] *= 1.15
    # Hills (outdoor only).
    grade = np.zeros(n)
    if not indoor:
        for _ in range(int(rng.integers(0, 4))):
            a = int(rng.integers(5 * 60, n - 4 * 60))
            grade[a:a + int(rng.integers(60, 240))] = rng.uniform(-0.06, 0.06)
    # A stop (traffic light, water) in half the outdoor runs.
    stopped = np.zeros(n, bool)
    if not indoor and rng.random() < 0.5:
        a = int(rng.integers(15 * 60, n - 10 * 60))
        stopped[a:a + int(rng.integers(90, 300))] = True
    speed[stopped] = 0.0

    met = np.where(speed > 0.5, minetti_cost(grade) * speed, REST_W_PER_KG)

    # True core temperature (1 s steps).
    ct = np.empty(n)
    x = 37.0 + rng.normal(0, 0.1)
    for i in range(n):
        eq = 36.9 + heat_gain * 0.13 * (met[i] - REST_W_PER_KG) * (1 + 0.06 * max(wbgt - 12, 0))
        # Realistic slow passive cooling (~0.04 degC/min from 39.5); deliberately not the
        # model's 45 min, and no post-stop delay, so the evaluation isn't circular.
        tau = 25 * 60 if eq > x else 60 * 60
        creep = 0.004 / 60 * max(wbgt - 26, 0) * (met[i] > 4)
        x += (eq - x) / tau + creep
        ct[i] = x

    # True heart rate: lagged effort, non-linear heat response, dehydration drift.
    lag = np.empty(n)
    acc = REST_W_PER_KG
    for i in range(n):
        acc += (met[i] - acc) / 40
        lag[i] = acc
    t_min = np.arange(n) / 60
    hr = (52 + hr_offset + float(rng.uniform(6, 8)) * lag + 12 * (ct - 37) + 4 * (ct - 37) ** 2
          + 0.0015 * t_min * air)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.98 * noise[i - 1] + rng.normal(0, 0.5)
    hr = -3 * np.logaddexp(-(hr + noise) / 3, -196 / 3)

    cadence = np.where(speed > 0.5, 150 + 7 * speed + rng.normal(0, 1.5, n), 0.0)
    # Sensor artifact: soft lock toward cadence for a few minutes in some runs.
    if rng.random() < 0.3 and minutes > 30:
        a = int(rng.integers(10 * 60, n - 8 * 60))
        b = a + int(rng.integers(180, 360))
        if not stopped[a:b].any():
            hr[a:b] = cadence[a:b] + 4 + rng.normal(0, 0.8, b - a)

    alt = 20 + np.cumsum(grade * speed)
    dist = np.cumsum(speed)
    keep = ~stopped | (rng.random(n) < 0.3)  # auto-pause drops most stopped seconds
    idx = pd.date_range("2026-07-01 12:00", periods=n, freq="s", tz="UTC")
    df = pd.DataFrame({"heart_rate": np.round(hr), "speed": speed, "cadence_spm": cadence,
                       "distance": dist, "altitude": alt, "true_ct": ct}, index=idx)[keep]
    return df, {"air": air, "rh": rh, "wbgt": wbgt, "indoor": indoor, "minutes": minutes}
