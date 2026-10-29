"""Reference baseline: the published heart-rate-only extended Kalman filter.

OFFLINE REFERENCE ONLY. This module re-implements the method published in
Buller et al. (2013), Physiological Measurement 34(7):781-798, so that HeatSense's own
model can be compared against it on recorded data. The method is patented by the US Army
(US 10,702,165). It is run on a computer against recorded runs and is never shipped in a
watch or mobile app. See README.md.

Model, updated once per minute:

    state:        CT_t = a * CT_{t-1} + w,    w ~ N(0, gamma)
    observation:  HR_t = b2*CT_t^2 + b1*CT_t + b0 + v,    v ~ N(0, sigma)

The observation curve is linearised at the predicted state (extended Kalman filter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineParams:
    """Published population parameters (Buller et al., 2013)."""

    a: float = 1.0
    gamma: float = 0.022**2  # process variance, degC^2 per minute
    b0: float = -7887.1
    b1: float = 384.4286
    b2: float = -4.5714
    sigma: float = 18.88**2  # observation variance, bpm^2
    ct0: float = 37.1  # starting core temperature, degC
    v0: float = 0.0  # starting variance


def expected_hr(ct: np.ndarray | float, p: BaselineParams = BaselineParams()):
    """Heart rate the model expects at a given core temperature."""
    ct = np.asarray(ct, dtype=float)
    return p.b2 * ct**2 + p.b1 * ct + p.b0


def run_baseline(
    hr_per_minute: np.ndarray | pd.Series,
    params: BaselineParams = BaselineParams(),
) -> pd.DataFrame:
    """Run the filter over a sequence of 1-minute mean heart rates.

    Missing minutes (NaN) skip the observation update: the estimate carries forward
    and its uncertainty grows, which is the correct Kalman behaviour for a gap.

    Returns a DataFrame with columns ``ct`` (estimate, degC), ``ct_sd`` (1 standard
    deviation, degC) and ``gain``.
    """
    p = params
    index = hr_per_minute.index if isinstance(hr_per_minute, pd.Series) else None
    hr = np.asarray(hr_per_minute, dtype=float)

    n = hr.size
    ct = np.empty(n)
    var = np.empty(n)
    gain = np.full(n, np.nan)

    x, v = p.ct0, p.v0
    for t in range(n):
        # Time update.
        x_pred = p.a * x
        v_pred = p.a**2 * v + p.gamma

        if np.isnan(hr[t]):
            x, v = x_pred, v_pred
        else:
            # Observation update, linearised at x_pred.
            c = 2.0 * p.b2 * x_pred + p.b1
            k = v_pred * c / (c**2 * v_pred + p.sigma)
            x = x_pred + k * (hr[t] - expected_hr(x_pred, p))
            v = (1.0 - k * c) * v_pred
            gain[t] = k

        ct[t], var[t] = x, v

    return pd.DataFrame({"ct": ct, "ct_sd": np.sqrt(var), "gain": gain}, index=index)
