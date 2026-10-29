"""Accuracy metrics against a reference core temperature."""

from __future__ import annotations

import numpy as np


def compare(estimate, reference, skip_minutes: int = 0) -> dict[str, float]:
    """Bias, RMSE and 95% limits of agreement (Bland-Altman) of estimate - reference.

    ``skip_minutes`` drops the warm-up, when every estimator is still dominated by its
    starting value. Report results both with and without it.
    """
    est = np.asarray(estimate, dtype=float)[skip_minutes:]
    ref = np.asarray(reference, dtype=float)[skip_minutes:]
    ok = ~(np.isnan(est) | np.isnan(ref))
    err = est[ok] - ref[ok]
    if err.size == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "loa_low": np.nan, "loa_high": np.nan}
    bias = float(err.mean())
    sd = float(err.std(ddof=1)) if err.size > 1 else 0.0
    return {
        "n": int(err.size),
        "bias": bias,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
    }
