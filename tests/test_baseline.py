import numpy as np
import pandas as pd
import pytest

from heatsense.baseline import BaselineParams, expected_hr, run_baseline


def test_expected_hr_is_increasing_in_physiological_range():
    ct = np.linspace(36.5, 40.5, 50)
    assert np.all(np.diff(expected_hr(ct)) > 0)


def test_constant_hr_converges_to_matching_temperature():
    # Pick a temperature, compute the HR the model expects there, feed it for 3 hours.
    target = 38.6
    hr = np.full(180, float(expected_hr(target)))
    out = run_baseline(hr)
    assert out["ct"].iloc[-1] == pytest.approx(target, abs=0.05)


def test_rising_hr_gives_rising_estimate():
    hr = np.linspace(110, 165, 90)
    ct = run_baseline(hr)["ct"].to_numpy()
    assert ct[-1] > ct[0] + 1.0
    # Smooth: no single-minute jump bigger than a few hundredths of a degree.
    assert np.max(np.abs(np.diff(ct))) < 0.06


def test_gap_carries_estimate_and_grows_uncertainty():
    hr = np.r_[np.full(30, 150.0), np.full(10, np.nan), np.full(5, 150.0)]
    out = run_baseline(hr)
    gap = out.iloc[30:40]
    assert np.allclose(gap["ct"], out["ct"].iloc[29])
    assert np.all(np.diff(gap["ct_sd"]) > 0)


def test_keeps_series_index():
    idx = pd.date_range("2026-07-01 07:00", periods=5, freq="min")
    out = run_baseline(pd.Series([100, 110, 120, 125, 130.0], index=idx))
    assert out.index.equals(idx)


def test_starting_point_is_configurable():
    out = run_baseline(np.full(1, np.nan), BaselineParams(ct0=37.5))
    assert out["ct"].iloc[0] == pytest.approx(37.5)
