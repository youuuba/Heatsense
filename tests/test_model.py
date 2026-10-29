import dataclasses

import numpy as np
import pandas as pd
import pytest

from heatsense.effort import REST_W_PER_KG, minetti_cost
from heatsense.evaluate import evaluate_synthetic, prepare
from heatsense.model import (PhysicsParams, Profile, build_features, calibrate, ct_equilibrium,
                             physics_only, predict_hr, run_heatsense, strain_zones)


def _feat(minutes=60, met=10.0, wbgt=18.0, hr=150.0, paused=None):
    n = minutes
    m = pd.DataFrame({"hr": np.full(n, hr), "paused": np.zeros(n, bool) if paused is None else paused,
                      "hr_coverage": np.ones(n)}, index=pd.RangeIndex(n, name="minute"))
    e = pd.DataFrame({"met_power": np.full(n, met)}, index=m.index)
    e.loc[m["paused"].to_numpy(), "met_power"] = REST_W_PER_KG
    return build_features(m, e, wbgt)


def test_minetti_cost_flat_and_hills():
    assert minetti_cost(0.0) == pytest.approx(3.6)
    assert minetti_cost(0.1) > minetti_cost(0.0) > minetti_cost(-0.1)


def test_physics_heats_with_effort_and_heat_and_cools_at_rest():
    p = PhysicsParams()
    assert ct_equilibrium(10, 18, p) > ct_equilibrium(6, 18, p)
    assert ct_equilibrium(10, 30, p) > ct_equilibrium(10, 18, p)
    paused = np.r_[np.zeros(40, bool), np.ones(20, bool)]
    ct = physics_only(_feat(paused=paused))
    assert ct[39] > 38.0
    assert ct[-1] < ct[39] - 0.3  # cools during the stop


def test_heart_rate_saturates_at_max():
    prof = Profile(hr_max=190)
    assert predict_hr(41.0, 14.0, 90, prof) < 190.5


def test_filter_never_changes_parameters():
    prof = Profile(hr_base=100.0)
    before = dataclasses.asdict(prof)
    run_heatsense(_feat(hr=175.0), prof)
    assert dataclasses.asdict(prof) == before


def test_higher_than_expected_hr_raises_estimate():
    low = run_heatsense(_feat(hr=140.0))["ct"].iloc[-1]
    high = run_heatsense(_feat(hr=180.0))["ct"].iloc[-1]
    assert high > low + 0.2


def test_missing_hr_falls_back_to_physics_with_growing_uncertainty():
    f = _feat()
    f.loc[30:, "hr"] = np.nan
    out = run_heatsense(f)
    assert out["innovation"].iloc[30:].isna().all()
    assert out["ct_sd"].iloc[-1] > out["ct_sd"].iloc[29]


def test_calibration_learns_heart_rate_offset():
    lo = calibrate([prepare(7000 + i, 1.0, -10.0)["feat"] for i in range(5)])
    hi = calibrate([prepare(7000 + i, 1.0, +10.0)["feat"] for i in range(5)])
    # Same runs, heart rate shifted by 20 bpm: the fitted curve should shift by ~20 bpm.
    m = 10.0
    shift = (hi.hr_base + hi.hr_per_effort * m) - (lo.hr_base + lo.hr_per_effort * m)
    assert shift == pytest.approx(20.0, abs=4.0)
    assert hi.heat_gain == 1.0  # never fitted from heart rate alone


def test_calibration_with_too_little_data_returns_default():
    prof = calibrate([_feat(minutes=6)])
    assert prof.source == "population default"
    assert "not enough data" in prof.notes[0]


def test_profile_roundtrip(tmp_path):
    prof = Profile(hr_base=111.0, notes=["x"])
    prof.save(tmp_path / "p.json")
    assert Profile.load(tmp_path / "p.json") == prof


def test_zones():
    f = _feat(minutes=30)
    est = pd.DataFrame({"ct": np.r_[np.full(10, 37.5), np.linspace(37.5, 39.0, 10), np.full(10, 39.9)],
                        "ct_sd": np.full(30, 0.1)}, index=f.index)
    z = strain_zones(est, f)
    assert z.iloc[5] == "green"
    assert z.iloc[-1] == "red"
    f.loc[25:, "hr"] = np.nan
    assert strain_zones(est, f).iloc[-1] == "check"


def test_heatsense_beats_reference_baseline_on_simulated_athlete():
    # Regression guard for the headline result (small, fast version of `heatsense evaluate`).
    summary, *_ = evaluate_synthetic(n_calibration=5, n_test=8, heat_gain=1.2, hr_offset=5.0, seed0=4242)
    rmse = summary["rmse_C"]
    assert rmse["HeatSense, calibrated"] < rmse["Army baseline (HR only)"]
