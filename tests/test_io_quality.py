import numpy as np
import pandas as pd
import pytest

from heatsense.io_fit import read_fit, to_minutes
from heatsense.quality import check_hr, flag_cadence_lock, flag_flatline, flag_spikes, summarize
from heatsense.synthetic import make_hot_run


def test_read_fit_roundtrip(fit_path, synthetic_run):
    df = read_fit(fit_path)
    assert len(df) == len(synthetic_run)
    assert df.index.tz is not None
    assert df["speed"].mean() == pytest.approx(synthetic_run["speed"].mean(), rel=0.01)
    # Garmin stores strides/min; the reader converts to steps/min.
    assert df["cadence_spm"].mean() == pytest.approx(synthetic_run["cadence_spm"].mean(), abs=2)
    assert df["core_temp"].notna().all()


def test_gps_dropped_by_default(fit_path):
    assert "lat" not in read_fit(fit_path).columns
    kept = read_fit(fit_path, keep_gps=True)
    assert kept["lat"].iloc[0] == pytest.approx(51.50, abs=1e-4)


def test_cadence_lock_is_caught():
    run = make_hot_run(minutes=50, cadence_lock=(40, 43), dropout=None)
    flags = flag_cadence_lock(run["heart_rate"], run["cadence_spm"])
    window = flags.iloc[40 * 60 + 30 : 43 * 60 - 30]
    assert window.mean() > 0.9
    assert flags.iloc[: 38 * 60].mean() < 0.01


def test_hr_near_cadence_but_independent_is_not_flagged():
    # Regression from a real run: HR drifts 161 -> 168 while cadence wanders 157-167
    # independently. Close values, no tracking: this is real heart rate.
    rng = np.random.default_rng(0)
    n = 80 * 60
    idx = pd.date_range("2026-10-12", periods=n, freq="s", tz="UTC")
    hr = pd.Series(np.round(161 + 7 * np.arange(n) / n + rng.normal(0, 1, n)), index=idx)
    walk = np.cumsum(rng.normal(0, 0.05, n))
    cadence = pd.Series(163 + 4 * np.sin(np.arange(n) / 400) + walk - walk.mean(), index=idx)
    assert flag_cadence_lock(hr, cadence).mean() < 0.02


def _soft_lock_run(seed=1):
    # Mirrors a real run (2026-10-18): HR ~178 runs 14 bpm above cadence ~164, then snaps
    # down to ~169 (near cadence) at minute 20 for 6 minutes and snaps back, at steady pace.
    rng = np.random.default_rng(seed)
    n = 40 * 60
    idx = pd.date_range("2026-10-18", periods=n, freq="s", tz="UTC")
    hr = np.full(n, 178.0) + rng.normal(0, 1.0, n)
    hr[20 * 60 : 26 * 60] = 169 + rng.normal(0, 1.0, 6 * 60)
    cadence = 164 + rng.normal(0, 1.5, n)
    speed = np.full(n, 2.6) + rng.normal(0, 0.03, n)
    return pd.DataFrame({"heart_rate": np.round(hr), "cadence_spm": cadence, "speed": speed}, index=idx)


def test_soft_lock_plateau_is_caught():
    from heatsense.quality import flag_soft_lock
    run = _soft_lock_run()
    f = flag_soft_lock(run["heart_rate"], run["cadence_spm"], run["speed"])
    assert f.iloc[20 * 60 + 20 : 26 * 60 - 20].mean() > 0.95
    assert f.iloc[: 19 * 60].mean() == 0
    assert f.iloc[27 * 60 :].mean() == 0


def test_soft_lock_ignores_effort_change():
    # Same drop, but because the runner slowed down: that's real physiology, not a fault.
    from heatsense.quality import flag_soft_lock
    run = _soft_lock_run()
    run.iloc[20 * 60 - 10 : 26 * 60 + 10, run.columns.get_loc("speed")] = 2.0
    f = flag_soft_lock(run["heart_rate"], run["cadence_spm"], run["speed"])
    assert f.mean() < 0.02


def test_spikes_and_flatline():
    idx = pd.date_range("2026-07-01", periods=200, freq="s", tz="UTC")
    hr = pd.Series(np.r_[np.linspace(120, 140, 100), np.full(100, 140.0)], index=idx)
    hr.iloc[50] = 200
    assert flag_spikes(hr).iloc[50]
    assert flag_flatline(hr, min_seconds=60).iloc[150]


def test_timer_pause_keeps_real_timeline():
    # Regression from a real run: auto-pause left a ~5-minute hole in the records.
    run = make_hot_run(minutes=30, cadence_lock=None, dropout=None)
    run = run.drop(run.index[10 * 60 : 15 * 60])  # timer off from minute 10 to 15
    minutes = to_minutes(check_hr(run))
    assert len(minutes) == 30  # the 5 empty minutes are kept, not collapsed
    assert minutes.loc[10:14, "recorded"].eq(0).all()
    assert minutes.loc[10:14, "paused"].all()
    assert minutes.loc[10:14, "hr"].isna().all()
    assert not minutes.loc[16:28, "paused"].any()
    assert minutes.loc[12, "time"] == run.index[0] + pd.Timedelta(minutes=12)


def test_standing_stop_with_timer_running_is_paused():
    run = make_hot_run(minutes=20, cadence_lock=None, dropout=None)
    stop = slice(8 * 60, 11 * 60)
    run.iloc[stop, run.columns.get_loc("speed")] = 0.0
    run.iloc[stop, run.columns.get_loc("cadence_spm")] = 0.0
    minutes = to_minutes(check_hr(run))
    assert minutes.loc[8:10, "paused"].all()
    assert minutes.loc[8:10, "recorded"].eq(1).all()


def test_checked_minutes_mark_bad_minutes_as_gaps():
    run = make_hot_run(minutes=60, cadence_lock=(40, 43), dropout=(55, 56))
    checked = check_hr(run)
    minutes = to_minutes(checked)
    assert len(minutes) == 60
    assert minutes.loc[41, "hr"] != minutes.loc[41, "hr"]  # NaN during cadence lock
    assert np.isnan(minutes.loc[55, "hr"])  # NaN during dropout
    assert minutes["hr"].notna().sum() >= 55
    s = summarize(checked)
    assert 0.9 < s["usable"] < 1.0
