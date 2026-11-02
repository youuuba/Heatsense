import numpy as np
import pandas as pd
import pytest

from heatsense import cli
from heatsense.cli import choose_weather, expand_paths
from heatsense.io_fit import to_minutes
from heatsense.model import PhysicsParams, Profile
from heatsense.quality import check_hr
from heatsense.synthetic import make_hot_run

LONDON = (51.5, -0.12)


@pytest.fixture
def minutes():
    return to_minutes(check_hr(make_hot_run(minutes=20)))


def _touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


def test_wildcard_expands_to_sorted_matches(tmp_path):
    _touch(tmp_path, "b.fit", "a.fit", "notes.txt")
    got = expand_paths([tmp_path / "*.fit"])
    assert [p.name for p in got] == ["a.fit", "b.fit"]


def test_literal_paths_pass_through(tmp_path):
    _touch(tmp_path, "run.fit")
    assert [p.name for p in expand_paths([tmp_path / "run.fit"])] == ["run.fit"]


def test_same_file_listed_once(tmp_path):
    _touch(tmp_path, "run.fit")
    # Overlapping patterns plus the literal path all point at one file.
    got = expand_paths([tmp_path / "*.fit", tmp_path / "r*.fit", tmp_path / "run.fit"])
    assert len(got) == 1


def test_no_match_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="No files match"):
        expand_paths([tmp_path / "*.fit"])


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="File not found"):
        expand_paths([tmp_path / "missing.fit"])


def test_weather_from_open_meteo(minutes, monkeypatch):
    def fake(m, lat, lon):
        n = len(m)
        return pd.DataFrame({"air_temp": np.full(n, 31.0), "rh": np.full(n, 50.0),
                             "wbgt_est": np.full(n, 29.0)}, index=m.index)
    monkeypatch.setattr(cli, "run_weather", fake)
    wbgt, wx, source, note = choose_weather(minutes, LONDON)
    assert source == "open-meteo" and wx is not None
    assert np.allclose(wbgt, 29.0) and "Open-Meteo" in note


def test_weather_manual(minutes):
    wbgt, wx, source, note = choose_weather(minutes, LONDON, fixed_wbgt=24.0)
    assert (wbgt, wx, source) == (24.0, None, "manual")
    assert "manually" in note


def test_indoor_is_labelled_assumption(minutes):
    wbgt, _, source, note = choose_weather(minutes, None)
    assert wbgt == PhysicsParams().indoor_wbgt == 18.0
    assert source == "indoor-assumed" and "ASSUMED" in note


def test_failed_lookup_is_conservative_not_mild(minutes, monkeypatch):
    def broken(*a):
        raise ConnectionError("offline")
    monkeypatch.setattr(cli, "run_weather", broken)
    wbgt, wx, source, note = choose_weather(minutes, LONDON)
    assert wbgt == PhysicsParams().unknown_wbgt == 28.0 and wx is None
    assert source == "unknown-conservative"
    assert "ConnectionError" in note and "ASSUMED" in note


def test_skipped_lookup_is_conservative(minutes):
    wbgt, _, source, _ = choose_weather(minutes, LONDON, weather=False)
    assert (wbgt, source) == (28.0, "unknown-conservative")


def test_source_reaches_csv_console_and_plot_note(tmp_path, capsys, monkeypatch):
    notes = []
    monkeypatch.setattr(cli, "plot_run", lambda *a, note=None, **k: notes.append(note))
    cli._analyze(make_hot_run(minutes=20), "t", tmp_path / "x.png", tmp_path / "x.csv", Profile())
    table = pd.read_csv(tmp_path / "x.csv")
    assert (table["wx_source"] == "indoor-assumed").all()  # synthetic run has no GPS
    assert "Weather source: indoor-assumed" in capsys.readouterr().out
    assert "Weather source: indoor-assumed" in notes[0]
