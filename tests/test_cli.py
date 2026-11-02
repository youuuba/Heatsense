import pytest

from heatsense.cli import expand_paths


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
