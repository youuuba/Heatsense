"""Shared fixtures: write a real .FIT file from a synthetic run."""

import pytest

from heatsense.synthetic import make_hot_run


def write_fit(df, path, with_gps=True, with_core=False):
    from fit_tool.fit_file_builder import FitFileBuilder
    from fit_tool.profile.messages.file_id_message import FileIdMessage
    from fit_tool.profile.messages.record_message import RecordMessage
    from fit_tool.profile.profile_type import FileType, Manufacturer

    builder = FitFileBuilder(auto_define=True, min_string_size=50)
    start_ms = int(df.index[0].timestamp() * 1000)

    fid = FileIdMessage()
    fid.type = FileType.ACTIVITY
    fid.manufacturer = Manufacturer.DEVELOPMENT.value
    fid.product = 0
    fid.serial_number = 12345
    fid.time_created = start_ms
    builder.add(fid)

    distance = 0.0
    for ts, row in df.iterrows():
        rec = RecordMessage()
        rec.timestamp = int(ts.timestamp() * 1000)
        if row["heart_rate"] == row["heart_rate"]:  # not NaN
            rec.heart_rate = int(row["heart_rate"])
        rec.enhanced_speed = float(row["speed"])
        rec.cadence = int(row["cadence_spm"] // 2)
        distance += float(row["speed"])
        rec.distance = distance
        if with_gps:
            rec.position_lat = 51.50
            rec.position_long = -0.12
        if with_core:
            rec.core_temperature = float(row["true_ct"])
        builder.add(rec)

    builder.build().to_file(str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_run():
    return make_hot_run(minutes=30)


@pytest.fixture(scope="session")
def fit_path(tmp_path_factory, synthetic_run):
    return write_fit(synthetic_run, tmp_path_factory.mktemp("fit") / "run.fit", with_core=True)
