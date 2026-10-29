import numpy as np
import pandas as pd
import pytest

from heatsense.weather import for_minutes, heat_index_c, hourly_frame, wbgt_bom_c, weather_url


def test_url_rounds_coordinates_for_privacy():
    url = weather_url(51.501234, -0.124321, "2026-09-21")
    assert "latitude=51.50" in url and "longitude=-0.12" in url
    assert "51.5012" not in url


def test_heat_index_matches_nws_table():
    # NWS chart: 90 F / 60% RH -> ~100 F;  95 F / 50% -> ~105 F.
    assert heat_index_c((90 - 32) * 5 / 9, 60) == pytest.approx((100 - 32) * 5 / 9, abs=0.7)
    assert heat_index_c((95 - 32) * 5 / 9, 50) == pytest.approx((105 - 32) * 5 / 9, abs=0.7)
    # Mild weather: heat index stays near air temperature.
    assert heat_index_c(18, 50) == pytest.approx(18, abs=1.5)


def test_wbgt_estimate_is_sane():
    assert 18 < wbgt_bom_c(20, 50) < 23
    assert wbgt_bom_c(32, 70) > 30  # hot and humid: high heat load
    assert wbgt_bom_c(30, 80) > wbgt_bom_c(30, 30)


def test_interpolates_hourly_to_minutes():
    data = {"hourly": {
        "time": ["2026-09-21T15:00", "2026-09-21T16:00", "2026-09-21T17:00"],
        "temperature_2m": [20.0, 22.0, 24.0], "relative_humidity_2m": [60, 55, 50],
        "wind_speed_10m": [2, 3, 4], "shortwave_radiation": [400, 500, 600]}}
    hourly = hourly_frame(data)
    times = pd.Series(pd.to_datetime(["2026-09-21 15:30", "2026-09-21 16:45"], utc=True))
    w = for_minutes(times, hourly)
    assert w["air_temp"].tolist() == pytest.approx([21.0, 23.5])
    assert {"heat_index", "wbgt_est"} <= set(w.columns)


def test_run_weather_uses_cache_and_crosses_midnight(tmp_path):
    import json
    from heatsense.weather import run_weather
    for day, temps in [("2026-09-16", [20.0 + h * 0.1 for h in range(24)]),
                       ("2026-09-17", [18.0] * 24)]:
        times = [f"{day}T{h:02d}:00" for h in range(24)]
        data = {"hourly": {"time": times, "temperature_2m": temps,
                           "relative_humidity_2m": [60] * 24, "wind_speed_10m": [2] * 24,
                           "shortwave_radiation": [0] * 24}}
        (tmp_path / f"51.50_-0.12_{day}.json").write_text(json.dumps(data))
    minutes = pd.DataFrame({"time": pd.date_range("2026-09-16 23:30", periods=60, freq="min", tz="UTC")})
    w = run_weather(minutes, 51.501, -0.122, cache_dir=tmp_path)  # no network needed
    assert len(w) == 60
    assert w["air_temp"].iloc[0] == pytest.approx(22.3 + 0.5 * (18.0 - 22.3), abs=0.01)
    assert w["air_temp"].iloc[-1] == pytest.approx(18.0, abs=0.01)
