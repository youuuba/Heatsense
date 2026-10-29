"""Historical weather for a run, plus heat-stress indices.

Data source: Open-Meteo (free, no API key, https://open-meteo.com). Its historical-forecast
API covers 2022 onward with only a short delay, so recent runs work.

Privacy: coordinates are rounded to 0.01 degrees (about 1 km) before any request, so the
lookup never contains an exact route or home location.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]
DEFAULT_CACHE = Path.home() / ".cache" / "heatsense" / "weather"


def weather_url(lat: float, lon: float, day: str) -> str:
    q = {
        "latitude": f"{round(lat, 2):.2f}",
        "longitude": f"{round(lon, 2):.2f}",
        "start_date": day,
        "end_date": day,
        "hourly": ",".join(HOURLY),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    return f"{API}?{urllib.parse.urlencode(q)}"


def fetch_day(lat: float, lon: float, day: str, cache_dir: Path | None = DEFAULT_CACHE) -> dict:
    """Open-Meteo JSON for one UTC day, cached on disk (weather for a past day never changes)."""
    url = weather_url(lat, lon, day)
    key = f"{round(lat, 2):.2f}_{round(lon, 2):.2f}_{day}.json"
    if cache_dir is not None and (cache_dir / key).exists():
        return json.loads((cache_dir / key).read_text())
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.load(resp)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / key).write_text(json.dumps(data))
    return data


def hourly_frame(data: dict) -> pd.DataFrame:
    h = data["hourly"]
    df = pd.DataFrame({k: h[k] for k in HOURLY}, index=pd.to_datetime(h["time"], utc=True))
    return df.rename(columns={
        "temperature_2m": "air_temp", "relative_humidity_2m": "rh",
        "wind_speed_10m": "wind", "shortwave_radiation": "solar",
    }).astype(float)


def for_minutes(times: pd.Series, hourly: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate hourly weather to each minute's timestamp, then add indices."""
    t = pd.DatetimeIndex(times)
    x_h = hourly.index.asi8.astype(float)
    x_m = t.asi8.astype(float)
    out = pd.DataFrame({c: np.interp(x_m, x_h, hourly[c].to_numpy()) for c in hourly.columns},
                       index=times.index)
    out["heat_index"] = heat_index_c(out["air_temp"], out["rh"])
    out["wbgt_est"] = wbgt_bom_c(out["air_temp"], out["rh"])
    return out


def run_weather(minutes: pd.DataFrame, lat: float, lon: float, cache_dir=DEFAULT_CACHE):
    """Weather columns for a minute table (handles runs that cross midnight UTC)."""
    days = sorted({ts.strftime("%Y-%m-%d") for ts in minutes["time"]})
    hourly = pd.concat([hourly_frame(fetch_day(lat, lon, d, cache_dir)) for d in days])
    return for_minutes(minutes["time"], hourly.sort_index())


def heat_index_c(t_c, rh):
    """US National Weather Service heat index (Rothfusz regression), in degC.

    Below about 27 degC the regression doesn't apply and the simple Steadman form is used,
    which stays close to air temperature.
    """
    t_c, rh = np.asarray(t_c, float), np.asarray(rh, float)
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
            - 6.83783e-3 * t**2 - 5.481717e-2 * rh**2 + 1.22874e-3 * t**2 * rh
            + 8.5282e-4 * t * rh**2 - 1.99e-6 * t**2 * rh**2)
    adj = np.where((rh < 13) & (t >= 80) & (t <= 112),
                   -((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95)) / 17, 0, None)), 0.0)
    adj = np.where((rh > 85) & (t >= 80) & (t <= 87), ((rh - 85) / 10) * ((87 - t) / 5), adj)
    hi_f = np.where((simple + t) / 2 >= 80, full + adj, simple)
    return (hi_f - 32) * 5 / 9


def wbgt_bom_c(t_c, rh):
    """Approximate outdoor WBGT, degC (Australian Bureau of Meteorology formula).

    WBGT ~ 0.567*Ta + 0.393*e + 3.94, with e the vapour pressure in hPa. It assumes
    moderately high sun and light wind, so it is a rough heat-load index, not a measured
    WBGT. A full model (e.g. Liljegren) using sun and wind is a planned upgrade.
    """
    t_c, rh = np.asarray(t_c, float), np.asarray(rh, float)
    e = rh / 100 * 6.105 * np.exp(17.27 * t_c / (237.7 + t_c))
    return 0.567 * t_c + 0.393 * e + 3.94
