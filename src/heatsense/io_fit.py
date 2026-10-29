"""Read Garmin .FIT activity files into a tidy per-second table."""

from __future__ import annotations

from pathlib import Path

import fitdecode
import numpy as np
import pandas as pd

# Columns HeatSense uses. Anything else in the file is ignored.
COLUMNS = ["heart_rate", "speed", "cadence_spm", "distance", "altitude", "device_temp", "core_temp"]
GPS_COLUMNS = ["lat", "lon"]

_SEMICIRCLE_TO_DEG = 180.0 / 2**31


def read_fit(path: str | Path, keep_gps: bool = False) -> pd.DataFrame:
    """Return one row per record message, indexed by UTC timestamp.

    Columns:
        heart_rate   bpm
        speed        m/s (enhanced_speed preferred)
        cadence_spm  steps per minute, both feet (Garmin stores strides per minute)
        distance     m
        altitude     m
        device_temp  degC from the watch's sensor (reads skin/air, NOT core temperature)
        core_temp    degC from a paired core-temperature sensor (e.g. CORE), if present;
                     this is the validation reference

    GPS is dropped by default so plots and shared CSVs don't reveal where you live.
    """
    rows = []
    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if not (isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record"):
                continue
            get = lambda name: frame.get_value(name, fallback=None)  # noqa: E731
            ts = get("timestamp")
            if ts is None:
                continue

            cadence = get("cadence")
            if cadence is not None:
                frac = get("fractional_cadence") or 0.0
                cadence = 2.0 * (cadence + frac)

            row = {
                "timestamp": ts,
                "heart_rate": get("heart_rate"),
                "speed": _first(get("enhanced_speed"), get("speed")),
                "cadence_spm": cadence,
                "distance": get("distance"),
                "altitude": _first(get("enhanced_altitude"), get("altitude")),
                "device_temp": get("temperature"),
                "core_temp": get("core_temperature"),
            }
            if keep_gps:
                lat, lon = get("position_lat"), get("position_long")
                row["lat"] = lat * _SEMICIRCLE_TO_DEG if lat is not None else None
                row["lon"] = lon * _SEMICIRCLE_TO_DEG if lon is not None else None
            rows.append(row)

    cols = ["timestamp", *COLUMNS, *(GPS_COLUMNS if keep_gps else [])]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise ValueError(f"No record messages with timestamps found in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df.apply(pd.to_numeric, errors="coerce").astype(float)


def to_minutes(df: pd.DataFrame, hr_col: str = "heart_rate", min_coverage: float = 0.5):
    """Aggregate a per-second table (with an ``hr_ok`` mask) to 1-minute rows.

    Minutes are wall-clock minutes since the start, and EVERY minute appears, including
    ones with no records at all (watch timer paused). Collapsing those would make the
    filter think the minute after a 5-minute break came straight after the one before it.

    Columns:
        time         wall-clock start of the minute (UTC)
        hr           mean of good heart-rate samples; NaN if fewer than
                     ``min_coverage`` x 60 seconds were good
        hr_coverage  good heart-rate seconds / 60
        recorded     recorded seconds / 60 (0 = timer paused the whole minute)
        moving       moving seconds / 60 (speed > 1 m/s and cadence > 100 spm)
        paused       True if the timer was off or the runner was stopped most of the minute
        speed, cadence_spm   means over recorded seconds
    """
    ok = df["hr_ok"] if "hr_ok" in df else pd.Series(True, index=df.index)
    elapsed = (df.index - df.index[0]).total_seconds()
    minute = (elapsed // 60).astype(int)
    moving = (df["speed"].fillna(0) > 1.0) & (df["cadence_spm"].fillna(0) > 100)
    g = df.assign(_minute=minute, _good=ok.astype(float), _moving=moving.astype(float),
                  _hr_good=df[hr_col].where(ok)).groupby("_minute")

    full = pd.RangeIndex(0, int(minute.max()) + 1, name="minute")
    out = pd.DataFrame({
        "hr": g["_hr_good"].mean(),
        "hr_coverage": g["_good"].sum() / 60,
        "recorded": g.size() / 60,
        "moving": g["_moving"].sum() / 60,
        "speed": g["speed"].mean(),
        "cadence_spm": g["cadence_spm"].mean(),
    }).reindex(full)
    out[["hr_coverage", "recorded", "moving"]] = out[["hr_coverage", "recorded", "moving"]].fillna(0.0)
    out.insert(0, "time", df.index[0] + pd.to_timedelta(full.to_numpy(), unit="min"))
    out["paused"] = out["moving"] < 0.5
    out.loc[out["hr_coverage"] < min_coverage, "hr"] = np.nan
    return out


def _first(*values):
    for v in values:
        if v is not None:
            return v
    return None
