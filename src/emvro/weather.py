"""Hourly weather features via Open-Meteo archive (no API key)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

# Central NYC — good enough for citywide EMS hourly regimes
DEFAULT_LAT = 40.758
DEFAULT_LON = -73.985

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weathercode",
    "cloudcover",
    "windspeed_10m",
    "visibility",
]

WEATHER_FEATURE_COLUMNS = [
    "wx_temp_c",
    "wx_humidity",
    "wx_precip_mm",
    "wx_rain_mm",
    "wx_snow_mm",
    "wx_cloudcover",
    "wx_wind_kmh",
    "wx_visibility_m",
    "wx_is_precip",
    "wx_is_snow",
]


def fetch_hourly_weather(
    start_date: str,
    end_date: str,
    *,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    timeout: int = 120,
) -> pd.DataFrame:
    """Return hourly weather rows for [start_date, end_date] (YYYY-MM-DD)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/New_York",
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def weather_lookup_table(
    datetimes: Iterable,
    *,
    cache_path: Path | str | None = None,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Build an hourly weather table covering the min/max of ``datetimes``.

    Caches CSV at ``cache_path`` when provided.
    """
    dts = pd.to_datetime(pd.Series(list(datetimes)), errors="coerce").dropna()
    if dts.empty:
        return pd.DataFrame(columns=["wx_hour"] + WEATHER_FEATURE_COLUMNS)

    start = dts.min().floor("D").date().isoformat()
    end = dts.max().ceil("D").date().isoformat()

    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["time"])
        cmin, cmax = cached["time"].min(), cached["time"].max()
        if cmin <= dts.min() and cmax >= dts.max():
            return _normalize_weather(cached)

    raw = fetch_hourly_weather(start, end, lat=lat, lon=lon)
    if cache_path is not None and len(raw):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache_path, index=False)
    return _normalize_weather(raw)


def _normalize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or not len(raw):
        return pd.DataFrame(columns=["wx_hour"] + WEATHER_FEATURE_COLUMNS)
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    out = pd.DataFrame(
        {
            "wx_hour": df["time"].dt.floor("h"),
            "wx_temp_c": pd.to_numeric(df.get("temperature_2m"), errors="coerce"),
            "wx_humidity": pd.to_numeric(df.get("relative_humidity_2m"), errors="coerce"),
            "wx_precip_mm": pd.to_numeric(df.get("precipitation"), errors="coerce"),
            "wx_rain_mm": pd.to_numeric(df.get("rain"), errors="coerce"),
            "wx_snow_mm": pd.to_numeric(df.get("snowfall"), errors="coerce"),
            "wx_cloudcover": pd.to_numeric(df.get("cloudcover"), errors="coerce"),
            "wx_wind_kmh": pd.to_numeric(df.get("windspeed_10m"), errors="coerce"),
            "wx_visibility_m": pd.to_numeric(df.get("visibility"), errors="coerce"),
        }
    )
    out["wx_is_precip"] = (out["wx_precip_mm"].fillna(0) > 0).astype(int)
    out["wx_is_snow"] = (out["wx_snow_mm"].fillna(0) > 0).astype(int)
    return out.drop_duplicates(subset=["wx_hour"]).sort_values("wx_hour")


def join_weather_features(
    incidents: pd.DataFrame,
    *,
    datetime_col: str = "incident_datetime",
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    """Left-join hourly weather onto an incident / feature table."""
    out = incidents.copy()
    if datetime_col not in out.columns:
        for c in WEATHER_FEATURE_COLUMNS:
            out[c] = pd.NA
        return out

    out["_dt"] = pd.to_datetime(out[datetime_col], errors="coerce")
    out["wx_hour"] = out["_dt"].dt.floor("h")
    wx = weather_lookup_table(out["_dt"], cache_path=cache_path)
    out = out.merge(wx, on="wx_hour", how="left")
    return out.drop(columns=["_dt", "wx_hour"], errors="ignore")
