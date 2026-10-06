"""Traffic + weather routing conditions (ASI evaluation variables).

Matches the ASI Gemini eval framing:
  - Clear weather baseline
  - Severe weather (rain / snow)
  - Congestion / rush regimes (hour-based traffic prior; TomTom-style live flow
    can plug in later when an API key is available)
  - Dedicated bus-lane / ROW advantage for EMVs vs civilian GPS

Open-Meteo hourly weather (already cached for EMS windows) can seed real
snapshots; named presets cover the slide story without a live traffic feed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..weather import WEATHER_FEATURE_COLUMNS, _normalize_weather, fetch_hourly_weather


@dataclass(frozen=True)
class RoutingConditions:
    """One traffic/weather regime applied when building edge costs."""

    id: str
    label: str
    hour: int
    congestion: float
    wx_precip_mm: float = 0.0
    wx_snow_mm: float = 0.0
    wx_visibility_m: float = 10000.0
    wx_wind_kmh: float = 10.0
    wx_temp_c: float = 18.0
    description: str = ""
    source: str = "preset"  # preset | open_meteo

    @property
    def wx_is_precip(self) -> int:
        return int(self.wx_precip_mm > 0.05 or self.wx_snow_mm > 0.0)

    @property
    def wx_is_snow(self) -> int:
        return int(self.wx_snow_mm > 0.0)

    def weather_factor_civilian(self) -> float:
        """Civilian slowdown from precip / snow / low visibility / wind."""
        f = 1.0
        if self.wx_precip_mm > 0.05:
            f *= 1.0 + min(0.22, 0.06 + 0.04 * self.wx_precip_mm)
        if self.wx_snow_mm > 0.0:
            f *= 1.0 + min(0.35, 0.14 + 0.08 * self.wx_snow_mm)
        if self.wx_visibility_m < 2000:
            f *= 1.12
        elif self.wx_visibility_m < 4000:
            f *= 1.05
        if self.wx_wind_kmh >= 45:
            f *= 1.06
        return float(f)

    def weather_factor_emv(self) -> float:
        """EMVs still slow in weather, but less than civilian GPS traffic."""
        civ = self.weather_factor_civilian()
        # Keep ~45% of the weather penalty (lights/sirens + priority corridors)
        return float(1.0 + 0.45 * (civ - 1.0))

    def row_bonus(self) -> float:
        """
        Extra EMV right-of-way discount that *grows* with congestion.

        At free-flow (~1.05) ≈ 1.00; at heavy rush (~1.55) ≈ 0.88.
        Models that prefer primary/bus edges amplify this further in graph.py.
        """
        c = max(1.0, float(self.congestion))
        # Map congestion 1.0→1.6 onto bonus 1.0→0.86
        t = min(1.0, max(0.0, (c - 1.0) / 0.60))
        return float(1.0 - 0.14 * t)

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            {
                "wx_is_precip": self.wx_is_precip,
                "wx_is_snow": self.wx_is_snow,
                "weather_factor_civilian": round(self.weather_factor_civilian(), 4),
                "weather_factor_emv": round(self.weather_factor_emv(), 4),
                "row_bonus": round(self.row_bonus(), 4),
            }
        )
        return d


# ASI-style evaluation grid (traffic amount × weather)
CONDITION_PRESETS: dict[str, RoutingConditions] = {
    "clear_offpeak": RoutingConditions(
        id="clear_offpeak",
        label="Clear · off-peak",
        hour=11,
        congestion=1.15,
        description="Baseline clear midday; light congestion.",
    ),
    "clear_rush": RoutingConditions(
        id="clear_rush",
        label="Clear · rush hour",
        hour=17,
        congestion=1.55,
        description="PM rush prior (hour-based congestion; TomTom-style when available).",
    ),
    "rain_rush": RoutingConditions(
        id="rain_rush",
        label="Rain · rush hour",
        hour=17,
        congestion=1.60,
        wx_precip_mm=4.5,
        wx_visibility_m=3500,
        wx_temp_c=14.0,
        description="Heavy rain + rush — civilian GPS slows hard; EMV uses ROW corridors.",
    ),
    "snow_am": RoutingConditions(
        id="snow_am",
        label="Snow · morning",
        hour=8,
        congestion=1.50,
        wx_snow_mm=1.8,
        wx_precip_mm=1.2,
        wx_visibility_m=1200,
        wx_temp_c=-2.0,
        wx_wind_kmh=28.0,
        description="Snow / low visibility — prefer wider primary & bus-lane avenues.",
    ),
    "night_clear": RoutingConditions(
        id="night_clear",
        label="Clear · night",
        hour=2,
        congestion=1.05,
        wx_temp_c=12.0,
        description="Overnight free-flow; EMV ROW advantage is smaller.",
    ),
}


def congestion_for_hour(hour: int) -> float:
    """Hour-of-day congestion prior (TomTom-style stand-in)."""
    h = int(hour) % 24
    if h in (7, 8, 9, 16, 17, 18, 19):
        return 1.55
    if h in (6, 10, 15, 20):
        return 1.35
    if h >= 22 or h < 5:
        return 1.05
    if h == 5:
        return 1.12
    return 1.20


# Weather knobs independent of clock hour (slider × weather UI)
WEATHER_PROFILES: dict[str, dict[str, float]] = {
    "clear": {
        "wx_precip_mm": 0.0,
        "wx_snow_mm": 0.0,
        "wx_visibility_m": 10000.0,
        "wx_wind_kmh": 10.0,
        "wx_temp_c": 18.0,
    },
    "rain": {
        "wx_precip_mm": 4.5,
        "wx_snow_mm": 0.0,
        "wx_visibility_m": 3500.0,
        "wx_wind_kmh": 18.0,
        "wx_temp_c": 14.0,
    },
    "snow": {
        "wx_precip_mm": 1.2,
        "wx_snow_mm": 1.8,
        "wx_visibility_m": 1200.0,
        "wx_wind_kmh": 28.0,
        "wx_temp_c": -2.0,
    },
}


def conditions_at(hour: int, weather: str = "clear") -> RoutingConditions:
    """Build conditions for a clock hour + weather mode (map slider UI)."""
    weather = (weather or "clear").lower()
    if weather not in WEATHER_PROFILES:
        raise KeyError(f"Unknown weather {weather!r}; use {list(WEATHER_PROFILES)}")
    h = int(hour) % 24
    cong = congestion_for_hour(h)
    # Bad weather slightly worsens traffic beyond the hour prior
    if weather == "rain":
        cong = min(1.75, cong + 0.08)
    elif weather == "snow":
        cong = min(1.80, cong + 0.12)
    wx = WEATHER_PROFILES[weather]
    label_wx = {"clear": "Clear", "rain": "Rain", "snow": "Snow"}[weather]
    return RoutingConditions(
        id=f"{weather}_h{h:02d}",
        label=f"{label_wx} · {h:02d}:00",
        hour=h,
        congestion=float(cong),
        description=f"Slider state hour={h} weather={weather}",
        source="slider",
        **wx,
    )


def list_condition_ids() -> list[str]:
    return list(CONDITION_PRESETS.keys())


def get_conditions(condition_id: str) -> RoutingConditions:
    if condition_id not in CONDITION_PRESETS:
        known = ", ".join(CONDITION_PRESETS)
        raise KeyError(f"Unknown condition {condition_id!r}. Known: {known}")
    return CONDITION_PRESETS[condition_id]


def conditions_from_weather_row(
    row: pd.Series | dict,
    *,
    hour: int,
    congestion: float | None = None,
    condition_id: str = "open_meteo",
    label: str | None = None,
) -> RoutingConditions:
    """Build conditions from an Open-Meteo / weather_hourly row."""
    if congestion is None:
        if hour in (7, 8, 9, 16, 17, 18, 19):
            congestion = 1.45
        elif hour >= 22 or hour < 6:
            congestion = 1.05
        else:
            congestion = 1.20

    def _f(key_a, key_b=None, default=0.0):
        for k in (key_a, key_b):
            if k and k in row and row[k] == row[k]:
                try:
                    return float(row[k])
                except (TypeError, ValueError):
                    pass
        return float(default)

    precip = _f("wx_precip_mm", "precipitation", 0.0)
    snow = _f("wx_snow_mm", "snowfall", 0.0)
    vis = _f("wx_visibility_m", "visibility", 10000.0)
    if vis != vis or vis <= 0:
        vis = 10000.0
    wind = _f("wx_wind_kmh", "windspeed_10m", 10.0)
    temp = _f("wx_temp_c", "temperature_2m", 18.0)
    stamp = row.get("wx_hour") or row.get("time") or ""
    return RoutingConditions(
        id=condition_id,
        label=label or f"Open-Meteo {stamp} (h={hour})",
        hour=int(hour),
        congestion=float(congestion),
        wx_precip_mm=precip,
        wx_snow_mm=snow,
        wx_visibility_m=vis,
        wx_wind_kmh=wind,
        wx_temp_c=temp,
        description="Real Open-Meteo hourly snapshot for NYC.",
        source="open_meteo",
    )


def load_open_meteo_snapshot(
    *,
    when: str,
    cache_path: Path | str | None = None,
    lat: float = 40.758,
    lon: float = -73.985,
) -> RoutingConditions:
    """
    Load one NYC hourly weather snapshot (ISO datetime) from cache or API.

    Example: when='2024-06-03T17:00:00'
    """
    ts = pd.Timestamp(when).tz_localize(None)
    hour = int(ts.hour)
    cache_path = Path(cache_path) if cache_path else None

    raw = None
    if cache_path is not None and cache_path.exists():
        raw = pd.read_csv(cache_path, parse_dates=["time"])
    if raw is None or not len(raw):
        day = ts.date().isoformat()
        raw = fetch_hourly_weather(day, day, lat=lat, lon=lon)
        if cache_path is not None and len(raw):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # append-safe rewrite of day only when empty
            if not cache_path.exists():
                raw.to_csv(cache_path, index=False)

    wx = _normalize_weather(raw)
    if not len(wx):
        raise RuntimeError(f"No Open-Meteo rows for {when}")
    wx["wx_hour"] = pd.to_datetime(wx["wx_hour"])
    target = ts.floor("h")
    hit = wx[wx["wx_hour"] == target]
    if hit.empty:
        # nearest hour
        wx = wx.assign(_dt=abs((wx["wx_hour"] - target).dt.total_seconds()))
        hit = wx.nsmallest(1, "_dt")
    row = hit.iloc[0]
    return conditions_from_weather_row(
        row,
        hour=hour,
        condition_id=f"open_meteo_{target.strftime('%Y%m%d_%H')}",
        label=f"Open-Meteo {target}",
    )


def pick_harsh_open_meteo_hours(
    weather_csv: Path | str,
    *,
    n: int = 3,
) -> list[RoutingConditions]:
    """Pick the harshest precip/snow hours from a cached weather CSV for demos."""
    path = Path(weather_csv)
    if not path.exists():
        return []
    raw = pd.read_csv(path, parse_dates=["time"])
    wx = _normalize_weather(raw)
    if not len(wx):
        return []
    score = (
        wx["wx_precip_mm"].fillna(0) * 2.0
        + wx["wx_snow_mm"].fillna(0) * 6.0
        + (10000 - wx["wx_visibility_m"].fillna(10000).clip(0, 10000)) / 5000.0
    )
    wx = wx.assign(_score=score)
    top = wx.nlargest(max(n * 3, n), "_score").drop_duplicates(subset=["wx_hour"]).head(n)
    out: list[RoutingConditions] = []
    for _, row in top.iterrows():
        hour = int(pd.Timestamp(row["wx_hour"]).hour)
        out.append(
            conditions_from_weather_row(
                row,
                hour=hour,
                condition_id=f"wx_{pd.Timestamp(row['wx_hour']).strftime('%Y%m%d_%H')}",
                label=f"Open-Meteo {pd.Timestamp(row['wx_hour'])}",
            )
        )
    return out


# Re-export for callers that only import conditions
__all__ = [
    "RoutingConditions",
    "CONDITION_PRESETS",
    "list_condition_ids",
    "get_conditions",
    "conditions_from_weather_row",
    "load_open_meteo_snapshot",
    "pick_harsh_open_meteo_hours",
    "WEATHER_FEATURE_COLUMNS",
]
