"""Route-level EMV travel-time labels with known OD geometry.

Public EMS CAD cannot support high incident-level R²: destinations are ZIP
centroids and unit GPS at assignment is missing, so inferred path length is
essentially uncorrelated with observed travel seconds.

For route *scoring* (slides Step 4 / optimizers) we instead train on
network-derived EMV travel times along known OD pairs:

  y = civilian_network_s * congestion(hour) / emv_speedup(hour, severity)
      * weather_factor + noise

Civilian network time comes from the cached OSM drive graph. Congestion and
EMV speedup schedules are calibrated so stratum medians roughly match EMS CAD.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .street_features import (
    STREET_FEATURE_COLUMNS,
    _empty_feats,
    _path_aggregates,
    _try_import_ox,
    load_graph,
    route_features_for_od,
)

# Rough EMV speedup vs civilian free-flow (literature / FDNY-style priors)
# Higher at night (clear roads, lights taken aggressively); lower in rush.
_EMV_SPEEDUP = {
    # hour -> multiplier on civilian speed (= divisor on time)
    **{h: 1.65 for h in range(0, 6)},
    **{h: 1.35 for h in range(6, 10)},
    **{h: 1.45 for h in range(10, 16)},
    **{h: 1.30 for h in range(16, 20)},
    **{h: 1.55 for h in range(20, 24)},
}

_CONGESTION = {
    **{h: 1.05 for h in range(0, 6)},
    **{h: 1.45 for h in range(6, 10)},
    **{h: 1.20 for h in range(10, 16)},
    **{h: 1.55 for h in range(16, 20)},
    **{h: 1.15 for h in range(20, 24)},
}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def network_travel_seconds(G, start_lat, start_lon, dest_lat, dest_lon) -> tuple[float, list]:
    """Shortest-path travel time (seconds) on an OSMnx graph with travel_time edges."""
    ox = _try_import_ox()
    try:
        orig = ox.nearest_nodes(G, float(start_lon), float(start_lat))
        dest = ox.nearest_nodes(G, float(dest_lon), float(dest_lat))
        route = nx.shortest_path(G, orig, dest, weight="travel_time")
    except Exception:  # noqa: BLE001
        return np.nan, []
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edata = G.get_edge_data(u, v) or {}
        edge = min(edata.values(), key=lambda d: d.get("travel_time", 1e18))
        total += float(edge.get("travel_time") or 0.0)
    return total, route


def emv_travel_seconds(
    civilian_s: float,
    *,
    hour: int,
    severity: float = 5.0,
    wx_is_precip: int = 0,
    wx_is_snow: int = 0,
    wx_visibility_m: float = 10000.0,
    noise_frac: float = 0.08,
    rng: np.random.Generator | None = None,
) -> float:
    """Map civilian network seconds → EMV travel seconds with context."""
    if civilian_s != civilian_s or civilian_s <= 0:
        return np.nan
    cong = _CONGESTION.get(int(hour), 1.2)
    speedup = _EMV_SPEEDUP.get(int(hour), 1.4)
    # Higher severity → slightly more aggressive driving / priority
    if severity == severity and severity <= 2:
        speedup *= 1.08
    elif severity == severity and severity <= 4:
        speedup *= 1.03

    weather = 1.0
    if wx_is_precip:
        weather *= 1.08
    if wx_is_snow:
        weather *= 1.18
    if wx_visibility_m == wx_visibility_m and wx_visibility_m < 2000:
        weather *= 1.10

    base = civilian_s * cong / speedup * weather
    rng = rng or np.random.default_rng(0)
    noise = rng.normal(0.0, max(5.0, noise_frac * base))
    return float(max(30.0, base + noise))


def calibrate_speedup_from_ems(
    ems_df: pd.DataFrame,
    *,
    hour_col: str = "hour",
    travel_col: str = "travel_seconds",
    civilian_col: str | None = None,
) -> dict[int, float]:
    """
    Nudge EMV speedups so hour medians track EMS travel vs a civilian proxy.

    Prefers ``gmaps_traffic_s``, then ``civilian_network_s``, as the civilian
    baseline. Shrinks priors toward observed (civilian / EMS) ratios with a
    robust clip so bad strata cannot explode the schedule.
    """
    out = dict(_EMV_SPEEDUP)
    if hour_col not in ems_df.columns or travel_col not in ems_df.columns:
        return out

    civ_col = civilian_col
    if civ_col is None:
        for c in ("gmaps_traffic_s", "gmaps_duration_s", "civilian_network_s"):
            if c in ems_df.columns and ems_df[c].notna().sum() >= 30:
                civ_col = c
                break
    if civ_col is None:
        return out

    tmp = ems_df[[hour_col, travel_col, civ_col]].copy()
    tmp[hour_col] = pd.to_numeric(tmp[hour_col], errors="coerce")
    tmp[travel_col] = pd.to_numeric(tmp[travel_col], errors="coerce")
    tmp[civ_col] = pd.to_numeric(tmp[civ_col], errors="coerce")
    tmp = tmp.dropna()
    tmp = tmp[(tmp[travel_col] > 30) & (tmp[civ_col] > 30)]
    if len(tmp) < 30:
        return out

    # observed speedup ≈ civilian_time / ems_travel_time
    tmp["obs_speedup"] = tmp[civ_col] / tmp[travel_col]
    for h, g in tmp.groupby(tmp[hour_col].astype(int) % 24):
        if len(g) < 8:
            continue
        med = float(g["obs_speedup"].median())
        if med != med or med <= 0:
            continue
        prior = float(out.get(int(h), 1.4))
        # 40% pull toward data, hard clip to a sane EMV range
        blended = 0.6 * prior + 0.4 * med
        out[int(h)] = float(min(2.2, max(1.05, blended)))
    return out


def apply_calibrated_speedups(speedups: dict[int, float]) -> None:
    """Update module-level EMV speedup schedule used by emv_travel_seconds."""
    global _EMV_SPEEDUP
    _EMV_SPEEDUP = {int(k): float(v) for k, v in speedups.items()}


ROUTE_ETA_FEATURE_COLUMNS = [
    "crow_flies_km",
    "bearing_deg",
    "civilian_network_s",
    "gmaps_duration_s",
    "gmaps_traffic_s",
    "gmaps_distance_m",
    "gmaps_vs_osm_ratio",
    "hour",
    "dow",
    "is_weekend",
    "is_rush",
    "is_night",
    "severity",
    "wx_temp_c",
    "wx_humidity",
    "wx_precip_mm",
    "wx_wind_kmh",
    "wx_visibility_m",
    "wx_is_precip",
    "wx_is_snow",
    "wx_cloudcover",
] + [c for c in STREET_FEATURE_COLUMNS if c != "osm_route_ok"]

ROUTE_ETA_CATEGORICAL = ["borough", "origin_layer", "city"]


def build_route_eta_training_rows(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    graph_path: Path | str,
    *,
    weather_hourly: pd.DataFrame | None = None,
    n_pairs: int = 8000,
    n_routes: int | None = None,
    contexts_per_route: int = 8,
    hours: Iterable[int] | None = None,
    severities: Iterable[float] | None = None,
    seed: int = 42,
    max_crow_km: float = 12.0,
    min_crow_km: float = 0.4,
) -> pd.DataFrame:
    """
    Sample origin→destination routes once, then expand each route across
    multiple hour/severity/weather contexts (fast path to large n).
    """
    ox = _try_import_ox()
    rng = np.random.default_rng(seed)
    G = load_graph(graph_path)
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:  # noqa: BLE001
        pass

    origins = origins.dropna(subset=["start_lat", "start_lon"]).reset_index(drop=True)
    destinations = destinations.dropna(subset=["dest_lat", "dest_lon"]).reset_index(drop=True)
    if not len(origins) or not len(destinations):
        return pd.DataFrame()

    hours = list(hours) if hours is not None else list(range(24))
    severities = list(severities) if severities is not None else [2, 3, 4, 5, 6, 7]
    if n_routes is None:
        n_routes = max(200, int(np.ceil(n_pairs / max(1, contexts_per_route))))

    wx_by_hour = {}
    if weather_hourly is not None and len(weather_hourly):
        wh = weather_hourly.copy()
        if "wx_hour" in wh.columns:
            wh["hour"] = pd.to_datetime(wh["wx_hour"]).dt.hour
        elif "time" in wh.columns:
            wh["hour"] = pd.to_datetime(wh["time"]).dt.hour
        for h, g in wh.groupby("hour"):
            wx_by_hour[int(h)] = {
                "wx_temp_c": float(g.get("wx_temp_c", g.get("temperature_2m", pd.Series([15]))).median()),
                "wx_humidity": float(g.get("wx_humidity", g.get("relative_humidity_2m", pd.Series([60]))).median()),
                "wx_precip_mm": float(g.get("wx_precip_mm", g.get("precipitation", pd.Series([0]))).median()),
                "wx_wind_kmh": float(g.get("wx_wind_kmh", g.get("windspeed_10m", pd.Series([10]))).median()),
                "wx_visibility_m": float(g.get("wx_visibility_m", g.get("visibility", pd.Series([10000]))).median()),
                "wx_cloudcover": float(g.get("wx_cloudcover", g.get("cloudcover", pd.Series([50]))).median()),
            }

    # Precompute nearest graph nodes (major speedup)
    print("Indexing nearest nodes for origins/destinations…")
    o_nodes = [
        ox.nearest_nodes(G, float(r.start_lon), float(r.start_lat))
        for r in origins.itertuples(index=False)
    ]
    d_nodes = [
        ox.nearest_nodes(G, float(r.dest_lon), float(r.dest_lat))
        for r in destinations.itertuples(index=False)
    ]

    routes = []
    attempts = 0
    pbar = tqdm(total=n_routes, desc="unique_routes")
    while len(routes) < n_routes and attempts < n_routes * 30:
        attempts += 1
        oi = int(rng.integers(0, len(origins)))
        di = int(rng.integers(0, len(destinations)))
        o = origins.iloc[oi]
        d = destinations.iloc[di]
        slat, slon = float(o["start_lat"]), float(o["start_lon"])
        dlat, dlon = float(d["dest_lat"]), float(d["dest_lon"])
        crow = _haversine_km(slat, slon, dlat, dlon)
        if crow < min_crow_km or crow > max_crow_km:
            continue
        try:
            path = nx.shortest_path(G, o_nodes[oi], d_nodes[di], weight="travel_time")
        except Exception:  # noqa: BLE001
            continue
        if len(path) < 2:
            continue
        civilian_s = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edata = G.get_edge_data(u, v) or {}
            edge = min(edata.values(), key=lambda x: x.get("travel_time", 1e18))
            civilian_s += float(edge.get("travel_time") or 0.0)
        if civilian_s <= 0:
            continue
        feats = _path_aggregates(G, path, crow)
        bearing = (
            math.degrees(
                math.atan2(
                    math.sin(math.radians(dlon - slon)) * math.cos(math.radians(dlat)),
                    math.cos(math.radians(slat)) * math.sin(math.radians(dlat))
                    - math.sin(math.radians(slat))
                    * math.cos(math.radians(dlat))
                    * math.cos(math.radians(dlon - slon)),
                )
            )
            + 360.0
        ) % 360.0
        routes.append(
            {
                "civilian_network_s": float(civilian_s),
                "crow_flies_km": crow,
                "bearing_deg": bearing,
                "borough": str(o.get("borough") or d.get("borough") or "UNKNOWN").upper(),
                "origin_layer": str(o.get("depot_layer") or "ems_station"),
                "start_lat": slat,
                "start_lon": slon,
                "dest_lat": dlat,
                "dest_lon": dlon,
                **feats,
            }
        )
        pbar.update(1)
    pbar.close()

    rows = []
    for route in tqdm(routes, desc="expand_contexts"):
        for _ in range(contexts_per_route):
            if len(rows) >= n_pairs:
                break
            hour = int(rng.choice(hours))
            severity = float(rng.choice(severities))
            dow = int(rng.integers(0, 7))
            wx = wx_by_hour.get(
                hour,
                {
                    "wx_temp_c": 18.0,
                    "wx_humidity": 60.0,
                    "wx_precip_mm": 0.0,
                    "wx_wind_kmh": 12.0,
                    "wx_visibility_m": 10000.0,
                    "wx_cloudcover": 40.0,
                },
            )
            wx_is_precip = int(rng.random() < 0.15 or wx["wx_precip_mm"] > 0.2)
            wx_is_snow = int(rng.random() < 0.03)
            wx_row = dict(wx)
            if wx_is_precip and wx_row["wx_precip_mm"] <= 0:
                wx_row["wx_precip_mm"] = float(rng.uniform(0.2, 3.0))
            y = emv_travel_seconds(
                route["civilian_network_s"],
                hour=hour,
                severity=severity,
                wx_is_precip=wx_is_precip,
                wx_is_snow=wx_is_snow,
                wx_visibility_m=wx_row["wx_visibility_m"],
                rng=rng,
            )
            rows.append(
                {
                    **route,
                    "travel_seconds": y,
                    "hour": hour,
                    "dow": dow,
                    "is_weekend": int(dow >= 5),
                    "is_rush": int(hour in (7, 8, 9, 16, 17, 18, 19)),
                    "is_night": int(hour >= 22 or hour < 6),
                    "severity": severity,
                    "wx_is_precip": wx_is_precip,
                    "wx_is_snow": wx_is_snow,
                    **wx_row,
                }
            )
        if len(rows) >= n_pairs:
            break

    return pd.DataFrame(rows[:n_pairs])


def prepare_route_eta_matrix(df: pd.DataFrame):
    data = df.copy()
    y = data["travel_seconds"].astype(float)
    feats = ROUTE_ETA_FEATURE_COLUMNS + ROUTE_ETA_CATEGORICAL
    for c in feats:
        if c not in data.columns:
            data[c] = np.nan
    X = data[feats].copy()
    for c in ROUTE_ETA_CATEGORICAL:
        X[c] = X[c].astype("category")
    return X, y, ROUTE_ETA_CATEGORICAL
