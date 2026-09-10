"""OSM drive-network path aggregates for inferred ambulance OD pairs.

LION is the long-term NYC centerline source; OSM (via OSMnx) is the practical
open substitute until a local LION extract is wired in.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

STREET_FEATURE_COLUMNS = [
    "osm_path_km",
    "osm_n_edges",
    "osm_n_nodes",
    "osm_mean_speed_kmh",
    "osm_min_speed_kmh",
    "osm_primary_share",
    "osm_motorway_share",
    "osm_residential_share",
    "osm_circuity",  # path_km / crow_flies_km
    "osm_route_ok",
]

# NYC boroughs bbox as (left, bottom, right, top) = (west, south, east, north) for OSMnx 2.x
NYC_BBOX = (-74.26, 40.49, -73.70, 40.92)


def _try_import_ox():
    try:
        import osmnx as ox  # noqa: WPS433

        return ox
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "osmnx is required for street path features. "
            "Install with: pip install osmnx geopandas"
        ) from exc


def build_nyc_drive_graph(out_path: Path | str, *, network_type: str = "drive") -> Path:
    """Download and cache an OSM drive graph covering NYC."""
    ox = _try_import_ox()
    # Keep HTTP response cache under data/raw (gitignored), not repo-root ./cache
    cache_dir = Path(out_path).resolve().parent / "osmnx_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    G = ox.graph_from_bbox(
        bbox=NYC_BBOX,  # (left, bottom, right, top)
        network_type=network_type,
        simplify=True,
    )
    # Keep a projected copy helpful for length; osmnx stores length on edges in meters
    ox.save_graphml(G, out_path)
    return out_path


def load_graph(path: Path | str):
    ox = _try_import_ox()
    return ox.load_graphml(Path(path))


def _edge_speed_kmh(data: dict[str, Any]) -> float:
    # osmnx >=1.6 may set 'speed_kph'; else parse maxspeed
    if "speed_kph" in data and data["speed_kph"] not in (None, ""):
        try:
            return float(data["speed_kph"])
        except (TypeError, ValueError):
            pass
    ms = data.get("maxspeed")
    if ms is None:
        return np.nan
    if isinstance(ms, list):
        ms = ms[0] if ms else None
    if ms is None:
        return np.nan
    text = str(ms).lower().replace("mph", "").strip()
    try:
        val = float(text.split()[0])
    except (ValueError, IndexError):
        return np.nan
    if "mph" in str(data.get("maxspeed", "")).lower() or val <= 70:
        # Heuristic: values <=70 without unit often mph in US extracts
        if "mph" in str(data.get("maxspeed", "")).lower():
            return val * 1.60934
    return val


def _highway_class(data: dict[str, Any]) -> str:
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    return str(hw or "").lower()


def _path_aggregates(G, route: list, crow_km: float) -> dict[str, float]:
    lengths = []
    speeds = []
    highways = []
    for u, v in zip(route[:-1], route[1:]):
        edata = G.get_edge_data(u, v)
        if not edata:
            continue
        # Multigraph: pick first / shortest edge
        edge = min(edata.values(), key=lambda d: d.get("length", 0) or 0)
        lengths.append(float(edge.get("length") or 0.0))
        speeds.append(_edge_speed_kmh(edge))
        highways.append(_highway_class(edge))

    path_m = float(np.nansum(lengths)) if lengths else np.nan
    path_km = path_m / 1000.0 if path_m == path_m else np.nan
    speed_arr = np.array(speeds, dtype=float)
    n = len(highways) or 1
    primary = sum(1 for h in highways if h in {"primary", "primary_link", "trunk", "trunk_link"})
    motorway = sum(1 for h in highways if h in {"motorway", "motorway_link"})
    residential = sum(1 for h in highways if h in {"residential", "living_street", "unclassified"})

    circuity = (
        float(path_km / crow_km)
        if path_km == path_km and crow_km and crow_km > 0.05
        else np.nan
    )
    return {
        "osm_path_km": path_km,
        "osm_n_edges": float(len(lengths)),
        "osm_n_nodes": float(len(route)),
        "osm_mean_speed_kmh": float(np.nanmean(speed_arr)) if len(speed_arr) else np.nan,
        "osm_min_speed_kmh": float(np.nanmin(speed_arr)) if np.isfinite(speed_arr).any() else np.nan,
        "osm_primary_share": primary / n,
        "osm_motorway_share": motorway / n,
        "osm_residential_share": residential / n,
        "osm_circuity": circuity,
        "osm_route_ok": 1.0,
    }


def _empty_feats() -> dict[str, float]:
    return {c: (0.0 if c == "osm_route_ok" else np.nan) for c in STREET_FEATURE_COLUMNS}


def route_features_for_od(
    G,
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    crow_km: float | None = None,
) -> dict[str, float]:
    """Shortest-path aggregates between start and dest on graph G."""
    ox = _try_import_ox()
    if any(x != x for x in (start_lat, start_lon, dest_lat, dest_lon)):
        return _empty_feats()
    try:
        orig = ox.nearest_nodes(G, start_lon, start_lat)
        dest = ox.nearest_nodes(G, dest_lon, dest_lat)
        route = nx.shortest_path(G, orig, dest, weight="length")
    except Exception:  # noqa: BLE001
        return _empty_feats()

    if crow_km is None or crow_km != crow_km:
        # haversine fallback
        r = 6371.0
        p1, p2 = map(math.radians, [start_lat, dest_lat])
        dphi = math.radians(dest_lat - start_lat)
        dlmb = math.radians(dest_lon - start_lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        crow_km = 2 * r * math.asin(math.sqrt(a))

    return _path_aggregates(G, route, float(crow_km))


def attach_street_features(
    df: pd.DataFrame,
    graph_path: Path | str,
    *,
    limit: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add OSM path features for each row with start_/dest_ coordinates.

    Routes unique rounded OD keys once, then maps back (much faster than
    per-row shortest paths). ``limit`` optionally caps input rows (smoke tests).
    """
    graph_path = Path(graph_path)
    out = df.copy()
    for c in STREET_FEATURE_COLUMNS:
        out[c] = np.nan

    if not graph_path.exists():
        out["osm_route_ok"] = 0.0
        return out

    if limit is not None:
        work = out.iloc[:limit].copy()
    else:
        work = out

    G = load_graph(graph_path)
    ox = _try_import_ox()
    try:
        G = ox.add_edge_speeds(G)
    except Exception:  # noqa: BLE001
        pass

    work = work.copy()
    work["od_key"] = list(
        zip(
            work["start_lat"].round(4),
            work["start_lon"].round(4),
            work["dest_lat"].round(4),
            work["dest_lon"].round(4),
        )
    )
    # One representative row per unique OD
    reps = work.drop_duplicates(subset=["od_key"], keep="first")
    cache: dict[tuple, dict[str, float]] = {}
    iterator = reps.itertuples(index=False)
    if show_progress:
        iterator = tqdm(reps.itertuples(index=False), total=len(reps), desc="osm_unique_od", unit="od")

    for row in iterator:
        key = row.od_key
        if any(pd.isna(x) for x in key):
            cache[key] = _empty_feats()
            continue
        crow = float(row.crow_flies_km) if pd.notna(getattr(row, "crow_flies_km", np.nan)) else None
        cache[key] = route_features_for_od(
            G,
            float(key[0]),
            float(key[1]),
            float(key[2]),
            float(key[3]),
            crow_km=crow,
        )

    mapped = work["od_key"].map(cache)
    for c in STREET_FEATURE_COLUMNS:
        vals = mapped.map(lambda d, col=c: (d or {}).get(col, np.nan))
        out.loc[work.index, c] = vals.to_numpy()

    if limit is not None and limit < len(out):
        out.loc[out.index[limit:], "osm_route_ok"] = 0.0
    return out
