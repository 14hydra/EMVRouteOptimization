"""OSM drive-network path aggregates for inferred firetruck OD pairs.

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
    "osm_travel_s",  # path_km / mean_speed → seconds
    "osm_route_ok",
]

# Extra OSM way tags needed for street-characteristic analysis. osmnx's defaults
# drop bus-lane, bike-lane and parking tags, so bus-lane logic in
# routing/graph.py and routing/emv_corridors.py can never fire on a default graph.
RICH_WAY_TAGS = [
    "busway",
    "busway:left",
    "busway:right",
    "lanes:bus",
    "lanes:bus:forward",
    "lanes:bus:backward",
    "bus:lanes",
    "lanes:forward",
    "lanes:backward",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "parking:lane:both",
    "parking:lane:left",
    "parking:lane:right",
    "sidewalk",
    "surface",
    "lit",
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


def build_drive_graph(
    out_path: Path | str,
    bbox: tuple[float, float, float, float],
    *,
    network_type: str = "drive",
    rich_tags: bool = False,
) -> Path:
    """Download and cache an OSM drive graph for an arbitrary bbox (W,S,E,N).

    ``rich_tags`` additionally keeps bus-lane / bike-lane / parking-lane tags
    (see ``RICH_WAY_TAGS``) for street-characteristic analysis.
    """
    ox = _try_import_ox()
    if rich_tags:
        ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | set(RICH_WAY_TAGS))
    cache_dir = Path(out_path).resolve().parent / "osmnx_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    G = ox.graph_from_bbox(bbox=bbox, network_type=network_type, simplify=True)
    ox.save_graphml(G, out_path)
    return out_path


def build_nyc_drive_graph(
    out_path: Path | str, *, network_type: str = "drive", rich_tags: bool = False
) -> Path:
    """Download and cache an OSM drive graph covering NYC."""
    return build_drive_graph(out_path, NYC_BBOX, network_type=network_type, rich_tags=rich_tags)


# San Francisco approx bbox (W, S, E, N)
SF_BBOX = (-122.52, 37.70, -122.35, 37.84)


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
    mean_spd = float(np.nanmean(speed_arr)) if len(speed_arr) else np.nan
    travel_s = (
        float(path_km / mean_spd * 3600.0)
        if path_km == path_km and mean_spd == mean_spd and mean_spd > 1.0
        else np.nan
    )
    return {
        "osm_path_km": path_km,
        "osm_n_edges": float(len(lengths)),
        "osm_n_nodes": float(len(route)),
        "osm_mean_speed_kmh": mean_spd,
        "osm_min_speed_kmh": float(np.nanmin(speed_arr)) if np.isfinite(speed_arr).any() else np.nan,
        "osm_primary_share": primary / n,
        "osm_motorway_share": motorway / n,
        "osm_residential_share": residential / n,
        "osm_circuity": circuity,
        "osm_travel_s": travel_s,
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
    workers: int = 8,
) -> pd.DataFrame:
    """
    Add OSM path features for each row with start_/dest_ coordinates.

    Fast path: batch ``nearest_nodes`` for unique coordinates, then route unique
    node-pairs once and map features back to rows.
    """
    graph_path = Path(graph_path)
    out = df.copy()
    for c in STREET_FEATURE_COLUMNS:
        out[c] = np.nan

    if not graph_path.exists():
        out["osm_route_ok"] = 0.0
        return out

    work = out.iloc[:limit].copy() if limit is not None else out.copy()
    need = ["start_lat", "start_lon", "dest_lat", "dest_lon"]
    if any(c not in work.columns for c in need):
        out["osm_route_ok"] = 0.0
        return out

    G = load_graph(graph_path)
    ox = _try_import_ox()
    try:
        G = ox.add_edge_speeds(G)
    except Exception:  # noqa: BLE001
        pass

    work["s_lat"] = work["start_lat"].round(4)
    work["s_lon"] = work["start_lon"].round(4)
    work["d_lat"] = work["dest_lat"].round(4)
    work["d_lon"] = work["dest_lon"].round(4)
    work["od_key"] = list(zip(work["s_lat"], work["s_lon"], work["d_lat"], work["d_lon"]))

    pts = pd.concat(
        [
            work[["s_lat", "s_lon"]].rename(columns={"s_lat": "lat", "s_lon": "lon"}),
            work[["d_lat", "d_lon"]].rename(columns={"d_lat": "lat", "d_lon": "lon"}),
        ],
        ignore_index=True,
    ).dropna().drop_duplicates()

    if not len(pts):
        out["osm_route_ok"] = 0.0
        return out

    node_ids = ox.nearest_nodes(G, pts["lon"].to_numpy(), pts["lat"].to_numpy())
    pt_to_node = dict(zip(zip(pts["lat"], pts["lon"]), node_ids))

    reps = work.drop_duplicates(subset=["od_key"], keep="first")
    # Build unique (orig_node, dest_node) jobs with crow-flies.
    node_jobs: dict[tuple[Any, Any], list[tuple[tuple, float | None]]] = {}
    empty_keys = []
    for row in reps.itertuples(index=False):
        key = row.od_key
        if any(pd.isna(x) for x in key):
            empty_keys.append(key)
            continue
        orig = pt_to_node.get((key[0], key[1]))
        dest = pt_to_node.get((key[2], key[3]))
        if orig is None or dest is None:
            empty_keys.append(key)
            continue
        crow = float(row.crow_flies_km) if pd.notna(getattr(row, "crow_flies_km", np.nan)) else None
        node_jobs.setdefault((orig, dest), []).append((key, crow))

    cache: dict[tuple, dict[str, float]] = {k: _empty_feats() for k in empty_keys}
    items = list(node_jobs.items())
    iterator = items
    if show_progress:
        iterator = tqdm(items, total=len(items), desc="osm_node_pairs", unit="pair")

    for (orig, dest), key_crow_list in iterator:
        try:
            route = nx.shortest_path(G, orig, dest, weight="length")
            # Use first crow; path aggregates only need one crow for circuity.
            crow = key_crow_list[0][1]
            if crow is None or crow != crow:
                # reconstruct from first key lat/lon
                k0 = key_crow_list[0][0]
                r = 6371.0
                p1, p2 = map(math.radians, [k0[0], k0[2]])
                dphi = math.radians(k0[2] - k0[0])
                dlmb = math.radians(k0[3] - k0[1])
                a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
                crow = 2 * r * math.asin(math.sqrt(a))
            feats = _path_aggregates(G, route, float(crow))
        except Exception:  # noqa: BLE001
            feats = _empty_feats()
        for key, _ in key_crow_list:
            cache[key] = feats

    mapped = work["od_key"].map(cache)
    for c in STREET_FEATURE_COLUMNS:
        vals = mapped.map(lambda d, col=c: (d or {}).get(col, np.nan))
        out.loc[work.index, c] = vals.to_numpy()

    if limit is not None and limit < len(out):
        out.loc[out.index[limit:], "osm_route_ok"] = 0.0
    return out
