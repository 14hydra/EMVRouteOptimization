"""Street-characteristic features for inferred EMS routes.

Joins NYC-published street attributes onto the OSM drive graph and aggregates
them (length-weighted) along each origin→destination route:

- **NYC Centerline** (``inkn-q76z``): measured street width (ft), travel /
  parking lane counts, posted speed.
- **NYC DOT Bus Lanes – Local Streets** (``ycrg-ses3``): bus-lane segments.
- **OSM tags** (graph built with ``--rich-tags``): bus lanes, bike lanes,
  one-ways, bridges/tunnels, traffic signals.

OSM's own ``width`` tag covers <1% of NYC edges, so it is not used.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import shape
from tqdm import tqdm

from .street_features import _try_import_ox, load_graph

BASE = "https://data.cityofnewyork.us/resource"
CENTERLINE_ID = "inkn-q76z"
BUS_LANES_ID = "ycrg-ses3"

# NY Long Island State Plane (US ft) is the local metric CRS; use meters instead.
METRIC_CRS = "EPSG:32618"  # UTM 18N

CENTERLINE_SELECT = (
    "the_geom,streetwidth,number_travel_lanes,number_park_lanes,number_total_lanes,"
    "posted_speed,trafdir,rw_type,snow_priority,full_street_name"
)

# Route-level street characteristics used in the regression / heatmap, in
# display order, with (label, group).
STREET_FEATURES: dict[str, tuple[str, str]] = {
    "street_width_ft": ("Street width (ft)", "Road geometry"),
    "travel_lanes": ("Travel lanes", "Road geometry"),
    "park_lanes": ("Parking lanes", "Road geometry"),
    "oneway_share": ("One-way share", "Road geometry"),
    "bus_lane_share": ("Bus-lane share", "Priority lanes"),
    "bike_lane_share": ("Bike-lane share", "Priority lanes"),
    "arterial_share": ("Arterial share", "Road class"),
    "expressway_share": ("Expressway share", "Road class"),
    "posted_speed_mph": ("Posted speed (mph)", "Road class"),
    "signals_per_km": ("Signals per km", "Friction"),
    "turns_per_km": ("Sharp turns per km", "Friction"),
    "block_length_m": ("Mean block length (m)", "Friction"),
    "bridge_tunnel_share": ("Bridge/tunnel share", "Friction"),
    "circuity": ("Circuity", "Route shape"),
}


# --------------------------------------------------------------------------
# External data
# --------------------------------------------------------------------------
def _soda_page(
    dataset_id: str,
    params: dict[str, Any],
    *,
    retries: int = 8,
    timeout: float = 180.0,
) -> list[dict[str, Any]]:
    """One SODA page, retrying the 5xx / timeout failures the portal returns often."""
    url = f"{BASE}/{dataset_id}.json"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            fatal = status is not None and status < 500 and status != 429
            if fatal or attempt == retries - 1:
                raise
            time.sleep(min(60.0, 2.0 * 2**attempt))
    raise RuntimeError("unreachable")


def fetch_geo_dataset(
    dataset_id: str,
    out_path: Path | str,
    *,
    select: str | None = None,
    page_size: int = 10000,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """Download a SODA dataset with its ``the_geom`` column to GeoJSON (cached)."""
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return gpd.read_file(out_path)

    rows: list[dict[str, Any]] = []
    offset = 0
    pbar = tqdm(desc=f"download:{dataset_id}", unit="row")
    while True:
        params: dict[str, Any] = {"$limit": page_size, "$offset": offset, "$order": ":id"}
        if select:
            params["$select"] = select
        page = _soda_page(dataset_id, params)
        if not page:
            break
        rows.extend(page)
        pbar.update(len(page))
        if len(page) < page_size:
            break
        offset += page_size
    pbar.close()

    geoms = [shape(r["the_geom"]) if r.get("the_geom") else None for r in rows]
    attrs = pd.DataFrame([{k: v for k, v in r.items() if k != "the_geom"} for r in rows])
    gdf = gpd.GeoDataFrame(attrs, geometry=geoms, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf


def _explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge multi-part lines into LineStrings, one row per part."""
    out = gdf.copy()
    out["geometry"] = shapely.line_merge(out.geometry.values)
    out = out.explode(index_parts=False, ignore_index=True)
    return out[out.geometry.geom_type == "LineString"].reset_index(drop=True)


# --------------------------------------------------------------------------
# Edge-level attributes
# --------------------------------------------------------------------------
def _azimuth_mod180(line_geoms, dists, half_window: float = 8.0) -> np.ndarray:
    """Local line direction (deg, mod 180) near ``dists`` along each line."""
    lengths = shapely.length(line_geoms)
    d0 = np.clip(dists - half_window, 0, lengths)
    d1 = np.clip(dists + half_window, 0, lengths)
    p0 = shapely.line_interpolate_point(line_geoms, d0)
    p1 = shapely.line_interpolate_point(line_geoms, d1)
    dx = shapely.get_x(p1) - shapely.get_x(p0)
    dy = shapely.get_y(p1) - shapely.get_y(p0)
    return np.degrees(np.arctan2(dx, dy)) % 180.0


def _match_edges_to_lines(
    edges: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    *,
    max_dist_m: float,
    max_angle_deg: float,
) -> pd.DataFrame:
    """For each edge, the nearest *parallel* line within ``max_dist_m``.

    Returns a frame indexed like ``edges`` with columns ``line_idx`` and
    ``match_dist_m`` (rows without a match are absent).
    """
    edge_geoms = edges.geometry.values
    mids = shapely.line_interpolate_point(edge_geoms, 0.5, normalized=True)
    tree = shapely.STRtree(lines.geometry.values)
    e_idx, l_idx = tree.query(mids, predicate="dwithin", distance=max_dist_m)
    if len(e_idx) == 0:
        return pd.DataFrame(columns=["line_idx", "match_dist_m"])

    line_geoms = lines.geometry.values[l_idx]
    mid_pts = mids[e_idx]
    dist = shapely.distance(mid_pts, line_geoms)

    az_line = _azimuth_mod180(line_geoms, shapely.line_locate_point(line_geoms, mid_pts))
    e_geoms = edge_geoms[e_idx]
    az_edge = _azimuth_mod180(e_geoms, 0.5 * shapely.length(e_geoms))
    dang = np.abs(az_line - az_edge)
    dang = np.minimum(dang, 180.0 - dang)

    cand = pd.DataFrame({"e": e_idx, "l": l_idx, "dist": dist, "dang": dang})
    cand = cand[cand["dang"] <= max_angle_deg]
    best = cand.sort_values("dist").drop_duplicates("e", keep="first")
    return pd.DataFrame(
        {"line_idx": best["l"].to_numpy(), "match_dist_m": best["dist"].to_numpy()},
        index=best["e"].to_numpy(),
    )


def _tag_contains(series: pd.Series, needles: tuple[str, ...]) -> pd.Series:
    """True where a (possibly list-valued / stringified) OSM tag contains a needle."""
    s = series.astype(str).str.lower()
    out = pd.Series(False, index=series.index)
    for n in needles:
        out |= s.str.contains(rf"\b{n}\b", regex=True, na=False)
    return out & series.notna()


_NEGATIVE_OSM = frozenset({"", "nan", "none", "no", "0", "false", "null"})


def _tag_positive(series: pd.Series) -> pd.Series:
    """True for OSM tags that affirm a feature (not just present-as-null/'no')."""
    s = series.astype(str).str.strip().str.lower()
    return series.notna() & ~s.isin(_NEGATIVE_OSM)


def build_edge_table(
    G,
    centerline: gpd.GeoDataFrame,
    bus_lanes: gpd.GeoDataFrame,
    *,
    max_dist_m: float = 25.0,
    max_angle_deg: float = 25.0,
) -> pd.DataFrame:
    """One row per OSM edge (indexed by (u, v, key)) with street characteristics."""
    ox = _try_import_ox()
    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(METRIC_CRS)
    edges = edges.reset_index()  # u, v, key columns
    n = len(edges)

    cl = _explode_lines(centerline.to_crs(METRIC_CRS))
    bl = _explode_lines(bus_lanes.to_crs(METRIC_CRS))

    # --- Centerline attributes -------------------------------------------
    m = _match_edges_to_lines(edges, cl, max_dist_m=max_dist_m, max_angle_deg=max_angle_deg)
    num = lambda c: pd.to_numeric(cl[c], errors="coerce")  # noqa: E731
    cl_num = pd.DataFrame(
        {
            "street_width_ft": num("streetwidth"),
            "travel_lanes": num("number_travel_lanes"),
            "park_lanes": num("number_park_lanes"),
            "posted_speed_mph": num("posted_speed"),
        }
    )
    # LION uses 0 for "unknown" width / speed.
    for c in ("street_width_ft", "posted_speed_mph"):
        cl_num.loc[cl_num[c] <= 0, c] = np.nan
    cl_num.loc[cl_num["travel_lanes"] <= 0, "travel_lanes"] = np.nan

    table = pd.DataFrame(index=range(n))
    for c in cl_num.columns:
        table[c] = np.nan
        table.loc[m.index, c] = cl_num[c].to_numpy()[m["line_idx"].to_numpy()]
    table["lion_matched"] = table.index.isin(m.index)

    # --- DOT bus lanes (geometry match) + OSM bus tags --------------------
    mb = _match_edges_to_lines(edges, bl, max_dist_m=15.0, max_angle_deg=max_angle_deg)
    table["bus_dot"] = table.index.isin(mb.index)
    osm_bus = pd.Series(False, index=edges.index)
    for col in ("busway", "busway:left", "busway:right"):
        if col in edges.columns:
            osm_bus |= _tag_positive(edges[col])
    for col in ("lanes:bus", "lanes:bus:forward", "lanes:bus:backward"):
        if col in edges.columns:
            # Numeric lane counts: any positive count counts; "0"/"no" do not.
            raw = edges[col]
            num = pd.to_numeric(raw, errors="coerce")
            osm_bus |= (num.fillna(0) > 0) | (_tag_positive(raw) & num.isna())
    if "bus:lanes" in edges.columns:
        osm_bus |= _tag_contains(edges["bus:lanes"], ("designated",))
    table["bus_osm"] = osm_bus.to_numpy()
    table["bus_lane"] = (table["bus_dot"] | table["bus_osm"]).astype(float)

    # --- OSM-only attributes ----------------------------------------------
    bike = pd.Series(False, index=edges.index)
    for col in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
        if col in edges.columns:
            bike |= _tag_contains(edges[col], ("lane", "track", "opposite_lane", "opposite_track"))
    table["bike_lane"] = bike.to_numpy().astype(float)

    hw = edges["highway"].astype(str).str.lower()
    table["arterial"] = hw.str.contains(r"\b(?:primary|trunk|secondary)(?:_link)?\b", regex=True).astype(float).to_numpy()
    table["expressway"] = hw.str.contains(r"\bmotorway(?:_link)?\b", regex=True).astype(float).to_numpy()
    table["oneway"] = edges["oneway"].astype(str).str.lower().isin(["true", "yes", "1"]).astype(float).to_numpy()
    if "bridge" in edges.columns or "tunnel" in edges.columns:
        bt = pd.Series(False, index=edges.index)
        for col in ("bridge", "tunnel"):
            if col in edges.columns:
                bt |= edges[col].notna() & ~edges[col].astype(str).str.lower().isin(["no", "nan"])
        table["bridge_tunnel"] = bt.to_numpy().astype(float)
    else:
        table["bridge_tunnel"] = 0.0

    table["length_m"] = pd.to_numeric(edges["length"], errors="coerce").to_numpy()
    table.index = pd.MultiIndex.from_frame(edges[["u", "v", "key"]])
    return table


# --------------------------------------------------------------------------
# Route-level aggregation
# --------------------------------------------------------------------------
def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))


def prepare_graph(graph_path: Path | str, *, cache_path: Path | str | None = None):
    """Load the rich graph with speeds / travel times / edge bearings.

    Parsing a metro-scale GraphML and re-deriving the osmnx attributes costs
    minutes; ``cache_path`` pickles the finished graph so re-runs pay seconds.
    The cache is invalidated by the source graph's mtime and size.
    """
    ox = _try_import_ox()
    graph_path = Path(graph_path)
    stamp = (str(graph_path.resolve()), graph_path.stat().st_mtime_ns, graph_path.stat().st_size)

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            blob = pickle.loads(cache_path.read_bytes())
            if blob.get("stamp") == stamp:
                return blob["graph"]
        except Exception:  # noqa: BLE001 - a stale/corrupt cache must never be fatal
            pass

    G = load_graph(graph_path)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.bearing.add_edge_bearings(G)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_bytes(pickle.dumps({"stamp": stamp, "graph": G}, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(cache_path)
    return G


# Feature columns of ``build_edge_table`` consumed by ``route_characteristics``.
_EDGE_ATTRS = (
    "street_width_ft",
    "travel_lanes",
    "park_lanes",
    "posted_speed_mph",
    "oneway",
    "bus_lane",
    "bike_lane",
    "arterial",
    "expressway",
    "bridge_tunnel",
    "lion_matched",
)


class EdgeAttrs:
    """Column-major view of a ``build_edge_table`` frame for per-route lookups.

    ``DataFrame.reindex`` on a million-row MultiIndex costs milliseconds *per
    route*; here each column is a float array and each ``(u, v, key)`` maps to a
    row offset, so a route becomes one ``dict`` walk plus fancy indexing. Index
    ``-1`` is a NaN sentinel for edges absent from the table.
    """

    __slots__ = ("pos", "cols")

    def __init__(self, edge_table: pd.DataFrame):
        self.pos = {k: i for i, k in enumerate(edge_table.index)}
        self.cols = {}
        for c in _EDGE_ATTRS:
            arr = np.empty(len(edge_table) + 1, dtype=float)
            arr[:-1] = pd.to_numeric(edge_table[c], errors="coerce").to_numpy(dtype=float)
            arr[-1] = np.nan  # sentinel row hit by index -1
            self.cols[c] = arr

    def take(self, keys: list[tuple]) -> dict[str, np.ndarray]:
        idx = np.fromiter((self.pos.get(k, -1) for k in keys), dtype=np.int64, count=len(keys))
        return {c: arr[idx] for c, arr in self.cols.items()}


def _as_edge_attrs(edge_table) -> EdgeAttrs:
    return edge_table if isinstance(edge_table, EdgeAttrs) else EdgeAttrs(edge_table)


def signal_nodes(G) -> set:
    """Graph nodes tagged as traffic signals (computed once, not per route)."""
    return {
        n
        for n, hw in G.nodes(data="highway")
        if hw is not None and "traffic_signals" in str(hw)
    }


def route_characteristics(
    G,
    edge_table,
    path: list,
    crow_km: float,
    *,
    turn_threshold_deg: float = 45.0,
    signals_set: set | None = None,
) -> dict[str, float]:
    """Length-weighted street characteristics along a node path.

    ``edge_table`` may be a ``build_edge_table`` frame or a prebuilt
    :class:`EdgeAttrs`; pass the latter when routing many paths.
    """
    attrs = _as_edge_attrs(edge_table)
    if signals_set is None:
        signals_set = signal_nodes(G)

    keys, lengths, bearings = [], [], []
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v)
        if not edata:
            continue
        k, edge = min(edata.items(), key=lambda kv: kv[1].get("travel_time", 1e18))
        keys.append((u, v, k))
        lengths.append(float(edge.get("length") or 0.0))
        bearings.append(float(edge.get("bearing", np.nan)))
    if not keys:
        return {}

    w = np.asarray(lengths, dtype=float)
    total_m = float(w.sum())
    if total_m <= 0:
        return {}
    sub = attrs.take(keys)
    matched = sub["lion_matched"] > 0

    # Sharp turns between consecutive edges
    b = np.asarray(bearings, dtype=float)
    if len(b) > 1:
        d = np.abs(b[:-1] - b[1:]) % 360.0
        d = np.minimum(d, 360.0 - d)
        turns = int(np.count_nonzero(d >= turn_threshold_deg))
    else:
        turns = 0
    signals = sum(1 for node in path[1:-1] if node in signals_set)

    km = total_m / 1000.0
    return {
        "path_km": km,
        "n_edges": float(len(keys)),
        "lion_match_share": float(w[matched].sum() / total_m),
        "street_width_ft": _wmean(sub["street_width_ft"], w),
        "travel_lanes": _wmean(sub["travel_lanes"], w),
        "park_lanes": _wmean(sub["park_lanes"], w),
        "posted_speed_mph": _wmean(sub["posted_speed_mph"], w),
        "oneway_share": _wmean(sub["oneway"], w),
        "bus_lane_share": _wmean(sub["bus_lane"], w),
        "bike_lane_share": _wmean(sub["bike_lane"], w),
        "arterial_share": _wmean(sub["arterial"], w),
        "expressway_share": _wmean(sub["expressway"], w),
        "bridge_tunnel_share": _wmean(sub["bridge_tunnel"], w),
        "signals_per_km": signals / km if km > 0 else np.nan,
        "turns_per_km": turns / km if km > 0 else np.nan,
        "block_length_m": total_m / len(keys),
        "circuity": km / crow_km if crow_km and crow_km > 0.05 else np.nan,
    }


def route_paths_for_nodes(
    G,
    pairs: list[tuple],
    *,
    weight: str = "travel_time",
    show_progress: bool = True,
) -> dict[tuple, list]:
    """Shortest node path for each unique ``(origin, dest)`` node pair.

    ``nx.shortest_path`` runs a bidirectional Dijkstra, which on the NYC drive
    graph costs ~8 ms for an EMS-length trip. Solving a whole Dijkstra tree per
    origin instead costs ~440 ms, so it only wins for origins with dozens of
    destinations — far more than dispatch data produces. Dedupe plus the caller's
    path cache are what actually keep this cheap.
    """
    paths: dict[tuple, list] = {}
    todo = [(o, d) for o, d in dict.fromkeys(pairs) if o != d]
    it = tqdm(todo, desc="routing", unit="od") if show_progress else todo
    for o, d in it:
        try:
            paths[(o, d)] = nx.shortest_path(G, o, d, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return paths


def route_table_for_od(
    G,
    edge_table: pd.DataFrame,
    od: pd.DataFrame,
    *,
    show_progress: bool = True,
    path_cache: Path | str | None = None,
) -> pd.DataFrame:
    """Route each unique (start, dest) node pair once; return features per ``od_id``.

    ``od`` needs columns od_id, start_lat, start_lon, dest_lat, dest_lon, crow_km.
    ``path_cache`` persists the node paths (which depend only on the graph, not
    on ``edge_table``), so re-running after an edge-table change skips routing.
    """
    ox = _try_import_ox()
    # One nearest_nodes call: each call rebuilds a KD-tree over every graph node.
    n = len(od)
    xs = np.concatenate([od["start_lon"].to_numpy(float), od["dest_lon"].to_numpy(float)])
    ys = np.concatenate([od["start_lat"].to_numpy(float), od["dest_lat"].to_numpy(float)])
    nodes = list(ox.nearest_nodes(G, xs, ys))
    od = od.assign(o_node=nodes[:n], d_node=nodes[n:])

    pairs = list(zip(od["o_node"], od["d_node"]))

    cached: dict[tuple, list] = {}
    cache_file = Path(path_cache) if path_cache else None
    if cache_file and cache_file.exists():
        try:
            cached = pickle.loads(cache_file.read_bytes())
        except Exception:  # noqa: BLE001
            cached = {}
    todo = [p for p in dict.fromkeys(pairs) if p not in cached and p[0] != p[1]]
    if todo:
        cached.update(route_paths_for_nodes(G, todo, show_progress=show_progress))
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
            tmp.write_bytes(pickle.dumps(cached, protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(cache_file)

    attrs = _as_edge_attrs(edge_table)
    signals_set = signal_nodes(G)
    feats_by_pair: dict[tuple, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for r in od.itertuples(index=False):
        key = (r.o_node, r.d_node)
        if key not in feats_by_pair:
            path = cached.get(key)
            feats_by_pair[key] = (
                route_characteristics(
                    G, attrs, path, float(r.crow_km), signals_set=signals_set
                )
                if path
                else {}
            )
        feats = dict(feats_by_pair[key])
        if "path_km" in feats:  # circuity depends on this row's own crow-flies distance
            feats["circuity"] = feats["path_km"] / r.crow_km if r.crow_km > 0.05 else np.nan
        rows.append({"od_id": r.od_id, **feats})
    return pd.DataFrame(rows)
