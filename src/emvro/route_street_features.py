"""Street characteristics along the likely response route.

For each destination, trace the fastest free-flow path on the drive graph from
its first-due engine's firehouse (nearest firehouse when unknown) and summarise
the streets on it, length-weighted: width, lanes, parking lanes, posted speed
(NYC Centerline), bus / bike lanes (DOT + OSM), road class, one-ways,
bridges/tunnels, traffic signals and sharp turns.

Edge attributes come from ``emvro.street_analysis.build_edge_table`` on the
rich-tag graph (``build_osm_graph.py --rich-tags``). One SciPy Dijkstra per
firehouse (with predecessors) covers every destination it serves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

ROUTE_STREET_COLUMNS = [
    "rt_path_km",
    "rt_street_width_ft",
    "rt_travel_lanes",
    "rt_park_lanes",
    "rt_posted_speed_mph",
    "rt_bus_lane_share",
    "rt_bike_lane_share",
    "rt_arterial_share",
    "rt_expressway_share",
    "rt_oneway_share",
    "rt_bridge_tunnel_share",
    "rt_signals_per_km",
    "rt_turns_per_km",
    "rt_block_length_m",
    "rt_lion_match_share",
]
_MEAN_COLS = ["street_width_ft", "travel_lanes", "park_lanes", "posted_speed_mph"]
_SHARE_COLS = {"bus_lane": "bus_lane_share", "bike_lane": "bike_lane_share", "arterial": "arterial_share",
               "expressway": "expressway_share", "oneway": "oneway_share", "bridge_tunnel": "bridge_tunnel_share"}


class StreetGraph:
    """CSR travel-time graph with per-edge street attributes (fastest parallel edge)."""

    def __init__(self, graph_path: Path | str, edge_table: pd.DataFrame):
        import osmnx as ox

        G = ox.load_graphml(Path(graph_path))
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.bearing.add_edge_bearings(G)
        nodes = list(G.nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        self.lat = np.array([G.nodes[n]["y"] for n in nodes])
        self.lon = np.array([G.nodes[n]["x"] for n in nodes])
        self.signal = np.array(["traffic_signals" in str(G.nodes[n].get("highway", "")) for n in nodes])

        rec = []
        for u, v, k, d in G.edges(keys=True, data=True):
            rec.append((idx[u], idx[v], u, v, k, max(float(d.get("travel_time") or 0), 0.01),
                        float(d.get("length") or 0), float(d.get("bearing", np.nan))))
        e = pd.DataFrame(rec, columns=["ui", "vi", "u", "v", "key", "t", "len", "bearing"])
        e = e.sort_values("t").drop_duplicates(["ui", "vi"]).reset_index(drop=True)
        attrs = edge_table.reindex(pd.MultiIndex.from_frame(e[["u", "v", "key"]]))
        self.attr = {c: attrs[c].to_numpy(float) for c in _MEAN_COLS + list(_SHARE_COLS) + ["lion_matched"]}
        self.len = e["len"].to_numpy()
        self.bearing = e["bearing"].to_numpy()
        n = len(nodes)
        # Store edge row + 1 in a parallel matrix so (u, v) -> edge attributes is O(1).
        self.time = sparse.csr_matrix((e["t"], (e["ui"], e["vi"])), shape=(n, n))
        self.eid = sparse.csr_matrix((np.arange(len(e)) + 1, (e["ui"], e["vi"])), shape=(n, n))

    def nearest(self, lat, lon) -> np.ndarray:
        from scipy.spatial import cKDTree

        k = np.cos(np.radians(40.7))
        if not hasattr(self, "_tree"):
            self._tree = cKDTree(np.c_[self.lon * k, self.lat])
        return self._tree.query(np.c_[np.asarray(lon) * k, np.asarray(lat)])[1]


def _summarise(sg: StreetGraph, path: np.ndarray) -> dict[str, float]:
    if len(path) < 2:
        return {}
    ids = np.asarray(sg.eid[path[:-1], path[1:]]).ravel().astype(int) - 1
    ids = ids[ids >= 0]
    if not len(ids):
        return {}
    w = sg.len[ids]
    tot = w.sum()
    if tot <= 0:
        return {}
    km = tot / 1000.0
    out = {"rt_path_km": km}
    for c in _MEAN_COLS:
        v = sg.attr[c][ids]
        ok = np.isfinite(v)
        out[f"rt_{c}"] = float(np.average(v[ok], weights=w[ok])) if ok.any() and w[ok].sum() > 0 else np.nan
    for c, name in _SHARE_COLS.items():
        out[f"rt_{name}"] = float(np.average(np.nan_to_num(sg.attr[c][ids]), weights=w))
    out["rt_lion_match_share"] = float(np.average(np.nan_to_num(sg.attr["lion_matched"][ids]), weights=w))
    b = sg.bearing[ids]
    db = np.abs(np.diff(b)) % 360
    db = np.minimum(db, 360 - db)
    out["rt_turns_per_km"] = float(np.nansum(db >= 45) / km)
    out["rt_signals_per_km"] = float(sg.signal[path[1:-1]].sum() / km)
    out["rt_block_length_m"] = float(tot / len(ids))
    return out


def route_street_features(
    dests: pd.DataFrame,
    houses: pd.DataFrame,
    engine_house: dict[int, int],
    sg: StreetGraph,
    *,
    limit_s: float = 1500.0,
) -> pd.DataFrame:
    """Route street features per destination (aligned to ``dests.index``).

    ``dests`` needs dest_lat, dest_lon, first_due_engine; ``houses`` is
    ``network_origins.firehouse_table`` output.
    """
    d_node = sg.nearest(dests["dest_lat"].to_numpy(), dests["dest_lon"].to_numpy())
    h_node = sg.nearest(houses["lat"].to_numpy(), houses["lon"].to_numpy())
    eng = pd.to_numeric(dests["first_due_engine"], errors="coerce").to_numpy()
    house = np.array([engine_house.get(int(e), -1) if e == e else -1 for e in eng])
    # No first-due engine: route from the geographically nearest house instead.
    miss = house < 0
    if miss.any():
        from scipy.spatial import cKDTree

        k = np.cos(np.radians(40.7))
        t = cKDTree(np.c_[houses["lon"].to_numpy() * k, houses["lat"].to_numpy()])
        house[miss] = t.query(np.c_[dests["dest_lon"].to_numpy()[miss] * k, dests["dest_lat"].to_numpy()[miss]])[1]

    rows: list[dict] = [dict() for _ in range(len(dests))]
    for h in np.unique(house):
        sel = np.flatnonzero(house == h)
        _, pred = dijkstra(sg.time, directed=True, indices=int(h_node[h]), limit=limit_s, return_predecessors=True)
        src = int(h_node[h])
        for i in sel:
            tgt = int(d_node[i])
            if tgt == src:
                continue
            path = [tgt]
            while path[-1] != src and pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            if path[-1] != src:
                continue  # unreachable within limit
            rows[i] = _summarise(sg, np.array(path[::-1]))
    return pd.DataFrame(rows, index=dests.index).reindex(columns=ROUTE_STREET_COLUMNS)
