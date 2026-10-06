"""Street characteristics along each trip's shortest legal route.

Two steps:

1. ``edge_feature_table(G)``: one row per directed graph edge with its street
   characteristics. Road class, one-way, speed limit, lanes, bridge/tunnel and
   roundabout come from the osmnx graph. Point features that osmnx drops when it
   simplifies the graph (traffic signals, traffic calming such as speed humps,
   pedestrian crossings, give-way/stop signs) and bus lanes are read from the
   Geofabrik .osm.pbf: a point feature is matched to an edge when its exact
   coordinate is one of the edge's interior vertices or its end junction.

2. ``RouteFeatureRouter.route_features(...)``: for each origin (station), one
   Dijkstra shortest-distance tree, then the sum of every edge feature from the
   origin to every junction by pointer jumping (log2(depth) vectorised passes),
   plus turns between consecutive edges. Each trip reads its sums at its
   destination junction.

The route is the shortest-distance legal route (one-way streets respected), not
the route the engine actually drove; features describe that assumed route.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .road_distance import PBF_PATH, ROOT, RoadRouter

EDGE_FEATURES_PATH = ROOT / "data" / "processed" / "london_edge_features.parquet"

# Road class groups (OSM highway=*), coarse enough to have support everywhere.
ROAD_CLASS = {
    "motorway": "major", "motorway_link": "major", "trunk": "major", "trunk_link": "major",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "residential": "residential", "living_street": "residential",
    "unclassified": "minor", "busway": "minor", "road": "minor",
}
CLASSES = ["major", "primary", "secondary", "tertiary", "residential", "minor"]
POINT_KINDS = ["signals", "calming", "crossing", "give_way"]
BUS_LANE_KEYS = ("busway", "busway:both", "busway:left", "busway:right", "bus:lanes", "lanes:bus",
                 "lanes:psv", "psv:lanes", "bus:lanes:forward", "bus:lanes:backward")

# Route-level columns produced by RouteFeatureRouter.route_features.
ROUTE_FEATURES = (
    [f"share_{c}" for c in CLASSES]
    + ["share_oneway", "share_20mph", "share_bus_lane", "share_bridge_tunnel", "mean_lanes",
       "signals_per_km", "calming_per_km", "crossings_per_km", "give_way_per_km",
       "junctions_per_km", "turns_per_km", "sharp_turns_per_km", "roundabouts_per_km"]
)


def _first(x):
    return x[0] if isinstance(x, list) and x else x


def _mph(x) -> float:
    vals = [float(m) for m in re.findall(r"\d+", str(x))] if x is not None else []
    return min(vals) if vals else np.nan


def _lanes(x) -> float:
    vals = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(x))] if x is not None else []
    return float(np.mean(vals)) if vals else np.nan


def _point_kind(tags) -> str | None:
    hw = tags.get("highway")
    if hw == "traffic_signals" or tags.get("crossing") == "traffic_signals":
        return "signals"
    if "traffic_calming" in tags and tags["traffic_calming"] != "no":
        return "calming"
    if hw == "crossing":
        return "crossing"
    if hw in ("give_way", "stop"):
        return "give_way"
    return None


def _pbf_points_and_bus_ways(pbf: Path) -> tuple[dict[tuple[float, float], str], set[int]]:
    import osmium

    points: dict[tuple[float, float], str] = {}
    for n in osmium.FileProcessor(str(pbf), osmium.osm.NODE):
        if len(n.tags):
            kind = _point_kind(n.tags)
            if kind and n.location.valid():
                points[(round(n.location.lon, 7), round(n.location.lat, 7))] = kind
    bus: set[int] = set()
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        t = w.tags
        if "highway" in t and any(k in t and t[k] not in ("no", "") for k in BUS_LANE_KEYS):
            bus.add(w.id)
    return points, bus


def _bearing(x1, y1, x2, y2) -> float:
    """Compass bearing (degrees) of a short segment, equirectangular approximation."""
    dx = (x2 - x1) * np.cos(np.radians((y1 + y2) / 2))
    return float(np.degrees(np.arctan2(dx, y2 - y1)) % 360)


def edge_feature_table(G, pbf: Path = PBF_PATH, cache: Path = EDGE_FEATURES_PATH) -> pd.DataFrame:
    """One row per directed edge (u, v, key) with street characteristics."""
    if cache.exists():
        return pd.read_parquet(cache)
    points, bus_ways = _pbf_points_and_bus_ways(pbf)
    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if "geometry" in d:
            coords = list(d["geometry"].coords)
        else:
            coords = [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        counts = dict.fromkeys(POINT_KINDS, 0)
        for x, y in coords[1:]:  # interior vertices + end junction (start counted on the previous edge)
            kind = points.get((round(x, 7), round(y, 7)))
            if kind:
                counts[kind] += 1
        osmids = d["osmid"] if isinstance(d["osmid"], list) else [d["osmid"]]
        hw = _first(d.get("highway"))
        rows.append(
            {
                "u": u, "v": v, "key": k,
                "length": float(d["length"]),
                "road_class": ROAD_CLASS.get(hw, "minor"),
                "oneway": bool(d.get("oneway") in (True, "True")),
                "maxspeed_mph": _mph(d.get("maxspeed")),
                "lanes": _lanes(d.get("lanes")),
                "bridge_tunnel": bool(d.get("bridge") or d.get("tunnel")),
                "roundabout": _first(d.get("junction")) in ("roundabout", "circular"),
                "bus_lane": any(int(o) in bus_ways for o in osmids),
                **counts,
                "bearing_start": _bearing(*coords[0], *coords[1]),
                "bearing_end": _bearing(*coords[-2], *coords[-1]),
            }
        )
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


class RouteFeatureRouter(RoadRouter):
    """RoadRouter that also sums street characteristics along each shortest route."""

    def __init__(self, G, edges: pd.DataFrame):
        from scipy.sparse import csr_matrix

        super().__init__(G)
        e = edges.assign(ui=edges["u"].map(self.index), vi=edges["v"].map(self.index))
        # Same edge Dijkstra uses: the shortest of any parallel edges.
        e = e.sort_values("length").drop_duplicates(["ui", "vi"]).reset_index(drop=True)
        n = len(self.nodes)
        self.edge_id = csr_matrix((np.arange(1, len(e) + 1), (e["ui"], e["vi"])), shape=(n, n))
        L = e["length"].to_numpy()
        cols = {f"len_{c}": L * (e["road_class"] == c) for c in CLASSES}
        cols.update(
            len_total=L,
            len_oneway=L * e["oneway"],
            len_20mph=L * (e["maxspeed_mph"] <= 20),
            len_bus_lane=L * e["bus_lane"],
            len_bridge_tunnel=L * e["bridge_tunnel"],
            lanes_x_len=np.nan_to_num(e["lanes"].to_numpy()) * L,
            len_lanes_tagged=L * e["lanes"].notna(),
            n_junctions=np.ones(len(e)),
            n_roundabout=e["roundabout"].astype(float),
            **{f"n_{k}": e[k].astype(float) for k in POINT_KINDS},
        )
        self.edge_vals = pd.DataFrame(cols)
        self.bearing_start = e["bearing_start"].to_numpy()
        self.bearing_end = e["bearing_end"].to_numpy()

    def _tree_sums(self, origin: int) -> tuple[pd.DataFrame, np.ndarray]:
        """Feature sums from ``origin`` to every node along its shortest-distance tree."""
        from scipy.sparse.csgraph import dijkstra

        dist, pred = dijkstra(self.csr, directed=True, indices=origin, return_predecessors=True)
        n = len(pred)
        nodes = np.arange(n)
        has_parent = pred >= 0
        parent = np.where(has_parent, pred, nodes)
        eid = np.zeros(n, dtype=int)  # 1-based edge id into this node; 0 = root / unreachable
        eid[has_parent] = np.asarray(self.edge_id[pred[has_parent], nodes[has_parent]]).ravel()
        vals = np.zeros((n, self.edge_vals.shape[1] + 2))
        vals[has_parent, : self.edge_vals.shape[1]] = self.edge_vals.to_numpy()[eid[has_parent] - 1]
        # Turn at the parent junction: incoming edge's end bearing vs this edge's start bearing.
        pe = eid[parent]
        turn_ok = has_parent & (pe > 0)
        diff = np.abs(self.bearing_end[pe[turn_ok] - 1] - self.bearing_start[eid[turn_ok] - 1]) % 360
        angle = np.minimum(diff, 360 - diff)
        vals[turn_ok, -2] = angle > 45
        vals[turn_ok, -1] = angle > 120
        # Pointer jumping: after k passes each node holds the sum over its 2^k nearest ancestors' edges.
        S, P = vals, parent
        while True:
            S = S + S[P] * (P != nodes)[:, None]
            P_next = P[P]
            if np.array_equal(P_next, P):
                break
            P = P_next
        cols = list(self.edge_vals.columns) + ["n_turns", "n_sharp_turns"]
        return pd.DataFrame(S, columns=cols), dist

    def route_features(self, start_lat, start_lon, dest_lat, dest_lon) -> pd.DataFrame:
        """ROUTE_FEATURES for each trip (rows aligned with the inputs)."""
        s_idx, _ = self.snap(np.asarray(start_lat), np.asarray(start_lon))
        d_idx, _ = self.snap(np.asarray(dest_lat), np.asarray(dest_lon))
        blocks = []
        for o in np.unique(s_idx):
            tree, _ = self._tree_sums(o)
            mask = s_idx == o
            blocks.append(tree.iloc[d_idx[mask]].set_index(np.flatnonzero(mask)))
        sums = pd.concat(blocks).sort_index()
        L = sums["len_total"].replace(0, np.nan)
        km = L / 1000
        out = pd.DataFrame(index=sums.index)
        for c in CLASSES:
            out[f"share_{c}"] = sums[f"len_{c}"] / L
        out["share_oneway"] = sums["len_oneway"] / L
        out["share_20mph"] = sums["len_20mph"] / L
        out["share_bus_lane"] = sums["len_bus_lane"] / L
        out["share_bridge_tunnel"] = sums["len_bridge_tunnel"] / L
        out["mean_lanes"] = sums["lanes_x_len"] / sums["len_lanes_tagged"].replace(0, np.nan)
        out["signals_per_km"] = sums["n_signals"] / km
        out["calming_per_km"] = sums["n_calming"] / km
        out["crossings_per_km"] = sums["n_crossing"] / km
        out["give_way_per_km"] = sums["n_give_way"] / km
        out["junctions_per_km"] = sums["n_junctions"] / km
        out["turns_per_km"] = sums["n_turns"] / km
        out["sharp_turns_per_km"] = sums["n_sharp_turns"] / km
        out["roundabouts_per_km"] = sums["n_roundabout"] / km
        return out[ROUTE_FEATURES].reset_index(drop=True)
