"""Lightweight firetruck routing on an OSM drive GraphML (no NYC/weather deps).

Prefers arterials, penalises residential/service, and blocks long contraflow —
same policy as the former ``emvro.routing.firetruck`` overlay, but self-contained
for the London-only tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox

WEIGHT_KEY = "weight_firetruck"
SECONDS_KEY = "firetruck_s"
MAX_CONTRAFLOW_M = 80.0
CONTRAFLOW_EXTRA_S = 12.0
RESIDENTIAL_PENALTY = 1.12
PRIMARY_BONUS = 0.94
DEADEND_PENALTY = 1.35


def _highway(data: dict) -> str:
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    return str(hw or "").lower()


def _base_seconds(data: dict) -> float:
    length = float(data.get("length") or data.get("length_m") or 0.0)
    # Prefer posted maxspeed when present; else ~35 kph apparatus free-flow.
    ms = data.get("maxspeed")
    kph = 35.0
    if ms is not None:
        try:
            import re

            nums = [float(x) for x in re.findall(r"\d+", str(ms))]
            if nums:
                kph = max(15.0, min(nums[0], 70.0))
        except Exception:  # noqa: BLE001
            pass
    if length <= 0:
        return 1.0
    return (length / 1000.0) / kph * 3600.0


def annotate_firetruck_costs(G: nx.MultiDiGraph) -> dict[str, Any]:
    n_blocked = n_contra = n_res = n_dead = 0
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        length = float(data.get("length") or data.get("length_m") or 0.0)
        hw = _highway(data)
        cost = float(data.get("travel_time") or data.get("emv_s") or 0.0)
        if cost <= 0:
            cost = _base_seconds(data)

        oneway = str(data.get("oneway", "")).lower() in {"true", "yes", "1", "-1"}
        # Heuristic: edges tagged junction=roundabout / oneway already in digraph;
        # treat explicit "contraflow" / "lanes:backward" emergency tags if present.
        is_contra = data.get("emv_corridor_kind") == "contraflow" or str(
            data.get("oneway:bicycle") or ""
        ).lower() in {"yes", "true"}
        if is_contra:
            if length > MAX_CONTRAFLOW_M:
                cost = 1e12
                n_blocked += 1
            else:
                cost += CONTRAFLOW_EXTRA_S
                n_contra += 1

        if hw in {"residential", "living_street", "service", "unclassified"}:
            cost *= RESIDENTIAL_PENALTY
            n_res += 1
        elif hw in {"trunk", "trunk_link", "primary", "primary_link"} or data.get("busway"):
            cost *= PRIMARY_BONUS

        if data.get("noexit") in (True, "yes", "True", "1") or hw == "turning_circle":
            cost *= DEADEND_PENALTY
            n_dead += 1

        data[SECONDS_KEY] = float(cost)
        data[WEIGHT_KEY] = float(cost)
        if "length_m" not in data and length:
            data["length_m"] = length

    stats = {
        "contraflow_blocked": n_blocked,
        "contraflow_kept": n_contra,
        "residential_penalized": n_res,
        "deadend_penalized": n_dead,
        "weight_key": WEIGHT_KEY,
    }
    G.graph["firetruck_routing"] = stats
    return stats


def prepare_firetruck_graph(graph_path: Path | str, *, hour: int = 17) -> nx.MultiDiGraph:
    """Load GraphML and annotate firetruck edge weights. ``hour`` reserved for traffic priors."""
    del hour  # reserved
    G = ox.load_graphml(Path(graph_path))
    annotate_firetruck_costs(G)
    return G


def nearest_node(G: nx.MultiDiGraph, lon: float, lat: float):
    return ox.distance.nearest_nodes(G, float(lon), float(lat))


def firetruck_route_latlon(G, lon1: float, lat1: float, lon2: float, lat2: float):
    o = nearest_node(G, lon1, lat1)
    d = nearest_node(G, lon2, lat2)
    try:
        path = nx.shortest_path(G, o, d, weight=WEIGHT_KEY)
        ok = True
        meta: dict[str, Any] = {}
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        path = []
        ok = False
        meta = {"error": str(exc)}

    class _R:
        pass

    r = _R()
    r.ok = ok
    r.node_path = path
    r.meta = meta
    r.origin = o
    r.dest = d
    return r


def path_firetruck_stats(G, path) -> tuple[float, float, int]:
    if len(path) < 2:
        return float("nan"), float("nan"), 0
    total_t = total_d = 0.0
    n = 0
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            return float("nan"), float("nan"), 0
        edge = min(
            edata.values(),
            key=lambda d: d.get(SECONDS_KEY, d.get("travel_time", 1e18)),
        )
        total_t += float(edge.get(SECONDS_KEY) or edge.get("travel_time") or 0.0)
        total_d += float(edge.get("length_m") or edge.get("length") or 0.0)
        n += 1
    return total_t, total_d, n
