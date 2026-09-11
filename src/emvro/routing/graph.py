"""Shared graph helpers and route result types for EMV routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

from ..street_features import load_graph, _try_import_ox


@dataclass
class RouteResult:
    model: str
    node_path: list[Any]
    travel_seconds: float
    distance_m: float
    n_edges: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.node_path) and len(self.node_path) >= 2


def prepare_routing_graph(
    graph_path: Path | str,
    *,
    hour: int = 12,
    congestion: float | None = None,
) -> nx.MultiDiGraph:
    """Load OSM graph and attach EMV-oriented edge costs."""
    ox = _try_import_ox()
    G = load_graph(graph_path)
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:  # noqa: BLE001
        pass

    # Congestion multiplier by hour (simple BPR-style prior)
    if congestion is None:
        if hour in (7, 8, 9, 16, 17, 18, 19):
            congestion = 1.45
        elif hour >= 22 or hour < 6:
            congestion = 1.05
        else:
            congestion = 1.20

    # EMV can use bus lanes / move faster — discount travel_time slightly on
    # primary/trunk and residential (proxy until LION bus-lane tags are wired).
    for u, v, k, data in G.edges(keys=True, data=True):
        base = float(data.get("travel_time") or 0.0)
        length = float(data.get("length") or 0.0)
        if base <= 0 and length > 0:
            speed = float(data.get("speed_kph") or 30.0)
            base = (length / 1000.0) / max(speed, 5.0) * 3600.0
        hw = data.get("highway")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
        hw = str(hw or "").lower()
        emv_factor = 0.85  # default EMV advantage vs civilian
        if hw in {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}:
            emv_factor = 0.80
        elif hw in {"residential", "living_street", "unclassified"}:
            emv_factor = 0.90
        # Bus-lane proxy: osmnx may expose "lanes:bus" rarely; keep hook
        if data.get("busway") or data.get("lanes:bus"):
            emv_factor *= 0.92

        data["civilian_s"] = base * congestion
        data["emv_s"] = base * congestion * emv_factor
        data["length_m"] = length
        # Default routing weight for EMV Dijkstra control/variants
        data["weight_emv"] = data["emv_s"]
        data["weight_length"] = length if length > 0 else 1.0
    return G


def nearest_node(G, lon: float, lat: float):
    ox = _try_import_ox()
    return ox.nearest_nodes(G, float(lon), float(lat))


def path_stats(G, path: Sequence) -> tuple[float, float, int]:
    """Return (emv_seconds, distance_m, n_edges) along a node path."""
    if len(path) < 2:
        return np.nan, np.nan, 0
    total_t = 0.0
    total_d = 0.0
    n = 0
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            return np.nan, np.nan, 0
        edge = min(edata.values(), key=lambda d: d.get("emv_s", d.get("travel_time", 1e18)))
        total_t += float(edge.get("emv_s") or edge.get("travel_time") or 0.0)
        total_d += float(edge.get("length_m") or edge.get("length") or 0.0)
        n += 1
    return total_t, total_d, n


def dijkstra_route(
    G,
    origin,
    dest,
    *,
    weight: str = "weight_emv",
    model_name: str = "dijkstra",
) -> RouteResult:
    try:
        path = nx.shortest_path(G, origin, dest, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return RouteResult(model_name, [], np.nan, np.nan, 0)
    t, d, n = path_stats(G, path)
    return RouteResult(model_name, list(path), t, d, n, meta={"weight": weight})


def control_shortest_distance(G, origin, dest) -> RouteResult:
    """Civilian-style control: minimize distance (standard GPS proxy)."""
    return dijkstra_route(G, origin, dest, weight="weight_length", model_name="control_distance")


def control_civilian_time(G, origin, dest) -> RouteResult:
    """Civilian-style control: minimize congested travel_time."""
    # Temporarily expose civilian weight
    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight_civilian"] = float(data.get("civilian_s") or data.get("travel_time") or 1.0)
    return dijkstra_route(G, origin, dest, weight="weight_civilian", model_name="control_civilian_time")


def reconstruct_path_from_parents(parents: dict, origin, dest) -> list:
    if dest not in parents and dest != origin:
        return []
    path = [dest]
    cur = dest
    while cur != origin:
        cur = parents.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


EdgeCostFn = Callable[[dict], float]
