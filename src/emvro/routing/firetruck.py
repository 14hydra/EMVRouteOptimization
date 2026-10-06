"""Firetruck-specific routing costs on the EMV drive graph.

Generic ``weight_emv`` is tuned for light EMS demos (busways + short contraflow).
Fire apparatus are larger, slower to clear tight residential, and should use
contraflow more conservatively. This module writes ``weight_firetruck`` /
``firetruck_s`` on each edge after ``prepare_routing_graph``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .graph import RouteResult, dijkstra_route, nearest_node, prepare_routing_graph

# Apparatus constraints relative to generic EMV demo costs.
MAX_FIRETRUCK_CONTRAFLOW_M = 80.0  # shorter than generic 120 m
CONTRAFLOW_RISK_EXTRA_S = 12.0  # on top of emv_s risk already in graph
RESIDENTIAL_PENALTY = 1.12  # prefer arterials for engines/ladders
PRIMARY_BONUS = 0.94  # slight preference for primary/trunk when already legal
DEADEND_PENALTY = 1.35

WEIGHT_KEY = "weight_firetruck"
SECONDS_KEY = "firetruck_s"


def _highway(data: dict) -> str:
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    return str(hw or "").lower()


def annotate_firetruck_costs(G: nx.MultiDiGraph) -> dict[str, Any]:
    """Derive firetruck edge weights from existing ``emv_s`` / corridor tags."""
    n_contra_blocked = 0
    n_contra_kept = 0
    n_residential = 0
    n_deadend = 0

    for _u, _v, _k, data in G.edges(keys=True, data=True):
        base = float(data.get("emv_s") or data.get("travel_time") or 0.0)
        length = float(data.get("length_m") or data.get("length") or 0.0)
        hw = _highway(data)
        is_contra = data.get("emv_corridor_kind") == "contraflow"
        has_bus = bool(
            data.get("emv_corridor_kind") == "busway"
            or data.get("busway")
            or data.get("lanes:bus")
            or data.get("has_bus_lane")
        )

        cost = base if base > 0 else (length / 1000.0) / 25.0 * 3600.0

        if is_contra:
            if length > MAX_FIRETRUCK_CONTRAFLOW_M:
                # Too long for apparatus — treat as unavailable.
                cost = 1e12
                n_contra_blocked += 1
            else:
                cost = cost + CONTRAFLOW_RISK_EXTRA_S
                n_contra_kept += 1

        if hw in {"residential", "living_street", "service", "unclassified"}:
            cost *= RESIDENTIAL_PENALTY
            n_residential += 1
        elif hw in {"trunk", "trunk_link", "primary", "primary_link"} or has_bus:
            cost *= PRIMARY_BONUS

        # Cul-de-sac / dead-end tags when present.
        if data.get("noexit") in (True, "yes", "True", "1") or str(data.get("highway", "")).lower() == "turning_circle":
            cost *= DEADEND_PENALTY
            n_deadend += 1

        data[SECONDS_KEY] = float(cost)
        data[WEIGHT_KEY] = float(cost)

    stats = {
        "contraflow_blocked": n_contra_blocked,
        "contraflow_kept": n_contra_kept,
        "residential_penalized": n_residential,
        "deadend_penalized": n_deadend,
        "weight_key": WEIGHT_KEY,
        "max_contraflow_m": MAX_FIRETRUCK_CONTRAFLOW_M,
    }
    G.graph["firetruck_routing"] = stats
    return stats


def prepare_firetruck_graph(
    graph_path: Path | str,
    *,
    hour: int = 17,
    congestion: float | None = None,
    conditions: Any | None = None,
) -> nx.MultiDiGraph:
    """Load OSM drive graph with EMV privileges, then firetruck cost overlay."""
    G = prepare_routing_graph(
        graph_path, hour=hour, congestion=congestion, conditions=conditions
    )
    annotate_firetruck_costs(G)
    return G


def apply_firetruck_to_existing(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Annotate an already-prepared routing graph with firetruck weights."""
    annotate_firetruck_costs(G)
    return G


def firetruck_route(G, origin, dest) -> RouteResult:
    """Dijkstra on ``weight_firetruck``; falls back to ``weight_emv`` if missing."""
    weight = WEIGHT_KEY
    # Probe one edge
    sample = next(iter(G.edges(data=True)), None)
    if sample is None or WEIGHT_KEY not in sample[2]:
        weight = "weight_emv"
    return dijkstra_route(G, origin, dest, weight=weight, model_name="firetruck")


def firetruck_route_latlon(G, lon1: float, lat1: float, lon2: float, lat2: float) -> RouteResult:
    o = nearest_node(G, lon1, lat1)
    d = nearest_node(G, lon2, lat2)
    r = firetruck_route(G, o, d)
    r.meta["origin_node"] = o
    r.meta["dest_node"] = d
    return r


def path_firetruck_stats(G, path) -> tuple[float, float, int]:
    """(seconds, metres, n_edges) along a node path using firetruck costs."""
    if len(path) < 2:
        return float("nan"), float("nan"), 0
    # Prefer firetruck_s when present
    total_t = 0.0
    total_d = 0.0
    n = 0
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            return float("nan"), float("nan"), 0
        edge = min(
            edata.values(),
            key=lambda d: d.get(SECONDS_KEY, d.get("emv_s", d.get("travel_time", 1e18))),
        )
        total_t += float(edge.get(SECONDS_KEY) or edge.get("emv_s") or edge.get("travel_time") or 0.0)
        total_d += float(edge.get("length_m") or edge.get("length") or 0.0)
        n += 1
    return total_t, total_d, n


# Re-export helpers used by the planner
__all__ = [
    "WEIGHT_KEY",
    "SECONDS_KEY",
    "prepare_firetruck_graph",
    "apply_firetruck_to_existing",
    "annotate_firetruck_costs",
    "firetruck_route",
    "firetruck_route_latlon",
    "path_firetruck_stats",
    "nearest_node",
    "RouteResult",
]
