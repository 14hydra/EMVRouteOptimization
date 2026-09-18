"""Shared graph helpers and route result types for EMV routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

from ..street_features import load_graph, _try_import_ox
from .emv_corridors import annotate_emv_road_privileges, CIVILIAN_BLOCKED


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
        # Google Maps control stores a lat/lon polyline, not OSM nodes.
        if self.model == "control_google_maps":
            return (
                self.travel_seconds == self.travel_seconds
                and float(self.travel_seconds) > 0
            )
        return bool(self.node_path) and len(self.node_path) >= 2


def apply_routing_conditions(
    G: nx.MultiDiGraph,
    *,
    hour: int = 12,
    congestion: float | None = None,
    conditions: Any | None = None,
) -> nx.MultiDiGraph:
    """Recompute civilian_s / emv_s on an already-loaded privileged graph."""
    wx_civ = 1.0
    wx_emv = 1.0
    row_bonus = 1.0
    cond_meta: dict[str, Any] = {}

    if conditions is not None:
        hour = int(getattr(conditions, "hour", hour))
        congestion = float(getattr(conditions, "congestion"))
        wx_civ = float(conditions.weather_factor_civilian())
        wx_emv = float(conditions.weather_factor_emv())
        row_bonus = float(conditions.row_bonus())
        cond_meta = conditions.to_meta() if hasattr(conditions, "to_meta") else {}
    elif congestion is None:
        from .conditions import congestion_for_hour

        congestion = congestion_for_hour(hour)

    congestion = float(congestion)

    for u, v, k, data in G.edges(keys=True, data=True):
        base = float(data.get("travel_time") or 0.0)
        length = float(data.get("length") or data.get("length_m") or 0.0)
        if base <= 0 and length > 0:
            speed = float(data.get("speed_kph") or 30.0)
            base = (length / 1000.0) / max(speed, 5.0) * 3600.0
        hw = data.get("highway")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
        hw = str(hw or "").lower()

        emv_factor = 0.85
        if hw in {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}:
            emv_factor = 0.78
        elif hw in {"secondary", "secondary_link"}:
            emv_factor = 0.82
        elif hw in {"residential", "living_street", "unclassified"}:
            emv_factor = 0.92

        has_bus = bool(
            data.get("emv_corridor_kind") == "busway"
            or data.get("busway")
            or data.get("lanes:bus")
        )
        is_contra = data.get("emv_corridor_kind") == "contraflow"
        civilian_forbidden = bool(data.get("civilian_forbidden")) or has_bus or is_contra

        # Bus lanes: real EMV privilege — mild speedup vs mixed traffic.
        if has_bus:
            emv_factor *= 0.88

        # Contraflow: tactical, risky — PENALTY vs legal EMV travel so routers
        # only take it when the path save is clearly worth it (not free speedup).
        # ~+25% time + fixed intersection/clearance risk per edge.
        contraflow_risk_s = 0.0
        if is_contra:
            emv_factor *= 1.25
            contraflow_risk_s = 8.0  # seconds: yield, verify clear, re-enter

        # Congestion-scaled ROW bonus applies to legal priority roads + busways.
        # Do NOT apply it to contraflow (that would erase the risk penalty).
        if (
            hw
            in {
                "motorway",
                "motorway_link",
                "trunk",
                "trunk_link",
                "primary",
                "primary_link",
                "secondary",
                "secondary_link",
            }
            or has_bus
        ) and not is_contra:
            emv_factor *= row_bonus

        edge_wx_civ = wx_civ
        edge_wx_emv = wx_emv
        # Milder weather on wide arterials / busways. Contraflow is *worse* in
        # bad weather (harder to see oncoming traffic) — keep full EMV wx hit.
        if wx_civ > 1.01 and (
            hw in {"trunk", "trunk_link", "primary", "primary_link"} or has_bus
        ) and not is_contra:
            edge_wx_emv = 1.0 + 0.30 * (wx_civ - 1.0)
        elif is_contra and wx_civ > 1.01:
            edge_wx_emv = 1.0 + 0.70 * (wx_civ - 1.0)

        if civilian_forbidden:
            data["civilian_s"] = CIVILIAN_BLOCKED
        else:
            data["civilian_s"] = base * congestion * edge_wx_civ
        data["emv_s"] = base * congestion * edge_wx_emv * emv_factor + contraflow_risk_s
        data["length_m"] = length
        data["weight_emv"] = data["emv_s"]
        data["weight_length"] = length if length > 0 else 1.0
        data["has_bus_lane"] = int(has_bus)
        data["civilian_forbidden"] = civilian_forbidden
        if civilian_forbidden and not data.get("emv_corridor"):
            data["emv_corridor"] = True
            data["emv_corridor_kind"] = data.get("emv_corridor_kind") or "restricted"

    G.graph["routing_hour"] = hour
    G.graph["routing_congestion"] = congestion
    G.graph["routing_conditions"] = cond_meta
    return G


def prepare_routing_graph(
    graph_path: Path | str,
    *,
    hour: int = 12,
    congestion: float | None = None,
    conditions: Any | None = None,
) -> nx.MultiDiGraph:
    """
    Load OSM graph and attach civilian + EMV edge costs.

    When ``conditions`` (a ``RoutingConditions``) is provided, weather and a
    congestion-scaled right-of-way bonus are applied so EMV advantage grows
    under rush / rain / snow — the ASI evaluation story.
    """
    ox = _try_import_ox()
    G = load_graph(graph_path)
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:  # noqa: BLE001
        pass

    # Busways + short urban contraflow (Google Maps won't use these).
    # Contraflow is length-capped and cost-penalized — see emv_corridors.py.
    privilege_stats = annotate_emv_road_privileges(G)
    apply_routing_conditions(
        G, hour=hour, congestion=congestion, conditions=conditions
    )
    G.graph["emv_privileges"] = privilege_stats
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
    """Civilian-style control: minimize congested travel_time (OSM prior)."""
    # Temporarily expose civilian weight
    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight_civilian"] = float(data.get("civilian_s") or data.get("travel_time") or 1.0)
    return dijkstra_route(G, origin, dest, weight="weight_civilian", model_name="control_civilian_time")


def control_google_maps(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    departure_time: str | int | None = "now",
    cache_path: Path | str | None = None,
    include_polyline: bool = True,
) -> RouteResult:
    """
    Primary civilian control: Google Maps Directions (traffic-aware when available).

    This is the phone-GPS baseline used against EMV routers on the slides.
    """
    from ..gmaps import GoogleMapsControl

    gmaps = GoogleMapsControl(cache_path=cache_path)
    if not gmaps.available:
        return RouteResult(
            "control_google_maps",
            [],
            np.nan,
            np.nan,
            0,
            meta={"error": "missing_api_key"},
        )
    r = gmaps.driving_seconds(
        origin_lat,
        origin_lon,
        dest_lat,
        dest_lon,
        departure_time=departure_time,
        include_polyline=include_polyline,
    )
    if not r.get("ok"):
        return RouteResult(
            "control_google_maps",
            [],
            np.nan,
            np.nan,
            0,
            meta={k: r.get(k) for k in ("error", "status", "message")},
        )
    secs = r.get("duration_in_traffic_s") or r.get("duration_s")
    poly = r.get("polyline_latlons") or []
    return RouteResult(
        "control_google_maps",
        [],  # Google path is continuous lat/lon, not OSM nodes
        float(secs),
        float(r.get("distance_m") or np.nan),
        max(0, len(poly) - 1),
        meta={
            "duration_s": r.get("duration_s"),
            "duration_in_traffic_s": r.get("duration_in_traffic_s"),
            "polyline_latlons": poly,
            "cached": r.get("cached", False),
            "source": "google_maps_directions",
        },
    )


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
