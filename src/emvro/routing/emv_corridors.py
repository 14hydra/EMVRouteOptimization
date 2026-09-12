"""EMV-only road privileges that civilian GPS (Google Maps) will not use.

Public OSM drive graphs rarely tag NYC bus lanes. To demonstrate the ASI
story — EMVs using special right-of-way that civilian routing deems
untakeable — we annotate:

1. Real ``highway=busway`` / bus-lane tags when present (civilian blocked).
2. **Emergency contraflow**: reverse edges on primary/trunk/secondary one-ways.
   Google Maps respects one-ways; EMVs with lights/sirens may use the reverse
   direction. Those reverse edges exist only for EMV costs (civilian weight
   is infinite).
"""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx


_PRIORITY_HW = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}

CIVILIAN_BLOCKED = 1e12  # effectively untakeable for civilian Dijkstra


def _highway(data: dict) -> str:
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    return str(hw or "").lower()


def _is_oneway(data: dict) -> bool:
    ow = data.get("oneway")
    if ow is True or ow in ("yes", "true", "1", "-1"):
        return True
    return False


def _has_bus_privilege(data: dict) -> bool:
    hw = _highway(data)
    if hw in {"busway", "bus_guideway"}:
        return True
    if data.get("busway") or data.get("lanes:bus"):
        return True
    if data.get("lanes:bus:forward") or data.get("lanes:bus:backward"):
        return True
    return False


def annotate_emv_road_privileges(G: nx.MultiDiGraph) -> dict[str, int]:
    """
    Mutate ``G`` in place:

    - Flag busway edges as ``civilian_forbidden`` / ``emv_corridor``.
    - Add reverse **EMV-only contraflow** edges on priority one-ways that have
      no existing reverse.

    Call this *before* attaching civilian_s / emv_s costs.
    """
    bus_n = 0
    for _, _, _, data in G.edges(keys=True, data=True):
        if _has_bus_privilege(data):
            data["civilian_forbidden"] = True
            data["emv_corridor"] = True
            data["emv_corridor_kind"] = "busway"
            bus_n += 1

    # Snapshot edges before we add contraflow (avoid iterating new edges)
    base_edges = list(G.edges(keys=True, data=True))
    contra_n = 0
    for u, v, k, data in base_edges:
        if not _is_oneway(data):
            continue
        hw = _highway(data)
        if hw not in _PRIORITY_HW:
            continue
        # Skip if any reverse edge already exists
        if G.has_edge(v, u):
            continue
        length = float(data.get("length") or data.get("length_m") or 0.0)
        rev = {
            "length": length,
            "highway": data.get("highway"),
            "name": data.get("name"),
            "speed_kph": data.get("speed_kph"),
            "travel_time": data.get("travel_time"),
            "oneway": False,
            "civilian_forbidden": True,
            "emv_corridor": True,
            "emv_corridor_kind": "contraflow",
            "emv_contraflow_of": f"{u}->{v}:{k}",
        }
        G.add_edge(v, u, **rev)
        contra_n += 1

    G.graph["emv_privileges"] = {
        "busway_edges": bus_n,
        "contraflow_edges_added": contra_n,
    }
    return G.graph["emv_privileges"]


def path_corridor_segments(G, path: Sequence) -> list[dict[str, Any]]:
    """Return lat/lon segments along ``path`` that use EMV-only corridors."""
    segs: list[dict[str, Any]] = []
    if len(path) < 2:
        return segs
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            continue
        # Prefer the EMV-tagged parallel edge if several exist
        edge = min(
            edata.values(),
            key=lambda d: (
                0 if d.get("emv_corridor") else 1,
                float(d.get("emv_s") or d.get("travel_time") or 1e18),
            ),
        )
        if not edge.get("emv_corridor") and not edge.get("civilian_forbidden"):
            continue
        u_d, v_d = G.nodes[u], G.nodes[v]
        segs.append(
            {
                "kind": edge.get("emv_corridor_kind") or "emv_only",
                "coords": [
                    [float(u_d.get("y", u_d.get("lat"))), float(u_d.get("x", u_d.get("lon")))],
                    [float(v_d.get("y", v_d.get("lat"))), float(v_d.get("x", v_d.get("lon")))],
                ],
            }
        )
    return segs


def path_uses_emv_corridor(G, path: Sequence) -> bool:
    return bool(path_corridor_segments(G, path))


def count_corridor_edges(G, path: Sequence) -> int:
    return len(path_corridor_segments(G, path))
