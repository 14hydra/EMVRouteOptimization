"""EMV-only road privileges that civilian GPS (Google Maps) will not use.

Public OSM drive graphs rarely tag NYC bus lanes. To demonstrate the ASI
story — EMVs using special right-of-way that civilian routing deems
untakeable — we annotate:

1. Real ``highway=busway`` / bus-lane tags when present (civilian blocked).
2. **Emergency contraflow**: short reverse edges on urban arterial one-ways.
   Google Maps respects one-ways; EMVs with lights/sirens may briefly use the
   reverse direction for tactical shortcuts — not free-flow wrong-way driving.

Realism constraints (tuned to NYC/SF EMS practice):
- No motorways / freeway ramps (wrong-way there is not normal ops).
- Prefer primary/trunk; secondary only for short blocks.
- Cap contraflow segment length (~120 m) so paths cannot chain long reverse runs.
- Contraflow is *costlier* than legal EMV travel (see ``graph.py``) so routers
  only take it when the time save clearly justifies the risk.
"""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx


# Freeways / ramps are excluded — wrong-way there is not realistic EMS practice.
_CONTRAFLOW_HW_ALWAYS = {
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
}
# Secondary one-ways only if the block is short (grid tactical move).
_CONTRAFLOW_HW_SHORT_ONLY = {
    "secondary",
    "secondary_link",
}

# ~120 m ≈ one NYC midtown short block; longer reverse runs are unrealistic.
MAX_CONTRAFLOW_LENGTH_M = 120.0
# Secondary contraflow only for even shorter blocks.
MAX_SECONDARY_CONTRAFLOW_LENGTH_M = 80.0

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


def _edge_length_m(data: dict) -> float:
    return float(data.get("length") or data.get("length_m") or 0.0)


def _contraflow_allowed(hw: str, length_m: float) -> bool:
    """Return True if this one-way edge may get an EMV-only reverse."""
    if length_m <= 0 or length_m > MAX_CONTRAFLOW_LENGTH_M:
        return False
    if hw in _CONTRAFLOW_HW_ALWAYS:
        return True
    if hw in _CONTRAFLOW_HW_SHORT_ONLY and length_m <= MAX_SECONDARY_CONTRAFLOW_LENGTH_M:
        return True
    return False


def annotate_emv_road_privileges(G: nx.MultiDiGraph) -> dict[str, int]:
    """
    Mutate ``G`` in place:

    - Flag busway edges as ``civilian_forbidden`` / ``emv_corridor``.
    - Add reverse **EMV-only contraflow** edges on short urban arterial
      one-ways that have no existing reverse (no motorways).

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
    skipped_motorway = 0
    skipped_long = 0
    for u, v, k, data in base_edges:
        if not _is_oneway(data):
            continue
        hw = _highway(data)
        length = _edge_length_m(data)
        if hw in {"motorway", "motorway_link"}:
            skipped_motorway += 1
            continue
        if not _contraflow_allowed(hw, length):
            if hw in _CONTRAFLOW_HW_ALWAYS | _CONTRAFLOW_HW_SHORT_ONLY:
                skipped_long += 1
            continue
        # Skip if any reverse edge already exists
        if G.has_edge(v, u):
            continue
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
            # Risk metadata for cost model / routers
            "emv_contraflow_risk": True,
            "emv_max_contraflow_m": MAX_CONTRAFLOW_LENGTH_M,
        }
        G.add_edge(v, u, **rev)
        contra_n += 1

    G.graph["emv_privileges"] = {
        "busway_edges": bus_n,
        "contraflow_edges_added": contra_n,
        "contraflow_skipped_motorway_candidates": skipped_motorway,
        "contraflow_skipped_too_long": skipped_long,
        "contraflow_max_length_m": MAX_CONTRAFLOW_LENGTH_M,
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


def path_contraflow_length_m(G, path: Sequence) -> float:
    """Total meters of contraflow used along a node path."""
    total = 0.0
    if len(path) < 2:
        return total
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            continue
        edge = min(
            edata.values(),
            key=lambda d: (
                0 if d.get("emv_corridor_kind") == "contraflow" else 1,
                float(d.get("emv_s") or 1e18),
            ),
        )
        if edge.get("emv_corridor_kind") == "contraflow":
            total += float(edge.get("length") or edge.get("length_m") or 0.0)
    return total
