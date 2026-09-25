#!/usr/bin/env python3
"""
Build optimal EMV patrol posts + patrol loops from historical incident density.

Pairs with the inferred starting-location pipeline (`docs/starting_locations.md`):
`build_od_pairs.py` infers where a unit *probably started*; this script chooses
where units *should* be posted so the next incident is closer.

Objective (see also summary.json):
    demand  w_c   = historical FDNY incidents in grid cell c
    cost    t(p,c)= EMV travel seconds post p -> cell c
    minimize  E[T] = Σ w_c · min_p t(p,c) / Σ w_c          (--objective response_time)
    maximize  C(T) = Σ w_c · 1[min_p t(p,c) ≤ T] / Σ w_c   (--objective coverage)

Candidate posts come from the same inferred layers used for starts (FDNY firehouses,
synthetic alarm-box CSLs) plus high-demand cell anchors. Home bases
("anchors") are restricted to real facilities; each unit patrols a short loop
anchor -> posts -> anchor.

Outputs (default `data/figures/patrol/`):
    patrol_posts.csv         selected posts + per-post demand / response metrics
    patrol_segments.csv      patrol loop legs with travel time, distance, geometry
    patrol_demand_cells.csv  demand grid with response time + coverage flag
    patrol_map.html          interactive Folium map
    summary.json             objective, parameters, metrics, baseline comparison

Usage:
  PYTHONPATH=src python scripts/build_patrol_routes.py
  PYTHONPATH=src python scripts/build_patrol_routes.py --posts 30 --units 8 --objective coverage
  PYTHONPATH=src python scripts/build_patrol_routes.py --no-graph      # surrogate only (fast)
  PYTHONPATH=src python scripts/build_patrol_routes.py --demo          # synthetic data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.patrol import (  # noqa: E402
    TravelTimeSurrogate,
    add_demand_cell_candidates,
    build_candidate_posts,
    build_demand_grid,
    build_unit_plans,
    graph_nearest_nodes,
    graph_segment,
    graph_time_matrix,
    haversine_km,
    posture_metrics,
    screen_candidates,
    select_posts,
    select_unit_anchors,
    synthetic_demo_data,
)

UNIT_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#008080",
    "#9a6324",
    "#800000",
    "#808000",
    "#000075",
    "#f032e6",
    "#469990",
]

LAYER_LABELS = {
    "fdny_firehouse": "FDNY firehouse",
    "csl": "Synthetic CSL (alarm-box intersection)",
    "demand_cell": "Demand-cell anchor",
    "ems_station": "EMS station (legacy)",
    "hospital_bay": "Hospital bay (legacy)",
}


def _maybe_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None


def _unit_color(unit_id: int) -> str:
    return UNIT_COLORS[(unit_id - 1) % len(UNIT_COLORS)]


def _response_color(seconds: float, threshold_s: float) -> str:
    r = seconds / max(threshold_s, 1.0)
    if r <= 0.5:
        return "#1a9850"
    if r <= 0.75:
        return "#91cf60"
    if r <= 1.0:
        return "#fee08b"
    if r <= 1.5:
        return "#fc8d59"
    return "#d73027"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_inputs(args) -> dict:
    """Load FDNY incidents + firehouse/CSL candidate layers, or synthesize a demo."""
    if args.demo:
        incidents, stations, csls = synthetic_demo_data()
        return {
            "mode": "demo_synthetic",
            "incidents": incidents,
            "layers": [(stations, "fdny_firehouse"), (csls, "csl")],
            "od": incidents,
        }

    od_path = args.od
    od = _maybe_csv(od_path)
    if od is None or "dest_lat" not in od.columns:
        print(f"! No usable OD file at {od_path}; falling back to --demo data")
        incidents, stations, csls = synthetic_demo_data()
        return {
            "mode": "demo_synthetic_fallback",
            "incidents": incidents,
            "layers": [(stations, "fdny_firehouse"), (csls, "csl")],
            "od": incidents,
        }

    firehouses = _maybe_csv(args.raw / "fdny_firehouses.csv")
    csls = _maybe_csv(args.processed / "synthetic_csls.csv")
    layers = [(firehouses, "fdny_firehouse"), (csls, "csl")]

    if getattr(args, "legacy_ems", False):
        stations = _maybe_csv(args.raw / "ems_stations.csv")
        hospitals = _maybe_csv(args.raw / "hospital_bays.csv")
        if hospitals is not None:
            from emvro.depots import filter_hospital_bays

            hospitals = filter_hospital_bays(hospitals)
        layers.extend([(stations, "ems_station"), (hospitals, "hospital_bay")])

    if all(df is None or not len(df) for df, _ in layers):
        # Degrade to the depots already embedded in the OD table.
        cols = ["depot_name", "depot_layer", "start_lat", "start_lon", "borough"]
        if all(c in od.columns for c in cols):
            uniq = od[cols].dropna(subset=["start_lat", "start_lon"]).drop_duplicates()
            layers = [
                (g.rename(columns={"depot_name": "facname"}), str(lyr))
                for lyr, g in uniq.groupby("depot_layer")
            ]
            print(f"! Facility CSVs missing; recovered {len(uniq)} depots from OD table")
        else:
            raise SystemExit("No candidate post layers available")

    return {"mode": "nyc_firetrucks", "incidents": od, "layers": layers, "od": od}


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


def build_plan(args) -> dict:
    data = load_inputs(args)
    incidents = data["incidents"]

    surrogate = TravelTimeSurrogate.calibrate(
        data["od"],
        detour_factor=args.detour_factor,
        default_speed_kph=args.speed_kph,
        enabled=args.calibrate_speed,
    )
    print(
        f"Travel-time surrogate: {surrogate.speed_kph:.1f} kph effective "
        f"(x{surrogate.detour_factor} detour, source={surrogate.meta.get('speed_source')})"
    )

    cells = build_demand_grid(
        incidents,
        cell_km=args.cell_km,
        severity_col=args.severity_col if args.severity_weighted else None,
        min_incidents=args.min_cell_incidents,
    )
    weights = cells["weight"].to_numpy(dtype=float)
    print(f"Demand grid: {len(cells)} cells at {args.cell_km} km, {weights.sum():.0f} weighted incidents")

    candidates = build_candidate_posts(data["layers"])
    candidates = add_demand_cell_candidates(candidates, cells, top_n=args.demand_candidates)
    T_sur = surrogate.matrix(candidates, cells)
    candidates, T_sur = screen_candidates(
        candidates, T_sur, weights, max_candidates=args.max_candidates
    )
    print(
        f"Candidate posts: {len(candidates)} "
        f"({candidates['layer'].value_counts().to_dict()})"
    )

    threshold_s = args.threshold_min * 60.0
    sel = select_posts(
        T_sur,
        weights,
        args.posts,
        objective=args.objective,
        threshold_s=threshold_s,
        max_swap_rounds=args.swap_rounds,
    )
    print(
        f"Selected {len(sel.indices)} posts (objective={args.objective}, "
        f"greedy={sel.greedy_value:.3f} -> swapped={sel.value:.3f} in {sel.n_swaps} swaps)"
    )

    posts = candidates.iloc[sel.indices].reset_index(drop=True)
    posts.insert(0, "post_id", np.arange(1, len(posts) + 1))

    # --- postures to score side by side (all on the SAME travel-time model) --
    postures = [{"label": "patrol_plan", "rows": list(sel.indices), "role": "plan"}]
    anchor_rows = np.flatnonzero(candidates["is_anchor"].to_numpy())
    if len(anchor_rows):
        # Same number of units, but held at existing facilities: the honest
        # "what do we give up by not going on-road?" comparison.
        k_station = select_posts(
            T_sur,
            weights,
            args.posts,
            objective=args.objective,
            threshold_s=threshold_s,
            allowed=[int(i) for i in anchor_rows],
            max_swap_rounds=args.swap_rounds,
        )
        postures.append(
            {
                "label": f"best_{args.posts}_facilities_only",
                "rows": list(k_station.indices),
                "role": "baseline",
                "equal_cost": True,
                "note": (
                    "Equal-cost comparison: same number of units, but restricted to existing "
                    "facilities (no on-road CSL-style posts)."
                ),
            }
        )
        postures.append(
            {
                "label": "all_facilities_stationary",
                "rows": [int(i) for i in anchor_rows],
                "role": "baseline",
                "equal_cost": False,
                "note": (
                    "Context only, not an equal-cost comparison: assumes a staffed unit sitting "
                    "at every FDNY firehouse (and legacy EMS layers if --legacy-ems)."
                ),
            }
        )

    return {
        "data_mode": data["mode"],
        "surrogate": surrogate,
        "cells": cells,
        "weights": weights,
        "candidates": candidates,
        "T_sur": T_sur,
        "T_final": T_sur,  # replaced by graph times when refinement succeeds
        "row_index": {int(r): int(r) for r in range(len(candidates))},
        "graph_resolved": None,
        "selection": sel,
        "posts": posts,
        "postures": postures,
        "threshold_s": threshold_s,
        "incidents_n": int(len(incidents)),
        "reselected_on_graph": False,
    }


def _rebuild_posts(plan: dict, rows: list[int]) -> pd.DataFrame:
    """Refresh the post table (and post ids) after a re-selection."""
    posts = plan["candidates"].iloc[rows].reset_index(drop=True)
    posts.insert(0, "post_id", np.arange(1, len(posts) + 1))
    plan["posts"] = posts
    plan["postures"][0]["rows"] = list(rows)
    return posts


def build_units(plan: dict, args) -> list:
    """Pick home-base anchors and order each unit's patrol loop.

    Anchors and loop order use the surrogate: loop *ordering* only needs relative
    proximity, and the exact leg times are recomputed on the graph afterwards.
    """
    posts = plan["posts"]
    anchor_sel = select_unit_anchors(
        plan["candidates"],
        posts,
        posts["assigned_weight"].to_numpy(dtype=float),
        args.units,
        plan["surrogate"],
    )
    plans = build_unit_plans(
        plan["candidates"], posts, anchor_sel, plan["surrogate"], balance=args.balance
    )
    print(f"Units: {len(plans)} home bases -> " + ", ".join(p.anchor_name or "?" for p in plans))
    plan["plans"] = plans
    return plans


# --------------------------------------------------------------------------- #
# Optional OSM graph refinement (exact EMV times + real loop geometry)
# --------------------------------------------------------------------------- #


def refine_with_graph(plan: dict, args) -> dict:
    """Recompute selected-post response times and loop legs on the OSM graph."""
    info = {"used": False, "reason": None, "graph": None}
    if args.no_graph:
        info["reason"] = "disabled (--no-graph)"
        return info
    graph_path = args.graph
    if not graph_path or not Path(graph_path).exists():
        info["reason"] = f"graph not found: {graph_path}"
        print(f"! {info['reason']} — using crow-flies surrogate only")
        return info

    try:
        from emvro.routing import prepare_routing_graph
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"routing import failed: {exc}"
        print(f"! {info['reason']}")
        return info

    print(f"Loading OSM graph {Path(graph_path).name} (hour={args.hour})…")
    try:
        G = prepare_routing_graph(graph_path, hour=args.hour)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"graph load failed: {exc}"
        print(f"! {info['reason']} — using crow-flies surrogate only")
        return info

    cells = plan["cells"]
    candidates = plan["candidates"]
    weights = plan["weights"]
    T_sur = plan["T_sur"]
    print(f"Graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges; snapping points…")

    # Sources = every posture's locations plus the best surrogate-screened
    # candidates, so the graph-time pass can also *re-pick* posts instead of only
    # re-scoring the surrogate's choice.
    union = {int(r) for p in plan["postures"] for r in p["rows"]}
    if args.graph_candidates > 0:
        score = (T_sur * weights[None, :]).sum(axis=1)
        union |= {int(i) for i in np.argsort(score)[: args.graph_candidates]}
    union_rows = sorted(union)
    row_index = {r: i for i, r in enumerate(union_rows)}
    src_nodes = graph_nearest_nodes(
        G, candidates.iloc[union_rows]["lon"], candidates.iloc[union_rows]["lat"]
    )
    cell_nodes = graph_nearest_nodes(G, cells["lon"], cells["lat"])

    cutoff = float(args.threshold_min * 60.0 * args.cutoff_factor)
    print(f"Dijkstra from {len(src_nodes)} candidate locations (cutoff {cutoff/60:.0f} min)…")
    T_graph, resolved = graph_time_matrix(
        G,
        src_nodes,
        cell_nodes,
        cutoff_s=cutoff,
        fallback=T_sur[union_rows, :],
        progress_every=25,
    )

    plan["T_final"] = T_graph
    plan["row_index"] = row_index
    plan["graph_resolved"] = resolved

    # ---- stage 2 of the optimizer: re-select posts on exact EMV times -------
    threshold_s = plan["threshold_s"]
    anchor_union = [row_index[r] for r in union_rows if bool(candidates.iloc[r]["is_anchor"])]
    sel2 = select_posts(
        T_graph,
        weights,
        args.posts,
        objective=args.objective,
        threshold_s=threshold_s,
        max_swap_rounds=args.swap_rounds,
    )
    back = {i: r for r, i in row_index.items()}
    _rebuild_posts(plan, [back[i] for i in sel2.indices])
    plan["selection_graph"] = sel2
    plan["reselected_on_graph"] = True
    print(
        f"Re-selected posts on graph times (objective {sel2.greedy_value:.3f} -> "
        f"{sel2.value:.3f} in {sel2.n_swaps} swaps)"
    )
    if anchor_union:
        k_station2 = select_posts(
            T_graph,
            weights,
            args.posts,
            objective=args.objective,
            threshold_s=threshold_s,
            allowed=anchor_union,
            max_swap_rounds=args.swap_rounds,
        )
        for posture in plan["postures"]:
            if posture.get("equal_cost"):
                posture["rows"] = [back[i] for i in k_station2.indices]

    plan["graph_nodes"] = {
        int(candidates.iloc[r]["candidate_id"]): src_nodes[i] for r, i in row_index.items()
    }
    plan["G"] = G
    plan_rows = [row_index[int(r)] for r in plan["postures"][0]["rows"]]
    cells_on_graph = float(resolved[plan_rows, :].any(axis=0).mean())
    info.update(
        used=True,
        graph=str(graph_path),
        hour=args.hour,
        cutoff_s=cutoff,
        sources=len(src_nodes),
        pairs_resolved_share=round(float(resolved.mean()), 4),
        plan_cells_with_graph_time_share=round(cells_on_graph, 4),
        nodes=int(G.number_of_nodes()),
        edges=int(G.number_of_edges()),
    )
    print(
        f"Graph times: {resolved.mean()*100:.1f}% of source→cell pairs within cutoff; "
        f"{cells_on_graph*100:.1f}% of demand cells reached by a patrol post"
    )
    return info


def _headline(plan_metrics: dict, baselines: list) -> dict:
    """Patrol posture vs the same number of units held at existing facilities."""
    base = next((b for b in baselines if b.get("is_equal_cost_baseline")), None)
    if not base:
        return {}
    e_plan = plan_metrics["expected_response_s"]
    e_base = base["expected_response_s"]
    return {
        "baseline": base["label"],
        "expected_response_s_baseline": e_base,
        "expected_response_s_patrol": e_plan,
        "expected_response_saved_s": round(e_base - e_plan, 1),
        "expected_response_pct_faster": round(100.0 * (e_base - e_plan) / max(e_base, 1e-9), 2),
        "coverage_share_baseline": base["coverage_share_within_threshold"],
        "coverage_share_patrol": plan_metrics["coverage_share_within_threshold"],
        "coverage_points_gained": round(
            100.0
            * (
                plan_metrics["coverage_share_within_threshold"]
                - base["coverage_share_within_threshold"]
            ),
            2,
        ),
    }


def build_loop_segments(plan: dict, graph_info: dict) -> pd.DataFrame:
    """One row per patrol leg (anchor→post→…→anchor) with time, distance, geometry."""
    posts = plan["posts"].set_index("post_id")
    surrogate = plan["surrogate"]
    G = plan.get("G")
    nodes = plan.get("graph_nodes") or {}  # candidate_id -> graph node
    rows = []

    for up in plan["plans"]:
        # (kind, candidate_id, name, lon, lat) per stop; the loop closes on the anchor.
        stops = [("anchor", up.anchor_candidate_id, up.anchor_name, up.anchor_lon, up.anchor_lat)]
        for pid in up.post_order:
            r = posts.loc[pid]
            stops.append(
                (
                    "post",
                    int(r["candidate_id"]),
                    str(r["post_name"]),
                    float(r["lon"]),
                    float(r["lat"]),
                )
            )
        stops.append(stops[0])

        loop_s = 0.0
        loop_km = 0.0
        leg_no = 0
        for i in range(len(stops) - 1):
            a, b = stops[i], stops[i + 1]
            # A home base is often itself a good post: skip the zero-length leg.
            if a[1] == b[1] or float(haversine_km(a[3], a[4], b[3], b[4])) < 0.025:
                continue
            seg = None
            if G is not None and nodes:
                na, nb = nodes.get(a[1]), nodes.get(b[1])
                if na is not None and nb is not None and na != nb:
                    s = graph_segment(G, na, nb)
                    if s["ok"]:
                        seg = {
                            "seconds": s["seconds"],
                            "km": s["meters"] / 1000.0,
                            "latlons": s["latlons"],
                            "source": "osm_graph_emv",
                        }
            if seg is None:
                km = float(haversine_km(a[3], a[4], b[3], b[4]) * surrogate.detour_factor)
                seg = {
                    "seconds": float(surrogate.seconds(a[3], a[4], b[3], b[4])),
                    "km": km,
                    "latlons": [[a[4], a[3]], [b[4], b[3]]],
                    "source": "haversine_surrogate",
                }
            loop_s += seg["seconds"]
            loop_km += seg["km"]
            leg_no += 1
            rows.append(
                {
                    "unit_id": up.unit_id,
                    "seq": leg_no,
                    "from_kind": a[0],
                    "from_name": a[2],
                    "from_lat": a[4],
                    "from_lon": a[3],
                    "to_kind": b[0],
                    "to_name": b[2],
                    "to_lat": b[4],
                    "to_lon": b[3],
                    "travel_seconds": round(seg["seconds"], 1),
                    "distance_km": round(seg["km"], 3),
                    "geometry_source": seg["source"],
                    "path_latlons": json.dumps([[round(y, 6), round(x, 6)] for y, x in seg["latlons"]]),
                }
            )
        up.loop_seconds = loop_s
        up.loop_km = loop_km

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #


def build_map(plan: dict, segments: pd.DataFrame, summary: dict, out: Path) -> Path:
    import folium
    from branca.element import Element
    from folium.plugins import HeatMap

    cells = plan["cells"]
    posts = plan["posts"]
    weights = plan["weights"]
    resp = plan["response_s"]
    threshold_s = plan["threshold_s"]

    center = [float(cells["lat"].mean()), float(cells["lon"].mean())]
    # Never use tiles.openstreetmap.org — OSM's volunteer CDN 403s Folium apps
    # (osm.wiki/Blocked). Match visualize_data.py: Esri default + Carto fallbacks.
    m = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri &copy; OpenStreetMap contributors",
        name="Esri streets",
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Carto light",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Carto voyager",
        show=False,
    ).add_to(m)

    # --- demand -----------------------------------------------------------
    fg_heat = folium.FeatureGroup(name="Incident demand (heat)", show=True)
    wmax = max(float(weights.max()), 1e-9)
    HeatMap(
        [[float(r["lat"]), float(r["lon"]), float(r["weight"]) / wmax] for _, r in cells.iterrows()],
        radius=16,
        blur=22,
        min_opacity=0.25,
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    fg_resp = folium.FeatureGroup(name="Demand cells · modelled response time", show=False)
    dlat = float(cells["cell_dlat"].iloc[0]) / 2.0
    dlon = float(cells["cell_dlon"].iloc[0]) / 2.0
    for i, r in cells.iterrows():
        secs = float(resp[i])
        pid = int(posts.iloc[int(plan["assignment"][i])]["post_id"])
        folium.Rectangle(
            bounds=[
                [float(r["lat"]) - dlat, float(r["lon"]) - dlon],
                [float(r["lat"]) + dlat, float(r["lon"]) + dlon],
            ],
            color=None,
            fill=True,
            fill_color=_response_color(secs, threshold_s),
            fill_opacity=0.55,
            weight=0,
            popup=(
                f"<b>Demand cell {int(r['cell_id'])}</b><br>"
                f"{int(r['n_incidents'])} incidents ({float(r['share'])*100:.2f}% of demand)<br>"
                f"nearest post: #{pid}<br>"
                f"modelled response: <b>{secs/60:.1f} min</b>"
            ),
            tooltip=f"{secs/60:.1f} min · {int(r['n_incidents'])} incidents",
        ).add_to(fg_resp)
    fg_resp.add_to(m)

    # --- candidates not chosen -------------------------------------------
    fg_cand = folium.FeatureGroup(name="Candidate posts (not selected)", show=False)
    chosen_ids = set(int(x) for x in posts["candidate_id"])
    for _, r in plan["candidates"].iterrows():
        if int(r["candidate_id"]) in chosen_ids:
            continue
        folium.CircleMarker(
            [float(r["lat"]), float(r["lon"])],
            radius=2.5,
            color="#888",
            fill=True,
            fill_opacity=0.5,
            tooltip=f"candidate · {LAYER_LABELS.get(r['layer'], r['layer'])}",
            popup=f"{r['post_name']}<br>{LAYER_LABELS.get(r['layer'], r['layer'])}",
        ).add_to(fg_cand)
    fg_cand.add_to(m)

    # --- loops, posts, anchors -------------------------------------------
    fg_loops = folium.FeatureGroup(name="Patrol loops", show=True)
    fg_posts = folium.FeatureGroup(name="Patrol posts", show=True)
    fg_anchor = folium.FeatureGroup(name="Home bases (inferred depots)", show=True)

    unit_by_post = {}
    for up in plan["plans"]:
        for order, pid in enumerate(up.post_order, start=1):
            unit_by_post[int(pid)] = (up.unit_id, order)

    for up in plan["plans"]:
        color = _unit_color(up.unit_id)
        legs = segments[segments["unit_id"] == up.unit_id]
        for _, leg in legs.iterrows():
            coords = json.loads(leg["path_latlons"])
            if len(coords) < 2:
                continue
            folium.PolyLine(
                coords,
                color=color,
                weight=4,
                opacity=0.8,
                dash_array=None if leg["geometry_source"] == "osm_graph_emv" else "6 8",
                popup=(
                    f"<b>Unit {up.unit_id} leg {int(leg['seq'])}</b><br>"
                    f"{leg['from_name']} → {leg['to_name']}<br>"
                    f"{float(leg['travel_seconds'])/60:.1f} min · {float(leg['distance_km']):.2f} km<br>"
                    f"<i>{leg['geometry_source']}</i>"
                ),
                tooltip=f"Unit {up.unit_id} leg {int(leg['seq'])} ({float(leg['travel_seconds'])/60:.1f} min)",
            ).add_to(fg_loops)

        folium.Marker(
            [up.anchor_lat, up.anchor_lon],
            icon=folium.Icon(color="darkblue", icon="home", prefix="glyphicon"),
            popup=(
                f"<b>Unit {up.unit_id} home base</b><br>{up.anchor_name}<br>"
                f"{LAYER_LABELS.get(up.anchor_layer, up.anchor_layer)}<br>"
                f"loop: {up.loop_seconds/60:.1f} min · {up.loop_km:.2f} km<br>"
                f"demand share: {up.demand_share*100:.1f}%"
            ),
            tooltip=f"Unit {up.unit_id} home base · {up.anchor_name}",
        ).add_to(fg_anchor)

    for _, r in posts.iterrows():
        pid = int(r["post_id"])
        unit_id, order = unit_by_post.get(pid, (0, 0))
        color = _unit_color(unit_id) if unit_id else "#333"
        share = float(r["demand_share"])
        folium.CircleMarker(
            [float(r["lat"]), float(r["lon"])],
            radius=5 + 14 * share,
            color="#111",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=(
                f"<b>Patrol post #{pid}</b> (unit {unit_id}, stop {order})<br>"
                f"{r['post_name']}<br>{LAYER_LABELS.get(r['layer'], r['layer'])}<br>"
                f"demand share: {share*100:.1f}%<br>"
                f"mean response to its cells: {float(r['mean_response_s'])/60:.1f} min<br>"
                f"cells served: {int(r['cells_assigned'])}"
            ),
            tooltip=f"Post #{pid} · unit {unit_id} · {share*100:.1f}% of demand",
        ).add_to(fg_posts)
        folium.Marker(
            [float(r["lat"]), float(r["lon"])],
            icon=folium.DivIcon(
                html=(
                    '<div style="font:10px/1 system-ui,sans-serif;font-weight:700;color:#111;'
                    'text-shadow:0 0 3px #fff,0 0 3px #fff;transform:translate(8px,-6px);">'
                    f"{pid}</div>"
                )
            ),
        ).add_to(fg_posts)

    fg_loops.add_to(m)
    fg_posts.add_to(m)
    fg_anchor.add_to(m)

    # --- panel ------------------------------------------------------------
    plan_m = summary["metrics"]["patrol_plan"]
    rows = [plan_m] + summary["metrics"]["baselines"]
    table = ""
    for row in rows:
        gain = ""
        if row is not plan_m and row.get("expected_response_s"):
            # negative = patrol posture is faster than this baseline
            delta = (plan_m["expected_response_s"] - row["expected_response_s"]) / row[
                "expected_response_s"
            ]
            color = "#1a7f37" if delta < 0 else "#a40e26"
            gain = f" <span style='color:{color};'>(patrol {delta*100:+.1f}%)</span>"
        bold = "font-weight:700;" if row is plan_m else ""
        table += (
            f"<tr style='{bold}'><td style='padding:1px 6px 1px 0;'>{row['label']}</td>"
            f"<td style='text-align:right;padding:1px 6px;'>{row['n_posts']}</td>"
            f"<td style='text-align:right;padding:1px 6px;'>{row['expected_response_s']/60:.2f}</td>"
            f"<td style='text-align:right;padding:1px 6px;'>{row['p90_response_s']/60:.2f}</td>"
            f"<td style='text-align:right;padding:1px 0;'>"
            f"{row['coverage_share_within_threshold']*100:.1f}%{gain}</td></tr>"
        )

    unit_rows = "".join(
        f"<div><span style='color:{_unit_color(up.unit_id)};font-weight:700;'>━━</span> "
        f"Unit {up.unit_id}: {len(up.post_order)} posts · loop {up.loop_seconds/60:.0f} min · "
        f"{up.demand_share*100:.0f}% demand<br>"
        f"<span style='color:#555;font-size:11px;'>base: {up.anchor_name}</span></div>"
        for up in plan["plans"]
    )

    panel = f"""
    <div id="patrol-panel" style="position:fixed;top:14px;left:14px;z-index:9999;
         background:rgba(255,255,255,0.97);padding:12px 14px;border:1px solid #888;
         border-radius:10px;font:13px/1.45 system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.18);
         width:400px;max-width:92vw;max-height:92vh;overflow:auto;">
      <div style="font-weight:700;font-size:14px;">EMV patrol posts &amp; loops</div>
      <div style="color:#555;font-size:12px;margin:4px 0 8px;">
        {summary['objective']['plain_english']}
      </div>
      <div style="background:#f0f4f8;border:1px solid #d0d7de;border-radius:6px;padding:8px 10px;
           font-size:12px;margin-bottom:8px;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="color:#555;font-size:11px;text-align:left;">
            <th>posture</th><th style="text-align:right;">posts</th>
            <th style="text-align:right;">E[T] min</th><th style="text-align:right;">p90 min</th>
            <th style="text-align:right;">≤{summary['parameters']['threshold_min']} min</th>
          </tr>
          {table}
        </table>
      </div>
      <div style="font-size:12px;margin-bottom:8px;">{unit_rows}</div>
      <div style="font-size:11px;color:#555;">
        Travel time: {summary['travel_time']['description']}<br>
        Demand: {summary['demand']['incidents']} incidents → {summary['demand']['cells']} cells
        ({summary['parameters']['cell_km']} km grid)<br>
        Data: {summary['data_mode']} · travel-only (no call-processing / chute time)
      </div>
      <div style="font-size:11px;color:#555;margin-top:6px;">
        Toggle layers (top right) for the response-time surface and unused candidates.
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(panel))
    folium.LayerControl(collapsed=True).add_to(m)

    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--od", type=Path, default=ROOT / "data" / "processed" / "od_pairs_inferred_starts.csv")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "patrol")
    p.add_argument("--posts", type=int, default=24, help="Number of patrol posts to place")
    p.add_argument("--units", type=int, default=6, help="Number of patrol units (one loop each)")
    p.add_argument("--cell-km", type=float, default=0.75, help="Demand grid cell size (km)")
    p.add_argument("--min-cell-incidents", type=int, default=2)
    p.add_argument(
        "--objective",
        choices=["response_time", "coverage"],
        default="response_time",
        help="response_time = p-median (minimize expected travel); coverage = max demand within threshold",
    )
    p.add_argument("--threshold-min", type=float, default=8.0, help="Coverage threshold (minutes)")
    p.add_argument("--severity-weighted", action="store_true", help="Up-weight acute calls in demand")
    p.add_argument("--severity-col", default="initial_severity_level_code")
    p.add_argument("--demand-candidates", type=int, default=60, help="Top demand cells added as candidates")
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--swap-rounds", type=int, default=8)
    p.add_argument("--balance", type=float, default=1.35, help="Per-unit demand capacity slack")
    p.add_argument("--speed-kph", type=float, default=28.0, help="Prior effective EMV speed (door to scene)")
    p.add_argument("--detour-factor", type=float, default=1.30)
    p.add_argument(
        "--calibrate-speed",
        action="store_true",
        help="Try fitting effective speed from OD medians (usually rejected: inferred starts are too close)",
    )
    # Prefer the rich GraphML (bus-lane / contraflow tags) so EMV corridor edge
    # costs actually fire; fall back to the plain drive graph if rich is absent.
    _rich = ROOT / "data" / "raw" / "nyc_drive_rich.graphml"
    _plain = ROOT / "data" / "raw" / "nyc_drive.graphml"
    p.add_argument("--graph", type=Path, default=_rich if _rich.exists() else _plain)
    p.add_argument("--no-graph", action="store_true", help="Skip OSM graph; crow-flies surrogate only")
    p.add_argument("--hour", type=int, default=17, help="Hour of day for graph congestion")
    p.add_argument("--cutoff-factor", type=float, default=2.5, help="Dijkstra cutoff = threshold x factor")
    p.add_argument(
        "--graph-candidates",
        type=int,
        default=150,
        help="Extra surrogate-screened candidates given exact graph times so posts can be re-picked",
    )
    p.add_argument("--demo", action="store_true", help="Run on synthetic demand (no NYC data needed)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args)
    graph_info = refine_with_graph(plan, args)

    posts = plan["posts"]
    weights = plan["weights"]
    threshold_s = plan["threshold_s"]

    # Score plan + baselines on the SAME final time matrix (graph if available).
    T = plan["T_final"]
    ridx = plan["row_index"]
    sub = T[[ridx[int(r)] for r in plan["postures"][0]["rows"]], :]
    assigned = sub.argmin(axis=0)
    resp = sub.min(axis=0)
    plan["response_s"] = resp
    plan["assignment"] = assigned

    baselines = []
    for posture in plan["postures"][1:]:
        r = T[[ridx[int(x)] for x in posture["rows"]], :].min(axis=0)
        m = posture_metrics(
            r,
            weights,
            threshold_s=threshold_s,
            label=posture["label"],
            n_posts=len(posture["rows"]),
        )
        m["note"] = posture.get("note")
        if posture.get("equal_cost"):
            m["is_equal_cost_baseline"] = True
        baselines.append(m)

    # Per-post demand under the final times (graph refinement can reassign cells).
    post_weight = np.array([weights[assigned == i].sum() for i in range(len(posts))])
    posts["assigned_weight"] = post_weight
    posts["demand_share"] = post_weight / max(post_weight.sum(), 1e-9)

    build_units(plan, args)

    # per-post response metrics under the final (graph or surrogate) times
    mean_resp, cells_assigned, cov_assigned = [], [], []
    for i in range(len(posts)):
        mask = assigned == i
        w = weights[mask]
        r = resp[mask]
        tot = max(w.sum(), 1e-9)
        mean_resp.append(float((w * r).sum() / tot) if mask.any() else float("nan"))
        cells_assigned.append(int(mask.sum()))
        cov_assigned.append(float((w * (r <= threshold_s)).sum() / tot) if mask.any() else float("nan"))
    posts["mean_response_s"] = np.round(mean_resp, 1)
    posts["cells_assigned"] = cells_assigned
    posts["coverage_share_assigned"] = np.round(cov_assigned, 4)

    unit_by_post = {}
    for up in plan["plans"]:
        for order, pid in enumerate(up.post_order, start=1):
            unit_by_post[int(pid)] = (up.unit_id, order)
    posts["unit_id"] = [unit_by_post.get(int(p), (0, 0))[0] for p in posts["post_id"]]
    posts["loop_seq"] = [unit_by_post.get(int(p), (0, 0))[1] for p in posts["post_id"]]
    posts["response_source"] = "osm_graph_emv" if graph_info.get("used") else "haversine_surrogate"

    share_by_post = dict(zip(posts["post_id"].astype(int), posts["demand_share"]))
    for up in plan["plans"]:
        up.demand_share = float(sum(share_by_post.get(int(pid), 0.0) for pid in up.post_order))

    segments = build_loop_segments(plan, graph_info)

    plan_metrics = posture_metrics(
        resp, weights, threshold_s=threshold_s, label="patrol_plan", n_posts=len(posts)
    )

    summary = {
        "data_mode": plan["data_mode"],
        "objective": {
            "mode": args.objective,
            "formal": (
                "minimize sum_c w_c * min_{p in P} t(p,c) / sum_c w_c"
                if args.objective == "response_time"
                else f"maximize sum_c w_c * 1[min_{{p in P}} t(p,c) <= {threshold_s:.0f}s] / sum_c w_c"
            ),
            "plain_english": (
                "Place {k} patrol posts so the demand-weighted expected EMV travel time to the "
                "next incident is as low as possible.".format(k=len(posts))
                if args.objective == "response_time"
                else "Place {k} patrol posts so as much historical incident demand as possible is "
                "within {t:.0f} minutes of a posted unit.".format(k=len(posts), t=args.threshold_min)
            ),
            "decision_variables": "which k candidate locations become patrol posts, and the loop order per unit",
            "algorithm": (
                "greedy seed (p-median / max-coverage) + bounded k-medoids swap improvement; "
                "unit home bases chosen by p-median over post demand; loops = nearest-neighbour + 2-opt TSP"
            ),
            "excluded": "call-processing time, chute/turnout time, unit availability queueing, multi-call concurrency",
        },
        "parameters": {
            "posts": len(posts),
            "units": len(plan["plans"]),
            "cell_km": args.cell_km,
            "threshold_min": args.threshold_min,
            "severity_weighted": bool(args.severity_weighted),
            "demand_candidates": args.demand_candidates,
            "max_candidates": args.max_candidates,
            "swap_rounds": args.swap_rounds,
            "balance": args.balance,
            "hour": args.hour,
        },
        "demand": {
            "incidents": plan["incidents_n"],
            "cells": int(len(plan["cells"])),
            "weight_total": round(float(weights.sum()), 1),
            "top_cell_share": round(float(plan["cells"]["share"].iloc[0]), 4),
        },
        "candidates": {
            "total": int(len(plan["candidates"])),
            "by_layer": plan["candidates"]["layer"].value_counts().to_dict(),
            "anchor_layers": sorted(
                plan["candidates"].loc[plan["candidates"]["is_anchor"], "layer"].unique().tolist()
            ),
        },
        "travel_time": {
            "screening": "calibrated crow-flies surrogate",
            "scoring": "OSM graph Dijkstra on EMV edge costs" if graph_info.get("used") else "surrogate",
            "surrogate": plan["surrogate"].meta,
            "graph": graph_info,
            "description": (
                f"OSM EMV Dijkstra (hour {args.hour}), surrogate fallback beyond cutoff"
                if graph_info.get("used")
                else f"crow-flies x{plan['surrogate'].detour_factor} at "
                f"{plan['surrogate'].speed_kph:.1f} kph effective"
            ),
        },
        "search": {
            "stage1_surrogate": {
                "greedy_objective": round(float(plan["selection"].greedy_value), 4),
                "final_objective": round(float(plan["selection"].value), 4),
                "swaps_applied": int(plan["selection"].n_swaps),
                "candidates": int(len(plan["candidates"])),
            },
            "stage2_graph": (
                {
                    "greedy_objective": round(float(plan["selection_graph"].greedy_value), 4),
                    "final_objective": round(float(plan["selection_graph"].value), 4),
                    "swaps_applied": int(plan["selection_graph"].n_swaps),
                    "candidates": int(len(plan["row_index"])),
                    "note": "posts re-picked on exact EMV graph times; reported metrics use this set",
                }
                if plan.get("reselected_on_graph")
                else None
            ),
        },
        "metrics": {
            "scored_on": "osm_graph_emv" if graph_info.get("used") else "haversine_surrogate",
            "patrol_plan": plan_metrics,
            "baselines": baselines,
            "headline_vs_equal_cost_baseline": _headline(plan_metrics, baselines),
        },
        "units": [
            {
                "unit_id": up.unit_id,
                "home_base": up.anchor_name,
                "home_base_layer": up.anchor_layer,
                "posts": up.post_order,
                "demand_share": round(up.demand_share, 4),
                "loop_minutes": round(up.loop_seconds / 60.0, 2),
                "loop_km": round(up.loop_km, 2),
            }
            for up in plan["plans"]
        ],
        "pairs_with_inferred_starts": (
            "Candidate posts and home bases are the same inferred layers used as OD start "
            "locations (fdny_firehouse / synthetic CSL, see docs/starting_locations.md). "
            "Patrol posts are the prescriptive version of the hybrid on-road start: instead of "
            "assuming a unit happened to be at a CSL, they say which CSL-like posts to hold."
        ),
        "limits": [
            "Demand destinations are ZIP-centroid level (HIPAA aggregation), so cell centroids are "
            "coarse: finer grids do not add resolution until block-level incident geography exists.",
            (
                "Effective surrogate speed is a prior, not fitted: inferred starts sit a median "
                "~0.3 km from the ZIP-centroid destination, so OD-implied speeds (~6 kph) are an "
                "artifact of the start inference rather than driving speed."
                if plan["surrogate"].meta.get("speed_source") != "median_observed_od"
                else "Surrogate speed fitted from inferred-start OD medians, not CAD unit GPS."
            ),
            "Static demand: no hour-of-day or day-of-week re-posting yet (one --hour per run).",
            "No queueing model: assumes the nearest post has an available unit.",
            "Patrol loops are rotation paths, not dispatch routes; response is measured from the "
            "post, so a unit caught mid-leg is slower than reported.",
        ],
        "outputs": {},
    }

    posts_cols = [
        "post_id",
        "unit_id",
        "loop_seq",
        "candidate_id",
        "post_name",
        "layer",
        "borough",
        "lat",
        "lon",
        "is_anchor",
        "assigned_weight",
        "demand_share",
        "cells_assigned",
        "mean_response_s",
        "coverage_share_assigned",
        "response_source",
    ]
    posts_out = args.out_dir / "patrol_posts.csv"
    posts[posts_cols].sort_values(["unit_id", "loop_seq"]).to_csv(posts_out, index=False)

    seg_out = args.out_dir / "patrol_segments.csv"
    segments.to_csv(seg_out, index=False)

    cells_out = args.out_dir / "patrol_demand_cells.csv"
    cells_df = plan["cells"].copy()
    cells_df["nearest_post_id"] = [int(posts.iloc[int(a)]["post_id"]) for a in assigned]
    cells_df["response_s"] = np.round(resp, 1)
    cells_df["within_threshold"] = (resp <= threshold_s).astype(int)
    if plan.get("graph_resolved") is not None:
        res_plan = plan["graph_resolved"][[ridx[int(r)] for r in plan["postures"][0]["rows"]], :]
        cells_df["response_from_graph"] = res_plan[assigned, np.arange(len(cells_df))].astype(int)
    cells_df.drop(columns=["_ix", "_iy"], errors="ignore").to_csv(cells_out, index=False)

    map_out = args.out_dir / "patrol_map.html"
    summary["outputs"] = {
        "patrol_posts_csv": str(posts_out),
        "patrol_segments_csv": str(seg_out),
        "patrol_demand_cells_csv": str(cells_out),
        "patrol_map_html": str(map_out),
    }
    build_map(plan, segments, summary, map_out)

    summary_out = args.out_dir / "summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print("\n--- patrol plan ---")
    print(f"objective        : {args.objective} ({summary['objective']['formal']})")
    print(f"expected response: {plan_metrics['expected_response_s']/60:.2f} min "
          f"(p90 {plan_metrics['p90_response_s']/60:.2f} min)")
    print(f"coverage ≤{args.threshold_min:g} min : {plan_metrics['coverage_share_within_threshold']*100:.1f}% of demand")
    for b in baselines:
        delta = (plan_metrics["expected_response_s"] - b["expected_response_s"]) / max(
            b["expected_response_s"], 1e-9
        )
        tag = " [equal #units]" if b.get("is_equal_cost_baseline") else " [unlimited #units]"
        print(
            f"  vs {b['label']:<32s} {b['expected_response_s']/60:.2f} min "
            f"({b['coverage_share_within_threshold']*100:.1f}% covered) → patrol {delta*100:+.1f}%{tag}"
        )
    print("\nWrote", posts_out)
    print("Wrote", seg_out)
    print("Wrote", cells_out)
    print("Wrote", map_out)
    print("Wrote", summary_out)


if __name__ == "__main__":
    main()
