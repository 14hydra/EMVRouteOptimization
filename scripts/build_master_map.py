#!/usr/bin/env python3
"""
Build one master Folium map covering NYC + San Francisco:

  • Dataset OD routes
  • Example model trips (Google Maps, OSM, MIPSSTW+MCS, DRL, GBDT) with best-model highlight
  • ROW-win examples

Usage:
  PYTHONPATH=src python scripts/build_master_map.py
  PYTHONPATH=src python scripts/build_master_map.py --dataset-routes 30 --dataset-markers 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Diverse SF OD scenarios (fire/EMS-ish → neighborhoods)
SF_MAP_SCENARIOS = [
    {
        "id": "mission_to_fidi",
        "label": "Mission → Financial District",
        "origin": {"lat": 37.7599, "lon": -122.4148},
        "dest": {"lat": 37.7946, "lon": -122.3999},
    },
    {
        "id": "sunset_to_castro",
        "label": "Sunset → Castro",
        "origin": {"lat": 37.7530, "lon": -122.4950},
        "dest": {"lat": 37.7609, "lon": -122.4350},
    },
    {
        "id": "bayview_to_soma",
        "label": "Bayview → SOMA",
        "origin": {"lat": 37.7312, "lon": -122.3880},
        "dest": {"lat": 37.7785, "lon": -122.4050},
    },
    {
        "id": "richmond_to_union",
        "label": "Richmond → Union Square",
        "origin": {"lat": 37.7799, "lon": -122.4800},
        "dest": {"lat": 37.7879, "lon": -122.4075},
    },
    {
        "id": "marina_to_tenderloin",
        "label": "Marina → Tenderloin",
        "origin": {"lat": 37.8020, "lon": -122.4360},
        "dest": {"lat": 37.7840, "lon": -122.4140},
    },
]


def _load_vrm():
    spec = importlib.util.spec_from_file_location(
        "visualize_route_models", ROOT / "scripts" / "visualize_route_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fmt_zip(z) -> str:
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return "?"
    try:
        return str(int(float(z)))
    except (TypeError, ValueError):
        s = str(z).strip()
        return s[:-2] if s.endswith(".0") else (s or "?")


def _node_ll(G, node):
    d = G.nodes[node]
    return [float(d.get("y", d.get("lat"))), float(d.get("x", d.get("lon")))]


def _path_ll(G, path):
    return [_node_ll(G, n) for n in path]


def _offset(coords, model_name: str, index: int = 0):
    if len(coords) < 3:
        return coords
    offsets = {
        "control_google_maps": 0.00002,
        "control_civilian_time": 0.0,
        "civilian_gps": 0.0,
        "mipsstw_mcs": 0.00005,
        "composite_drl": -0.00005,
        "gbdt_router": 0.00009,
        "emv_dijkstra": -0.00009,
        "emv_row": -0.00009,
    }
    dy = offsets.get(model_name, 0.0) + (index % 3) * 0.00001
    out = [coords[0]]
    for lat, lon in coords[1:-1]:
        out.append([lat + dy, lon + dy * 0.55])
    out.append(coords[-1])
    return out


def _city_configs(vrm):
    return [
        {
            "id": "nyc",
            "label": "NYC",
            "center": [40.74, -73.95],
            "zoom": 11,
            "graph": ROOT / "data" / "raw" / "nyc_drive.graphml",
            "od": ROOT / "data" / "processed" / "od_pairs_inferred_starts.csv",
            "row_wins": ROOT / "data" / "figures" / "route_models" / "row_wins.csv",
            "scenarios": vrm.MAP_SCENARIOS,
            "facilities": True,
        },
        {
            "id": "sf",
            "label": "SF",
            "center": [37.76, -122.44],
            "zoom": 12,
            "graph": ROOT / "data" / "raw" / "sf_drive.graphml",
            "od": ROOT / "data" / "processed" / "od_pairs_sf_ems.csv",
            "row_wins": ROOT / "data" / "figures" / "route_models" / "sf" / "row_wins.csv",
            "scenarios": SF_MAP_SCENARIOS,
            "facilities": False,
        },
    ]


def build_master_map(
    *,
    out: Path,
    dataset_markers: int = 200,
    dataset_routes: int = 30,
    hour: int = 17,
    seed: int = 42,
    cities: list[str] | None = None,
) -> Path:
    import folium
    from branca.element import Element
    from folium.plugins import MarkerCluster

    from emvro.gmaps import load_api_key_from_dotenv
    from emvro.routing import (
        apply_routing_conditions,
        conditions_at,
        control_civilian_time,
        control_google_maps,
        nearest_node,
        prepare_routing_graph,
        solve_composite_drl,
        solve_gbdt_route,
        solve_mipsstw_mcs,
    )
    from emvro.routing.emv_corridors import path_corridor_segments
    from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model
    from emvro.routing.graph import dijkstra_route

    vrm = _load_vrm()
    MODEL_COLORS = vrm.MODEL_COLORS
    map_label = vrm._map_model_label
    corridor_label = vrm._corridor_kind_label
    configs = _city_configs(vrm)
    if cities:
        want = {c.lower() for c in cities}
        configs = [c for c in configs if c["id"] in want]
    configs = [c for c in configs if c["graph"].exists() and c["od"].exists()]
    if not configs:
        raise SystemExit("No city configs available (need graph + OD CSV)")

    load_api_key_from_dotenv(ROOT / ".env")
    gmaps_cache = ROOT / "data" / "processed" / "gmaps_cache.json"

    m = folium.Map(location=configs[0]["center"], zoom_start=configs[0]["zoom"], tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri streets",
    ).add_to(m)

    layer_js: dict[str, str] = {}
    layer_sets: dict[str, list[str]] = {
        "dataset": [],
        "examples": [],
        "row_wins": [],
        "facilities": [],
    }
    city_layers: dict[str, list[str]] = {c["id"]: [] for c in configs}
    city_views = {c["id"]: {"center": c["center"], "zoom": c["zoom"]} for c in configs}
    example_best_rows: list[dict] = []
    # Example route polylines live in one holder; trip/model FeatureGroups are
    # checkboxes only. Visibility = trip ON ∧ model ON (no duplicate parents).
    example_route_registry: list[dict] = []
    stats_bits: list[str] = []

    def _fg(name: str, show: bool = False, *, city_id: str | None = None, kind: str | None = None):
        fg = folium.FeatureGroup(name=name, show=show)
        fg.add_to(m)
        layer_js[name] = fg.get_name()
        if city_id:
            city_layers[city_id].append(name)
        if kind and kind in layer_sets:
            layer_sets[kind].append(name)
        return fg

    # Hidden holder for all example polylines (not in LayerControl)
    example_route_holder = folium.FeatureGroup(name="_example_routes_holder", show=True, control=False)
    example_route_holder.add_to(m)

    emv_model_names = ("mipsstw_mcs", "composite_drl", "gbdt_router")
    map_models = [
        ("control_google_maps", dict(weight=3, opacity=0.45, dash="4 8")),
        ("control_civilian_time", dict(weight=2, opacity=0.3, dash="10 8")),
        ("mipsstw_mcs", dict(weight=4, opacity=0.55, dash=None)),
        ("composite_drl", dict(weight=4, opacity=0.55, dash=None)),
        ("gbdt_router", dict(weight=3, opacity=0.5, dash="4 8")),
    ]

    all_bounds: list = []
    default_on: list[str] = []

    for ci, city in enumerate(configs):
        cid, clabel = city["id"], city["label"]
        print(f"\n=== {clabel} ===")
        print(f"Loading {city['graph'].name}…")
        G = prepare_routing_graph(city["graph"], hour=hour)
        print("Training GBDT edge model…")
        edge_df = build_edge_training_frame(G, hours=[hour], max_edges=4000)
        bundle = train_gbdt_edge_model(edge_df)

        show_city = ci == 0  # NYC (or first) visible by default

        # ---- Dataset ----
        ds_markers = _fg(
            f"{clabel} · Dataset · OD markers", show=False, city_id=cid, kind="dataset"
        )
        ds_civ = _fg(
            f"{clabel} · Dataset · Civilian GPS routes",
            show=False,
            city_id=cid,
            kind="dataset",
        )
        ds_emv = _fg(
            f"{clabel} · Dataset · EMV Dijkstra routes",
            show=False,
            city_id=cid,
            kind="dataset",
        )

        od = pd.read_csv(city["od"], low_memory=False)
        for c in ("start_lat", "start_lon", "dest_lat", "dest_lon", "travel_seconds"):
            if c in od.columns:
                od[c] = pd.to_numeric(od[c], errors="coerce")
        usable = od.dropna(subset=["start_lat", "start_lon", "dest_lat", "dest_lon"]).copy()
        markers_n = min(dataset_markers, len(usable))
        routes_n = min(dataset_routes, len(usable))
        mark_sample = usable.sample(n=markers_n, random_state=seed + ci)
        route_sample = usable.sample(n=routes_n, random_state=seed + 10 + ci)
        print(f"  dataset markers={markers_n}, routes={routes_n}")

        for _, r in mark_sample.iterrows():
            layer = str(r.get("depot_layer") or "other")
            start_color = {
                "csl": "#9467bd",
                "ems_station": "#1f77b4",
                "hospital_bay": "#2ca02c",
            }.get(layer, "#7f7f7f")
            zip_s = _fmt_zip(r.get("zipcode"))
            travel = r.get("travel_seconds")
            travel_txt = (
                f"{float(travel)/60:.1f} min observed"
                if pd.notna(travel)
                else "travel time n/a"
            )
            folium.CircleMarker(
                [float(r["start_lat"]), float(r["start_lon"])],
                radius=4,
                color=start_color,
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"<b>{clabel} dataset start</b><br>{layer}: {r.get('depot_name')}<br>"
                    f"{travel_txt}"
                ),
                tooltip=f"{clabel} start · {layer}",
            ).add_to(ds_markers)
            folium.CircleMarker(
                [float(r["dest_lat"]), float(r["dest_lon"])],
                radius=4,
                color="#ff7f0e",
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"<b>{clabel} dataset destination</b><br>ZIP {zip_s}<br>"
                    f"id={r.get('incident_id')}"
                ),
                tooltip=f"{clabel} dest · ZIP {zip_s}",
            ).add_to(ds_markers)
            folium.PolyLine(
                [
                    [float(r["start_lat"]), float(r["start_lon"])],
                    [float(r["dest_lat"]), float(r["dest_lon"])],
                ],
                color="#bbbbbb",
                weight=1,
                opacity=0.25,
                dash_array="2 6",
            ).add_to(ds_markers)

        routed_ok = 0
        for i, (_, r) in enumerate(route_sample.iterrows()):
            try:
                origin = nearest_node(G, float(r["start_lon"]), float(r["start_lat"]))
                dest = nearest_node(G, float(r["dest_lon"]), float(r["dest_lat"]))
            except Exception:
                continue
            if origin == dest:
                continue
            o_ll, d_ll = _node_ll(G, origin), _node_ll(G, dest)
            label = (
                f"{str(r.get('borough') or '?').title()} → ZIP {_fmt_zip(r.get('zipcode'))}"
            )
            civ = control_civilian_time(G, origin, dest)
            emv = dijkstra_route(
                G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"
            )
            for name, result, fg, color, weight, dash, opacity in [
                ("civilian_gps", civ, ds_civ, "#7f8c8d", 3, "8 6", 0.55),
                ("emv_dijkstra", emv, ds_emv, "#2980b9", 4, None, 0.85),
            ]:
                if not result.ok or not result.node_path:
                    continue
                coords = _path_ll(G, result.node_path)
                coords[0], coords[-1] = list(o_ll), list(d_ll)
                all_bounds.extend(coords)
                mins = result.travel_seconds / 60.0
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    dash_array=dash,
                    popup=(
                        f"<b>{clabel} dataset route</b><br>{label}<br>"
                        f"{map_label(name)}<br>{mins:.2f} min"
                    ),
                    tooltip=f"{clabel}: {map_label(name)} ({mins:.1f} min)",
                ).add_to(fg)
            routed_ok += 1
        print(f"  dataset routes drawn: {routed_ok}")

        # ---- Examples ----
        ex_models = {
            "control_google_maps": _fg(
                f"{clabel} · Examples · Google Maps",
                show=show_city,
                city_id=cid,
                kind="examples",
            ),
            "control_civilian_time": _fg(
                f"{clabel} · Examples · OSM civilian GPS",
                show=False,
                city_id=cid,
                kind="examples",
            ),
            "mipsstw_mcs": _fg(
                f"{clabel} · Examples · MIPSSTW + MCS",
                show=False,
                city_id=cid,
                kind="examples",
            ),
            "composite_drl": _fg(
                f"{clabel} · Examples · Composite DRL",
                show=False,
                city_id=cid,
                kind="examples",
            ),
            "gbdt_router": _fg(
                f"{clabel} · Examples · GBDT router",
                show=False,
                city_id=cid,
                kind="examples",
            ),
        }
        best_fg = _fg(
            f"{clabel} · Examples · Best EMV route",
            show=show_city,
            city_id=cid,
            kind="examples",
        )
        if show_city:
            default_on.extend(
                [
                    f"{clabel} · Examples · Google Maps",
                    f"{clabel} · Examples · Best EMV route",
                ]
            )

        scenarios = city["scenarios"]
        print(f"  routing {len(scenarios)} example trips…")
        for i, sc in enumerate(scenarios):
            trip_fg = _fg(
                f"{clabel} · Examples · Trip {i+1}: {sc['label']}",
                show=show_city,
                city_id=cid,
                kind="examples",
            )
            if show_city:
                default_on.append(f"{clabel} · Examples · Trip {i+1}: {sc['label']}")

            o, d = sc["origin"], sc["dest"]
            try:
                origin = nearest_node(G, o["lon"], o["lat"])
                dest = nearest_node(G, d["lon"], d["lat"])
            except Exception as exc:  # noqa: BLE001
                print(f"  skip example {sc['id']}: {exc}")
                continue
            o_ll, d_ll = _node_ll(G, origin), _node_ll(G, dest)

            solved = {
                "control_google_maps": control_google_maps(
                    o["lat"], o["lon"], d["lat"], d["lon"], cache_path=gmaps_cache
                ),
                "control_civilian_time": control_civilian_time(G, origin, dest),
                "mipsstw_mcs": solve_mipsstw_mcs(
                    G, origin, dest, n_iterations=18, n_nests=10, seed=42 + i + 100 * ci
                ),
                "composite_drl": solve_composite_drl(
                    G, origin, dest, episodes=80, seed=42 + i + 100 * ci
                ),
                "gbdt_router": solve_gbdt_route(
                    G, origin, dest, bundle=bundle, hour=hour
                ),
            }

            best_name = None
            best_secs = float("inf")
            for name in emv_model_names:
                r = solved[name]
                if r.ok and r.travel_seconds == r.travel_seconds and r.travel_seconds < best_secs:
                    best_secs = float(r.travel_seconds)
                    best_name = name
            best_label = map_label(best_name) if best_name else "n/a"
            best_min = best_secs / 60.0 if best_name else float("nan")
            gmaps_r = solved["control_google_maps"]
            gmaps_min = (
                gmaps_r.travel_seconds / 60.0
                if gmaps_r.ok and gmaps_r.travel_seconds == gmaps_r.travel_seconds
                else None
            )
            vs_gmaps = None
            if gmaps_min and best_name and gmaps_min > 0:
                vs_gmaps = 100.0 * (1.0 - best_min / gmaps_min)

            example_best_rows.append(
                {
                    "city": clabel,
                    "trip": i + 1,
                    "label": sc["label"],
                    "best_model": best_name,
                    "best_label": best_label,
                    "best_min": None if not best_name else round(best_min, 2),
                    "gmaps_min": None if gmaps_min is None else round(gmaps_min, 2),
                    "pct_faster_vs_gmaps": None if vs_gmaps is None else round(vs_gmaps, 1),
                }
            )

            folium.CircleMarker(
                o_ll,
                radius=9,
                color="#0b3d5c",
                fill=True,
                fill_color="#2e86c1",
                fill_opacity=1.0,
                weight=2,
                popup=(
                    f"<b>{clabel} example start {i+1}</b><br>{sc['label']}<br>"
                    f"<b>Best EMV: {best_label}</b>"
                    + (f" ({best_min:.2f} min)" if best_name else "")
                ),
                tooltip=f"{clabel} Ex {i+1} start · best={best_label}",
            ).add_to(trip_fg)
            folium.CircleMarker(
                d_ll,
                radius=9,
                color="#7d3c00",
                fill=True,
                fill_color="#e67e22",
                fill_opacity=1.0,
                weight=2,
                popup=f"<b>{clabel} example dest {i+1}</b><br>{sc['label']}",
                tooltip=f"{clabel} Ex {i+1} dest",
            ).add_to(trip_fg)

            badge = (
                f"{clabel} Ex {i+1} BEST: {best_label} · {best_min:.1f} min"
                if best_name
                else f"{clabel} Ex {i+1}"
            )
            if vs_gmaps is not None:
                badge += f" ({vs_gmaps:.0f}% vs GMaps)"
            folium.Marker(
                [(o_ll[0] + d_ll[0]) / 2, (o_ll[1] + d_ll[1]) / 2],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font:12px/1.25 system-ui,sans-serif;font-weight:700;'
                        f'background:rgba(255,255,255,.96);padding:4px 8px;border-radius:6px;'
                        f'border:2px solid #1f4e79;box-shadow:0 1px 4px rgba(0,0,0,.2);'
                        f'white-space:nowrap;color:#111;">{badge}</div>'
                    )
                ),
            ).add_to(trip_fg)

            for name, style in map_models:
                result = solved[name]
                if not result.ok:
                    continue
                if name == "control_google_maps":
                    coords = list(result.meta.get("polyline_latlons") or [])
                    if len(coords) < 2:
                        continue
                else:
                    if not result.node_path:
                        continue
                    coords = _path_ll(G, result.node_path)
                coords = _offset(coords, name, i + 10 * ci)
                coords[0], coords[-1] = list(o_ll), list(d_ll)
                all_bounds.extend(coords)
                mins = result.travel_seconds / 60.0
                km = (result.distance_m or 0) / 1000.0
                is_best = name == best_name
                role = " ★ BEST EMV" if is_best else ""
                popup = (
                    f"<b>{clabel} example {i+1}: {sc['label']}</b><br>"
                    f"{map_label(name)}{role}<br>{mins:.2f} min · {km:.2f} km"
                )
                tip = f"{clabel} Ex {i+1}: {map_label(name)} ({mins:.1f} min){role}"
                # Single parent only (holder). Trip/model layers are checkboxes;
                # JS shows a route only when BOTH its trip and model are enabled.
                pl = folium.PolyLine(
                    coords,
                    color=MODEL_COLORS.get(name, "#333"),
                    weight=style["weight"] + (2 if is_best else 0),
                    opacity=0.95 if is_best else style["opacity"],
                    dash_array=None if is_best else style["dash"],
                    popup=popup,
                    tooltip=tip,
                )
                pl.add_to(example_route_holder)
                example_route_registry.append(
                    {
                        "layer": pl.get_name(),
                        "city": cid,
                        "trip": i + 1,
                        "trip_layer": f"{clabel} · Examples · Trip {i+1}: {sc['label']}",
                        "model": name,
                        "model_layer": f"{clabel} · Examples · "
                        + {
                            "control_google_maps": "Google Maps",
                            "control_civilian_time": "OSM civilian GPS",
                            "mipsstw_mcs": "MIPSSTW + MCS",
                            "composite_drl": "Composite DRL",
                            "gbdt_router": "GBDT router",
                        }[name],
                    }
                )
                if is_best:
                    halo = folium.PolyLine(coords, color="#ffffff", weight=12, opacity=0.85)
                    halo.add_to(example_route_holder)
                    example_route_registry.append(
                        {
                            "layer": halo.get_name(),
                            "city": cid,
                            "trip": i + 1,
                            "trip_layer": f"{clabel} · Examples · Trip {i+1}: {sc['label']}",
                            "model": "best_emv",
                            "model_layer": f"{clabel} · Examples · Best EMV route",
                        }
                    )
                    hi = folium.PolyLine(
                        coords,
                        color=MODEL_COLORS.get(name, "#8e44ad"),
                        weight=7,
                        opacity=1.0,
                        popup=popup,
                        tooltip=f"BEST · {tip}",
                    )
                    hi.add_to(example_route_holder)
                    example_route_registry.append(
                        {
                            "layer": hi.get_name(),
                            "city": cid,
                            "trip": i + 1,
                            "trip_layer": f"{clabel} · Examples · Trip {i+1}: {sc['label']}",
                            "model": "best_emv",
                            "model_layer": f"{clabel} · Examples · Best EMV route",
                        }
                    )
            print(f"  example {sc['id']} best={best_name}")

        # ---- ROW wins ----
        rw_civ = _fg(
            f"{clabel} · ROW wins · Civilian GPS", show=False, city_id=cid, kind="row_wins"
        )
        rw_emv = _fg(
            f"{clabel} · ROW wins · EMV Dijkstra", show=False, city_id=cid, kind="row_wins"
        )
        rw_gold = _fg(
            f"{clabel} · ROW wins · EMV-only corridors",
            show=False,
            city_id=cid,
            kind="row_wins",
        )
        rw_mark = _fg(
            f"{clabel} · ROW wins · markers", show=False, city_id=cid, kind="row_wins"
        )

        rw_path = city["row_wins"]
        if rw_path.exists():
            print(f"  routing ROW wins from {rw_path}…")
            wins = pd.read_csv(rw_path)
            apply_routing_conditions(G, conditions=conditions_at(17, "rain"))
            for _, w in wins.iterrows():
                try:
                    origin = nearest_node(G, float(w["start_lon"]), float(w["start_lat"]))
                    dest = nearest_node(G, float(w["dest_lon"]), float(w["dest_lat"]))
                except Exception:
                    continue
                o_ll, d_ll = _node_ll(G, origin), _node_ll(G, dest)
                rank = int(w.get("rank") or 0)
                label = w.get("label") or f"ROW win {rank}"
                if isinstance(label, str) and "ZIP" in label and ".0" in label:
                    parts = label.split("ZIP")
                    label = parts[0] + "ZIP " + _fmt_zip(parts[-1].strip())
                title = f"{clabel} ROW win {rank}: {label}"

                folium.CircleMarker(
                    o_ll,
                    radius=7,
                    color="#1f4e79",
                    fill=True,
                    fill_color="#3498db",
                    popup=f"<b>{title}</b><br>Start",
                    tooltip=f"{clabel} ROW start {rank}",
                ).add_to(rw_mark)
                folium.CircleMarker(
                    d_ll,
                    radius=7,
                    color="#b35900",
                    fill=True,
                    fill_color="#e67e22",
                    popup=f"<b>{title}</b><br>Destination",
                    tooltip=f"{clabel} ROW dest {rank}",
                ).add_to(rw_mark)

                civ = control_civilian_time(G, origin, dest)
                emv = dijkstra_route(
                    G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"
                )
                civ_m = civ.travel_seconds / 60.0 if civ.ok else float("nan")
                emv_m = emv.travel_seconds / 60.0 if emv.ok else float("nan")
                saved = civ_m - emv_m if civ.ok and emv.ok else float("nan")
                popup = (
                    f"<b>{title}</b><br>Conditions: Rain · 17:00<br>"
                    f"Civilian GPS: {civ_m:.2f} min → EMV Dijkstra: {emv_m:.2f} min<br>"
                    f"Faster by: <b>{saved:.2f} min</b>"
                    if civ.ok and emv.ok
                    else f"<b>{title}</b>"
                )
                if civ.ok and civ.node_path:
                    coords = _path_ll(G, civ.node_path)
                    coords[0], coords[-1] = list(o_ll), list(d_ll)
                    all_bounds.extend(coords)
                    folium.PolyLine(
                        coords,
                        color="#7f8c8d",
                        weight=4,
                        opacity=0.55,
                        dash_array="10 8",
                        popup=popup,
                        tooltip=f"{clabel} ROW {rank} civilian ({civ_m:.1f} min)",
                    ).add_to(rw_civ)
                if emv.ok and emv.node_path:
                    coords = _path_ll(G, emv.node_path)
                    coords[0], coords[-1] = list(o_ll), list(d_ll)
                    all_bounds.extend(coords)
                    folium.PolyLine(
                        coords,
                        color="#8e44ad",
                        weight=6,
                        opacity=0.95,
                        popup=popup,
                        tooltip=f"{clabel} ROW {rank} EMV ({emv_m:.1f} min)",
                    ).add_to(rw_emv)
                    for seg in path_corridor_segments(G, emv.node_path):
                        folium.PolyLine(
                            seg["coords"],
                            color="#f1c40f",
                            weight=8,
                            opacity=0.9,
                            popup=f"<b>{corridor_label(seg.get('kind'))}</b><br>{title}",
                            tooltip=corridor_label(seg.get("kind")),
                        ).add_to(rw_gold)
                folium.Marker(
                    [(o_ll[0] + d_ll[0]) / 2, (o_ll[1] + d_ll[1]) / 2],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font:11px/1.2 sans-serif;background:rgba(255,255,255,.94);'
                            f'padding:2px 6px;border:1px solid #888;border-radius:4px;white-space:nowrap;">'
                            f"{clabel} ROW {rank}: {saved:.1f} min faster</div>"
                        )
                    ),
                ).add_to(rw_mark)
            apply_routing_conditions(G, conditions=conditions_at(hour, "clear"))
            print(f"  ROW wins mapped: {len(wins)}")
        else:
            print(f"  no ROW wins CSV at {rw_path} (skip)")

        # ---- Facilities (NYC) ----
        if city.get("facilities"):
            fac_ems = MarkerCluster(name=f"{clabel} · Facilities · EMS stations", show=False)
            fac_ems.add_to(m)
            layer_js[f"{clabel} · Facilities · EMS stations"] = fac_ems.get_name()
            city_layers[cid].append(f"{clabel} · Facilities · EMS stations")
            layer_sets["facilities"].append(f"{clabel} · Facilities · EMS stations")
            fac_hosp = MarkerCluster(
                name=f"{clabel} · Facilities · Hospital bays", show=False
            )
            fac_hosp.add_to(m)
            layer_js[f"{clabel} · Facilities · Hospital bays"] = fac_hosp.get_name()
            city_layers[cid].append(f"{clabel} · Facilities · Hospital bays")
            layer_sets["facilities"].append(f"{clabel} · Facilities · Hospital bays")

            stations = ROOT / "data" / "raw" / "ems_stations.csv"
            hospitals = ROOT / "data" / "raw" / "hospital_bays.csv"
            if stations.exists():
                sdf = pd.read_csv(stations)
                for _, r in sdf.iterrows():
                    if pd.isna(r.get("latitude")):
                        continue
                    folium.Marker(
                        [float(r["latitude"]), float(r["longitude"])],
                        icon=folium.Icon(color="blue", icon="plus-sign"),
                        popup=f"EMS station<br>{r.get('facname')}",
                    ).add_to(fac_ems)
            if hospitals.exists():
                hdf = pd.read_csv(hospitals)
                for _, r in hdf.iterrows():
                    if pd.isna(r.get("latitude")):
                        continue
                    folium.Marker(
                        [float(r["latitude"]), float(r["longitude"])],
                        icon=folium.Icon(color="green", icon="plus-sign"),
                        popup=f"Hospital bay<br>{r.get('facname')}",
                    ).add_to(fac_hosp)

        stats_bits.append(f"{clabel}: {markers_n} OD · {routed_ok} routed · {len(scenarios)} examples")

    # Don't fit to all cities at once (spans continent) — start on first city
    m.location = configs[0]["center"]
    m.options["zoom"] = configs[0]["zoom"]

    best_csv = out.with_name("master_map_example_best.csv")
    pd.DataFrame(example_best_rows).to_csv(best_csv, index=False)

    best_rows_html = ""
    for row in example_best_rows:
        g = "—" if row["gmaps_min"] is None else f"{row['gmaps_min']:.1f} min"
        b = "—" if row["best_min"] is None else f"{row['best_min']:.1f} min"
        pct = (
            ""
            if row["pct_faster_vs_gmaps"] is None
            else f" · {row['pct_faster_vs_gmaps']:.0f}% faster"
        )
        best_rows_html += (
            f"<div style='margin:2px 0;'><b>{row['city']} Ex {row['trip']}</b> "
            f"{row['label']}<br>"
            f"<span style='color:#1f4e79;'>Best: {row['best_label']} ({b})</span>"
            f" vs GMaps {g}{pct}</div>"
        )

    city_ids = [c["id"] for c in configs]
    city_btn_html = "".join(
        (
            f'<button type="button" data-city="{c["id"]}" class="emv-city'
            f'{" active" if i == 0 else ""}">{c["label"]}</button>'
        )
        for i, c in enumerate(configs)
    )
    panel = f"""
    <div id="emv-master-panel" style="position:fixed;top:14px;left:14px;z-index:9999;
         background:rgba(255,255,255,0.97);padding:12px 14px;border:1px solid #888;
         border-radius:10px;font:13px/1.4 system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.18);
         width:380px;max-width:92vw;max-height:92vh;overflow:auto;">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px;">EMV master map</div>
      <div style="color:#555;font-size:12px;margin-bottom:8px;">
        Pick a <b>city</b>, then a <b>case</b> (examples / dataset / ROW wins).
        Example mode highlights the best EMV model per trip.
      </div>
      <div style="font-weight:600;font-size:12px;margin-bottom:4px;">City</div>
      <div id="emv-city-btns" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;">
        {city_btn_html}
        <button type="button" data-city="both" class="emv-city">Both</button>
      </div>
      <div style="font-weight:600;font-size:12px;margin-bottom:4px;">Case</div>
      <div id="emv-mode-btns" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;">
        <button type="button" data-mode="examples" class="emv-mode active">Examples</button>
        <button type="button" data-mode="dataset" class="emv-mode">Dataset</button>
        <button type="button" data-mode="row_wins" class="emv-mode">ROW wins</button>
        <button type="button" data-mode="all" class="emv-mode">Show all</button>
      </div>
      <div style="background:#f0f4f8;border:1px solid #d0d7de;border-radius:6px;padding:8px 10px;
           margin-bottom:10px;font-size:12px;max-height:220px;overflow:auto;">
        <div style="font-weight:700;margin-bottom:4px;">Example trips — best EMV model</div>
        {best_rows_html}
      </div>
      <div style="font-size:12px;color:#333;">
        <div><span style="color:#111;">- - -</span> Google Maps (civilian)</div>
        <div><span style="color:#8e44ad;font-weight:700;">━━</span> Best EMV route (highlighted)</div>
        <div style="color:#666;margin-top:2px;">Optional rivals: MIPSSTW+MCS · Composite DRL · GBDT</div>
        <div><span style="color:#f1c40f;font-weight:700;">━━</span> EMV-only corridor</div>
      </div>
      <div style="margin-top:8px;font-size:11px;color:#666;">
        {" · ".join(stats_bits)} · hour {hour}:00
      </div>
    </div>
    <style>
      #emv-master-panel .emv-mode, #emv-master-panel .emv-city {{
        flex:1 1 40%; padding:7px 8px; border:1px solid #bbb; border-radius:6px;
        background:#fff; cursor:pointer; font-weight:600; font-size:12px;
      }}
      #emv-master-panel .emv-mode.active, #emv-master-panel .emv-city.active {{
        background:#1f4e79; color:#fff; border-color:#1f4e79;
      }}
    </style>
    <script>
    (function() {{
      var SETS = {json.dumps(layer_sets)};
      var CITY_LAYERS = {json.dumps(city_layers)};
      var CITY_VIEWS = {json.dumps(city_views)};
      var LAYERS = {json.dumps(layer_js)};
      var DEFAULT_ON = {json.dumps(default_on)};
      var CITY_IDS = {json.dumps(city_ids)};
      var ROUTE_REGISTRY = {json.dumps(example_route_registry)};
      var ROUTE_HOLDER = {json.dumps(example_route_holder.get_name())};
      var city = CITY_IDS[0] || "nyc";
      var mode = "examples";
      var mapRef = null;

      function findMap() {{
        if (mapRef) return mapRef;
        for (var k in window) {{
          if (k.indexOf("map_") === 0 && window[k] && window[k].eachLayer) {{
            mapRef = window[k];
            return mapRef;
          }}
        }}
        return null;
      }}
      function layerObj(jsName) {{ return window[jsName] || null; }}

      function setVisible(names, on) {{
        var map = findMap();
        if (!map) return;
        names.forEach(function(name) {{
          var lyr = layerObj(LAYERS[name]);
          if (!lyr) return;
          if (on) {{ if (!map.hasLayer(lyr)) map.addLayer(lyr); }}
          else {{ if (map.hasLayer(lyr)) map.removeLayer(lyr); }}
        }});
      }}

      function isLayerOn(name) {{
        var map = findMap();
        var lyr = layerObj(LAYERS[name]);
        return !!(map && lyr && map.hasLayer(lyr));
      }}

      // Example polylines are single-parented under ROUTE_HOLDER. A route is
      // visible only when its trip layer AND model layer are both checked.
      function syncExampleRoutes() {{
        var map = findMap();
        if (!map) return;
        var holder = layerObj(ROUTE_HOLDER);
        if (holder && !map.hasLayer(holder)) map.addLayer(holder);
        var examplesMode = (mode === "examples" || mode === "all");
        ROUTE_REGISTRY.forEach(function(r) {{
          var lyr = layerObj(r.layer);
          if (!lyr) return;
          var cityOk = (city === "both" || city === r.city);
          var on = examplesMode && cityOk && isLayerOn(r.trip_layer) && isLayerOn(r.model_layer);
          if (holder) {{
            if (on) {{
              if (!holder.hasLayer(lyr)) holder.addLayer(lyr);
            }} else {{
              if (holder.hasLayer(lyr)) holder.removeLayer(lyr);
            }}
          }} else {{
            if (on) {{ if (!map.hasLayer(lyr)) map.addLayer(lyr); }}
            else {{ if (map.hasLayer(lyr)) map.removeLayer(lyr); }}
          }}
        }});
      }}

      function allNamed() {{
        var all = [];
        Object.keys(SETS).forEach(function(k) {{ all = all.concat(SETS[k]); }});
        return all;
      }}

      function activeCityLayers() {{
        if (city === "both") {{
          var all = [];
          CITY_IDS.forEach(function(id) {{ all = all.concat(CITY_LAYERS[id] || []); }});
          return all;
        }}
        return CITY_LAYERS[city] || [];
      }}

      function flyCity() {{
        var map = findMap();
        if (!map) return;
        if (city === "both") {{
          // Keep current view; user can zoom out
          return;
        }}
        var v = CITY_VIEWS[city];
        if (v) map.setView(v.center, v.zoom);
      }}

      function apply() {{
        var all = allNamed();
        setVisible(all, false);
        var citySet = {{}};
        activeCityLayers().forEach(function(n) {{ citySet[n] = true; }});

        if (mode === "all") {{
          setVisible(activeCityLayers(), true);
        }} else {{
          var wanted = (SETS[mode] || []).filter(function(n) {{ return !!citySet[n]; }});
          if (mode === "examples") {{
            // Prefer best + gmaps + trip markers; hide rival model layers
            wanted = wanted.filter(function(n) {{
              return n.indexOf("MIPSSTW") < 0 && n.indexOf("Composite DRL") < 0
                  && n.indexOf("GBDT router") < 0 && n.indexOf("OSM civilian") < 0;
            }});
          }}
          setVisible(wanted, true);
        }}

        document.querySelectorAll("#emv-city-btns .emv-city").forEach(function(btn) {{
          btn.classList.toggle("active", btn.getAttribute("data-city") === city);
        }});
        document.querySelectorAll("#emv-mode-btns .emv-mode").forEach(function(btn) {{
          btn.classList.toggle("active", btn.getAttribute("data-mode") === mode);
        }});
        flyCity();
        syncExampleRoutes();
      }}

      function bind() {{
        document.querySelectorAll("#emv-city-btns .emv-city").forEach(function(btn) {{
          btn.addEventListener("click", function() {{
            city = btn.getAttribute("data-city");
            apply();
          }});
        }});
        document.querySelectorAll("#emv-mode-btns .emv-mode").forEach(function(btn) {{
          btn.addEventListener("click", function() {{
            mode = btn.getAttribute("data-mode");
            apply();
          }});
        }});
        var map = findMap();
        if (map) {{
          map.on("overlayadd overlayremove", function() {{
            setTimeout(syncExampleRoutes, 0);
          }});
        }}
        // Start from declared defaults then sync via apply()
        setVisible(allNamed(), false);
        setVisible(DEFAULT_ON, true);
        apply();
      }}

      var tries = 0;
      (function wait() {{
        tries += 1;
        if (findMap() || tries > 50) bind();
        else setTimeout(wait, 50);
      }})();
    }})();
    </script>
    """
    m.get_root().html.add_child(Element(panel))
    folium.LayerControl(collapsed=True).add_to(m)

    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print("\nWrote", out)
    print("Best-model summary:", best_csv)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "figures" / "master_map.html",
    )
    p.add_argument("--dataset-markers", type=int, default=200)
    p.add_argument("--dataset-routes", type=int, default=30)
    p.add_argument("--hour", type=int, default=17)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Subset of cities (nyc sf). Default: all available.",
    )
    args = p.parse_args()

    build_master_map(
        out=args.out,
        dataset_markers=args.dataset_markers,
        dataset_routes=args.dataset_routes,
        hour=args.hour,
        seed=args.seed,
        cities=args.cities,
    )


if __name__ == "__main__":
    main()
