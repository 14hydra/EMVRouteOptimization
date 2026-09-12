#!/usr/bin/env python3
"""
Visualize the 3 route-optimization models vs control (improved).

Produces comparison charts, DRL learning curve, MCS fitness, and a Folium map.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

sns.set_theme(style="whitegrid", context="notebook")

MODEL_ORDER = [
    "control_civilian_time",
    "control_distance",
    "emv_dijkstra",
    "mipsstw_mcs",
    "composite_drl",
    "gbdt_router",
]
MAIN_MODELS = {"mipsstw_mcs", "composite_drl", "gbdt_router"}
MODEL_COLORS = {
    "control_distance": "#95a5a6",
    "control_civilian_time": "#7f8c8d",
    "emv_dijkstra": "#2980b9",
    "mipsstw_mcs": "#c0392b",
    "composite_drl": "#8e44ad",
    "gbdt_router": "#1e8449",
}
MODEL_LABELS = {
    "control_distance": "Control\nshortest dist.",
    "control_civilian_time": "Control\ncivilian time",
    "emv_dijkstra": "EMV\nDijkstra",
    "mipsstw_mcs": "MIPSSTW\n+ MCS",
    "composite_drl": "Composite\nDRL",
    "gbdt_router": "GBDT\nrouter",
}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["order"] = d["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return d.sort_values("order")


def plot_travel_time_bars(df: pd.DataFrame, out: Path):
    d = _ordered(df)
    labels = [MODEL_LABELS.get(m, m) for m in d["model"]]
    colors = [MODEL_COLORS.get(m, "#333") for m in d["model"]]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    bars = ax.bar(labels, d["travel_minutes"], color=colors, width=0.72, edgecolor="white")
    # Emphasize the 3 main models
    for bar, m in zip(bars, d["model"]):
        if m in MAIN_MODELS:
            bar.set_linewidth(2.0)
            bar.set_edgecolor("#111")
    best = d["travel_minutes"].min()
    ax.axhline(best, color="#111", ls="--", lw=1, alpha=0.5, label=f"best = {best:.2f} min")
    ax.set_ylabel("Travel time (minutes)")
    ax.set_title("Route optimization models — travel time (lower is better)")
    for i, (v, m) in enumerate(zip(d["travel_minutes"], d["model"])):
        if pd.notna(v):
            ax.text(i, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if m in MAIN_MODELS else "normal")
    ax.legend(loc="upper right")
    _save(fig, out)


def plot_main_models_focus(df: pd.DataFrame, out: Path):
    """Side-by-side: three main models vs civilian control only."""
    keep = ["control_civilian_time", "mipsstw_mcs", "composite_drl", "gbdt_router"]
    d = _ordered(df[df["model"].isin(keep)]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))

    labels = [MODEL_LABELS.get(m, m).replace("\n", " ") for m in d["model"]]
    colors = [MODEL_COLORS.get(m, "#333") for m in d["model"]]

    # Zoom y-axis so small EMV gains are visible
    ymin = float(d["travel_minutes"].min()) - 0.15
    ymax = float(d["travel_minutes"].max()) + 0.25
    axes[0].bar(labels, d["travel_minutes"], color=colors, edgecolor="white", width=0.7)
    axes[0].set_ylim(max(0, ymin), ymax)
    axes[0].set_ylabel("Minutes")
    axes[0].set_title("Travel time (zoomed)")
    axes[0].tick_params(axis="x", rotation=12)
    for i, (v, m) in enumerate(zip(d["travel_minutes"], d["model"])):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9,
                     fontweight="bold" if m == "composite_drl" else "normal")

    pct = d["pct_vs_civilian_control"].fillna(0.0)
    axes[1].bar(labels, pct, color=colors, edgecolor="white", width=0.7)
    axes[1].axhline(0, color="#333", lw=1)
    pad = max(0.5, float(np.nanmax(np.abs(pct))) * 0.35 + 0.3)
    axes[1].set_ylim(float(np.nanmin(pct)) - pad, float(np.nanmax(pct)) + pad)
    axes[1].set_ylabel("% vs civilian (+ = faster)")
    axes[1].set_title("Time vs civilian control")
    axes[1].tick_params(axis="x", rotation=12)
    for i, (v, m) in enumerate(zip(pct, d["model"])):
        axes[1].text(
            i,
            v + (0.08 if v >= 0 else -0.12),
            f"{v:+.2f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            fontweight="bold" if m == "composite_drl" else "normal",
        )

    fig.suptitle("Main testing models (slides Step 3)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_distance_bars(df: pd.DataFrame, out: Path):
    d = _ordered(df)
    labels = [MODEL_LABELS.get(m, m) for m in d["model"]]
    colors = [MODEL_COLORS.get(m, "#333") for m in d["model"]]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.bar(labels, d["distance_km"], color=colors, edgecolor="white")
    ax.set_ylabel("Path distance (km)")
    ax.set_title("Route length (hypothesis: longer path can still be faster)")
    for i, v in enumerate(d["distance_km"]):
        if pd.notna(v):
            ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    _save(fig, out)


def plot_time_vs_distance(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for _, r in _ordered(df).iterrows():
        m = r["model"]
        ax.scatter(
            r["distance_km"],
            r["travel_minutes"],
            s=220 if m in MAIN_MODELS else 120,
            color=MODEL_COLORS.get(m, "#333"),
            edgecolors="#111" if m in MAIN_MODELS else "white",
            linewidths=1.2,
            zorder=3,
            label=MODEL_LABELS.get(m, m).replace("\n", " "),
        )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Travel time (min)")
    ax.set_title("Time vs distance")
    ax.legend(fontsize=8, loc="best", framealpha=0.95)
    _save(fig, out)


def plot_mcs_fitness(history: list, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(history, color="#c0392b", lw=2.2)
    ax.fill_between(range(len(history)), history, alpha=0.12, color="#c0392b")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (seconds + soft TW)")
    ax.set_title("MIPSSTW + MCS — search convergence")
    _save(fig, out)


def plot_drl_curve(history: list, out: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    y = np.array(history, dtype=float)
    y_min = y.copy()
    y_min[~np.isfinite(y_min)] = np.nan
    y_plot = y_min / 60.0
    ax.plot(y_plot, color="#8e44ad", lw=1.4, alpha=0.75, label="episode travel time")
    finite = y_plot[np.isfinite(y_plot)]
    if len(finite) >= 5:
        # nan-safe moving average
        s = pd.Series(y_plot)
        smooth = s.rolling(7, min_periods=1, center=True).mean()
        ax.plot(smooth, color="#4a235a", lw=2.4, label="moving avg")
    if len(finite):
        ax.axhline(np.nanmin(finite), color="#111", ls="--", lw=1, label=f"best={np.nanmin(finite):.2f} min")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Travel time (min)")
    ax.set_title("Composite DRL — learning curve (corridor Q-learning)")
    ax.legend(fontsize=8, loc="best")
    _save(fig, out)


def plot_dashboard(df: pd.DataFrame, mcs_hist, drl_hist, out: Path):
    fig = plt.figure(figsize=(12.5, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    d = _ordered(df)
    labels = [MODEL_LABELS.get(m, m) for m in d["model"]]
    colors = [MODEL_COLORS.get(m, "#333") for m in d["model"]]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(labels, d["travel_minutes"], color=colors)
    ax0.set_title("Travel time (min)")
    ax0.tick_params(axis="x", labelsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(labels, d["pct_vs_civilian_control"], color=colors)
    ax1.axhline(0, color="#333", lw=1)
    ax1.set_title("% vs civilian control")
    ax1.tick_params(axis="x", labelsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    if mcs_hist:
        ax2.plot(mcs_hist, color="#c0392b", lw=2)
        ax2.set_title("MIPSSTW+MCS fitness")
        ax2.set_xlabel("Iteration")
    else:
        ax2.text(0.5, 0.5, "No MCS history", ha="center", va="center")
        ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[1, 1])
    if drl_hist:
        y = np.array(drl_hist, dtype=float) / 60.0
        ax3.plot(y, color="#8e44ad", lw=1.5)
        ax3.set_title("Composite DRL learning")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Minutes")
    else:
        ax3.text(0.5, 0.5, "No DRL history", ha="center", va="center")
        ax3.set_axis_off()

    fig.suptitle("EMV route models — overview", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


# Diverse NYC OD scenarios for the multi-route map (EMS station-ish → neighborhoods)
MAP_SCENARIOS = [
    {
        "id": "midtown_to_civic",
        "label": "Midtown → Civic Center",
        "origin": {"lat": 40.7580, "lon": -73.9855},
        "dest": {"lat": 40.7115, "lon": -74.0060},
    },
    {
        "id": "bronx_to_harlem",
        "label": "Bronx EMS → Harlem",
        "origin": {"lat": 40.83487, "lon": -73.92797},
        "dest": {"lat": 40.8116, "lon": -73.9465},
    },
    {
        "id": "brooklyn_to_downtown",
        "label": "Brooklyn EMS → Downtown Bk",
        "origin": {"lat": 40.67835, "lon": -73.99022},
        "dest": {"lat": 40.6920, "lon": -73.9870},
    },
    {
        "id": "queens_to_flushing",
        "label": "Queens corridor → Flushing",
        "origin": {"lat": 40.7465, "lon": -73.8910},
        "dest": {"lat": 40.7620, "lon": -73.8300},
    },
    {
        "id": "uli_west_to_east",
        "label": "Lower West Side → East Village",
        "origin": {"lat": 40.74958, "lon": -73.99985},
        "dest": {"lat": 40.7265, "lon": -73.9815},
    },
]


def _path_latlons(G, path):
    coords = []
    for n in path:
        node = G.nodes[n]
        coords.append([float(node.get("y", node.get("lat"))), float(node.get("x", node.get("lon")))])
    return coords


def _offset_coords(coords, model_name: str, index: int):
    """Tiny lateral offset on intermediate points only (keep OD ends pinned)."""
    if len(coords) < 3:
        return coords
    offsets = {
        "control_civilian_time": 0.0,
        "mipsstw_mcs": 0.00005,
        "composite_drl": -0.00005,
        "gbdt_router": 0.00009,
        "emv_dijkstra": -0.00009,
    }
    dy = offsets.get(model_name, 0.0) + (index % 3) * 0.000012
    out = [coords[0]]
    for lat, lon in coords[1:-1]:
        out.append([lat + dy, lon + dy * 0.55])
    out.append(coords[-1])
    return out


def _node_latlon(G, node):
    data = G.nodes[node]
    return [float(data.get("y", data.get("lat"))), float(data.get("x", data.get("lon")))]


def build_multi_route_map(
    graph_path: Path,
    out: Path,
    *,
    hour: int = 17,
    scenarios: list | None = None,
    n_scenarios: int = 5,
) -> Path | None:
    """
    Draw several OD trips, each with the main route models as separate polylines.
    Layer control: toggle per trip and per model family.
    """
    try:
        import folium
        from branca.element import Element
    except ImportError:
        print("folium not installed; skipping map")
        return None

    from emvro.routing import (
        control_civilian_time,
        nearest_node,
        prepare_routing_graph,
        solve_composite_drl,
        solve_gbdt_route,
        solve_mipsstw_mcs,
    )
    from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model

    scenarios = (scenarios or MAP_SCENARIOS)[:n_scenarios]
    print(f"Building multi-route map for {len(scenarios)} OD scenarios…")
    G = prepare_routing_graph(graph_path, hour=hour)
    edge_df = build_edge_training_frame(G, hours=[hour], max_edges=4000)
    bundle = train_gbdt_edge_model(edge_df)

    m = folium.Map(location=[40.74, -73.95], zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri streets",
    ).add_to(m)

    model_layers = {
        "control_civilian_time": folium.FeatureGroup(name="Control: civilian time", show=True),
        "mipsstw_mcs": folium.FeatureGroup(name="MIPSSTW + MCS", show=True),
        "composite_drl": folium.FeatureGroup(name="Composite DRL", show=True),
        "gbdt_router": folium.FeatureGroup(name="GBDT router", show=True),
    }
    for fg in model_layers.values():
        fg.add_to(m)

    all_bounds = []
    trip_rows = []

    map_models = [
        ("control_civilian_time", dict(weight=3, opacity=0.45, dash="8 6")),
        ("mipsstw_mcs", dict(weight=6, opacity=0.9, dash=None)),
        ("composite_drl", dict(weight=6, opacity=0.95, dash=None)),
        ("gbdt_router", dict(weight=5, opacity=0.85, dash="2 8")),
    ]

    for i, sc in enumerate(scenarios):
        o, d = sc["origin"], sc["dest"]
        try:
            origin = nearest_node(G, o["lon"], o["lat"])
            dest = nearest_node(G, d["lon"], d["lat"])
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {sc['id']}: {exc}")
            continue

        o_ll = _node_latlon(G, origin)
        d_ll = _node_latlon(G, dest)

        trip_fg = folium.FeatureGroup(name=f"Trip {i+1}: {sc['label']}", show=True)
        trip_fg.add_to(m)

        solved = {
            "control_civilian_time": control_civilian_time(G, origin, dest),
            "mipsstw_mcs": solve_mipsstw_mcs(
                G, origin, dest, n_iterations=10, n_nests=8, seed=42 + i
            ),
            "composite_drl": solve_composite_drl(G, origin, dest, episodes=50, seed=42 + i),
            "gbdt_router": solve_gbdt_route(G, origin, dest, bundle=bundle, hour=hour),
        }

        # Markers snapped to the actual routed graph nodes (not raw scenario coords)
        folium.CircleMarker(
            o_ll,
            radius=8,
            color="#1f4e79",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.95,
            popup=f"<b>Origin {i+1}</b><br>{sc['label']}",
            tooltip=f"O{i+1}: {sc['label']}",
        ).add_to(trip_fg)
        folium.CircleMarker(
            d_ll,
            radius=8,
            color="#b35900",
            fill=True,
            fill_color="#e67e22",
            fill_opacity=0.95,
            popup=f"<b>Dest {i+1}</b><br>{sc['label']}",
            tooltip=f"D{i+1}: {sc['label']}",
        ).add_to(trip_fg)
        folium.Marker(
            [(o_ll[0] + d_ll[0]) / 2, (o_ll[1] + d_ll[1]) / 2],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font:11px/1.2 sans-serif;background:rgba(255,255,255,.92);'
                    f'padding:2px 5px;border-radius:4px;border:1px solid #999;white-space:nowrap;">'
                    f"Trip {i+1}</div>"
                )
            ),
        ).add_to(trip_fg)

        for name, style in map_models:
            result = solved[name]
            if not result.ok:
                continue
            if not result.node_path or result.node_path[0] != origin or result.node_path[-1] != dest:
                print(f"  warn: {name} on {sc['id']} incomplete — forcing full-graph repair")
                from emvro.routing.composite_drl import annotate_composite_weights
                import networkx as nx

                annotate_composite_weights(G)
                try:
                    result.node_path = list(
                        nx.shortest_path(G, origin, dest, weight="weight_composite")
                    )
                    from emvro.routing.graph import path_stats

                    t, dist, n_e = path_stats(G, result.node_path)
                    result.travel_seconds, result.distance_m, result.n_edges = t, dist, n_e
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip draw {name}/{sc['id']}: {exc}")
                    continue

            coords = _path_latlons(G, result.node_path)
            if len(coords) < 2:
                continue
            # Pin ends exactly to OD markers, offset only the middle
            coords = _offset_coords(coords, name, i)
            coords[0] = list(o_ll)
            coords[-1] = list(d_ll)
            all_bounds.extend(coords)
            mins = result.travel_seconds / 60.0
            km = result.distance_m / 1000.0

            popup = (
                f"<b>Trip {i+1}: {sc['label']}</b><br>"
                f"{MODEL_LABELS.get(name, name).replace(chr(10), ' ')}<br>"
                f"{mins:.2f} min · {km:.2f} km · {result.n_edges} edges<br>"
                f"reaches dest: yes"
            )
            tip = (
                f"Trip {i+1}: "
                f"{MODEL_LABELS.get(name, name).replace(chr(10), ' ')} ({mins:.1f} min)"
            )
            # Folium/Leaflet: a layer can only have one parent. Duplicate the
            # polyline so model toggles AND trip toggles both work.
            for parent in (model_layers[name], trip_fg):
                folium.PolyLine(
                    coords,
                    color=MODEL_COLORS.get(name, "#333"),
                    weight=style["weight"],
                    opacity=style["opacity"],
                    dash_array=style["dash"],
                    popup=popup,
                    tooltip=tip,
                ).add_to(parent)

            # Explicit endpoint tick so a route never looks truncated
            if name == "composite_drl":
                folium.CircleMarker(
                    coords[-1],
                    radius=5,
                    color="#8e44ad",
                    fill=True,
                    fill_color="#8e44ad",
                    popup=f"DRL end — Trip {i+1}",
                ).add_to(trip_fg)

            trip_rows.append(
                {
                    "trip": sc["id"],
                    "label": sc["label"],
                    "model": name,
                    "travel_minutes": round(mins, 2),
                    "distance_km": round(km, 3),
                    "n_edges": result.n_edges,
                    "reaches_dest": True,
                    "end_lat": coords[-1][0],
                    "end_lon": coords[-1][1],
                }
            )
            if sc["id"] == "queens_to_flushing" and name == "composite_drl":
                print(
                    f"  trip4 DRL: {mins:.2f} min, {len(coords)} verts, "
                    f"end=({coords[-1][0]:.5f},{coords[-1][1]:.5f}) dest=({d_ll[0]:.5f},{d_ll[1]:.5f})"
                )
        print(f"  routed {sc['id']}")

    if all_bounds:
        m.fit_bounds(all_bounds, padding=(30, 30))

    legend = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:rgba(255,255,255,0.96);
                padding:12px 14px;border:1px solid #999;border-radius:8px;font:13px/1.45 sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,.15);max-width:300px;">
      <div style="font-weight:700;margin-bottom:6px;">Multi-route map</div>
      <div>Toggle <b>Trip N</b> or model layers. For traffic/weather + Google-untakeable
           EMV corridors, open <b>route_conditions_map.html</b>.</div>
      <hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">
      <div><span style="color:#7f8c8d;">╌ ╌</span> Civilian control</div>
      <div><span style="color:#c0392b;font-weight:700;">━━</span> MIPSSTW + MCS</div>
      <div><span style="color:#8e44ad;font-weight:700;">━━</span> Composite DRL</div>
      <div><span style="color:#1e8449;font-weight:700;">- - -</span> GBDT router</div>
      <div style="margin-top:6px;"><span style="color:#3498db;">●</span> Origin &nbsp;
           <span style="color:#e67e22;">●</span> Destination</div>
    </div>
    """
    m.get_root().html.add_child(Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))

    if trip_rows:
        pd.DataFrame(trip_rows).to_csv(out.with_name("multi_route_times.csv"), index=False)
    return out


# OD focus for traffic/weather/ROW condition map (Google-untakeable corridors)
CONDITION_MAP_ODS = [
    MAP_SCENARIOS[0],  # Midtown → Civic (dense one-ways → contraflow)
    MAP_SCENARIOS[4],  # Lower West → East Village
]

# Hours precomputed for the time slider (every 2h keeps HTML + build fast)
CONDITION_MAP_HOURS = list(range(0, 24, 2))
CONDITION_MAP_WEATHERS = ["clear", "rain", "snow"]


def build_conditions_route_map(
    graph_path: Path,
    out: Path,
    *,
    hours: list[int] | None = None,
    weathers: list[str] | None = None,
    od_scenarios: list | None = None,
    default_hour: int = 17,
    default_weather: str = "clear",
) -> Path | None:
    """
    Interactive map: time slider + weather buttons swap the *entire* route set.

    Folium checkboxes are not used for conditions. A custom control shows one
    (hour, weather) state at a time — civilian vs EMV paths + gold EMV-only
    segments for that regime.
    """
    try:
        import folium
        from branca.element import Element
    except ImportError:
        print("folium not installed; skipping conditions map")
        return None

    import json as _json

    from emvro.routing import (
        apply_routing_conditions,
        conditions_at,
        control_civilian_time,
        nearest_node,
        prepare_routing_graph,
    )
    from emvro.routing.emv_corridors import count_corridor_edges, path_corridor_segments
    from emvro.routing.graph import dijkstra_route

    hours = sorted({int(h) % 24 for h in (hours or CONDITION_MAP_HOURS)})
    weathers = list(weathers or CONDITION_MAP_WEATHERS)
    od_scenarios = od_scenarios or CONDITION_MAP_ODS
    default_hour = int(default_hour) % 24
    if default_hour not in hours:
        default_hour = hours[min(range(len(hours)), key=lambda i: abs(hours[i] - 17))]
    default_weather = default_weather if default_weather in weathers else weathers[0]

    print(
        f"Building slider conditions map: {len(od_scenarios)} ODs × "
        f"{len(hours)} hours × {len(weathers)} weathers…"
    )

    # Load graph once; reweight per (hour, weather)
    G = prepare_routing_graph(graph_path, conditions=conditions_at(default_hour, default_weather))
    priv = G.graph.get("emv_privileges") or {}
    print(
        f"  privileges busway={priv.get('busway_edges')} "
        f"contraflow={priv.get('contraflow_edges_added')}"
    )

    # Resolve OD nodes once
    od_nodes = []
    for sc in od_scenarios:
        o, d = sc["origin"], sc["dest"]
        try:
            origin = nearest_node(G, o["lon"], o["lat"])
            dest = nearest_node(G, d["lon"], d["lat"])
            od_nodes.append(
                {
                    "sc": sc,
                    "origin": origin,
                    "dest": dest,
                    "o_ll": _node_latlon(G, origin),
                    "d_ll": _node_latlon(G, dest),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {sc['id']}: {exc}")

    m = folium.Map(location=[40.74, -73.98], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri streets",
    ).add_to(m)

    # Permanent OD markers
    markers_fg = folium.FeatureGroup(name="od_markers", show=True)
    markers_fg.add_to(m)
    for od in od_nodes:
        sc = od["sc"]
        folium.CircleMarker(
            od["o_ll"],
            radius=8,
            color="#1f4e79",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.95,
            popup=f"<b>Origin</b><br>{sc['label']}",
            tooltip=f"O: {sc['label']}",
        ).add_to(markers_fg)
        folium.CircleMarker(
            od["d_ll"],
            radius=8,
            color="#b35900",
            fill=True,
            fill_color="#e67e22",
            fill_opacity=0.95,
            popup=f"<b>Dest</b><br>{sc['label']}",
            tooltip=f"D: {sc['label']}",
        ).add_to(markers_fg)

    layer_js_names: dict[str, str] = {}
    meta_by_key: dict[str, dict] = {}
    all_bounds: list = []
    rows: list[dict] = []

    for weather in weathers:
        for hour in hours:
            cond = conditions_at(hour, weather)
            key = f"{hour:02d}_{weather}"
            apply_routing_conditions(G, conditions=cond)

            # show=False for all; JS will enable the default
            fg = folium.FeatureGroup(name=f"state_{key}", show=False)
            fg.add_to(m)
            layer_js_names[key] = fg.get_name()

            civ_mins = []
            emv_mins = []
            corr_counts = []

            for od in od_nodes:
                sc = od["sc"]
                origin, dest = od["origin"], od["dest"]
                o_ll, d_ll = od["o_ll"], od["d_ll"]

                civ = control_civilian_time(G, origin, dest)
                emv = dijkstra_route(
                    G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"
                )

                for name, result, color, weight, dash, opacity in [
                    ("civilian_gps", civ, "#7f8c8d", 4, "10 8", 0.65),
                    ("emv_row", emv, "#8e44ad", 6, None, 0.95),
                ]:
                    if not result.ok or len(result.node_path) < 2:
                        continue
                    coords = _path_latlons(G, result.node_path)
                    coords[0] = list(o_ll)
                    coords[-1] = list(d_ll)
                    all_bounds.extend(coords)
                    mins = result.travel_seconds / 60.0
                    n_corr = count_corridor_edges(G, result.node_path)
                    if name == "civilian_gps":
                        civ_mins.append(mins)
                    else:
                        emv_mins.append(mins)
                        corr_counts.append(n_corr)

                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=weight,
                        opacity=opacity,
                        dash_array=dash,
                        popup=(
                            f"<b>{sc['label']}</b><br>{cond.label}<br>{name}<br>"
                            f"{mins:.2f} min · EMV-only edges: {n_corr}"
                        ),
                        tooltip=f"{cond.label}: {name} ({mins:.1f} min)",
                    ).add_to(fg)

                    if name == "emv_row":
                        for seg in path_corridor_segments(G, result.node_path):
                            folium.PolyLine(
                                seg["coords"],
                                color="#f1c40f",
                                weight=8,
                                opacity=0.85,
                                popup=(
                                    f"<b>EMV-only ({seg['kind']})</b><br>"
                                    f"{sc['label']} · {cond.label}<br>"
                                    "Civilian / Google Maps will not route here"
                                ),
                                tooltip=f"EMV-only: {seg['kind']}",
                            ).add_to(fg)

                    rows.append(
                        {
                            "trip": sc["id"],
                            "label": sc["label"],
                            "hour": hour,
                            "weather": weather,
                            "condition_label": cond.label,
                            "model": name,
                            "travel_minutes": round(mins, 2),
                            "n_edges": result.n_edges,
                            "emv_only_edges": n_corr,
                            "congestion": cond.congestion,
                            "weather_factor_civilian": round(
                                cond.weather_factor_civilian(), 4
                            ),
                        }
                    )

            civ_avg = float(np.mean(civ_mins)) if civ_mins else None
            emv_avg = float(np.mean(emv_mins)) if emv_mins else None
            saved = None
            pct = None
            if civ_avg and emv_avg and civ_avg > 0:
                saved = civ_avg - emv_avg
                pct = 100.0 * saved / civ_avg
            meta_by_key[key] = {
                "hour": hour,
                "weather": weather,
                "label": cond.label,
                "congestion": round(cond.congestion, 3),
                "wx_civ": round(cond.weather_factor_civilian(), 3),
                "civ_min": None if civ_avg is None else round(civ_avg, 2),
                "emv_min": None if emv_avg is None else round(emv_avg, 2),
                "saved_min": None if saved is None else round(saved, 2),
                "pct_faster": None if pct is None else round(pct, 1),
                "emv_only_edges": int(np.mean(corr_counts)) if corr_counts else 0,
            }
            if hour % 6 == 0 and weather == "clear":
                print(f"  routed hour={hour:02d} weather={weather}")

    if all_bounds:
        m.fit_bounds(all_bounds, padding=(28, 28))

    # Custom UI: time slider + weather (exclusive state switching)
    hours_js = _json.dumps(hours)
    layers_js = _json.dumps(layer_js_names)
    meta_js = _json.dumps(meta_by_key)
    default_key = f"{default_hour:02d}_{default_weather}"

    controls = f"""
    <div id="emv-cond-panel" style="position:fixed;top:16px;left:16px;z-index:9999;
         background:rgba(255,255,255,0.97);padding:14px 16px;border:1px solid #888;
         border-radius:10px;font:13px/1.4 system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.18);
         width:320px;max-width:92vw;">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px;">Time &amp; conditions</div>
      <div style="color:#555;margin-bottom:10px;font-size:12px;">
        Drag the hour slider or pick weather — routes <b>replace</b> for that regime
        (not model on/off toggles).
      </div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <label for="emv-hour" style="font-weight:600;">Hour of day</label>
        <span id="emv-hour-label" style="font-variant-numeric:tabular-nums;font-weight:700;">
          {default_hour:02d}:00
        </span>
      </div>
      <input id="emv-hour" type="range" min="0" max="23" step="2" value="{default_hour}"
             style="width:100%;margin:6px 0 12px 0;" />
      <div style="font-weight:600;margin-bottom:6px;">Weather</div>
      <div id="emv-wx-btns" style="display:flex;gap:6px;margin-bottom:12px;">
        <button type="button" data-wx="clear" class="emv-wx">Clear</button>
        <button type="button" data-wx="rain" class="emv-wx">Rain</button>
        <button type="button" data-wx="snow" class="emv-wx">Snow</button>
      </div>
      <div id="emv-stats" style="background:#f4f6f8;border-radius:6px;padding:8px 10px;
           font-size:12px;border:1px solid #dde2e8;"></div>
      <hr style="border:none;border-top:1px solid #ddd;margin:12px 0 8px;">
      <div style="font-size:12px;">
        <div><span style="color:#7f8c8d;">╌ ╌</span> Civilian GPS</div>
        <div><span style="color:#8e44ad;font-weight:700;">━━</span> EMV ROW route</div>
        <div><span style="color:#f1c40f;font-weight:700;">━━</span> Google-untakeable segment</div>
      </div>
    </div>
    <style>
      #emv-cond-panel .emv-wx {{
        flex:1; padding:7px 0; border:1px solid #bbb; border-radius:6px;
        background:#fff; cursor:pointer; font-weight:600;
      }}
      #emv-cond-panel .emv-wx.active {{
        background:#1f4e79; color:#fff; border-color:#1f4e79;
      }}
      #emv-cond-panel .emv-wx:hover {{ filter:brightness(0.97); }}
    </style>
    <script>
    (function() {{
      var HOURS = {hours_js};
      var LAYERS = {layers_js};
      var META = {meta_js};
      var weather = { _json.dumps(default_weather) };
      var mapRef = null;

      function nearestHour(h) {{
        var best = HOURS[0], bestD = 99;
        for (var i = 0; i < HOURS.length; i++) {{
          var d = Math.abs(HOURS[i] - h);
          if (d < bestD) {{ bestD = d; best = HOURS[i]; }}
        }}
        return best;
      }}

      function keyFor(h, w) {{
        var hh = nearestHour(h);
        return (hh < 10 ? "0" + hh : "" + hh) + "_" + w;
      }}

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

      function layerObj(name) {{
        return window[name] || null;
      }}

      function showState(h, w) {{
        var map = findMap();
        if (!map) return;
        var key = keyFor(h, w);
        Object.keys(LAYERS).forEach(function(k) {{
          var lyr = layerObj(LAYERS[k]);
          if (!lyr) return;
          if (map.hasLayer(lyr)) map.removeLayer(lyr);
        }});
        var active = layerObj(LAYERS[key]);
        if (active) map.addLayer(active);

        var lab = document.getElementById("emv-hour-label");
        if (lab) lab.textContent = (nearestHour(h) < 10 ? "0" : "") + nearestHour(h) + ":00";

        document.querySelectorAll("#emv-wx-btns .emv-wx").forEach(function(btn) {{
          btn.classList.toggle("active", btn.getAttribute("data-wx") === w);
        }});

        var st = META[key] || {{}};
        var el = document.getElementById("emv-stats");
        if (el) {{
          el.innerHTML =
            "<div style='font-weight:700;margin-bottom:4px;'>" + (st.label || key) + "</div>" +
            "<div>Congestion ×" + (st.congestion != null ? st.congestion : "—") +
            " · weather ×" + (st.wx_civ != null ? st.wx_civ : "—") + "</div>" +
            "<div style='margin-top:4px;'>Civilian <b>" + (st.civ_min != null ? st.civ_min + " min" : "—") +
            "</b> → EMV <b>" + (st.emv_min != null ? st.emv_min + " min" : "—") + "</b></div>" +
            "<div>Saved <b>" + (st.saved_min != null ? st.saved_min + " min" : "—") +
            "</b> (" + (st.pct_faster != null ? st.pct_faster + "%" : "—") +
            ") · EMV-only edges ~" + (st.emv_only_edges || 0) + "</div>";
        }}
      }}

      function bind() {{
        var slider = document.getElementById("emv-hour");
        if (!slider) return;
        slider.addEventListener("input", function() {{
          showState(parseInt(slider.value, 10), weather);
        }});
        document.querySelectorAll("#emv-wx-btns .emv-wx").forEach(function(btn) {{
          btn.addEventListener("click", function() {{
            weather = btn.getAttribute("data-wx");
            showState(parseInt(slider.value, 10), weather);
          }});
        }});
        showState({default_hour}, weather);
      }}

      // Folium finishes defining map_* after inline scripts; retry briefly.
      var tries = 0;
      (function waitMap() {{
        tries += 1;
        if (findMap() || tries > 40) bind();
        else setTimeout(waitMap, 50);
      }})();
    }})();
    </script>
    """
    m.get_root().html.add_child(Element(controls))

    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))

    if rows:
        pd.DataFrame(rows).to_csv(out.with_name("condition_route_map_times.csv"), index=False)

    # Sanity: default key exists
    print(f"  default state {default_key} layer={layer_js_names.get(default_key)}")
    print("  wrote", out)
    return out


def build_folium_paths(payload: dict, graph_path: Path, out: Path, routes: dict) -> Path | None:
    """Legacy single-OD map — prefer build_multi_route_map."""
    return build_multi_route_map(
        graph_path,
        out,
        hour=int(payload.get("hour") or 17),
        n_scenarios=5,
    )


def _run_all_routes(payload: dict, graph_path: Path):
    from emvro.routing import (
        control_civilian_time,
        control_shortest_distance,
        nearest_node,
        prepare_routing_graph,
        solve_composite_drl,
        solve_gbdt_route,
        solve_mipsstw_mcs,
    )
    from emvro.routing.graph import dijkstra_route
    from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model

    G = prepare_routing_graph(graph_path, hour=int(payload.get("hour") or 12))
    o, d = payload["origin"], payload["dest"]
    origin = nearest_node(G, o["lon"], o["lat"])
    dest = nearest_node(G, d["lon"], d["lat"])
    hour = int(payload.get("hour") or 12)
    deadline = payload.get("deadline_s")

    edge_df = build_edge_training_frame(G, hours=[hour], max_edges=5000)
    bundle = train_gbdt_edge_model(edge_df)

    routes = {
        "_G": G,
        "control_civilian_time": control_civilian_time(G, origin, dest),
        "control_distance": control_shortest_distance(G, origin, dest),
        "emv_dijkstra": dijkstra_route(G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"),
        "mipsstw_mcs": solve_mipsstw_mcs(
            G, origin, dest, deadline_s=deadline, n_iterations=18, n_nests=10, seed=42
        ),
        "composite_drl": solve_composite_drl(G, origin, dest, episodes=80, seed=42),
        "gbdt_router": solve_gbdt_route(G, origin, dest, bundle=bundle, hour=hour),
    }
    return routes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-json", type=Path, default=ROOT / "data" / "processed" / "route_models_demo.json")
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "route_models")
    p.add_argument("--skip-map", action="store_true")
    p.add_argument("--n-map-routes", type=int, default=5, help="How many OD trips to draw on the map")
    p.add_argument(
        "--skip-conditions-map",
        action="store_true",
        help="Skip traffic/weather/ROW conditions map",
    )
    p.add_argument("--rerun", action="store_true", help="Re-run routers and refresh demo JSON")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.graph.exists():
        raise SystemExit(f"Missing graph {args.graph}")

    # Always re-run for fresh DRL/MCS histories + improved routes when visualizing
    print("Re-running route models for visualization…")
    if args.demo_json.exists():
        payload = json.loads(args.demo_json.read_text())
    else:
        payload = {
            "origin": {"lat": 40.7580, "lon": -73.9855},
            "dest": {"lat": 40.7115, "lon": -74.0060},
            "hour": 17,
            "deadline_s": None,
        }

    routes = _run_all_routes(payload, args.graph)
    control_t = routes["control_civilian_time"].travel_seconds
    rows = []
    for name in MODEL_ORDER:
        r = routes[name]
        t = r.travel_seconds
        rows.append(
            {
                "model": name,
                "ok": r.ok,
                "travel_seconds": None if t != t else round(float(t), 1),
                "travel_minutes": None if t != t else round(float(t) / 60.0, 2),
                "distance_km": None if r.distance_m != r.distance_m else round(r.distance_m / 1000.0, 3),
                "n_edges": r.n_edges,
                "pct_vs_civilian_control": None
                if not (control_t == control_t and t == t and control_t > 0)
                else round(100.0 * (1.0 - float(t) / control_t), 2),
                "meta": {k: v for k, v in r.meta.items() if k not in {"train_curve", "fitness_history"}},
            }
        )
    payload["results"] = rows
    payload["deadline_s"] = payload.get("deadline_s")
    if args.rerun or True:
        # Persist refreshed demo numbers
        out_json = ROOT / "data" / "processed" / "route_models_demo.json"
        slim = dict(payload)
        out_json.write_text(json.dumps(slim, indent=2) + "\n")

    df = pd.DataFrame(rows)
    plot_travel_time_bars(df, args.out_dir / "01_travel_time_comparison.png")
    plot_distance_bars(df, args.out_dir / "02_distance_comparison.png")
    plot_main_models_focus(df, args.out_dir / "03_main_models_focus.png")
    plot_time_vs_distance(df, args.out_dir / "04_time_vs_distance.png")

    mcs_hist = routes["mipsstw_mcs"].meta.get("fitness_history") or []
    drl_hist = routes["composite_drl"].meta.get("train_curve") or []
    if mcs_hist:
        plot_mcs_fitness(mcs_hist, args.out_dir / "05_mcs_fitness.png")
    if drl_hist:
        plot_drl_curve(drl_hist, args.out_dir / "06_drl_learning_curve.png")
    plot_dashboard(df, mcs_hist, drl_hist, args.out_dir / "00_dashboard.png")

    map_path = None
    cond_map_path = None
    if not args.skip_map:
        map_path = build_multi_route_map(
            args.graph,
            args.out_dir / "route_models_map.html",
            hour=int(payload.get("hour") or 17),
            n_scenarios=args.n_map_routes,
        )
    if not args.skip_conditions_map:
        cond_map_path = build_conditions_route_map(
            args.graph,
            args.out_dir / "route_conditions_map.html",
        )

    summary = {
        "figures": sorted(p.name for p in args.out_dir.glob("*.png")),
        "map": str(map_path) if map_path else None,
        "conditions_map": str(cond_map_path) if cond_map_path else None,
        "results": rows,
        "drl_meta": routes["composite_drl"].meta,
    }
    (args.out_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(rows)[["model", "travel_minutes", "distance_km", "pct_vs_civilian_control"]].to_string(index=False))
    print("Wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
