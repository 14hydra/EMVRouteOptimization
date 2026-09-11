#!/usr/bin/env python3
"""
Visualize the 3 route-optimization models vs control.

Produces:
  - comparison bar charts (time / distance / % vs civilian)
  - Folium map with each model's path as a colored polyline
  - optional MCS fitness curve if present in demo JSON meta

Reads `data/processed/route_models_demo.json` by default (from run_route_models.py).
Can also re-run routing if --rerun is set.
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

# Display order + colors for the three main models + controls
MODEL_ORDER = [
    "control_distance",
    "control_civilian_time",
    "emv_dijkstra",
    "mipsstw_mcs",
    "composite_drl",
    "gbdt_router",
]
MODEL_COLORS = {
    "control_distance": "#7f8c8d",
    "control_civilian_time": "#95a5a6",
    "emv_dijkstra": "#3498db",
    "mipsstw_mcs": "#c0392b",
    "composite_drl": "#8e44ad",
    "gbdt_router": "#27ae60",
}
MODEL_LABELS = {
    "control_distance": "Control: shortest distance",
    "control_civilian_time": "Control: civilian time",
    "emv_dijkstra": "EMV Dijkstra",
    "mipsstw_mcs": "MIPSSTW + MCS",
    "composite_drl": "Composite DRL",
    "gbdt_router": "GBDT router",
}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_travel_time_bars(df: pd.DataFrame, out: Path):
    d = df.copy()
    d["label"] = d["model"].map(MODEL_LABELS).fillna(d["model"])
    d["order"] = d["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = [MODEL_COLORS.get(m, "#1f4e79") for m in d["model"]]
    ax.bar(d["label"], d["travel_minutes"], color=colors)
    ax.set_ylabel("Travel time (min)")
    ax.set_title("Route models vs control — travel time")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(d["travel_minutes"]):
        if pd.notna(v):
            ax.text(i, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out)


def plot_distance_bars(df: pd.DataFrame, out: Path):
    d = df.copy()
    d["label"] = d["model"].map(MODEL_LABELS).fillna(d["model"])
    d["order"] = d["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = [MODEL_COLORS.get(m, "#1f4e79") for m in d["model"]]
    ax.bar(d["label"], d["distance_km"], color=colors)
    ax.set_ylabel("Path distance (km)")
    ax.set_title("Route models vs control — distance")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(d["distance_km"]):
        if pd.notna(v):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out)


def plot_pct_saved(df: pd.DataFrame, out: Path):
    d = df.copy()
    d = d[d["model"] != "control_civilian_time"]
    d["label"] = d["model"].map(MODEL_LABELS).fillna(d["model"])
    d["order"] = d["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = [MODEL_COLORS.get(m, "#1f4e79") for m in d["model"]]
    ax.bar(d["label"], d["pct_vs_civilian_control"], color=colors)
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel("% time vs civilian control (+ = faster)")
    ax.set_title("Percent time saved vs civilian control")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out)


def plot_time_vs_distance(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for _, r in df.iterrows():
        ax.scatter(
            r["distance_km"],
            r["travel_minutes"],
            s=120,
            color=MODEL_COLORS.get(r["model"], "#1f4e79"),
            label=MODEL_LABELS.get(r["model"], r["model"]),
            zorder=3,
        )
        ax.annotate(
            MODEL_LABELS.get(r["model"], r["model"]).split(":")[-1].strip()[:18],
            (r["distance_km"], r["travel_minutes"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Travel time (min)")
    ax.set_title("Time vs distance (hypothesis: longer can be faster)")
    ax.legend(fontsize=8, loc="best")
    _save(fig, out)


def plot_mcs_fitness(payload: dict, out: Path):
    hist = None
    for row in payload.get("results", []):
        if row.get("model") == "mipsstw_mcs":
            # fitness_history may have been stripped; try re-load from full meta if present
            hist = (row.get("meta") or {}).get("fitness_history")
    if not hist:
        return False
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(hist, color="#c0392b", lw=2)
    ax.set_xlabel("MCS iteration")
    ax.set_ylabel("Best fitness (s + soft TW penalty)")
    ax.set_title("MIPSSTW + MCS — fitness convergence")
    _save(fig, out)
    return True


def build_folium_paths(payload: dict, graph_path: Path, out: Path) -> Path | None:
    """Draw each model path on a map (needs graph to decode node lat/lon)."""
    try:
        import folium
    except ImportError:
        print("folium not installed; skipping map")
        return None
    if not graph_path.exists():
        print("graph missing; skipping map")
        return None

    from emvro.routing.graph import prepare_routing_graph

    G = prepare_routing_graph(graph_path, hour=int(payload.get("hour") or 12))

    # Re-run paths so we have node sequences (demo JSON does not store full paths)
    from emvro.routing import (
        control_civilian_time,
        control_shortest_distance,
        nearest_node,
        solve_composite_drl,
        solve_gbdt_route,
        solve_mipsstw_mcs,
    )
    from emvro.routing.graph import dijkstra_route
    from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model

    o = payload["origin"]
    d = payload["dest"]
    origin = nearest_node(G, o["lon"], o["lat"])
    dest = nearest_node(G, d["lon"], d["lat"])
    hour = int(payload.get("hour") or 12)
    deadline = payload.get("deadline_s")

    edge_df = build_edge_training_frame(G, hours=[hour], max_edges=4000)
    bundle = train_gbdt_edge_model(edge_df)

    routes = {
        "control_civilian_time": control_civilian_time(G, origin, dest),
        "mipsstw_mcs": solve_mipsstw_mcs(
            G, origin, dest, deadline_s=deadline, n_iterations=8, n_nests=8
        ),
        "composite_drl": solve_composite_drl(G, origin, dest, episodes=10, seed=42),
        "gbdt_router": solve_gbdt_route(G, origin, dest, bundle=bundle, hour=hour),
        "emv_dijkstra": dijkstra_route(G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"),
        "control_distance": control_shortest_distance(G, origin, dest),
    }

    m = folium.Map(location=[o["lat"], d["lat"]], zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri streets",
    ).add_to(m)
    folium.Marker([o["lat"], o["lon"]], popup="Origin", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([d["lat"], d["lon"]], popup="Destination", icon=folium.Icon(color="orange")).add_to(m)

    for name, result in routes.items():
        if not result.ok:
            continue
        coords = []
        for n in result.node_path:
            node = G.nodes[n]
            coords.append([float(node.get("y", node.get("lat"))), float(node.get("x", node.get("lon")))])
        folium.PolyLine(
            coords,
            color=MODEL_COLORS.get(name, "#333"),
            weight=5 if name in {"mipsstw_mcs", "composite_drl", "gbdt_router"} else 3,
            opacity=0.85,
            popup=f"{MODEL_LABELS.get(name, name)} — {result.travel_seconds/60:.1f} min",
            tooltip=MODEL_LABELS.get(name, name),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    legend = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:rgba(255,255,255,0.95);
                padding:10px 12px;border:1px solid #999;border-radius:8px;font-size:12px;">
      <b>Route models</b><br>
      <span style="color:#c0392b;">━</span> MIPSSTW + MCS<br>
      <span style="color:#8e44ad;">━</span> Composite DRL<br>
      <span style="color:#27ae60;">━</span> GBDT router<br>
      <span style="color:#3498db;">━</span> EMV Dijkstra<br>
      <span style="color:#95a5a6;">━</span> Civilian control
    </div>
    """
    from branca.element import Element

    m.get_root().html.add_child(Element(legend))
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demo-json",
        type=Path,
        default=ROOT / "data" / "processed" / "route_models_demo.json",
    )
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "route_models")
    p.add_argument("--skip-map", action="store_true", help="Skip Folium path map (faster)")
    args = p.parse_args()

    if not args.demo_json.exists():
        raise SystemExit(
            f"Missing {args.demo_json}. Run: PYTHONPATH=src python scripts/run_route_models.py"
        )

    payload = json.loads(args.demo_json.read_text())
    df = pd.DataFrame(payload["results"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_travel_time_bars(df, args.out_dir / "01_travel_time_comparison.png")
    plot_distance_bars(df, args.out_dir / "02_distance_comparison.png")
    plot_pct_saved(df, args.out_dir / "03_pct_vs_civilian.png")
    plot_time_vs_distance(df, args.out_dir / "04_time_vs_distance.png")
    # Fitness history was stripped from demo JSON — re-run a short MCS for the curve
    from emvro.routing.graph import nearest_node, prepare_routing_graph
    from emvro.routing.mipsstw_mcs import solve_mipsstw_mcs

    if args.graph.exists():
        G = prepare_routing_graph(args.graph, hour=int(payload.get("hour") or 12))
        o, d = payload["origin"], payload["dest"]
        origin = nearest_node(G, o["lon"], o["lat"])
        dest = nearest_node(G, d["lon"], d["lat"])
        mcs = solve_mipsstw_mcs(
            G, origin, dest, deadline_s=payload.get("deadline_s"), n_iterations=20, n_nests=10
        )
        if mcs.meta.get("fitness_history"):
            plot_mcs_fitness(
                {"results": [{"model": "mipsstw_mcs", "meta": mcs.meta}]},
                args.out_dir / "05_mcs_fitness.png",
            )

    map_path = None
    if not args.skip_map:
        map_path = build_folium_paths(payload, args.graph, args.out_dir / "route_models_map.html")

    summary = {
        "figures": sorted(p.name for p in args.out_dir.glob("*.png")),
        "map": str(map_path) if map_path else None,
        "demo_json": str(args.demo_json),
        "models": df["model"].tolist(),
    }
    (args.out_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
