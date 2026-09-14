#!/usr/bin/env python3
"""
Demo of the 3 slide route-optimization models vs Google Maps control.

  MODEL 1: MIPSSTW + MCS
  MODEL 2: Composite DRL
  MODEL 3: GBDT (LightGBM) edge-cost router
  CONTROL: Google Maps (primary) + OSM civilian time (secondary)

Example:
  PYTHONPATH=src python scripts/run_route_models.py \\
    --origin-lat 40.758 --origin-lon -73.985 \\
    --dest-lat 40.712 --dest-lon -74.006
"""

from __future__ import annotations

import argparse
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

from emvro.gmaps import load_api_key_from_dotenv  # noqa: E402
from emvro.routing import (  # noqa: E402
    control_civilian_time,
    control_google_maps,
    control_shortest_distance,
    nearest_node,
    prepare_routing_graph,
    solve_composite_drl,
    solve_gbdt_route,
    solve_mipsstw_mcs,
)
from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model  # noqa: E402
from emvro.routing.graph import dijkstra_route  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument("--origin-lat", type=float, default=40.7580)
    p.add_argument("--origin-lon", type=float, default=-73.9855)
    p.add_argument("--dest-lat", type=float, default=40.7115)
    p.add_argument("--dest-lon", type=float, default=-74.0060)
    p.add_argument("--hour", type=int, default=17)
    p.add_argument("--deadline-s", type=float, default=None, help="Soft time window for MIPSSTW")
    p.add_argument("--mcs-iters", type=int, default=18)
    p.add_argument("--drl-episodes", type=int, default=80)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "route_models_demo.json")
    args = p.parse_args()

    load_api_key_from_dotenv(ROOT / ".env")

    if not args.graph.exists():
        raise SystemExit(f"Missing graph {args.graph}; run scripts/build_osm_graph.py first")

    print("Loading routing graph…")
    G = prepare_routing_graph(args.graph, hour=args.hour)
    origin = nearest_node(G, args.origin_lon, args.origin_lat)
    dest = nearest_node(G, args.dest_lon, args.dest_lat)
    print(f"OD nodes: {origin} → {dest} (hour={args.hour})")

    emv_dij = dijkstra_route(G, origin, dest, weight="weight_emv", model_name="emv_dijkstra")
    deadline = args.deadline_s
    if deadline is None and emv_dij.ok:
        deadline = float(emv_dij.travel_seconds) * 1.05

    print("Training GBDT edge model (on-the-fly sample)…")
    edge_df = build_edge_training_frame(G, hours=[args.hour], max_edges=6000)
    gbdt_bundle = train_gbdt_edge_model(
        edge_df, out_path=ROOT / "data" / "processed" / "models" / "gbdt_edge_costs.joblib"
    )

    print("Querying Google Maps control…")
    gmaps_ctrl = control_google_maps(
        args.origin_lat,
        args.origin_lon,
        args.dest_lat,
        args.dest_lon,
        cache_path=ROOT / "data" / "processed" / "gmaps_cache.json",
    )

    print("Running models…")
    results = {
        "control_google_maps": gmaps_ctrl,
        "control_civilian_time": control_civilian_time(G, origin, dest),
        "control_distance": control_shortest_distance(G, origin, dest),
        "emv_dijkstra": emv_dij,
        "mipsstw_mcs": solve_mipsstw_mcs(
            G,
            origin,
            dest,
            deadline_s=deadline,
            n_iterations=args.mcs_iters,
            n_nests=10,
        ),
        "composite_drl": solve_composite_drl(
            G, origin, dest, episodes=args.drl_episodes, seed=42
        ),
        "gbdt_router": solve_gbdt_route(G, origin, dest, bundle=gbdt_bundle, hour=args.hour),
    }

    rows = []
    for name, r in results.items():
        rows.append(
            {
                "model": name,
                "ok": r.ok,
                "travel_seconds": None if r.travel_seconds != r.travel_seconds else round(r.travel_seconds, 1),
                "travel_minutes": None
                if r.travel_seconds != r.travel_seconds
                else round(r.travel_seconds / 60.0, 2),
                "distance_km": None if r.distance_m != r.distance_m else round(r.distance_m / 1000.0, 3),
                "n_edges": r.n_edges,
                "meta": {
                    k: v
                    for k, v in r.meta.items()
                    if k not in {"fitness_history", "train_curve", "polyline_latlons"}
                },
            }
        )

    control_t = results["control_google_maps"].travel_seconds
    control_name = "control_google_maps"
    if control_t != control_t or not results["control_google_maps"].ok:
        control_t = results["control_civilian_time"].travel_seconds
        control_name = "control_civilian_time"

    for row in rows:
        t = row["travel_seconds"]
        if control_t == control_t and t is not None and control_t > 0:
            row["pct_vs_gmaps_control"] = round(100.0 * (1.0 - t / control_t), 2)
            row["pct_vs_civilian_control"] = row["pct_vs_gmaps_control"]
        else:
            row["pct_vs_gmaps_control"] = None
            row["pct_vs_civilian_control"] = None

    payload = {
        "origin": {"lat": args.origin_lat, "lon": args.origin_lon, "node": int(origin)},
        "dest": {"lat": args.dest_lat, "lon": args.dest_lon, "node": int(dest)},
        "hour": args.hour,
        "deadline_s": deadline,
        "primary_control": control_name,
        "results": rows,
        "note": (
            "Primary civilian control = Google Maps Directions (traffic-aware). "
            "OSM civilian time kept as secondary baseline. "
            "EMV models optimize ROW-aware edge costs (contraflow / busway)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    df = pd.DataFrame(rows)
    print(df[["model", "travel_minutes", "distance_km", "n_edges", "pct_vs_gmaps_control"]].to_string(index=False))
    print("Primary control:", control_name)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
