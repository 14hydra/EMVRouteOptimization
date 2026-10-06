#!/usr/bin/env python3
"""
Scaffold: fit a per-segment speed model (width, lanes, maxspeed, road type, hour)
from an OSM drive graph. See emvro.segment_speeds for the Zhan et al.-style idea.

If data/raw/london/london_drive.graphml does not exist this prints what is
needed + a JSON stub plan and exits 0. If it exists (or --graph is given), it
samples OD node pairs, builds a *synthetic* route-time target from a Kolesar
curve on crow-fly distance (fit on LFB 2024 if available, else the London default), then
fits a ridge on the routes' summed edge features and reports the error. This is
a lightweight plumbing check, NOT a calibrated model: replace the synthetic
target with observed LFB/GPS route times once a real graph + routes exist.

Run:  .venv/bin/python scripts/fit_segment_speeds.py [--graph PATH] [--n-od 400]
      .venv/bin/python scripts/fit_segment_speeds.py --selftest   # tiny synthetic grid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.kolesar import KolesarModel  # noqa: E402
from emvro.london_eval import haversine_km  # noqa: E402
from emvro.segment_speeds import (  # noqa: E402
    SegmentSpeedModel,
    feature_names,
    route_features,
    route_time_seconds,
)

DEFAULT_GRAPH = ROOT / "data" / "raw" / "london" / "london_drive.graphml"
OUT = ROOT / "data" / "processed" / "models" / "segment_speeds_london.json"


def stub_plan(graph_path: Path) -> dict:
    return {
        "status": "no_graph",
        "graph_expected_at": str(graph_path),
        "needed": [
            "Build an OSM drive graph for Greater London (bbox -0.50,51.28,0.35,51.70), e.g. "
            "osmnx.graph_from_bbox(..., network_type='drive') with useful_tags_way extended by "
            "width, lanes, maxspeed, highway; save to the path above (graphml).",
            "Route-level targets: LFB start station -> incident OD with observed drive seconds "
            "(data/raw/london/lfb_incidents_planner.csv minus ~60 s turnout).",
            "Then run the iterative Dijkstra re-route fit (emvro.segment_speeds.iterative_reroute_fit).",
        ],
        "model": "SegmentSpeedModel: slowness(s/km) = w . [1, 3600/maxspeed, lanes, width, road_type onehots, rush, sin/cos hour]",
        "features": feature_names(),
    }


def tiny_grid():
    import networkx as nx

    rng = np.random.default_rng(0)
    G = nx.MultiDiGraph()
    n = 12
    for i in range(n):
        for j in range(n):
            G.add_node((i, j), y=51.45 + 0.005 * i, x=-0.20 + 0.008 * j)
    kinds = [("primary", "48 km/h", 3), ("residential", "20 mph", 1), ("secondary", "30 mph", 2)]
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b not in G:
                    continue
                hw, ms, lanes = kinds[rng.integers(len(kinds))]
                length = float(haversine_km(G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"]) * 1000)
                for u, v in ((a, b), (b, a)):
                    G.add_edge(u, v, 0, length=length, highway=hw, maxspeed=ms, lanes=str(lanes))
    return G


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--n-od", type=int, default=400)
    ap.add_argument("--hour", type=int, default=14)
    ap.add_argument("--selftest", action="store_true", help="use a tiny synthetic grid instead of a graph file")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.selftest:
        G = tiny_grid()
    elif not args.graph.exists():
        plan = stub_plan(args.graph)
        print(json.dumps(plan, indent=2))
        print("\nNo London graph found; nothing fitted (exit 0).")
        return 0
    else:
        import osmnx as ox

        G = ox.load_graphml(args.graph)

    import networkx as nx

    # Kolesar target curve (LFB-fit if possible else the London default)
    try:
        from emvro.kolesar import fit_kolesar
        from emvro.london_eval import load_london

        _, inc = load_london(years=[2024])
        kol = fit_kolesar(inc["crow_km"], inc["drive_s"])
        src = "kolesar fit on LFB 2024 drive_s"
    except Exception:  # noqa: BLE001
        kol, src = KolesarModel.london_default(), "London default curve"

    rng = np.random.default_rng(42)
    nodes = list(G.nodes)
    feats, ys, km_list, crow_list = [], [], [], []
    base = SegmentSpeedModel()
    ws = "length"
    for _ in range(args.n_od * 3):
        if len(ys) >= args.n_od:
            break
        o, d = (nodes[i] for i in rng.choice(len(nodes), 2, replace=False))
        crow = float(haversine_km(G.nodes[o]["y"], G.nodes[o]["x"], G.nodes[d]["y"], G.nodes[d]["x"]))
        if crow < 0.3:
            continue
        try:
            p = nx.shortest_path(G, o, d, weight=ws)
        except nx.NetworkXNoPath:
            continue
        f, km = route_features(G, p, args.hour)
        feats.append(f); ys.append(float(kol.predict(crow))); km_list.append(km); crow_list.append(crow)
    if len(ys) < 20:
        print("Too few routable OD pairs; nothing fitted.")
        return 0
    X, y = np.array(feats), np.array(ys)
    idx = rng.permutation(len(y)); cut = int(0.8 * len(y)); tr, te = idx[:cut], idx[cut:]
    prior_mae = float(np.mean(np.abs(X[te] @ base.w - y[te])))
    model = SegmentSpeedModel().fit_routes(X[tr], y[tr])
    fit_mae = float(np.mean(np.abs(X[te] @ model.w - y[te])))
    example_path = nx.shortest_path(G, *[nodes[0], nodes[-1]], weight=ws)
    summary = {
        "status": "fitted_synthetic_target",
        "graph": "synthetic_grid" if args.selftest else str(args.graph),
        "target": f"synthetic: {src} on crow-fly km (plumbing check, not observed times)",
        "n_od": int(len(y)),
        "heldout_mae_s_prior": prior_mae,
        "heldout_mae_s_ridge": fit_mae,
        "mean_route_km": float(np.mean(km_list)),
        "example_route_seconds": route_time_seconds(G, example_path, model, args.hour),
        "model": model.to_dict(),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.selftest:
        args.out.write_text(json.dumps(summary, indent=2))
        print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
