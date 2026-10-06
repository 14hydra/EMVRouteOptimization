#!/usr/bin/env python3
"""Build a London OSM drive graph for firetruck routing / race demos.

Default: central London bbox covering expand-plan sites around Westminster.
Use --full to build Greater London via Geofabrik PBF (emvro.road_distance).

  PYTHONPATH=src .venv/bin/python scripts/build_london_drive_graph.py
  PYTHONPATH=src .venv/bin/python scripts/build_london_drive_graph.py --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_central(out: Path, bbox: tuple[float, float, float, float]) -> Path:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.timeout = 180
    # osmnx 2.x: bbox = (west, south, east, north)
    print("Downloading central London drive network", bbox)
    G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True)
    G = ox.truncate.largest_component(G, strongly=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out)
    print(f"nodes={len(G.nodes)} edges={len(G.edges)} -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="Greater London via Geofabrik PBF")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output graphml (default: london_drive_central.graphml or london_drive.graphml)",
    )
    ap.add_argument("--west", type=float, default=-0.22)
    ap.add_argument("--south", type=float, default=51.48)
    ap.add_argument("--east", type=float, default=-0.05)
    ap.add_argument("--north", type=float, default=51.54)
    args = ap.parse_args()

    if args.full:
        from emvro.road_distance import GRAPH_PATH, load_or_build_graph

        out = args.out or GRAPH_PATH
        G = load_or_build_graph(out)
        print(f"full London nodes={len(G.nodes)} edges={len(G.edges)} -> {out}")
        return 0

    out = args.out or (ROOT / "data" / "raw" / "london" / "london_drive_central.graphml")
    build_central(out, (args.west, args.south, args.east, args.north))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
