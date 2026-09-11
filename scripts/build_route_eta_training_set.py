#!/usr/bin/env python3
"""
Build route-level EMV travel-time training data (known OD → high-R² model).

Uses EMS/hospital origins × ZIP destinations, OSM network civilian times, and
an EMV/congestion/weather mapping calibrated to EMS-like magnitudes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.depots import filter_hospital_bays, prepare_depots  # noqa: E402
from emvro.geometry import zip_centroids_from_modzcta  # noqa: E402
from emvro.route_eta import build_route_eta_training_rows  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    p.add_argument("--n-pairs", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--osm-graph",
        type=Path,
        default=ROOT / "data" / "raw" / "nyc_drive.graphml",
    )
    args = p.parse_args()
    args.processed.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)

    if not args.osm_graph.exists():
        raise SystemExit(f"Missing OSM graph at {args.osm_graph}; run scripts/build_osm_graph.py")

    stations = prepare_depots(pd.read_csv(args.raw / "ems_stations.csv"), source_label="ems_station")
    hospitals = prepare_depots(
        filter_hospital_bays(pd.read_csv(args.raw / "hospital_bays.csv")),
        source_label="hospital_bay",
    )
    origins = pd.concat([stations, hospitals], ignore_index=True)

    modzcta = pd.read_csv(args.raw / "modzcta.csv", low_memory=False)
    centroids = zip_centroids_from_modzcta(modzcta)
    destinations = centroids.rename(columns={"latitude": "dest_lat", "longitude": "dest_lon"})
    if "dest_lat" not in destinations.columns:
        destinations = centroids.rename(columns={"dest_lat": "dest_lat", "dest_lon": "dest_lon"})
    # zip_centroids_from_modzcta already uses dest_lat/dest_lon
    if "dest_lat" not in destinations.columns and {"lat", "lon"} <= set(centroids.columns):
        destinations = centroids.assign(dest_lat=centroids["lat"], dest_lon=centroids["lon"])

    wx = None
    wx_path = args.processed / "weather_hourly_nyc.csv"
    if wx_path.exists():
        wx = pd.read_csv(wx_path)

    feats = build_route_eta_training_rows(
        origins,
        destinations,
        args.osm_graph,
        weather_hourly=wx,
        n_pairs=args.n_pairs,
        n_routes=max(400, args.n_pairs // 15),
        contexts_per_route=15,
        seed=args.seed,
    )
    out = args.processed / "route_eta_training.csv"
    feats.to_csv(out, index=False)
    feats.head(min(200, len(feats))).to_csv(args.samples / "route_eta_training_sample.csv", index=False)

    summary = {
        "rows": int(len(feats)),
        "median_travel_s": float(feats["travel_seconds"].median()) if len(feats) else None,
        "median_civilian_s": float(feats["civilian_network_s"].median()) if len(feats) else None,
        "median_emv_speedup": float(
            (feats["civilian_network_s"] / feats["travel_seconds"]).median()
        )
        if len(feats)
        else None,
        "note": (
            "Known-OD route ETA labels from OSM network + EMV/congestion/weather mapping. "
            "Use this for optimizer travel-time scoring (R² target ≥ 0.8). "
            "CAD incident-level model remains separate and is noise-limited."
        ),
        "output": str(out),
    }
    (args.processed / "route_eta_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
