#!/usr/bin/env python3
"""
Build the ambulance travel-time training table (slides Step 4).

Uses real EMS travel seconds as labels. Because CAD does not publish unit GPS,
features include multi-candidate ambulance origins (EMS station / hospital bay /
CSL) instead of a single assumed start. Fire trucks / firehouses are excluded.

Upgrades layered onto the base table:
  - hourly weather (Open-Meteo)
  - OSM path aggregates along the primary inferred OD (cached graphml)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.csl import build_csl_points  # noqa: E402
from emvro.depots import filter_hospital_bays  # noqa: E402
from emvro.geometry import zip_centroids_from_modzcta  # noqa: E402
from emvro.street_features import attach_street_features  # noqa: E402
from emvro.travel_time import build_travel_time_feature_table  # noqa: E402
from emvro.weather import join_weather_features  # noqa: E402


def _maybe(path: Path):
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    p.add_argument(
        "--osm-graph",
        type=Path,
        default=ROOT / "data" / "raw" / "nyc_drive.graphml",
    )
    p.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip OSM path features (weather + geometry only)",
    )
    p.add_argument(
        "--osm-limit",
        type=int,
        default=None,
        help="Optional cap on OSM routed rows (smoke tests)",
    )
    args = p.parse_args()
    args.processed.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)

    incidents = pd.read_csv(args.raw / "ems_incidents.csv", low_memory=False)
    modzcta = pd.read_csv(args.raw / "modzcta.csv", low_memory=False)
    centroids = zip_centroids_from_modzcta(modzcta)

    stations = _maybe(args.raw / "ems_stations.csv")
    hospitals = _maybe(args.raw / "hospital_bays.csv")
    if hospitals is not None:
        hospitals = filter_hospital_bays(hospitals)
    alarm_boxes = _maybe(args.raw / "alarm_boxes.csv")
    if alarm_boxes is not None and len(alarm_boxes):
        csls = build_csl_points(alarm_boxes, incidents)
    else:
        csls = _maybe(args.processed / "synthetic_csls.csv")

    od = _maybe(args.processed / "od_pairs_inferred_starts.csv")
    if od is not None and "depot_layer" in od.columns:
        od = od[
            od["depot_layer"].isin(["ems_station", "hospital_bay", "csl"])
            | od["depot_layer"].isna()
        ]

    feats = build_travel_time_feature_table(
        incidents,
        centroids,
        ems_stations=stations,
        hospital_bays=hospitals,
        csl_points=csls,
        od_pairs=od,
    )

    # Weather join on incident hour
    wx_cache = args.processed / "weather_hourly_nyc.csv"
    feats = join_weather_features(feats, cache_path=wx_cache)

    osm_ok = 0
    if not args.skip_osm and args.osm_graph.exists():
        feats = attach_street_features(
            feats, args.osm_graph, limit=args.osm_limit, show_progress=True
        )
        osm_ok = int((feats.get("osm_route_ok", 0) == 1).sum()) if "osm_route_ok" in feats else 0
    elif not args.skip_osm:
        print(
            f"OSM graph missing at {args.osm_graph}; "
            "run scripts/build_osm_graph.py (continuing without path features)"
        )

    out = args.processed / "travel_time_training.csv"
    feats.to_csv(out, index=False)
    feats.head(min(200, len(feats))).to_csv(
        args.samples / "travel_time_training_sample.csv", index=False
    )

    hour_counts = (
        feats["hour"].value_counts().sort_index().to_dict() if "hour" in feats.columns else {}
    )
    summary = {
        "scope": "ambulances_only",
        "label": "travel_seconds (EMS assignment → on-scene)",
        "rows": int(len(feats)),
        "median_travel_s": float(feats["travel_seconds"].median()) if len(feats) else None,
        "hour_coverage": {str(k): int(v) for k, v in hour_counts.items()},
        "primary_origin_layer_counts": feats["primary_origin_layer"].value_counts().to_dict()
        if len(feats)
        else {},
        "weather_nonnull": int(feats["wx_temp_c"].notna().sum())
        if "wx_temp_c" in feats.columns
        else 0,
        "osm_routes_ok": osm_ok,
        "note": (
            "Origins uncertain (multi-distance features). Weather from Open-Meteo. "
            "OSM path aggregates when nyc_drive.graphml is present. Firehouses excluded."
        ),
        "output": str(out),
    }
    (args.processed / "travel_time_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
