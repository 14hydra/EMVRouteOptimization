#!/usr/bin/env python3
"""Build OD pairs with inferred EMV start locations; write samples for Gemini upload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.csl import build_csl_from_zip_centroids, build_csl_points  # noqa: E402
from emvro.depots import infer_start_locations  # noqa: E402
from emvro.geometry import zip_centroids_from_modzcta  # noqa: E402


def _maybe_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    p.add_argument("--sample-n", type=int, default=200)
    p.add_argument(
        "--start-mode",
        choices=["static", "csl", "hybrid"],
        default="hybrid",
        help="static=stations/hospitals/firehouses; csl=synthetic CSLs; hybrid=CSL if closer",
    )
    args = p.parse_args()

    args.processed.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)

    ems = pd.read_csv(args.raw / "ems_incidents.csv", low_memory=False)
    firehouses = pd.read_csv(args.raw / "fdny_firehouses.csv", low_memory=False)
    modzcta = pd.read_csv(args.raw / "modzcta.csv", low_memory=False)
    ems_stations = _maybe_csv(args.raw / "ems_stations.csv")
    hospital_bays = _maybe_csv(args.raw / "hospital_bays.csv")
    if hospital_bays is not None:
        from emvro.depots import filter_hospital_bays

        hospital_bays = filter_hospital_bays(hospital_bays)
    alarm_boxes = _maybe_csv(args.raw / "alarm_boxes.csv")

    centroids = zip_centroids_from_modzcta(modzcta)
    centroids.to_csv(args.processed / "zip_centroids.csv", index=False)

    if alarm_boxes is not None and len(alarm_boxes):
        csl_points = build_csl_points(alarm_boxes, ems)
    else:
        csl_points = build_csl_from_zip_centroids(centroids, ems)
    csl_points.to_csv(args.processed / "synthetic_csls.csv", index=False)

    od = infer_start_locations(
        ems,
        firehouses,
        centroids,
        ems_stations=ems_stations,
        hospital_bays=hospital_bays,
        csl_points=csl_points,
        start_mode=args.start_mode,
    )

    prefer_cols = [
        c
        for c in [
            "incident_id",
            "cad_incident_id",
            "incident_datetime",
            "borough",
            "dispatch_area",
            "incident_dispatch_area",
            "zipcode",
            "travel_seconds",
            "incident_travel_tm_seconds_qy",
            "response_seconds",
            "incident_response_seconds_qy",
            "start_lat",
            "start_lon",
            "dest_lat",
            "dest_lon",
            "depot_id",
            "depot_name",
            "depot_layer",
            "start_source",
            "start_mode",
            "crow_flies_km",
            "prefer_borough",
            "initial_call_type",
            "final_call_type",
        ]
        if c in od.columns
    ]
    other = [c for c in od.columns if c not in prefer_cols]
    od = od[prefer_cols + other]

    out_all = args.processed / "od_pairs_inferred_starts.csv"
    od.to_csv(out_all, index=False)

    usable = od.dropna(subset=["start_lat", "start_lon", "dest_lat", "dest_lon"])
    sample = usable.head(args.sample_n)
    sample_path = args.samples / "od_pairs_sample.csv"
    sample.to_csv(sample_path, index=False)

    # Sample packs for Gemini
    firehouses.head(min(100, len(firehouses))).to_csv(
        args.samples / "fdny_firehouses_sample.csv", index=False
    )
    if ems_stations is not None:
        ems_stations.to_csv(args.samples / "ems_stations_sample.csv", index=False)
    if hospital_bays is not None:
        hospital_bays.to_csv(args.samples / "hospital_bays_sample.csv", index=False)
    csl_points.head(min(200, len(csl_points))).to_csv(
        args.samples / "synthetic_csls_sample.csv", index=False
    )

    summary = {
        "start_mode": args.start_mode,
        "ems_rows": int(len(ems)),
        "od_rows": int(len(od)),
        "usable_od_rows": int(len(usable)),
        "zip_centroids": int(len(centroids)),
        "firehouses": int(len(firehouses)),
        "ems_stations": int(len(ems_stations)) if ems_stations is not None else 0,
        "hospital_bays": int(len(hospital_bays)) if hospital_bays is not None else 0,
        "synthetic_csls": int(len(csl_points)),
        "depot_layer_counts": usable["depot_layer"].value_counts().to_dict()
        if len(usable) and "depot_layer" in usable.columns
        else {},
        "start_mode_counts": usable["start_mode"].value_counts().to_dict()
        if len(usable) and "start_mode" in usable.columns
        else {},
        "start_source_counts": usable["start_source"].value_counts().head(20).to_dict()
        if len(usable)
        else {},
        "sample_path": str(sample_path),
        "note": (
            "start_* are INFERRED. Layers: ems_station + hospital_bay (H+H/voluntary) "
            "preferred; fdny_firehouse fallback; synthetic_csl from alarm-box intersections "
            "volume-weighted by EMS ZIP demand. hybrid mode uses CSL when closer than static."
        ),
    }
    (args.processed / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    (args.samples / "DATASET_UPLOAD_NOTES.md").write_text(
        "# Sample upload pack for Gemini ASI chat\n\n"
        f"- `od_pairs_sample.csv`: {len(sample)} OD pairs (`start_mode={args.start_mode}`)\n"
        "- `ems_stations_sample.csv`: dedicated EMS/ambulance stations\n"
        "- `hospital_bays_sample.csv`: HOSPITAL / ACUTE CARE HOSPITAL (H+H + voluntary)\n"
        "- `synthetic_csls_sample.csv`: volume-weighted alarm-box intersection CSLs\n"
        "- `fdny_firehouses_sample.csv`: firehouse fallback\n\n"
        "## Starting-location fix\n\n"
        "Public EMS CAD has no unit GPS at assignment. We infer starts as:\n"
        "1. nearest EMS station or hospital bay in the dispatch-area borough\n"
        "2. else nearest FDNY firehouse\n"
        "3. in `hybrid` mode, a synthetic CSL (alarm-box intersection in a high-volume ZIP) "
        "wins when closer to the destination than the best static depot "
        "(proxy for an already-on-the-road unit)\n\n"
        "`start_source` / `depot_layer` / `start_mode` make the rule auditable per row.\n"
    )

    print(json.dumps(summary, indent=2))
    print("Wrote", out_all)
    print("Wrote", sample_path)


if __name__ == "__main__":
    main()
