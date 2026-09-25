#!/usr/bin/env python3
"""Build OD pairs with inferred firetruck start locations; write samples for upload."""

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


def normalize_fdny_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Map FDNY Fire Incident Dispatch columns onto the shared OD incident schema."""
    out = df.copy()
    rename = {
        "starfire_incident_id": "incident_id",
        "incident_datetime": "incident_datetime",
        "incident_borough": "borough",
        "incident_travel_tm_seconds_qy": "travel_seconds",
        "incident_response_seconds_qy": "response_seconds",
        "dispatch_response_seconds_qy": "dispatch_wait_seconds",
        "zipcode": "zipcode",
        "incident_classification": "initial_call_type",
        "incident_classification_group": "final_call_type",
    }
    for src, dst in rename.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    # Do NOT put alarm_box_borough into dispatch_area: depots.dispatch_area_borough
    # treats the first letter as an EMS sector code (B→Bronx), which mis-routes
    # BROOKLYN → Bronx. Keep borough as the only borough prior for FDNY rows.
    if "alarm_box_number" in out.columns:
        out["dispatch_area"] = out["alarm_box_number"].astype(str)
    # Keep rows that actually rolled fire apparatus when quantity columns exist.
    eng = pd.to_numeric(out.get("engines_assigned_quantity"), errors="coerce")
    lad = pd.to_numeric(out.get("ladders_assigned_quantity"), errors="coerce")
    if eng is not None and lad is not None:
        apparatus = eng.fillna(0) + lad.fillna(0)
        if apparatus.gt(0).any():
            out = out.loc[apparatus.gt(0)].copy()
    # Valid travel-time filter when the CAD flag is present.
    if "valid_incident_rspns_time_indc" in out.columns:
        flag = out["valid_incident_rspns_time_indc"].astype(str).str.upper()
        out = out.loc[flag.isin(["Y", "YES", "TRUE", "1"])].copy()
    if "travel_seconds" in out.columns:
        out["travel_seconds"] = pd.to_numeric(out["travel_seconds"], errors="coerce")
        out = out.loc[out["travel_seconds"].between(30, 3600)].copy()
    return out


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
        help="static=firehouses; csl=alarm-box CSLs; hybrid=CSL if closer than firehouse",
    )
    p.add_argument(
        "--legacy-ems",
        action="store_true",
        help="Also allow EMS stations / hospital bays as static start candidates (off by default)",
    )
    args = p.parse_args()

    args.processed.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)

    incidents_path = args.raw / "fdny_incidents.csv"
    if not incidents_path.exists():
        raise SystemExit(
            f"Missing {incidents_path}. Run: python scripts/download_datasets.py --limit 5000"
        )
    incidents = normalize_fdny_incidents(pd.read_csv(incidents_path, low_memory=False))
    firehouses = pd.read_csv(args.raw / "fdny_firehouses.csv", low_memory=False)
    modzcta = pd.read_csv(args.raw / "modzcta.csv", low_memory=False)
    alarm_boxes = _maybe_csv(args.raw / "alarm_boxes.csv")

    ems_stations = None
    hospital_bays = None
    if args.legacy_ems:
        ems_stations = _maybe_csv(args.raw / "ems_stations.csv")
        hospital_bays = _maybe_csv(args.raw / "hospital_bays.csv")
        if hospital_bays is not None:
            from emvro.depots import filter_hospital_bays

            hospital_bays = filter_hospital_bays(hospital_bays)

    centroids = zip_centroids_from_modzcta(modzcta)
    centroids.to_csv(args.processed / "zip_centroids.csv", index=False)

    if alarm_boxes is not None and len(alarm_boxes):
        csl_points = build_csl_points(alarm_boxes, incidents)
    else:
        csl_points = build_csl_from_zip_centroids(centroids, incidents)
    csl_points.to_csv(args.processed / "synthetic_csls.csv", index=False)

    od = infer_start_locations(
        incidents,
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
            "incident_datetime",
            "borough",
            "dispatch_area",
            "zipcode",
            "travel_seconds",
            "response_seconds",
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

    firehouses.head(min(100, len(firehouses))).to_csv(
        args.samples / "fdny_firehouses_sample.csv", index=False
    )
    csl_points.head(min(200, len(csl_points))).to_csv(
        args.samples / "synthetic_csls_sample.csv", index=False
    )

    summary = {
        "scope": "firetrucks",
        "start_mode": args.start_mode,
        "fdny_incident_rows": int(len(incidents)),
        "od_rows": int(len(od)),
        "usable_od_rows": int(len(usable)),
        "zip_centroids": int(len(centroids)),
        "firehouses": int(len(firehouses)),
        "synthetic_csls": int(len(csl_points)),
        "legacy_ems_layers": bool(args.legacy_ems),
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
            "start_* are INFERRED. Default layers: fdny_firehouse preferred; "
            "synthetic_csl from alarm-box intersections volume-weighted by FDNY ZIP demand. "
            "hybrid mode uses CSL when closer than the best firehouse."
        ),
    }
    (args.processed / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    (args.samples / "DATASET_UPLOAD_NOTES.md").write_text(
        "# Sample upload pack (firetruck scope)\n\n"
        f"- `od_pairs_sample.csv`: {len(sample)} OD pairs (`start_mode={args.start_mode}`)\n"
        "- `fdny_firehouses_sample.csv`: FDNY firehouse depots\n"
        "- `synthetic_csls_sample.csv`: volume-weighted alarm-box intersection CSLs\n\n"
        "## Starting-location fix\n\n"
        "Public FDNY CAD has no unit GPS at assignment. We infer starts as:\n"
        "1. nearest FDNY firehouse in the incident borough\n"
        "2. in `hybrid` mode, a synthetic CSL (alarm-box intersection in a high-volume ZIP) "
        "wins when closer to the destination than the firehouse "
        "(proxy for an already-on-the-road apparatus)\n\n"
        "`start_source` / `depot_layer` / `start_mode` make the rule auditable per row.\n"
    )

    print(json.dumps(summary, indent=2))
    print("Wrote", out_all)
    print("Wrote", sample_path)


if __name__ == "__main__":
    main()
