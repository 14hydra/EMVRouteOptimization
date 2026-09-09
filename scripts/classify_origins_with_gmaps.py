#!/usr/bin/env python3
"""
Use Google Maps as the civilian control model to label which EMS trips
could realistically have left an official depot vs must have been on-road.

For each OD row:
  1) Query Google Maps driving time from nearest static depot → destination
  2) Optionally from nearest CSL → destination
  3) Compare to observed incident_travel_tm_seconds_qy

If even an EMV-speedup of the station GMaps ETA is still slower than the
observed travel time, classify as on_road_required; else station_plausible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.depots import (  # noqa: E402
    _nearest_depot,
    combine_depot_layers,
    filter_hospital_bays,
)
from emvro.gmaps import GoogleMapsControl, classify_origin_with_gmaps  # noqa: E402


def _maybe_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _prepare_static_and_csl(raw: Path, processed: Path):
    firehouses = pd.read_csv(raw / "fdny_firehouses.csv", low_memory=False)
    ems_stations = _maybe_csv(raw / "ems_stations.csv")
    hospital_bays = _maybe_csv(raw / "hospital_bays.csv")
    if hospital_bays is not None:
        hospital_bays = filter_hospital_bays(hospital_bays)
    csl_points = _maybe_csv(processed / "synthetic_csls.csv")

    layers = [
        (ems_stations, "ems_station"),
        (hospital_bays, "hospital_bay"),
        (firehouses, "fdny_firehouse"),
    ]
    static = combine_depot_layers(layers)
    preferred = static[static["depot_layer"].isin(["ems_station", "hospital_bay"])]
    if preferred.empty:
        preferred = static
    fallback = static[static["depot_layer"] == "fdny_firehouse"]

    csls = None
    if csl_points is not None and len(csl_points):
        csls = combine_depot_layers([(csl_points, "csl")])
    return preferred, fallback, csls


def _nearest_in_boro(depots: pd.DataFrame, prefer_b, dest_lon, dest_lat):
    if prefer_b is not None:
        cand = depots[depots["borough"] == prefer_b]
        if len(cand):
            return _nearest_depot(cand, dest_lon, dest_lat)
    return _nearest_depot(depots, dest_lon, dest_lat)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--od", type=Path, default=ROOT / "data" / "processed" / "od_pairs_inferred_starts.csv")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    p.add_argument("--limit", type=int, default=200, help="Max rows to query (API cost control)")
    p.add_argument("--emv-speed-factor", type=float, default=0.75, help="EMV vs civilian speed (<1 faster)")
    p.add_argument("--slack", type=float, default=1.15, help="Tolerance multiplier on actual time")
    p.add_argument("--also-csl", action="store_true", default=True, help="Also query GMaps from nearest CSL")
    p.add_argument("--no-csl", action="store_true", help="Skip CSL GMaps queries")
    args = p.parse_args()
    also_csl = args.also_csl and not args.no_csl

    gmaps = GoogleMapsControl(cache_path=args.processed / "gmaps_cache.json")
    if not gmaps.available:
        print(
            "ERROR: GOOGLE_MAPS_API_KEY (or GMAPS_API_KEY) is not set.\n"
            "Create a Google Cloud key with Directions API enabled, then:\n"
            "  export GOOGLE_MAPS_API_KEY=your_key_here\n"
            "  PYTHONPATH=src python scripts/classify_origins_with_gmaps.py --limit 200"
        )
        sys.exit(2)

    od = pd.read_csv(args.od, low_memory=False)
    preferred, fallback, csls = _prepare_static_and_csl(args.raw, args.processed)

    # Prefer rows that have observed travel time + dest coords.
    travel_col = "travel_seconds" if "travel_seconds" in od.columns else "incident_travel_tm_seconds_qy"
    od[travel_col] = pd.to_numeric(od[travel_col], errors="coerce")
    usable = od.dropna(subset=[travel_col, "dest_lat", "dest_lon"]).copy()
    usable = usable[usable[travel_col] > 0].head(args.limit).reset_index(drop=True)

    preferred_by_boro = {b: g for b, g in preferred.groupby("borough") if b is not None}
    fallback_by_boro = {b: g for b, g in fallback.groupby("borough") if b is not None}

    rows = []
    for rec in tqdm(usable.itertuples(index=False), total=len(usable), desc="gmaps_control"):
        dest_lat = float(rec.dest_lat)
        dest_lon = float(rec.dest_lon)
        prefer_b = getattr(rec, "prefer_borough", None) or getattr(rec, "borough_norm", None)
        actual = float(getattr(rec, travel_col))

        # Nearest official depot (EMS/hospital preferred, else firehouse)
        cand = preferred_by_boro.get(prefer_b)
        if cand is None or not len(cand):
            cand = fallback_by_boro.get(prefer_b)
        if cand is None or not len(cand):
            cand = preferred if len(preferred) else fallback
        station, station_km = _nearest_depot(cand, dest_lon, dest_lat)

        st = gmaps.driving_seconds(station["start_lat"], station["start_lon"], dest_lat, dest_lon)
        gmaps_station_s = st.get("duration_in_traffic_s") or st.get("duration_s") if st.get("ok") else None

        gmaps_csl_s = None
        csl_name = None
        csl_km = None
        if also_csl and csls is not None and len(csls):
            csl, csl_km = _nearest_in_boro(csls, prefer_b, dest_lon, dest_lat)
            ct = gmaps.driving_seconds(csl["start_lat"], csl["start_lon"], dest_lat, dest_lon)
            gmaps_csl_s = ct.get("duration_in_traffic_s") or ct.get("duration_s") if ct.get("ok") else None
            csl_name = csl.get("depot_name")

        verdict = classify_origin_with_gmaps(
            actual,
            float(gmaps_station_s) if gmaps_station_s is not None else None,
            float(gmaps_csl_s) if gmaps_csl_s is not None else None,
            emv_speed_factor=args.emv_speed_factor,
            slack=args.slack,
        )

        rows.append(
            {
                "incident_id": getattr(rec, "incident_id", None),
                "borough": getattr(rec, "borough", None),
                "zipcode": getattr(rec, "zipcode", None),
                "prefer_borough": prefer_b,
                "actual_travel_s": actual,
                "station_name": station.get("depot_name"),
                "station_layer": station.get("depot_layer"),
                "station_km": station_km,
                "gmaps_station_s": gmaps_station_s,
                "gmaps_station_ok": st.get("ok"),
                "csl_name": csl_name,
                "csl_km": csl_km,
                "gmaps_csl_s": gmaps_csl_s,
                "origin_class": verdict["origin_class"],
                "best_match_origin": verdict.get("best_match_origin"),
                "expected_emv_station_s": verdict.get("expected_emv_station_s"),
                "expected_emv_csl_s": verdict.get("expected_emv_csl_s"),
                "best_match_abs_error_s": verdict.get("best_match_abs_error_s"),
                "reason": verdict.get("reason"),
                "dest_lat": dest_lat,
                "dest_lon": dest_lon,
                "start_lat_station": station.get("start_lat"),
                "start_lon_station": station.get("start_lon"),
            }
        )

    out = pd.DataFrame(rows)
    args.processed.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)
    out_path = args.processed / "origin_class_gmaps_control.csv"
    out.to_csv(out_path, index=False)
    out.head(min(150, len(out))).to_csv(args.samples / "origin_class_gmaps_sample.csv", index=False)

    counts = out["origin_class"].value_counts(dropna=False).to_dict()
    best = out["best_match_origin"].value_counts(dropna=False).to_dict() if "best_match_origin" in out else {}
    summary = {
        "rows": int(len(out)),
        "origin_class_counts": counts,
        "best_match_origin_counts": best,
        "emv_speed_factor": args.emv_speed_factor,
        "slack": args.slack,
        "control_model": "Google Maps Directions (civilian driving)",
        "rule": (
            "on_road_required if (gmaps_station_s * emv_speed_factor) > (actual_travel_s * slack); "
            "else station_plausible. best_match_origin compares EMV-adjusted station vs CSL ETAs "
            "to observed travel time."
        ),
        "output": str(out_path),
    }
    (args.processed / "origin_class_gmaps_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
