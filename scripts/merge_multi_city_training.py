#!/usr/bin/env python3
"""
Merge NYC + external-city OD tables into an expanded route-ETA training CSV.

SF rows use *observed* EMS travel_seconds as labels (real signal).
NYC rows keep the existing route-ETA / GMaps-enriched labels.

Usage:
  PYTHONPATH=src python scripts/merge_multi_city_training.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.route_eta import ROUTE_ETA_CATEGORICAL, ROUTE_ETA_FEATURE_COLUMNS  # noqa: E402


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ROUTE_ETA_FEATURE_COLUMNS + ROUTE_ETA_CATEGORICAL + ["travel_seconds"]:
        if c not in out.columns:
            out[c] = np.nan
    return out


def nyc_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["city"] = df.get("city", "nyc")
    if "origin_layer" not in df.columns and "depot_layer" in df.columns:
        df["origin_layer"] = df["depot_layer"]
    df["city"] = df["city"].fillna("nyc")
    return _ensure_cols(df)


def sf_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["start_lat", "start_lon", "dest_lat", "dest_lon", "travel_seconds"])
    out = pd.DataFrame(
        {
            "city": "san_francisco",
            "borough": df.get("borough"),
            "origin_layer": df.get("depot_layer", "ems_station"),
            "start_lat": df["start_lat"],
            "start_lon": df["start_lon"],
            "dest_lat": df["dest_lat"],
            "dest_lon": df["dest_lon"],
            "crow_flies_km": df.get("crow_flies_km"),
            "hour": df.get("hour"),
            "dow": df.get("dow"),
            "severity": df.get("severity"),
            "travel_seconds": df["travel_seconds"],
            "label_source": "sf_observed_ems",
        }
    )
    out["is_weekend"] = out["dow"].apply(
        lambda d: float(d in (5, 6)) if pd.notna(d) else np.nan
    )
    out["is_rush"] = out["hour"].apply(
        lambda h: float(h in (7, 8, 9, 16, 17, 18, 19)) if pd.notna(h) else np.nan
    )
    out["is_night"] = out["hour"].apply(
        lambda h: float(h >= 22 or h < 6) if pd.notna(h) else np.nan
    )
    # Bearing when possible
    if out["crow_flies_km"].isna().any():
        # leave NaN; model handles missing
        pass
    return _ensure_cols(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nyc",
        type=Path,
        default=ROOT / "data" / "processed" / "route_eta_gmaps_training.csv",
    )
    p.add_argument(
        "--nyc-fallback",
        type=Path,
        default=ROOT / "data" / "processed" / "route_eta_training.csv",
    )
    p.add_argument(
        "--sf",
        type=Path,
        default=ROOT / "data" / "processed" / "od_pairs_sf_ems.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "route_eta_multicity_training.csv",
    )
    args = p.parse_args()

    nyc_path = args.nyc if args.nyc.exists() else args.nyc_fallback
    if not nyc_path.exists():
        raise SystemExit(f"Missing NYC training CSV ({args.nyc} or {args.nyc_fallback})")

    frames = [nyc_frame(nyc_path)]
    print(f"NYC rows: {len(frames[0])} from {nyc_path.name}")

    if args.sf.exists():
        sf = sf_frame(args.sf)
        frames.append(sf)
        print(f"SF rows:  {len(sf)} from {args.sf.name}")
    else:
        print(f"SF file missing ({args.sf}); NYC-only merge")

    merged = pd.concat(frames, ignore_index=True)
    # Drop rows without a label
    merged = merged.dropna(subset=["travel_seconds"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    summary = {
        "rows": int(len(merged)),
        "by_city": merged["city"].value_counts(dropna=False).to_dict(),
        "out": str(args.out),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
