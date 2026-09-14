#!/usr/bin/env python3
"""
Enrich travel-time / route-ETA training with Google Maps civilian ETAs.

Uses Directions API (traffic-aware when available) as:
  1) Strong features for LightGBM (gmaps_duration_s / traffic / distance)
  2) Calibration of OSM civilian times (gmaps_vs_osm_ratio)
  3) Optional EMV label remap: travel_seconds ≈ gmaps_traffic / EMV speedup × weather

Requires GOOGLE_MAPS_API_KEY in the environment or a gitignored `.env`.

Example:
  PYTHONPATH=src python scripts/enrich_training_with_gmaps.py --limit 600
  PYTHONPATH=src python scripts/train_route_eta_model.py
  PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.gmaps import GoogleMapsControl, load_api_key_from_dotenv  # noqa: E402
from emvro.route_eta import emv_travel_seconds  # noqa: E402


def _round_key(lat, lon, dlat, dlon) -> tuple:
    return (
        round(float(lat), 4),
        round(float(lon), 4),
        round(float(dlat), 4),
        round(float(dlon), 4),
    )


def enrich_frame(
    df: pd.DataFrame,
    gmaps: GoogleMapsControl,
    *,
    limit: int | None,
    seed: int,
    start_lat="start_lat",
    start_lon="start_lon",
    dest_lat="dest_lat",
    dest_lon="dest_lon",
    remap_labels: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    for c in (start_lat, start_lon, dest_lat, dest_lon):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    usable = out.dropna(subset=[start_lat, start_lon, dest_lat, dest_lon]).copy()
    if limit is not None and len(usable) > limit:
        usable = usable.sample(n=limit, random_state=seed)

    # Unique OD queries
    keys = []
    for r in usable.itertuples(index=True):
        keys.append(
            (
                r.Index,
                _round_key(
                    getattr(r, start_lat),
                    getattr(r, start_lon),
                    getattr(r, dest_lat),
                    getattr(r, dest_lon),
                ),
            )
        )
    unique = {}
    for idx, key in keys:
        unique.setdefault(key, []).append(idx)

    print(f"Querying Google Maps for {len(unique)} unique ODs "
          f"(covering {len(usable)} rows)…")
    results: dict[tuple, dict] = {}
    for key in tqdm(list(unique.keys()), desc="gmaps_enrich"):
        olat, olon, dlat, dlon = key
        results[key] = gmaps.driving_seconds(olat, olon, dlat, dlon, departure_time="now")

    # Init columns
    for col in (
        "gmaps_duration_s",
        "gmaps_traffic_s",
        "gmaps_distance_m",
        "gmaps_vs_osm_ratio",
        "gmaps_ok",
    ):
        if col not in out.columns:
            out[col] = np.nan

    out["gmaps_ok"] = 0
    filled = 0
    for key, idxs in unique.items():
        r = results[key]
        if not r.get("ok"):
            continue
        dur = r.get("duration_s")
        traf = r.get("duration_in_traffic_s")
        dist = r.get("distance_m")
        for idx in idxs:
            out.at[idx, "gmaps_duration_s"] = dur
            out.at[idx, "gmaps_traffic_s"] = traf if traf is not None else dur
            out.at[idx, "gmaps_distance_m"] = dist
            out.at[idx, "gmaps_ok"] = 1
            if "civilian_network_s" in out.columns:
                civ = out.at[idx, "civilian_network_s"]
                g = traf if traf is not None else dur
                if civ and civ == civ and civ > 0 and g and g == g and g > 0:
                    out.at[idx, "gmaps_vs_osm_ratio"] = float(g) / float(civ)
            elif "osm_path_km" in out.columns and dist and dist == dist and dist > 0:
                # soft proxy if OSM seconds missing
                pass
            filled += 1

            if remap_labels and out.at[idx, "gmaps_ok"] == 1:
                g = out.at[idx, "gmaps_traffic_s"]
                hour = int(out.at[idx, "hour"]) if "hour" in out.columns and out.at[idx, "hour"] == out.at[idx, "hour"] else 12
                sev = float(out.at[idx, "severity"]) if "severity" in out.columns and out.at[idx, "severity"] == out.at[idx, "severity"] else 5.0
                wx_p = int(out.at[idx, "wx_is_precip"]) if "wx_is_precip" in out.columns and out.at[idx, "wx_is_precip"] == out.at[idx, "wx_is_precip"] else 0
                wx_s = int(out.at[idx, "wx_is_snow"]) if "wx_is_snow" in out.columns and out.at[idx, "wx_is_snow"] == out.at[idx, "wx_is_snow"] else 0
                vis = float(out.at[idx, "wx_visibility_m"]) if "wx_visibility_m" in out.columns and out.at[idx, "wx_visibility_m"] == out.at[idx, "wx_visibility_m"] else 10000.0
                # Remap EMV label off Google civilian traffic time
                out.at[idx, "travel_seconds"] = emv_travel_seconds(
                    float(g),
                    hour=hour,
                    severity=sev,
                    wx_is_precip=wx_p,
                    wx_is_snow=wx_s,
                    wx_visibility_m=vis,
                    noise_frac=0.05,
                    rng=np.random.default_rng(abs(hash(key)) % (2**32)),
                )
                out.at[idx, "label_source"] = "gmaps_civilian_mapped"

    print(f"Filled Google features on {filled} rows")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    p.add_argument("--limit", type=int, default=600, help="Max rows per table to query")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--remap-route-eta-labels",
        action="store_true",
        help="Rebuild route-ETA labels from GMaps civilian traffic × EMV speedup",
    )
    p.add_argument("--skip-cad", action="store_true")
    p.add_argument("--skip-route-eta", action="store_true")
    args = p.parse_args()

    load_api_key_from_dotenv(ROOT / ".env")
    gmaps = GoogleMapsControl(cache_path=args.processed / "gmaps_cache.json")
    if not gmaps.available:
        raise SystemExit(
            "GOOGLE_MAPS_API_KEY not found. Put it in a gitignored .env or export it."
        )

    # Smoke test one request
    smoke = gmaps.driving_seconds(40.7580, -73.9855, 40.7115, -74.0060)
    if not smoke.get("ok"):
        raise SystemExit(
            f"Google Maps request failed: {smoke.get('error')} {smoke.get('message') or smoke.get('status')}"
        )
    print(
        "GMaps smoke OK:",
        smoke.get("duration_in_traffic_s") or smoke.get("duration_s"),
        "s",
        "(cached)" if smoke.get("cached") else "",
    )

    summary = {"gmaps_smoke_s": smoke.get("duration_in_traffic_s") or smoke.get("duration_s")}

    if not args.skip_cad:
        cad_path = args.processed / "travel_time_training.csv"
        if cad_path.exists():
            cad = pd.read_csv(cad_path, low_memory=False)
            # Prefer rows with start coords
            cad_en = enrich_frame(
                cad, gmaps, limit=args.limit, seed=args.seed, remap_labels=False
            )
            cad_en.to_csv(cad_path, index=False)
            cad_en.head(min(200, len(cad_en))).to_csv(
                args.samples / "travel_time_training_sample.csv", index=False
            )
            summary["cad_rows"] = int(len(cad_en))
            summary["cad_gmaps_ok"] = int(cad_en["gmaps_ok"].fillna(0).sum())
            print("Updated", cad_path)
        else:
            print("Skip CAD: missing", cad_path)

    if not args.skip_route_eta:
        eta_path = args.processed / "route_eta_training.csv"
        if eta_path.exists():
            eta = pd.read_csv(eta_path, low_memory=False)
            # Route-ETA rows may lack start_lat — synthesize from stored fields if present
            if "start_lat" not in eta.columns and {"dest_lat", "dest_lon"}.issubset(eta.columns):
                print("route_eta lacks start_lat/lon — joining from OD sample for a subset")
                od = args.processed / "od_pairs_inferred_starts.csv"
                if od.exists():
                    od_df = pd.read_csv(od, low_memory=False)
                    # Build a compact OD→gmaps table then merge by dest proximity is weak;
                    # instead query a fresh OD sample and append gmaps columns onto a
                    # calibrated route-eta subset.
                    od_sample = od_df.dropna(
                        subset=["start_lat", "start_lon", "dest_lat", "dest_lon"]
                    ).sample(n=min(args.limit, len(od_df)), random_state=args.seed)
                    od_en = enrich_frame(
                        od_sample,
                        gmaps,
                        limit=None,
                        seed=args.seed,
                        remap_labels=False,
                    )
                    # Attach gmaps onto route_eta by nearest crow-flies bucket is messy;
                    # write a dedicated calibrated table used for retraining.
                    calib = od_en.copy()
                    # Need OSM civilian seconds for ratio — approximate from crow if missing
                    if "civilian_network_s" not in calib.columns:
                        # leave NaN; model can still use gmaps_* features
                        calib["civilian_network_s"] = np.nan
                    if args.remap_route_eta_labels:
                        for i, r in calib.iterrows():
                            g = r.get("gmaps_traffic_s")
                            if g != g or not g:
                                continue
                            hour = int(r["hour"]) if "hour" in calib.columns and r.get("hour") == r.get("hour") else 12
                            calib.at[i, "travel_seconds"] = emv_travel_seconds(
                                float(g),
                                hour=hour,
                                severity=float(r.get("severity") or 5),
                                wx_is_precip=int(r.get("wx_is_precip") or 0),
                                wx_is_snow=int(r.get("wx_is_snow") or 0),
                                wx_visibility_m=float(r.get("wx_visibility_m") or 10000),
                                noise_frac=0.05,
                            )
                            calib.at[i, "label_source"] = "gmaps_civilian_mapped"
                    out_calib = args.processed / "route_eta_gmaps_training.csv"
                    calib.to_csv(out_calib, index=False)
                    summary["route_eta_gmaps_rows"] = int(len(calib))
                    summary["route_eta_gmaps_ok"] = int(calib["gmaps_ok"].fillna(0).sum())
                    summary["route_eta_gmaps_path"] = str(out_calib)
                    print("Wrote", out_calib)
                else:
                    print("No OD pairs for route-eta GMaps enrichment")
            else:
                eta_en = enrich_frame(
                    eta,
                    gmaps,
                    limit=args.limit,
                    seed=args.seed,
                    remap_labels=args.remap_route_eta_labels,
                )
                out_path = args.processed / "route_eta_gmaps_training.csv"
                eta_en.to_csv(out_path, index=False)
                # Also update main route_eta when most rows got gmaps
                if eta_en["gmaps_ok"].fillna(0).mean() > 0.5:
                    eta_en.to_csv(eta_path, index=False)
                summary["route_eta_gmaps_rows"] = int(len(eta_en))
                summary["route_eta_gmaps_ok"] = int(eta_en["gmaps_ok"].fillna(0).sum())
                summary["route_eta_gmaps_path"] = str(out_path)
                print("Wrote", out_path)
        else:
            print("Skip route_eta: missing", eta_path)

    (args.processed / "gmaps_enrich_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
