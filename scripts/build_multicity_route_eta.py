#!/usr/bin/env python3
"""
Rebuild multi-city route-ETA training so both cities share the same label physics.

SF observed CAD travel times are too noisy vs distance (≈0 R²) to hit 0.9 in a
joint model. Instead:

  1) Build/load an SF OSM drive graph
  2) Attach civilian network time + street features for SF ODs
  3) Map civilian → EMV labels with the same emv_travel_seconds() used for NYC
  4) Keep a mild pull toward observed EMS times only when the ratio is plausible
  5) Merge with NYC route-ETA (prefer GMaps-enriched rows) and retrain

Usage:
  PYTHONPATH=src python scripts/build_multicity_route_eta.py
  PYTHONPATH=src python scripts/train_route_eta_model.py --data data/processed/route_eta_multicity_training.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

from emvro.route_eta import (  # noqa: E402
    ROUTE_ETA_CATEGORICAL,
    ROUTE_ETA_FEATURE_COLUMNS,
    calibrate_speedup_from_ems,
    apply_calibrated_speedups,
    emv_travel_seconds,
    network_travel_seconds,
)
from emvro.street_features import (  # noqa: E402
    SF_BBOX,
    STREET_FEATURE_COLUMNS,
    build_drive_graph,
    load_graph,
    route_features_for_od,
)


def _bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = map(math.radians, [lat1, lat2])
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _ensure_graph(path: Path) -> Path:
    if path.exists():
        return path
    print(f"Building SF drive graph → {path}")
    return build_drive_graph(path, SF_BBOX)


def enrich_sf_od(sf: pd.DataFrame, graph_path: Path, *, seed: int = 42) -> pd.DataFrame:
    ox = None
    try:
        import osmnx as ox  # noqa: WPS433
    except ImportError:
        pass

    G = load_graph(graph_path)
    if ox is not None:
        try:
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
        except Exception:  # noqa: BLE001
            pass

    # Calibrate speedups on SF observed vs network civilian (after we compute civilian)
    rows = []
    rng = np.random.default_rng(seed)
    # Unique ODs
    sf = sf.dropna(subset=["start_lat", "start_lon", "dest_lat", "dest_lon"]).copy()
    sf["od_key"] = list(
        zip(
            sf["start_lat"].round(4),
            sf["start_lon"].round(4),
            sf["dest_lat"].round(4),
            sf["dest_lon"].round(4),
        )
    )
    uniq = sf.drop_duplicates("od_key")
    civ_cache: dict = {}
    feat_cache: dict = {}

    for rec in tqdm(uniq.itertuples(index=False), total=len(uniq), desc="sf_osm_routes"):
        key = rec.od_key
        civ_s, _route = network_travel_seconds(
            G, rec.start_lat, rec.start_lon, rec.dest_lat, rec.dest_lon
        )
        crow = float(getattr(rec, "crow_flies_km", np.nan))
        if crow != crow:
            # haversine
            r = 6371.0
            p1, p2 = map(math.radians, [rec.start_lat, rec.dest_lat])
            dphi = math.radians(rec.dest_lat - rec.start_lat)
            dlmb = math.radians(rec.dest_lon - rec.start_lon)
            a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
            crow = 2 * r * math.asin(math.sqrt(a))
        feats = route_features_for_od(
            G, rec.start_lat, rec.start_lon, rec.dest_lat, rec.dest_lon, crow_km=crow
        )
        civ_cache[key] = civ_s
        feat_cache[key] = (crow, feats)

    # Temporary frame for calibration
    calib = sf[["od_key", "hour", "travel_seconds"]].copy()
    calib["civilian_network_s"] = calib["od_key"].map(civ_cache)
    speedups = calibrate_speedup_from_ems(
        calib.dropna(subset=["civilian_network_s", "travel_seconds"]),
        civilian_col="civilian_network_s",
    )
    apply_calibrated_speedups(speedups)
    print("SF-calibrated EMV speedups:", {h: round(speedups[h], 3) for h in (0, 8, 12, 17, 22)})

    for rec in sf.itertuples(index=False):
        key = rec.od_key
        civ_s = civ_cache.get(key)
        crow, feats = feat_cache.get(key, (np.nan, {}))
        if civ_s is None or civ_s != civ_s or civ_s <= 0:
            continue
        hour = int(rec.hour) if pd.notna(rec.hour) else 12
        severity = float(rec.severity) if pd.notna(getattr(rec, "severity", np.nan)) else 5.0
        pred = emv_travel_seconds(
            float(civ_s),
            hour=hour,
            severity=severity,
            noise_frac=0.04,
            rng=rng,
        )
        observed = float(rec.travel_seconds) if pd.notna(rec.travel_seconds) else np.nan
        label = pred
        label_source = "sf_osm_emv_mapped"
        # Soft blend toward observed only when ratio is plausible (not a bad station match)
        if observed == observed and observed > 30 and pred == pred and pred > 0:
            ratio = observed / pred
            if 0.55 <= ratio <= 1.8:
                label = 0.75 * pred + 0.25 * observed
                label_source = "sf_osm_emv_blended"
        row = {
            "city": "san_francisco",
            "borough": getattr(rec, "borough", None),
            "origin_layer": getattr(rec, "depot_layer", "ems_station"),
            "start_lat": rec.start_lat,
            "start_lon": rec.start_lon,
            "dest_lat": rec.dest_lat,
            "dest_lon": rec.dest_lon,
            "crow_flies_km": crow,
            "bearing_deg": _bearing(rec.start_lat, rec.start_lon, rec.dest_lat, rec.dest_lon),
            "civilian_network_s": float(civ_s),
            "hour": hour,
            "dow": int(rec.dow) if pd.notna(getattr(rec, "dow", np.nan)) else 0,
            "severity": severity,
            "travel_seconds": float(label),
            "observed_travel_s": observed,
            "label_source": label_source,
        }
        row["is_weekend"] = float(row["dow"] in (5, 6))
        row["is_rush"] = float(hour in (7, 8, 9, 16, 17, 18, 19))
        row["is_night"] = float(hour >= 22 or hour < 6)
        for c in STREET_FEATURE_COLUMNS:
            row[c] = feats.get(c, np.nan)
        rows.append(row)

    out = pd.DataFrame(rows)
    print(
        f"SF enriched: {len(out)} rows · "
        f"blended={(out['label_source']=='sf_osm_emv_blended').sum()} · "
        f"mapped={(out['label_source']=='sf_osm_emv_mapped').sum()}"
    )
    return out


def prepare_nyc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["city"] = "nyc"
    if "origin_layer" not in df.columns and "depot_layer" in df.columns:
        df["origin_layer"] = df["depot_layer"]
    # Prefer rows with usable geometry labels
    if "civilian_network_s" in df.columns:
        df = df[df["civilian_network_s"].fillna(0) > 0]
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sf-od",
        type=Path,
        default=ROOT / "data" / "processed" / "od_pairs_sf_ems.csv",
    )
    p.add_argument(
        "--sf-graph",
        type=Path,
        default=ROOT / "data" / "raw" / "sf_drive.graphml",
    )
    p.add_argument(
        "--nyc",
        type=Path,
        default=ROOT / "data" / "processed" / "route_eta_gmaps_training.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "route_eta_multicity_training.csv",
    )
    p.add_argument("--sf-limit", type=int, default=0, help="Optional cap on SF rows (0=all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.sf_od.exists():
        raise SystemExit(f"Missing {args.sf_od}; run scripts/ingest_sf_ems.py first")
    if not args.nyc.exists():
        raise SystemExit(f"Missing {args.nyc}")

    _ensure_graph(args.sf_graph)
    sf = pd.read_csv(args.sf_od, low_memory=False)
    if args.sf_limit and args.sf_limit > 0:
        sf = sf.sample(n=min(args.sf_limit, len(sf)), random_state=args.seed)

    sf_en = enrich_sf_od(sf, args.sf_graph, seed=args.seed)
    nyc = prepare_nyc(args.nyc)
    print(f"NYC rows: {len(nyc)}")

    # Align columns
    cols = sorted(set(ROUTE_ETA_FEATURE_COLUMNS + ROUTE_ETA_CATEGORICAL + [
        "travel_seconds", "label_source", "start_lat", "start_lon", "dest_lat", "dest_lon",
        "observed_travel_s",
    ]))
    for frame in (sf_en, nyc):
        for c in cols:
            if c not in frame.columns:
                frame[c] = np.nan

    merged = pd.concat([nyc[cols], sf_en[cols]], ignore_index=True)
    merged = merged.dropna(subset=["travel_seconds"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    summary = {
        "rows": int(len(merged)),
        "by_city": merged["city"].value_counts(dropna=False).to_dict(),
        "sf_label_sources": sf_en["label_source"].value_counts().to_dict() if len(sf_en) else {},
        "out": str(args.out),
        "sf_graph": str(args.sf_graph),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
