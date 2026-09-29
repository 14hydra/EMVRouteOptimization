#!/usr/bin/env python3
"""
Draft ingest: Paris Fire Brigade (BSPP) ENS data-challenge CSVs → OD pairs.

Unlike NYC/SF public CAD, this source includes **vehicle position before
departure** + intervention coords + ``delta departure-presentation`` (true
on-road travel seconds). That is the OD geometry we want for travel-time work.

Data (not redistributed here — BSPP terms restrict use):
  https://paris-fire-brigade.github.io/data-challenge/challenge.html
  Place files under ``data/raw/paris_bspp/``:
    x_train.csv, y_train.csv  (+ optional x_test.csv, y_test.csv)

Usage:
  # Demo schema smoke-test (synthetic rows, no download)
  PYTHONPATH=src python scripts/ingest_paris_bspp.py --demo --limit 500

  # After downloading challenge CSVs locally:
  PYTHONPATH=src python scripts/ingest_paris_bspp.py --limit 20000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Challenge column names (exact strings from the BSPP/ENS docs).
COL_ID = "emergency vehicle selection"
COL_DEST_LAT = "latitude intervention"
COL_DEST_LON = "longitude intervention"
COL_START_LAT = "latitude previous departure"
COL_START_LON = "longitude before departure"
COL_FROM_CENTER = "departed from its rescue center"
COL_SELECTION = "selection time"
COL_VEH_TYPE = "emergency vehicle type"
COL_CENTER = "rescue center"
COL_ALERT_CAT = "alert reason category"
COL_OSRM_DIST = "OSRM estimated distance"
COL_OSRM_DUR = "OSRM estimated duration"
COL_TRAVEL = "delta departure-presentation"
COL_TURNOUT = "delta selection-departure"
COL_TOTAL = "delta selection-presentation"

CHALLENGE_URL = "https://paris-fire-brigade.github.io/data-challenge/challenge.html"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM / normalize whitespace in headers."""
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def make_demo_tables(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic Paris-shaped tables so the draft pipeline runs without ENS files."""
    rng = np.random.default_rng(seed)
    # Rough Paris bbox
    dest_lat = rng.uniform(48.82, 48.90, n)
    dest_lon = rng.uniform(2.27, 2.42, n)
    from_center = rng.random(n) < 0.65
    # Starts near dest when "from center" (fake station), else on-road offset
    jitter = np.where(from_center, 0.008, 0.025)
    start_lat = dest_lat + rng.normal(0, jitter)
    start_lon = dest_lon + rng.normal(0, jitter)
    crow = np.array(
        [_haversine_km(a, b, c, d) for a, b, c, d in zip(start_lat, start_lon, dest_lat, dest_lon)]
    )
    travel = np.clip(crow / 28.0 * 3600 * rng.uniform(0.85, 1.35, n), 45, 1800).astype(int)
    turnout = rng.integers(20, 120, n)
    ids = np.arange(1, n + 1)
    sel = pd.to_datetime("2018-06-01") + pd.to_timedelta(rng.integers(0, 180 * 24 * 3600, n), unit="s")

    x = pd.DataFrame(
        {
            COL_ID: ids,
            "intervention": rng.integers(1000, 9999, n),
            COL_ALERT_CAT: rng.choice(["1", "2", "3", "4"], n),
            "alert reason": rng.choice(["A", "B", "C"], n),
            "intervention on public roads": rng.integers(0, 2, n),
            "floor": rng.integers(0, 8, n),
            "location of the event": rng.choice(["street", "apartment", "shop"], n),
            COL_DEST_LON: dest_lon,
            COL_DEST_LAT: dest_lat,
            "emergency vehicle": rng.integers(1, 200, n),
            COL_VEH_TYPE: rng.choice(["VSAV", "FPT", "EP", "VSR"], n),
            COL_CENTER: rng.integers(1, 80, n),
            COL_SELECTION: sel.astype(str),
            "date key selection": sel.strftime("%Y%m%d").astype(int),
            "time key selection": sel.strftime("%H%M%S").astype(int),
            "status preceding selection": rng.choice(["Returned", "Departed", "Presented"], n),
            COL_FROM_CENTER: from_center.astype(int),
            COL_START_LON: start_lon,
            COL_START_LAT: start_lat,
            "delta position gps previous departure-departure": rng.integers(0, 300, n),
            COL_OSRM_DIST: crow * 1000 * 1.25,
            COL_OSRM_DUR: crow / 25.0 * 3600,
        }
    )
    y = pd.DataFrame(
        {
            COL_ID: ids,
            COL_TURNOUT: turnout,
            COL_TRAVEL: travel,
            COL_TOTAL: turnout + travel,
        }
    )
    return x, y


def load_challenge_pair(
    raw_dir: Path,
    *,
    split: str = "train",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_path = raw_dir / f"x_{split}.csv"
    y_path = raw_dir / f"y_{split}.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path.name} and/or {y_path.name} in {raw_dir}.\n"
            f"Download from the ENS challenge (see {CHALLENGE_URL}), or pass --demo."
        )
    x = _norm_cols(pd.read_csv(x_path, low_memory=False))
    y = _norm_cols(pd.read_csv(y_path, low_memory=False))
    return x, y


def to_od_pairs(x: pd.DataFrame, y: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    """Join X/Y and map into the multi-city OD schema."""
    need_x = [COL_ID, COL_DEST_LAT, COL_DEST_LON, COL_START_LAT, COL_START_LON]
    need_y = [COL_ID, COL_TRAVEL]
    for c in need_x:
        if c not in x.columns:
            raise KeyError(f"x table missing column {c!r}; have {list(x.columns)[:20]}…")
    for c in need_y:
        if c not in y.columns:
            raise KeyError(f"y table missing column {c!r}; have {list(y.columns)}")

    m = x.merge(y, on=COL_ID, how="inner", suffixes=("", "_y"))
    m[COL_DEST_LAT] = pd.to_numeric(m[COL_DEST_LAT], errors="coerce")
    m[COL_DEST_LON] = pd.to_numeric(m[COL_DEST_LON], errors="coerce")
    m[COL_START_LAT] = pd.to_numeric(m[COL_START_LAT], errors="coerce")
    m[COL_START_LON] = pd.to_numeric(m[COL_START_LON], errors="coerce")
    m[COL_TRAVEL] = pd.to_numeric(m[COL_TRAVEL], errors="coerce")

    ok = (
        m[COL_DEST_LAT].between(48.5, 49.2)
        & m[COL_DEST_LON].between(1.8, 2.8)
        & m[COL_START_LAT].between(48.5, 49.2)
        & m[COL_START_LON].between(1.8, 2.8)
        & m[COL_TRAVEL].between(30, 3600)
    )
    m = m.loc[ok].copy()
    if limit is not None:
        m = m.head(int(limit))

    sel = pd.to_datetime(m[COL_SELECTION], errors="coerce") if COL_SELECTION in m.columns else pd.NaT
    from_center = pd.to_numeric(m.get(COL_FROM_CENTER), errors="coerce").fillna(0).astype(int)

    crow = [
        _haversine_km(a, b, c, d)
        for a, b, c, d in zip(
            m[COL_START_LAT], m[COL_START_LON], m[COL_DEST_LAT], m[COL_DEST_LON]
        )
    ]

    out = pd.DataFrame(
        {
            "incident_id": m[COL_ID].astype(str),
            "city": "paris",
            "borough": m[COL_CENTER].astype(str) if COL_CENTER in m.columns else pd.NA,
            "zipcode": pd.NA,
            "dest_lat": m[COL_DEST_LAT].to_numpy(),
            "dest_lon": m[COL_DEST_LON].to_numpy(),
            "start_lat": m[COL_START_LAT].to_numpy(),
            "start_lon": m[COL_START_LON].to_numpy(),
            "travel_seconds": m[COL_TRAVEL].astype(float).to_numpy(),
            "turnout_seconds": pd.to_numeric(m.get(COL_TURNOUT), errors="coerce"),
            "selection_to_onscene_seconds": pd.to_numeric(m.get(COL_TOTAL), errors="coerce"),
            "hour": sel.dt.hour if hasattr(sel, "dt") else pd.NA,
            "dow": sel.dt.dayofweek if hasattr(sel, "dt") else pd.NA,
            "selection_time": sel,
            "depot_layer": np.where(from_center == 1, "rescue_center", "on_road_gps"),
            "depot_name": m[COL_CENTER].astype(str) if COL_CENTER in m.columns else pd.NA,
            "start_mode": np.where(from_center == 1, "rescue_center", "pre_departure_gps"),
            "departed_from_rescue_center": from_center,
            "unit_type": m[COL_VEH_TYPE] if COL_VEH_TYPE in m.columns else pd.NA,
            "alert_reason_category": m[COL_ALERT_CAT] if COL_ALERT_CAT in m.columns else pd.NA,
            "osrm_distance_m": pd.to_numeric(m.get(COL_OSRM_DIST), errors="coerce"),
            "osrm_duration_s": pd.to_numeric(m.get(COL_OSRM_DUR), errors="coerce"),
            "crow_flies_km": crow,
            "label_source": "bspp_delta_departure_presentation",
        }
    )
    return out.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "paris_bspp")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--limit", type=int, default=None, help="Cap rows after filters")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Build synthetic Paris-shaped tables (no ENS download) for a draft smoke test",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "od_pairs_paris_bspp.csv",
    )
    args = p.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    readme = args.raw_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Paris BSPP challenge files (not committed)\n\n"
            f"Download `x_train.csv` / `y_train.csv` from the ENS challenge:\n"
            f"{CHALLENGE_URL}\n\n"
            "BSPP terms: data is for the challenge context unless BSPP agrees otherwise.\n"
            "Place CSVs in this folder, then run:\n\n"
            "```bash\n"
            "PYTHONPATH=src python scripts/ingest_paris_bspp.py --limit 20000\n"
            "```\n"
        )

    if args.demo:
        print("Building --demo synthetic Paris tables…")
        n = args.limit or 500
        x, y = make_demo_tables(n=n)
        # Persist demo raw for inspection
        x.to_csv(args.raw_dir / "x_train_demo.csv", index=False)
        y.to_csv(args.raw_dir / "y_train_demo.csv", index=False)
        source = "synthetic_demo"
    else:
        print(f"Loading {args.split} from {args.raw_dir}…")
        x, y = load_challenge_pair(args.raw_dir, split=args.split)
        source = str(args.raw_dir / f"x_{args.split}.csv")

    print(f"  x rows={len(x):,}  y rows={len(y):,}")
    od = to_od_pairs(x, y, limit=args.limit)
    print(f"  OD rows after geo/time filters: {len(od):,}")
    if len(od):
        print(
            f"  from rescue center: {float((od.departed_from_rescue_center==1).mean()):.1%}  "
            f"median travel={od.travel_seconds.median():.0f}s  "
            f"median crow={od.crow_flies_km.median():.2f}km"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    od.to_csv(args.out, index=False)
    summary = {
        "city": "paris",
        "source": source,
        "challenge_url": CHALLENGE_URL,
        "demo": bool(args.demo),
        "split": args.split,
        "rows": int(len(od)),
        "share_from_rescue_center": float((od["departed_from_rescue_center"] == 1).mean()) if len(od) else None,
        "share_on_road_gps_start": float((od["departed_from_rescue_center"] == 0).mean()) if len(od) else None,
        "median_travel_s": float(od["travel_seconds"].median()) if len(od) else None,
        "median_crow_km": float(od["crow_flies_km"].median()) if len(od) else None,
        "corr_crow_travel": float(od["crow_flies_km"].corr(od["travel_seconds"])) if len(od) > 10 else None,
        "out": str(args.out),
        "note": (
            "Draft ingest. travel_seconds = delta departure-presentation; "
            "start_* = pre-departure GPS (rescue center or on-road)."
        ),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
