#!/usr/bin/env python3
"""
Ingest San Francisco Fire/EMS CAD (DataSF nuek-vuh3) into our OD-ish schema.

Pulls first-unit medical responses with coordinates + response→on-scene travel time.
Station starts are approximated from SF Fire Station locations when available.

Usage:
  PYTHONPATH=src python scripts/ingest_sf_ems.py --limit 5000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SF_CALLS = "https://data.sf.gov/resource/nuek-vuh3.json"
# Fire stations / facilities (may 404 if renamed — we fall back to dest-only rows)
SF_STATIONS = "https://data.sfgov.org/resource/5tqz-smgj.json"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _fetch_json(url: str, params: dict, *, pages: int = 1, page_size: int = 5000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    for _ in range(pages):
        q = dict(params)
        q["$limit"] = page_size
        q["$offset"] = offset
        r = requests.get(url, params=q, timeout=120)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _parse_point(case_location) -> tuple[float | None, float | None]:
    if not isinstance(case_location, dict):
        return None, None
    coords = case_location.get("coordinates")
    if not coords or len(coords) < 2:
        return None, None
    lon, lat = float(coords[0]), float(coords[1])
    return lat, lon


def fetch_sf_medical(limit: int = 5000) -> pd.DataFrame:
    # Over-fetch because many rows lack response/on-scene timestamps.
    want = max(limit * 4, limit + 2000)
    pages = max(1, (want + 4999) // 5000)
    raw = _fetch_json(
        SF_CALLS,
        {
            "$where": (
                "call_type='Medical Incident' "
                "AND unit_sequence_in_call_dispatch='1'"
            ),
            "$order": "received_dttm DESC",
            "$select": (
                "call_number,incident_number,call_type,received_dttm,"
                "response_dttm,on_scene_dttm,station_area,zipcode_of_incident,"
                "priority,final_priority,unit_type,neighborhoods_analysis_boundaries,"
                "case_location,battalion"
            ),
        },
        pages=pages,
        page_size=5000,
    )
    rows = []
    for rec in raw:
        if len(rows) >= limit:
            break
        lat, lon = _parse_point(rec.get("case_location"))
        if lat is None:
            continue
        if not rec.get("response_dttm") or not rec.get("on_scene_dttm"):
            continue
        try:
            t0 = pd.to_datetime(rec["response_dttm"])
            t1 = pd.to_datetime(rec["on_scene_dttm"])
            travel = (t1 - t0).total_seconds()
        except Exception:  # noqa: BLE001
            continue
        if travel != travel or travel < 30 or travel > 3600:
            continue
        z = rec.get("zipcode_of_incident")
        try:
            z = int(float(z)) if z is not None and str(z).strip() else None
        except (TypeError, ValueError):
            z = z
        rows.append(
            {
                "incident_id": rec.get("incident_number") or rec.get("call_number"),
                "city": "san_francisco",
                "borough": rec.get("neighborhoods_analysis_boundaries"),
                "zipcode": z,
                "station_area": rec.get("station_area"),
                "dest_lat": lat,
                "dest_lon": lon,
                "travel_seconds": float(travel),
                "hour": int(t0.hour),
                "dow": int(t0.dayofweek),
                "severity": float(rec.get("final_priority") or rec.get("priority") or 5),
                "unit_type": rec.get("unit_type"),
                "received_dttm": rec.get("received_dttm"),
                "response_dttm": rec.get("response_dttm"),
                "on_scene_dttm": rec.get("on_scene_dttm"),
            }
        )
    return pd.DataFrame(rows)


def attach_station_starts(df: pd.DataFrame, stations_path: Path | None = None) -> pd.DataFrame:
    """
    Approximate start = nearest known SF fire station to the incident, preferring
    matching station_area when a station inventory CSV is provided.
    """
    out = df.copy()
    out["start_lat"] = pd.NA
    out["start_lon"] = pd.NA
    out["depot_layer"] = "unknown"
    out["depot_name"] = pd.NA
    out["start_mode"] = "unmatched"

    stations = None
    if stations_path and stations_path.exists():
        stations = pd.read_csv(stations_path)
    else:
        # Tiny built-in seed of central SF stations (lat, lon, area code, name)
        stations = pd.DataFrame(
            [
                {"station_area": "01", "latitude": 37.7955, "longitude": -122.3977, "facname": "Station 1"},
                {"station_area": "02", "latitude": 37.7810, "longitude": -122.4095, "facname": "Station 2"},
                {"station_area": "03", "latitude": 37.7879, "longitude": -122.4194, "facname": "Station 3"},
                {"station_area": "05", "latitude": 37.7749, "longitude": -122.4194, "facname": "Station 5"},
                {"station_area": "06", "latitude": 37.7599, "longitude": -122.4148, "facname": "Station 6"},
                {"station_area": "07", "latitude": 37.7485, "longitude": -122.4180, "facname": "Station 7"},
                {"station_area": "08", "latitude": 37.7600, "longitude": -122.4350, "facname": "Station 8"},
                {"station_area": "10", "latitude": 37.7390, "longitude": -122.3890, "facname": "Station 10"},
                {"station_area": "13", "latitude": 37.8000, "longitude": -122.4100, "facname": "Station 13"},
                {"station_area": "14", "latitude": 37.7300, "longitude": -122.4400, "facname": "Station 14"},
                {"station_area": "15", "latitude": 37.7250, "longitude": -122.4000, "facname": "Station 15"},
                {"station_area": "16", "latitude": 37.7800, "longitude": -122.4500, "facname": "Station 16"},
                {"station_area": "17", "latitude": 37.7200, "longitude": -122.4600, "facname": "Station 17"},
                {"station_area": "18", "latitude": 37.7600, "longitude": -122.4800, "facname": "Station 18"},
                {"station_area": "19", "latitude": 37.7400, "longitude": -122.4800, "facname": "Station 19"},
                {"station_area": "21", "latitude": 37.7800, "longitude": -122.4800, "facname": "Station 21"},
                {"station_area": "22", "latitude": 37.7600, "longitude": -122.5000, "facname": "Station 22"},
                {"station_area": "24", "latitude": 37.7400, "longitude": -122.4300, "facname": "Station 24"},
                {"station_area": "31", "latitude": 37.7800, "longitude": -122.5000, "facname": "Station 31"},
                {"station_area": "32", "latitude": 37.7300, "longitude": -122.3900, "facname": "Station 32"},
                {"station_area": "33", "latitude": 37.7200, "longitude": -122.4400, "facname": "Station 33"},
                {"station_area": "34", "latitude": 37.7300, "longitude": -122.4800, "facname": "Station 34"},
                {"station_area": "36", "latitude": 37.7600, "longitude": -122.4200, "facname": "Station 36"},
                {"station_area": "37", "latitude": 37.7800, "longitude": -122.4000, "facname": "Station 37"},
                {"station_area": "38", "latitude": 37.7800, "longitude": -122.4600, "facname": "Station 38"},
                {"station_area": "40", "latitude": 37.7400, "longitude": -122.5000, "facname": "Station 40"},
                {"station_area": "41", "latitude": 37.8000, "longitude": -122.4200, "facname": "Station 41"},
                {"station_area": "42", "latitude": 37.7300, "longitude": -122.4200, "facname": "Station 42"},
                {"station_area": "43", "latitude": 37.7200, "longitude": -122.4800, "facname": "Station 43"},
                {"station_area": "44", "latitude": 37.7400, "longitude": -122.3900, "facname": "Station 44"},
            ]
        )

    by_area = {
        str(r["station_area"]).zfill(2): r
        for _, r in stations.iterrows()
        if pd.notna(r.get("station_area"))
    }

    for i, row in out.iterrows():
        area = str(row.get("station_area") or "").zfill(2)
        st = by_area.get(area)
        if st is None:
            # nearest of inventory
            best, best_d = None, 1e9
            for _, s in stations.iterrows():
                d = _haversine_km(
                    float(row["dest_lat"]),
                    float(row["dest_lon"]),
                    float(s["latitude"]),
                    float(s["longitude"]),
                )
                if d < best_d:
                    best_d, best = d, s
            st = best
            mode = "nearest_station"
        else:
            mode = "station_area"
        if st is None:
            continue
        out.at[i, "start_lat"] = float(st["latitude"])
        out.at[i, "start_lon"] = float(st["longitude"])
        out.at[i, "depot_layer"] = "ems_station"
        out.at[i, "depot_name"] = st.get("facname")
        out.at[i, "start_mode"] = mode
        out.at[i, "crow_flies_km"] = _haversine_km(
            float(st["latitude"]),
            float(st["longitude"]),
            float(row["dest_lat"]),
            float(row["dest_lon"]),
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--stations", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "od_pairs_sf_ems.csv",
    )
    args = p.parse_args()

    print(f"Fetching up to {args.limit} SF medical first-unit responses…")
    df = fetch_sf_medical(limit=args.limit)
    print(f"  usable rows with travel time: {len(df)}")
    df = attach_station_starts(df, args.stations)
    matched = df["start_lat"].notna().sum()
    print(f"  with inferred station start: {matched}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = {
        "city": "san_francisco",
        "source": SF_CALLS,
        "rows": int(len(df)),
        "matched_starts": int(matched),
        "median_travel_s": float(df["travel_seconds"].median()) if len(df) else None,
        "out": str(args.out),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
