#!/usr/bin/env python3
"""
Build the London travel-time training set from LFB mobilisation records:
one row per fire engine trip, from the station it actually left to the incident.

Sources (London Datastore, Open Government Licence):
  - Mobilisation records (data/raw/london/lfb_mobilisations_*.csv): which
    station each engine was deployed from, whether it was at its home station
    or standing in at another one, and TravelTimeSeconds (leaving the station to
    arriving), recorded separately from turnout.
  - Incident records (data/raw/london/lfb_incidents_*.csv): incident location
    and the station ground it falls in.
  - Station locations (data/raw/london/london_firehouses.csv, built by
    scripts/build_london_firehouses.py): the real station buildings.
  - London drive graph (OpenStreetMap, emvro.road_distance): road_km is the
    shortest legal route from station to incident, one-way streets respected.

Kept trips: the engine left from an LFB station (DeployedFromLocation is Home
Station or Other Station), was not already out on the road ("On outside duty
when mobilised"), the address was not wrong ("Address incomplete/wrong"), and
the travel time is 30-1200 s.

Output: data/processed/lfb_travel_training.csv

Run:  .venv/bin/python scripts/build_lfb_travel_training.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw" / "london"

MOBILISATION_FILES = ["lfb_mobilisations_2021_2024.csv", "lfb_mobilisations_2025.csv"]
INCIDENT_FILES = ["lfb_incidents_2018_2023.csv", "lfb_incidents_2024_onwards.csv"]
EXCLUDED_DELAYS = {"On outside duty when mobilised", "Address incomplete/wrong"}
TRAVEL_CLIP_S = (30, 1200)
MAX_CROW_KM = 15.0

MOB_COLS = [
    "IncidentNumber", "CalYear", "HourOfCall", "DateAndTimeMobilised", "TravelTimeSeconds",
    "TurnoutTimeSeconds", "DeployedFromStation_Name", "DeployedFromLocation", "PumpOrder",
    "DelayCode_Description",
]
INC_COLS = [
    "IncidentNumber", "DateOfCall", "IncGeo_BoroughName", "IncidentStationGround",
    "Easting_m", "Northing_m", "Easting_rounded", "Northing_rounded",
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def load_incidents() -> pd.DataFrame:
    from pyproj import Transformer

    parts = []
    for f in INCIDENT_FILES:
        d = pd.read_csv(RAW / f, usecols=INC_COLS, dtype={"IncidentNumber": str}, low_memory=False)
        parts.append(d)
    inc = pd.concat(parts, ignore_index=True).drop_duplicates("IncidentNumber")
    # Exact grid reference when LFB publishes it, else the 50 m rounded one (always present).
    e = pd.to_numeric(inc["Easting_m"], errors="coerce").fillna(inc["Easting_rounded"])
    n = pd.to_numeric(inc["Northing_m"], errors="coerce").fillna(inc["Northing_rounded"])
    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(e.to_numpy(), n.to_numpy())
    return pd.DataFrame(
        {
            "IncidentNumber": inc["IncidentNumber"].str.strip(),
            "date_of_call": pd.to_datetime(inc["DateOfCall"], errors="coerce", dayfirst=False).dt.date,
            "borough": inc["IncGeo_BoroughName"].astype(str).str.upper(),
            "station_ground": inc["IncidentStationGround"],
            "dest_lat": lat,
            "dest_lon": lon,
            "dest_exact": pd.to_numeric(inc["Easting_m"], errors="coerce").notna().to_numpy(),
        }
    )


def load_mobilisations() -> pd.DataFrame:
    parts = [
        pd.read_csv(RAW / f, usecols=MOB_COLS, dtype={"IncidentNumber": str}, encoding="utf-8-sig", low_memory=False)
        for f in MOBILISATION_FILES
    ]
    m = pd.concat(parts, ignore_index=True)
    m["IncidentNumber"] = m["IncidentNumber"].str.strip()
    return m


def _load_stations() -> pd.DataFrame:
    path = RAW / "london_firehouses.csv"
    s = pd.read_csv(path)
    if "name" not in s.columns and "facilityname" in s.columns:
        s = s.rename(columns={"facilityname": "name", "latitude": "lat", "longitude": "lon"})
    if "lat" not in s.columns and "latitude" in s.columns:
        s = s.rename(columns={"latitude": "lat", "longitude": "lon"})
    return s.set_index("name")[["lat", "lon"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    ap.add_argument(
        "--crow-only",
        action="store_true",
        help="Skip road-graph distances (set road_km ≈ 1.25×crow). Use when london_drive.graphml is not built yet.",
    )
    args = ap.parse_args()

    stations = _load_stations()
    mob = load_mobilisations()
    counts = {"mobilisations": len(mob)}

    mob = mob[mob["DeployedFromLocation"].isin(["Home Station", "Other Station"])]
    counts["left_from_a_station"] = len(mob)
    mob = mob[~mob["DelayCode_Description"].isin(EXCLUDED_DELAYS)]
    counts["not_outside_duty_or_bad_address"] = len(mob)
    mob = mob[mob["DeployedFromStation_Name"].isin(stations.index)]
    counts["lfb_station"] = len(mob)
    travel = pd.to_numeric(mob["TravelTimeSeconds"], errors="coerce")
    mob = mob[travel.between(*TRAVEL_CLIP_S)].assign(travel_s=travel)
    counts["travel_30_1200s"] = len(mob)

    df = mob.merge(load_incidents(), on="IncidentNumber", how="inner")
    counts["joined_to_incident_location"] = len(df)

    df["start_lat"] = df["DeployedFromStation_Name"].map(stations["lat"])
    df["start_lon"] = df["DeployedFromStation_Name"].map(stations["lon"])
    df["crow_km"] = haversine_km(df["start_lat"], df["start_lon"], df["dest_lat"], df["dest_lon"])
    df = df[(df["crow_km"] > 0) & (df["crow_km"] <= MAX_CROW_KM)]
    counts["crow_km_0_15"] = len(df)

    graph_path = RAW / "london_drive.graphml"
    if args.crow_only or not graph_path.exists():
        df["road_km"] = df["crow_km"] * 1.25
        counts["routable_on_road_graph"] = int(len(df))
        counts["road_km_source"] = "crow_x1.25_proxy"
        if not args.crow_only:
            print("NOTE: london_drive.graphml missing — using crow×1.25 as road_km proxy")
    else:
        from emvro.road_distance import RoadRouter, load_or_build_graph

        router = RoadRouter(load_or_build_graph())
        df["road_km"] = router.road_km(df["start_lat"], df["start_lon"], df["dest_lat"], df["dest_lon"])
        df = df[np.isfinite(df["road_km"])]
        counts["routable_on_road_graph"] = len(df)
        counts["road_km_source"] = "osm_shortest_path"

    mobilised = pd.to_datetime(df["DateAndTimeMobilised"], format="%d/%m/%Y %H:%M", errors="coerce")
    out = pd.DataFrame(
        {
            "incident_number": df["IncidentNumber"],
            "cal_year": df["CalYear"].astype(int),
            "date_of_call": mobilised.dt.date.fillna(df["date_of_call"]),
            "hour_of_call": df["HourOfCall"].astype(int),
            "borough": df["borough"],
            "station_ground": df["station_ground"],
            "deployed_from_station": df["DeployedFromStation_Name"],
            "deployed_from_location": df["DeployedFromLocation"],
            "pump_order": df["PumpOrder"],
            # Engine came from outside the incident's own station ground (own engine busy / further away).
            "busy_flag": (df["DeployedFromStation_Name"] != df["station_ground"]).astype(int),
            "start_lat": df["start_lat"],
            "start_lon": df["start_lon"],
            "dest_lat": df["dest_lat"],
            "dest_lon": df["dest_lon"],
            "dest_exact": df["dest_exact"].astype(int),
            "crow_km": df["crow_km"],
            "road_km": df["road_km"],
            "turnout_s": pd.to_numeric(df["TurnoutTimeSeconds"], errors="coerce"),
            "drive_s": df["travel_s"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    for k, v in counts.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(f"{k:<34}{v:>10,}")
        else:
            print(f"{k:<34}{v}")
    print(out.groupby("cal_year").size().to_string())
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
