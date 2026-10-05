#!/usr/bin/env python3
"""
Build LFB incident extract for firehouse location planning (Google Doc schema).

Reads ``data/raw/london/lfb_incidents_2024_onwards.csv`` (or optional xlsx) and
writes ``data/raw/london/lfb_incidents_planner.csv``.

``travel_seconds`` is mapped from ``FirstPumpArriving_AttendanceTime``. That LFB
field measures mobilisation → first pump on scene (includes turnout / dispatch
delay), **not** pure driving time from station to incident.

Coordinates: ``dest_lat`` / ``dest_lon`` from ``Latitude`` / ``Longitude`` when
present; otherwise British National Grid ``Easting_m`` / ``Northing_m`` converted
to WGS84 (EPSG:27700 → EPSG:4326) via pyproj.

Filters: require coordinates and attendance; keep attendance in [30, 1800] seconds.

Usage:
  PYTHONPATH=src python scripts/build_lfb_planner_incidents.py
  python scripts/build_lfb_planner_incidents.py --input data/raw/london/LFB_Incident_data_from_2024_onwards.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "london" / "lfb_incidents_2024_onwards.csv"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "london" / "lfb_incidents_planner.csv"

ATTENDANCE_MIN_S = 30
ATTENDANCE_MAX_S = 1800

OUTPUT_COLUMNS = [
    "dest_lat",
    "dest_lon",
    "travel_seconds",
    "cal_year",
    "date_of_call",
    "hour_of_call",
    "borough",
    "station_ground",
    "deployed_from_station",
    "second_attendance_s",
    "second_deployed_from",
    "busy_flag",
]


def _load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _bng_to_wgs84(easting: pd.Series, northing: pd.Series) -> tuple[pd.Series, pd.Series]:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(
        easting.to_numpy(dtype=float),
        northing.to_numpy(dtype=float),
    )
    return pd.Series(lat, index=easting.index), pd.Series(lon, index=easting.index)


def build_planner(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    east = pd.to_numeric(df.get("Easting_m"), errors="coerce")
    north = pd.to_numeric(df.get("Northing_m"), errors="coerce")

    need_bng = lat.isna() | lon.isna()
    if need_bng.any():
        bng_ok = need_bng & east.notna() & north.notna()
        if bng_ok.any():
            bng_lat, bng_lon = _bng_to_wgs84(east.loc[bng_ok], north.loc[bng_ok])
            lat = lat.copy()
            lon = lon.copy()
            lat.loc[bng_ok] = bng_lat
            lon.loc[bng_ok] = bng_lon

    out["dest_lat"] = lat
    out["dest_lon"] = lon

    out["travel_seconds"] = pd.to_numeric(
        df.get("FirstPumpArriving_AttendanceTime"), errors="coerce"
    )
    out["cal_year"] = pd.to_numeric(df.get("CalYear"), errors="coerce").astype("Int64")
    out["date_of_call"] = df.get("DateOfCall")
    out["hour_of_call"] = pd.to_numeric(df.get("HourOfCall"), errors="coerce").astype("Int64")
    out["borough"] = df.get("IncGeo_BoroughName")
    out["station_ground"] = df.get("IncidentStationGround")
    out["deployed_from_station"] = df.get("FirstPumpArriving_DeployedFromStation")
    out["second_attendance_s"] = pd.to_numeric(
        df.get("SecondPumpArriving_AttendanceTime"), errors="coerce"
    )
    out["second_deployed_from"] = df.get("SecondPumpArriving_DeployedFromStation")

    ground = out["station_ground"].astype(str).str.strip()
    deployed = out["deployed_from_station"].astype(str).str.strip()
    both = (
        out["station_ground"].notna()
        & out["deployed_from_station"].notna()
        & (ground != "")
        & (deployed != "")
        & (ground.str.lower() != "nan")
        & (deployed.str.lower() != "nan")
    )
    out["busy_flag"] = (both & (ground != deployed)).astype(int)

    out = out.dropna(subset=["dest_lat", "dest_lon", "travel_seconds"])
    out = out.loc[out["travel_seconds"].between(ATTENDANCE_MIN_S, ATTENDANCE_MAX_S)].copy()
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def _print_summary(df: pd.DataFrame) -> None:
    print("\nRow counts by cal_year:")
    year_counts = df["cal_year"].value_counts(dropna=False).sort_index()
    for year, n in year_counts.items():
        print(f"  {year}: {n:,}")
    print(f"  total: {len(df):,}")

    busy = df["busy_flag"] == 1
    eligible = df["station_ground"].notna() & df["deployed_from_station"].notna()
    rate = float(busy.sum() / eligible.sum()) if eligible.any() else float("nan")
    print("\nBusy rate (deployed_from_station != station_ground, both present):")
    print(f"  busy incidents: {int(busy.sum()):,} / {int(eligible.sum()):,} eligible")
    print(f"  rate: {rate:.1%}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source CSV or xlsx")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Planner CSV path")
    args = p.parse_args()

    print(f"Reading {args.input} …")
    raw = _load_raw(args.input)
    print(f"  raw rows: {len(raw):,}")

    planner = build_planner(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    planner.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(planner):,} rows)")

    _print_summary(planner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
