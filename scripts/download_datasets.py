#!/usr/bin/env python3
"""Download chosen ASI datasets from NYC Open Data into data/raw/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.opendata import DATASETS, download_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    p.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Max EMS incident rows (facility tables download more fully)",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=[2022, 2023, 2024],
        help="EMS incident years to prefer when --start/--end not set",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="EMS window start ISO (e.g. 2024-06-01T00:00:00). Prefer over year+limit.",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="EMS window end ISO (e.g. 2024-06-07T23:59:59).",
    )
    p.add_argument(
        "--ems-only",
        action="store_true",
        help="Only refresh ems_incidents.csv (skip facilities / alarm boxes).",
    )
    p.add_argument(
        "--alarm-box-limit",
        type=int,
        default=15000,
        help="Max FDNY alarm-box rows for CSL candidates",
    )
    args = p.parse_args()

    print("Datasets:")
    for k, meta in DATASETS.items():
        print(f"  - {k}: {meta['id']} — {meta['description']}")

    if not args.ems_only:
        for key in ("fdny_firehouses", "ems_stations", "hospital_bays", "modzcta"):
            path = download_dataset(key, args.out, limit=5000)
            print("Wrote", path)

        boxes = download_dataset("alarm_boxes", args.out, limit=args.alarm_box_limit)
        print("Wrote", boxes)

        lion_note = args.out / "lion.README.txt"
        lion_note.write_text(
            "LION Open Data ID: 2v4z-66xt\n"
            "Street path features currently use a cached OSMNx drive graph "
            "(scripts/build_osm_graph.py) with LION as a planned swap-in.\n"
        )
        print("Wrote", lion_note)

    try:
        ems = download_dataset(
            "ems_incidents",
            args.out,
            limit=args.limit,
            years=None if (args.start and args.end) else args.years,
            start=args.start,
            end=args.end,
        )
    except Exception as exc:  # noqa: BLE001
        print("Primary EMS filter failed (%s); downloading without year/date filter…" % exc)
        ems = download_dataset("ems_incidents", args.out, limit=args.limit, years=None)
    print("Wrote", ems)


if __name__ == "__main__":
    main()
