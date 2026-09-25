#!/usr/bin/env python3
"""Download ASI datasets from NYC Open Data into data/raw/.

Default scope is **firetrucks** (FDNY incidents + firehouses + alarm boxes).
Pass ``--legacy-ems`` to also refresh EMS CAD / station tables for side comparisons.
"""

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
        help="Max FDNY incident rows (facility tables download more fully)",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=[2022, 2023, 2024],
        help="Incident years to prefer when --start/--end not set",
    )
    p.add_argument("--start", type=str, default=None, help="Incident window start ISO")
    p.add_argument("--end", type=str, default=None, help="Incident window end ISO")
    p.add_argument(
        "--incidents-only",
        action="store_true",
        help="Only refresh fdny_incidents.csv (skip facilities / alarm boxes).",
    )
    p.add_argument(
        "--alarm-box-limit",
        type=int,
        default=15000,
        help="Max FDNY alarm-box rows for CSL / patrol post candidates",
    )
    p.add_argument(
        "--legacy-ems",
        action="store_true",
        help="Also download legacy EMS incident / station / hospital tables",
    )
    args = p.parse_args()

    print("Datasets:")
    for k, meta in DATASETS.items():
        print(f"  - {k}: {meta['id']} — {meta['description']}")

    if not args.incidents_only:
        for key in ("fdny_firehouses", "modzcta"):
            path = download_dataset(key, args.out, limit=5000)
            print("Wrote", path)

        boxes = download_dataset("alarm_boxes", args.out, limit=args.alarm_box_limit)
        print("Wrote", boxes)

        if args.legacy_ems:
            for key in ("ems_stations", "hospital_bays"):
                path = download_dataset(key, args.out, limit=5000)
                print("Wrote", path)

        lion_note = args.out / "lion.README.txt"
        lion_note.write_text(
            "LION Open Data ID: 2v4z-66xt\n"
            "Street path features currently use a cached OSMnx drive graph "
            "(scripts/build_osm_graph.py) with LION as a planned swap-in.\n"
        )
        print("Wrote", lion_note)

    try:
        fdny = download_dataset(
            "fdny_incidents",
            args.out,
            limit=args.limit,
            years=None if (args.start and args.end) else args.years,
            start=args.start,
            end=args.end,
        )
    except Exception as exc:  # noqa: BLE001
        print("Primary FDNY filter failed (%s); downloading without year/date filter…" % exc)
        fdny = download_dataset("fdny_incidents", args.out, limit=args.limit, years=None)
    print("Wrote", fdny)

    if args.legacy_ems:
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
            print("Legacy EMS filter failed (%s); downloading without year/date filter…" % exc)
            ems = download_dataset("ems_incidents", args.out, limit=args.limit, years=None)
        print("Wrote", ems)


if __name__ == "__main__":
    main()
