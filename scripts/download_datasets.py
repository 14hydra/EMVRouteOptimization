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
        help="EMS incident years to prefer (SODA filter)",
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

    for key in ("fdny_firehouses", "ems_stations", "hospital_bays", "modzcta"):
        path = download_dataset(key, args.out, limit=5000)
        print("Wrote", path)

    # Alarm boxes are numerous; cap for local development.
    boxes = download_dataset("alarm_boxes", args.out, limit=args.alarm_box_limit)
    print("Wrote", boxes)

    try:
        ems = download_dataset(
            "ems_incidents",
            args.out,
            limit=args.limit,
            years=args.years,
        )
    except Exception as exc:  # noqa: BLE001
        print("Year filter failed (%s); downloading without year filter…" % exc)
        ems = download_dataset("ems_incidents", args.out, limit=args.limit, years=None)
    print("Wrote", ems)

    lion_note = args.out / "lion.README.txt"
    lion_note.write_text(
        "LION Open Data ID: 2v4z-66xt\n"
        "Download the full street centerline extract from NYC DCP / Open Data GeoJSON when ready.\n"
        "Synthetic CSLs currently use FDNY alarm-box intersections + EMS volume weights;\n"
        "OSM/LION intersection extraction is the next upgrade path.\n"
    )
    print("Wrote", lion_note)


if __name__ == "__main__":
    main()
