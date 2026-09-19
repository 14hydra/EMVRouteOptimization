#!/usr/bin/env python3
"""Build / cache an OSMNx drive graph covering NYC for path features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.street_features import build_nyc_drive_graph  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "raw" / "nyc_drive.graphml",
    )
    p.add_argument(
        "--rich-tags",
        action="store_true",
        help="Also keep bus-lane / bike-lane / parking-lane tags (needed by analyze_street_characteristics.py)",
    )
    args = p.parse_args()
    path = build_nyc_drive_graph(args.out, rich_tags=args.rich_tags)
    print("Wrote", path, "size_mb=", round(path.stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()
