#!/usr/bin/env python3
"""
Build data/raw/london/london_firehouses.csv: the real location of every London
Fire Brigade station, from OpenStreetMap (amenity=fire_station).

Each LFB station name used in the incident / mobilisation data is matched to an
OSM fire station by name ("Soho Fire Station" -> "Soho"). Eight stations are
unnamed in OSM or named differently ("Heston & Isleworth"); they are pinned to
specific OSM features in OSM_OVERRIDES. The four unnamed ones were checked
against the station's published address (postcode geocoded with postcodes.io
landed within ~100 m of the OSM building).

The coordinate is a point inside the station building's footprint (or the OSM
node), so it is the station itself, not the centre of its station ground.

Run:  .venv/bin/python scripts/build_london_firehouses.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "london"
BBOX = (-0.52, 51.27, 0.35, 51.70)  # (west, south, east, north), Greater London + margin

# LFB station name -> (OSM element, OSM id), for stations OSM leaves unnamed or names differently.
OSM_OVERRIDES = {
    "Bexley": ("way", 300944178),  # OSM "Bexleyheath Fire Station", Erith Road DA7 6BY
    "Enfield": ("way", 97874322),  # unnamed; 93 Carterhatch Lane EN1 4LA
    "Forest Hill": ("node", 26992679),  # unnamed; 155 Stanstead Road SE23 1HP
    "Wennington": ("way", 912781160),  # unnamed; Wennington Road RM13 9EE
    "Wimbledon": ("way", 1213265756),  # unnamed; 87 Kingston Road SW19 1JN
    "Heston": ("way", 975647650),  # OSM "Heston & Isleworth Fire Station (G38)", London Road TW7 4HR
    "Wallington": ("node", 33717225),  # OSM "Beddington & Wallington Fire Station"
    "Woodside": ("way", 478166743),  # OSM "H28 Woodside Fire Station", Long Lane CR0 7AL
}

# Sanity check: a matched station should sit inside (or next to) its own station ground.
MAX_KM_FROM_GROUND = 4.0


def _norm(name) -> str:
    s = re.sub(r"\(.*?\)", "", str(name).lower())
    s = re.sub(r"\b(fire|station|stn|community|lfb|london fire brigade)\b", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def fetch_osm() -> pd.DataFrame:
    import osmnx as ox

    ox.settings.use_cache = True
    g = ox.features_from_bbox(BBOX, {"amenity": "fire_station"}).reset_index()
    pt = g.geometry.representative_point()
    return pd.DataFrame(
        {
            "element": g["element"],
            "osm_id": g["id"].astype("int64"),
            "osm_name": g.get("name"),
            "operator": g.get("operator"),
            "lat": pt.y,
            "lon": pt.x,
        }
    )


def station_ground_centroids(planner: Path) -> pd.DataFrame:
    d = pd.read_csv(planner, usecols=["station_ground", "dest_lat", "dest_lon"])
    return d.groupby("station_ground")[["dest_lat", "dest_lon"]].median()


def match(osm: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    osm = osm.assign(_n=osm["osm_name"].map(_norm), _lfb=osm["operator"].fillna("").str.contains("London Fire"))
    rows = []
    for name in names:
        if name in OSM_OVERRIDES:
            el, oid = OSM_OVERRIDES[name]
            m, how = osm[(osm["element"] == el) & (osm["osm_id"] == oid)], "override"
        else:
            m, how = osm[osm["_n"] == _norm(name)], "name"
        if m.empty:
            rows.append({"name": name, "match": "missing"})
            continue
        # Duplicate OSM features for one station (node + building): prefer LFB-tagged, then building.
        best = m.sort_values(["_lfb", "element"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "name": name,
                "lat": best["lat"],
                "lon": best["lon"],
                "osm_element": best["element"],
                "osm_id": int(best["osm_id"]),
                "osm_name": best["osm_name"],
                "match": how,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", type=Path, default=RAW / "lfb_incidents_planner.csv")
    ap.add_argument("--out", type=Path, default=RAW / "london_firehouses.csv")
    args = ap.parse_args()

    ground = station_ground_centroids(args.planner)
    out = match(fetch_osm(), sorted(ground.index))

    missing = out[out["match"] == "missing"]["name"].tolist()
    if missing:
        print("No OSM match for:", missing, file=sys.stderr)
        return 1
    g = ground.loc[out["name"]]
    out["km_from_ground_centre"] = np.hypot(
        (out["lat"].to_numpy() - g["dest_lat"].to_numpy()) * 111.2,
        (out["lon"].to_numpy() - g["dest_lon"].to_numpy()) * 111.2 * np.cos(np.radians(51.5)),
    ).round(2)
    far = out[out["km_from_ground_centre"] > MAX_KM_FROM_GROUND]
    if len(far):
        print("Matched station far from its own ground (check):\n", far.to_string(), file=sys.stderr)
        return 1

    out["source"] = "OpenStreetMap amenity=fire_station"
    out.drop(columns="match").to_csv(args.out, index=False)
    print(f"{len(out)} stations -> {args.out}")
    print(f"distance station -> centre of its ground: median {out['km_from_ground_centre'].median():.2f} km, "
          f"max {out['km_from_ground_centre'].max():.2f} km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
