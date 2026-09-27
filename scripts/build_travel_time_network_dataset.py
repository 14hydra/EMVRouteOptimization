#!/usr/bin/env python3
"""
Build the FDNY travel-time table with street-network origin features.

Replaces the single guessed start (first-due house, else nearest house / synthetic
on-road point, measured crow-flies) with label-free network descriptions of all
plausible origins — see ``emvro.network_origins``. Scales to a full year+ of CAD.

  PYTHONPATH=src python scripts/build_travel_time_network_dataset.py \
      --raw data/raw/big --out data/processed/travel_time_network.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_od_pairs import normalize_fdny_incidents  # noqa: E402
from emvro.depots import normalize_borough  # noqa: E402
from emvro.first_due import locate_incidents_at_alarm_boxes  # noqa: E402
from emvro.geometry import zip_centroids_from_modzcta  # noqa: E402
from emvro.intersections import geocode_box_locations  # noqa: E402
from emvro.network_origins import (  # noqa: E402
    DriveGraph,
    assign_first_due_companies,
    company_to_house,
    firehouse_table,
    network_origin_features,
)
from emvro.weather import join_weather_features  # noqa: E402


# Rough borough centres: search anchor when an incident has no usable ZIP.
BOROUGH_CENTER = {
    "MANHATTAN": (40.7831, -73.9712),
    "BROOKLYN": (40.6500, -73.9496),
    "QUEENS": (40.7282, -73.7949),
    "BRONX": (40.8448, -73.8648),
    "STATEN ISLAND": (40.5795, -74.1502),
}


def _geocode_unlisted_boxes(inc: pd.DataFrame, centerline_path: Path) -> pd.DataFrame:
    import geopandas as gpd

    from emvro.street_analysis import CENTERLINE_ID, CENTERLINE_SELECT, fetch_geo_dataset

    un = inc[inc["box_lat"].isna() & inc["alarm_box_location"].notna()]
    locs = (
        un.groupby(["alarm_box_location", "borough"], dropna=False)
        .agg(zip_lat=("zip_lat", "first"), zip_lon=("zip_lon", "first"))  # first non-null
        .reset_index()
    )
    locs["max_km"] = 4.0
    boro = locs["borough"].map(normalize_borough)
    miss = locs["zip_lat"].isna()
    locs.loc[miss, "zip_lat"] = boro[miss].map(lambda b: BOROUGH_CENTER.get(b, (np.nan,))[0])
    locs.loc[miss, "zip_lon"] = boro[miss].map(lambda b: BOROUGH_CENTER.get(b, (np.nan, np.nan))[1])
    locs.loc[miss, "max_km"] = 15.0

    if centerline_path.exists():
        cl = gpd.read_file(centerline_path)
    else:
        cl = fetch_geo_dataset(CENTERLINE_ID, centerline_path, select=CENTERLINE_SELECT)
    locs = locs.join(geocode_box_locations(locs, cl))
    print(
        f"unlisted box locations: {len(locs):,}, geocoded {locs['geo_lat'].notna().mean():.1%}"
    )
    return locs[["alarm_box_location", "borough", "geo_lat", "geo_lon"]]


LOAD_WINDOWS_MIN = (15, 30, 60)


def _recent_load(times: pd.Series, keys: pd.Series, windows_min=LOAD_WINDOWS_MIN) -> pd.DataFrame:
    """Past-only workload per area: incidents in the preceding window(s) and time
    since the previous incident with the same key (excludes the row itself).

    A busy first-due company is the main reason the first-arriving unit comes
    from farther away than its house.
    """
    t = ((times - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    k = np.array([str(x) for x in keys], dtype=object)  # NaN -> "nan" (pandas 3 keeps NA in astype(str))
    order = np.lexsort((t, k))
    ts, ks = t[order], k[order]
    out = {f"n_prev_{w}m": np.zeros(len(t)) for w in windows_min}
    gap = np.full(len(t), np.nan)
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1, len(ks)]
    for a, b in zip(starts[:-1], starts[1:]):
        g = ts[a:b]
        pos = np.arange(b - a)
        for w in windows_min:
            out[f"n_prev_{w}m"][a:b] = pos - np.searchsorted(g, g - w * 60, side="left")
        gap[a + 1 : b] = np.diff(g)
    res = pd.DataFrame({c: np.empty(len(t)) for c in out})
    for c, v in out.items():
        res.loc[order, c] = v
    res["min_since_prev"] = np.nan
    res.loc[order, "min_since_prev"] = np.minimum(gap / 60.0, 24 * 60)
    res.loc[keys.isna().to_numpy(), :] = np.nan  # no area → no load, not one giant area
    return res


def _previous_design_features(inc: pd.DataFrame, houses: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    from emvro.depots import _haversine_km
    from emvro.network_origins import company_to_house

    listed = inc["box_lat"].notna().to_numpy()
    dlat = np.where(listed, inc["box_lat"], inc["zip_lat"]).astype(float)
    dlon = np.where(listed, inc["box_lon"], inc["zip_lon"]).astype(float)
    hlat, hlon = houses["lat"].to_numpy(), houses["lon"].to_numpy()
    k = np.cos(np.radians(40.7))
    tree = cKDTree(np.c_[hlon * k, hlat])
    ok = np.isfinite(dlat) & np.isfinite(dlon)
    near = np.zeros(len(inc), dtype=int)
    near[ok] = tree.query(np.c_[dlon[ok] * k, dlat[ok]])[1]
    crow_near = np.where(ok, _haversine_km(hlon[near], hlat[near], dlon, dlat), np.nan)

    eng_house = company_to_house(houses, "E")
    eng = pd.to_numeric(inc["first_due_engine"], errors="coerce").to_numpy()
    h = np.array([eng_house.get(int(e), -1) if e == e else -1 for e in eng])
    use = listed & (h >= 0)
    crow_primary = crow_near.copy()
    crow_primary[use] = _haversine_km(hlon[h[use]], hlat[h[use]], dlon[use], dlat[use])
    return pd.DataFrame(
        {
            "prev_dest_lat": dlat,
            "prev_dest_lon": dlon,
            "prev_dest_is_box": listed.astype(int),
            "prev_crow_km_primary": crow_primary,
            "prev_crow_km_nearest": crow_near,
        }
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "big")
    p.add_argument(
        "--centerline",
        type=Path,
        default=ROOT / "data" / "raw" / "centerline.geojson",
        help="NYC Centerline (inkn-q76z) GeoJSON; downloaded if missing",
    )
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument(
        "--rich-graph",
        type=Path,
        default=ROOT / "data" / "raw" / "nyc_drive_rich.graphml",
        help="Drive graph built with build_osm_graph.py --rich-tags (route street features)",
    )
    p.add_argument(
        "--edge-table",
        type=Path,
        default=ROOT / "data" / "processed" / "street_edge_table.pkl",
        help="Street attributes per edge from analyze_street_characteristics.py",
    )
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "travel_time_network.parquet")
    args = p.parse_args()

    raw = pd.read_csv(args.raw / "fdny_incidents.csv", low_memory=False)
    inc = normalize_fdny_incidents(raw).reset_index(drop=True)
    print(f"incidents: {len(raw):,} downloaded -> {len(inc):,} with apparatus + valid 30–3600 s travel")

    # --- Destination: alarm box, else ZIP centroid --------------------------
    alarm_boxes = pd.read_csv(args.raw / "alarm_boxes.csv", low_memory=False)
    inc = locate_incidents_at_alarm_boxes(inc, alarm_boxes)
    zips = zip_centroids_from_modzcta(pd.read_csv(args.raw / "modzcta.csv", low_memory=False))
    zips["zipcode"] = zips["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    inc["zipcode"] = inc["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    zc = zips.set_index("zipcode")
    lat_c = "dest_lat" if "dest_lat" in zc.columns else "lat"
    lon_c = "dest_lon" if "dest_lon" in zc.columns else "lon"
    inc["zip_lat"] = inc["zipcode"].map(zc[lat_c])
    inc["zip_lon"] = inc["zipcode"].map(zc[lon_c])

    # Boxes missing from the in-service listing: geocode the CAD intersection
    # string on NYC Centerline instead of dropping to the ZIP centroid.
    geo = _geocode_unlisted_boxes(inc, args.centerline)
    inc = inc.merge(geo, on=["alarm_box_location", "borough"], how="left")
    inc["dest_lat"] = inc["box_lat"].fillna(inc["geo_lat"]).fillna(inc["zip_lat"])
    inc["dest_lon"] = inc["box_lon"].fillna(inc["geo_lon"]).fillna(inc["zip_lon"])
    inc["dest_source"] = np.select(
        [inc["box_lat"].notna(), inc["geo_lat"].notna()],
        ["alarm_box", "intersection_geocode"],
        default="zip_centroid",
    )
    inc = inc.dropna(subset=["dest_lat", "dest_lon"]).reset_index(drop=True)
    print("dest source:", inc["dest_source"].value_counts().to_dict())

    # --- Network origin features on unique destinations ---------------------
    dests = inc[["dest_lat", "dest_lon"]].round(6).drop_duplicates().reset_index(drop=True)
    dests = dests.join(assign_first_due_companies(dests, args.raw / "fire_companies.geojson"))
    houses = firehouse_table(pd.read_csv(args.raw / "fdny_firehouses.csv"))
    g = DriveGraph.from_graphml(args.graph, cache=args.processed / "drive_csr.npz")
    net = network_origin_features(
        dests, houses, g, company_to_house(houses, "E"), company_to_house(houses, "L")
    )
    dests = pd.concat([dests, net], axis=1)

    # Street characteristics along the fastest route from the first-due house.
    if args.rich_graph.exists() and args.edge_table.exists():
        import pickle

        from emvro.route_street_features import StreetGraph, route_street_features

        sg = StreetGraph(args.rich_graph, pickle.loads(args.edge_table.read_bytes()))
        rt = route_street_features(dests, houses, company_to_house(houses, "E"), sg)
        dests = pd.concat([dests, rt], axis=1)
        net = pd.concat([net, rt], axis=1)
        print(f"route street features: {rt['rt_path_km'].notna().mean():.1%} of destinations")
    else:
        print("skipping route street features (need --rich-graph and --edge-table)")
    key = ["dest_lat", "dest_lon"]
    inc[key] = inc[key].round(6)
    inc = inc.merge(dests, on=key, how="left")
    print(f"unique destinations: {len(dests):,}")

    # --- Context features (all known at assignment time) --------------------
    dt = pd.to_datetime(inc["incident_datetime"], errors="coerce")
    num = lambda c: pd.to_numeric(inc.get(c), errors="coerce")  # noqa: E731
    out = pd.DataFrame(
        {
            "incident_id": inc["incident_id"].astype(str),
            "incident_datetime": dt,
            "travel_seconds": num("travel_seconds"),
            "dispatch_wait_seconds": num("dispatch_wait_seconds"),
            "borough": inc["borough"].map(normalize_borough),
            "zipcode": inc["zipcode"],
            "borobox": inc["borobox"].astype(str),
            "dest_source": inc["dest_source"],
            "dest_lat": inc["dest_lat"],
            "dest_lon": inc["dest_lon"],
            "first_due_engine": inc["first_due_engine"],
            "first_due_ladder": inc["first_due_ladder"],
            "call_type": inc["initial_call_type"].astype(str),
            "call_group": inc["final_call_type"].astype(str),
            "alarm_level": inc.get("highest_alarm_level", pd.Series(index=inc.index)).astype(str),
            "engines_assigned": num("engines_assigned_quantity"),
            "ladders_assigned": num("ladders_assigned_quantity"),
            "other_units_assigned": num("other_units_assigned_quantity"),
            "hour": dt.dt.hour,
            "dow": dt.dt.dayofweek,
            "month": dt.dt.month,
            "minute_of_day": dt.dt.hour * 60 + dt.dt.minute,
        }
    )
    for c in net.columns:
        out[c] = inc[c].to_numpy()

    # Unit availability proxies (past incidents only; known at dispatch time).
    dt_ok = out["incident_datetime"].fillna(pd.Timestamp("1970-01-01"))
    for name, key in (
        ("engine", out["first_due_engine"]),
        ("ladder", out["first_due_ladder"]),
        ("borough", out["borough"]),
    ):
        load = _recent_load(dt_ok, key)
        for c in load.columns:
            out[f"{name}_{c}"] = load[c].to_numpy()

    # Previous design, kept for a like-for-like comparison: destination = listed
    # box else ZIP centroid; origin = first-due engine house when the box is
    # listed, else nearest house; distances crow-flies.
    old = _previous_design_features(inc, houses)
    for c in old.columns:
        out[c] = old[c].to_numpy()
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_rush"] = out["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] < 6)).astype(int)

    out = join_weather_features(out, cache_path=args.processed / "weather_hourly_nyc_big.csv")

    # Previous-hour traffic congestion (NYC DOT Traffic Speeds).
    th, tl = args.raw / "traffic_hourly.parquet", args.raw / "traffic_links.csv"
    if th.exists() and tl.exists():
        from emvro.traffic import attach_traffic_features

        out = attach_traffic_features(out, pd.read_parquet(th), pd.read_csv(tl))
        print(f"traffic features: {out['traffic_city_index'].notna().mean():.1%} of incidents")
    else:
        print("skipping traffic features (run emvro.traffic.download_hourly_speeds)")
    out = out.sort_values("incident_datetime").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    summary = {
        "rows": int(len(out)),
        "date_min": str(out["incident_datetime"].min()),
        "date_max": str(out["incident_datetime"].max()),
        "dest_source": out["dest_source"].value_counts().to_dict(),
        "first_due_engine_share": float(out["first_due_engine"].notna().mean()),
        "first_due_ladder_share": float(out["first_due_ladder"].notna().mean()),
        "net_s_engine_nonnull": float(out["net_s_engine"].notna().mean()),
        "weather_nonnull": float(out["wx_temp_c"].notna().mean()),
        "median_travel_s": float(out["travel_seconds"].median()),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
