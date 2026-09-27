"""First-due firetruck start inference.

Chain:
  1. Locate the incident at its alarm-box coordinates (v57i-gtxb), joined by
     borough + box number — same geometry the city already geocoded, so we skip
     a separate Geosupport pass for unique boxes.
  2. Spatially join to FDNY Fire Company engine polygons (bst7-5464).
  3. Map the first-due engine number to its firehouse (hc8x-tcnd) via
     ``facilityname`` ("Engine 22/Ladder 13" → Engine 22).
  4. QC: crow-flies implied speed vs observed travel time; flag when the
     first-due house is not the nearest house (relocated / on-road likely).

Falls back to nearest-firehouse (and optional CSL hybrid) when any step fails.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .depots import (
    _haversine_km,
    _nearest_depot,
    _source_label,
    normalize_borough,
    prepare_depots,
    prepare_firehouses,
)

ENGINE_RE = re.compile(r"\bEngine\s+(\d+)\b", re.I)

# Alarm-box borough → borobox letter prefix used by v57i-gtxb.
BOX_BOROUGH_PREFIX = {
    "BRONX": "X",
    "BROOKLYN": "B",
    "MANHATTAN": "M",
    "QUEENS": "Q",
    "STATEN ISLAND": "R",
    "RICHMOND": "R",
    "RICHMOND / STATEN ISLAND": "R",
}

# Plausible EMV average speeds (km/h) over *crow-flies* OD. Crow-flies understates
# true path length, so the lower band is soft; the high band (origin too far) is
# the main on-road / wrong-company detector until network distances are wired in.
SPEED_KPH_LO = 1.5
SPEED_KPH_HI = 80.0


def _box_prefix(borough: Any) -> str | None:
    if borough is None or (isinstance(borough, float) and math.isnan(borough)):
        return None
    key = str(borough).strip().upper()
    if key in BOX_BOROUGH_PREFIX:
        return BOX_BOROUGH_PREFIX[key]
    if "STATEN" in key or "RICHMOND" in key:
        return "R"
    return BOX_BOROUGH_PREFIX.get(normalize_borough(borough) or "")


def alarm_box_join_keys(incidents: pd.DataFrame) -> pd.Series:
    """Build ``borobox`` keys (e.g. B1077) for joining to in-service alarm boxes."""
    box_num = pd.to_numeric(incidents.get("alarm_box_number"), errors="coerce")
    # Zero-pad to 4 digits to match borobox's numeric tail (B0801, M0897, …).
    num_str = box_num.map(lambda x: f"{int(x):04d}" if pd.notna(x) else pd.NA)
    pfx = incidents.get("alarm_box_borough")
    if pfx is None:
        pfx = pd.Series([pd.NA] * len(incidents), index=incidents.index)
    pfx = pfx.map(_box_prefix)
    # Fall back to incident borough when alarm_box_borough is missing / odd.
    if "borough" in incidents.columns or "incident_borough" in incidents.columns:
        boro_col = "borough" if "borough" in incidents.columns else "incident_borough"
        miss = pfx.isna()
        if miss.any():
            pfx = pfx.copy()
            pfx.loc[miss] = incidents.loc[miss, boro_col].map(_box_prefix)
    out = pd.Series(pd.NA, index=incidents.index, dtype=object)
    ok = pfx.notna() & num_str.notna()
    out.loc[ok] = pfx.loc[ok].astype(str) + num_str.loc[ok].astype(str)
    return out


def locate_incidents_at_alarm_boxes(
    incidents: pd.DataFrame,
    alarm_boxes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach alarm-box lat/lon; leave null when the box is not in the listing."""
    boxes = alarm_boxes.copy()
    boxes["borobox"] = boxes["borobox"].astype(str).str.strip()
    lat_col = "latitude" if "latitude" in boxes.columns else "lat"
    lon_col = "longitude" if "longitude" in boxes.columns else "lon"
    keep = boxes[["borobox", lat_col, lon_col]].rename(
        columns={lat_col: "box_lat", lon_col: "box_lon"}
    )
    if "location" in boxes.columns:
        keep = keep.assign(box_location=boxes["location"].values)
    keep["box_lat"] = pd.to_numeric(keep["box_lat"], errors="coerce")
    keep["box_lon"] = pd.to_numeric(keep["box_lon"], errors="coerce")
    keep = keep.dropna(subset=["box_lat", "box_lon"]).drop_duplicates("borobox")

    out = incidents.copy()
    out["borobox"] = alarm_box_join_keys(out)
    out = out.merge(keep, on="borobox", how="left")
    out["incident_geocode_source"] = np.where(
        out["box_lat"].notna(), "alarm_box_listing", "unmatched_alarm_box"
    )
    return out


def build_engine_firehouse_lookup(firehouses: pd.DataFrame) -> pd.DataFrame:
    """One row per engine number → firehouse coordinates."""
    houses = prepare_firehouses(firehouses)
    rows = []
    for _, r in houses.iterrows():
        name = str(r.get("depot_name") or "")
        for m in ENGINE_RE.findall(name):
            rows.append(
                {
                    "engine_num": int(m),
                    "depot_id": int(r["depot_id"]),
                    "depot_name": r.get("depot_name"),
                    "depot_layer": "fdny_firehouse",
                    "start_lat": float(r["start_lat"]),
                    "start_lon": float(r["start_lon"]),
                    "borough": r.get("borough"),
                    "bay_tag": r.get("bay_tag"),
                }
            )
    if not rows:
        raise ValueError("No Engine N tokens parsed from firehouse facilityname")
    lookup = pd.DataFrame(rows).drop_duplicates("engine_num", keep="first")
    return lookup


def load_engine_company_boundaries(path: Path | str):
    """Load bst7-5464 GeoJSON and keep engine polygons only."""
    import geopandas as gpd

    path = Path(path)
    gdf = gpd.read_file(path)
    type_col = "fire_co_type" if "fire_co_type" in gdf.columns else None
    num_col = "fire_co_num" if "fire_co_num" in gdf.columns else None
    if type_col is None or num_col is None:
        raise ValueError(f"Unexpected fire-company columns: {list(gdf.columns)}")
    eng = gdf[gdf[type_col].astype(str).str.upper().isin(["E", "ENGINE"])].copy()
    eng["engine_num"] = pd.to_numeric(eng[num_col], errors="coerce")
    eng = eng.dropna(subset=["engine_num"]).copy()
    eng["engine_num"] = eng["engine_num"].astype(int)
    if eng.crs is None:
        eng = eng.set_crs(4326)
    else:
        eng = eng.to_crs(4326)
    return eng


def assign_first_due_engine(
    located: pd.DataFrame,
    engine_boundaries,
    *,
    nearest_max_ft: float = 1500.0,
) -> pd.DataFrame:
    """Spatially join incident points to first-due engine polygons (NY State Plane).

    Prefer ``within``; if the alarm-box point sits just outside a boundary (common
    for curb intersections), fall back to nearest engine within ``nearest_max_ft``.
    """
    import geopandas as gpd

    out = located.copy()
    has_pt = out["box_lat"].notna() & out["box_lon"].notna()
    out["first_due_engine"] = np.nan
    out["first_due_join"] = "no_point"

    if not has_pt.any():
        return out

    pts = gpd.GeoDataFrame(
        out.loc[has_pt].copy(),
        geometry=gpd.points_from_xy(
            out.loc[has_pt, "box_lon"], out.loc[has_pt, "box_lat"]
        ),
        crs=4326,
    )
    # NY State Plane (ft) — more reliable for polygon-within than geographic CRS.
    pts_sp = pts.to_crs(2263)
    cos_sp = engine_boundaries.to_crs(2263)
    joined = gpd.sjoin(
        pts_sp,
        cos_sp[["engine_num", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    engines = joined["engine_num"].copy()
    how = pd.Series(
        np.where(engines.notna(), "within_engine", "outside_polygons"),
        index=joined.index,
    )

    miss = engines.isna()
    if miss.any():
        near = gpd.sjoin_nearest(
            pts_sp.loc[miss, ["geometry"]],
            cos_sp[["engine_num", "geometry"]],
            how="left",
            max_distance=nearest_max_ft,
            distance_col="_dist_ft",
        )
        near = near[~near.index.duplicated(keep="first")]
        hit = near["engine_num"].notna()
        engines.loc[near.index[hit]] = near.loc[hit, "engine_num"].to_numpy()
        how.loc[near.index[hit]] = "nearest_engine"
        how.loc[near.index[~hit]] = "outside_polygons"

    out.loc[has_pt, "first_due_engine"] = engines.to_numpy()
    out.loc[has_pt, "first_due_join"] = how.to_numpy()
    return out


def _nearest_firehouse_row(houses: pd.DataFrame, lon: float, lat: float):
    return _nearest_depot(houses, lon, lat)


def qc_first_due_origins(
    od: pd.DataFrame,
    firehouses: pd.DataFrame,
    *,
    speed_lo: float = SPEED_KPH_LO,
    speed_hi: float = SPEED_KPH_HI,
) -> pd.DataFrame:
    """Flag implausible implied speeds and non-nearest first-due houses."""
    out = od.copy()
    travel = pd.to_numeric(out.get("travel_seconds"), errors="coerce")
    dist = pd.to_numeric(out.get("crow_flies_km"), errors="coerce")
    hours = travel / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = dist / hours.replace(0, np.nan)
    out["implied_speed_kph"] = speed
    out["qc_speed_flag"] = "ok"
    out.loc[speed > speed_hi, "qc_speed_flag"] = "too_fast_origin_far"
    out.loc[speed < speed_lo, "qc_speed_flag"] = "too_slow_origin_close_or_error"
    out.loc[travel.isna() | dist.isna() | (travel <= 0), "qc_speed_flag"] = "unknown"

    houses = prepare_firehouses(firehouses)
    h_lon = houses["start_lon"].to_numpy()
    h_lat = houses["start_lat"].to_numpy()
    h_ids = houses["depot_id"].to_numpy()
    nearest_id = np.full(len(out), -1, dtype=int)
    nearest_km = np.full(len(out), np.nan)
    dlon = pd.to_numeric(out.get("dest_lon"), errors="coerce").to_numpy()
    dlat = pd.to_numeric(out.get("dest_lat"), errors="coerce").to_numpy()
    for i in range(len(out)):
        if np.isnan(dlon[i]) or np.isnan(dlat[i]):
            continue
        dists = _haversine_km(h_lon, h_lat, dlon[i], dlat[i])
        j = int(np.argmin(dists))
        nearest_id[i] = int(h_ids[j])
        nearest_km[i] = float(dists[j])
    out["nearest_firehouse_depot_id"] = nearest_id
    out["nearest_firehouse_km"] = nearest_km
    first_due_ids = pd.to_numeric(out.get("depot_id"), errors="coerce")
    out["qc_not_nearest_house"] = (
        (out.get("depot_layer") == "fdny_firehouse")
        & first_due_ids.notna()
        & (first_due_ids.astype(int) != out["nearest_firehouse_depot_id"])
        & (out["nearest_firehouse_depot_id"] >= 0)
        & (out.get("start_mode") == "first_due")
    )
    out["qc_keep"] = (out["qc_speed_flag"] == "ok") & (~out["qc_not_nearest_house"].fillna(False))
    return out


def infer_first_due_starts(
    incidents: pd.DataFrame,
    firehouses: pd.DataFrame,
    zip_centroids: pd.DataFrame,
    alarm_boxes: pd.DataFrame,
    engine_boundaries,
    *,
    csl_points: pd.DataFrame | None = None,
    fallback_mode: str = "hybrid",
) -> pd.DataFrame:
    """
    Infer firetruck starts via first-due company; fall back to nearest house / CSL.

    Destination prefers the alarm-box point; ZIP centroid is the fallback dest
    (and the only dest when the box listing miss).
    """
    from .depots import infer_start_locations  # local import avoids cycle at module load

    located = locate_incidents_at_alarm_boxes(incidents, alarm_boxes)
    located = assign_first_due_engine(located, engine_boundaries)
    lookup = build_engine_firehouse_lookup(firehouses)
    lookup_map = lookup.set_index("engine_num")

    zips = zip_centroids.copy()
    zips["zipcode"] = zips["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    located = located.copy()
    if "zipcode" in located.columns:
        located["zipcode"] = (
            located["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
        )
    located = located.merge(zips, on="zipcode", how="left", suffixes=("", "_zip"))

    # Destination: alarm box when known, else ZIP centroid columns from merge.
    dest_lat = np.where(
        located["box_lat"].notna(), located["box_lat"], located.get("dest_lat")
    )
    dest_lon = np.where(
        located["box_lon"].notna(), located["box_lon"], located.get("dest_lon")
    )
    # zip_centroids_from_modzcta may use lat/lon instead of dest_*.
    if "lat" in located.columns:
        dest_lat = np.where(pd.isna(dest_lat), located["lat"], dest_lat)
        dest_lon = np.where(pd.isna(dest_lon), located["lon"], dest_lon)
    located["dest_lat"] = pd.to_numeric(dest_lat, errors="coerce")
    located["dest_lon"] = pd.to_numeric(dest_lon, errors="coerce")
    located["dest_source"] = np.where(
        located["box_lat"].notna(), "alarm_box", "zip_centroid"
    )

    n = len(located)
    start_lat = np.full(n, np.nan)
    start_lon = np.full(n, np.nan)
    depot_id = np.full(n, -1, dtype=int)
    depot_name = np.array([None] * n, dtype=object)
    depot_layer = np.array([None] * n, dtype=object)
    start_source = np.array(["missing"] * n, dtype=object)
    start_mode = np.array(["first_due"] * n, dtype=object)
    crow = np.full(n, np.nan)
    engine_nums = np.full(n, np.nan)

    for i, (_, row) in enumerate(located.iterrows()):
        eng = row.get("first_due_engine")
        if pd.isna(eng):
            continue
        eng_i = int(eng)
        if eng_i not in lookup_map.index:
            continue
        house = lookup_map.loc[eng_i]
        # If duplicate index somehow, take first.
        if isinstance(house, pd.DataFrame):
            house = house.iloc[0]
        dlat, dlon = row.get("dest_lat"), row.get("dest_lon")
        if pd.isna(dlat) or pd.isna(dlon):
            continue
        start_lat[i] = float(house["start_lat"])
        start_lon[i] = float(house["start_lon"])
        depot_id[i] = int(house["depot_id"])
        depot_name[i] = house["depot_name"]
        depot_layer[i] = "fdny_firehouse"
        prefer_b = normalize_borough(row.get("borough") or row.get("incident_borough"))
        start_source[i] = (
            f"first_due_engine_{eng_i}_"
            + _source_label(house, prefer_b, scope="firehouse")
        )
        crow[i] = float(
            _haversine_km(
                float(house["start_lon"]),
                float(house["start_lat"]),
                float(dlon),
                float(dlat),
            )
        )
        engine_nums[i] = eng_i

    located["start_lat"] = start_lat
    located["start_lon"] = start_lon
    located["depot_id"] = depot_id
    located["depot_name"] = depot_name
    located["depot_layer"] = depot_layer
    located["start_source"] = start_source
    located["start_mode"] = start_mode
    located["crow_flies_km"] = crow
    located["first_due_engine"] = engine_nums

    # Fallback for rows that never got a first-due house.
    need_fb = located["start_lat"].isna() | located["dest_lat"].isna()
    if int(need_fb.sum()):
        # Remember alarm-box destinations so ZIP-centroid fallback doesn't wipe them.
        saved_dest = located.loc[need_fb, ["dest_lat", "dest_lon", "dest_source", "box_lat"]].copy()
        fb_inc = located.loc[need_fb].copy()
        fb = infer_start_locations(
            fb_inc,
            firehouses,
            zip_centroids,
            csl_points=csl_points,
            start_mode=fallback_mode,
        )
        for col in (
            "start_lat",
            "start_lon",
            "depot_id",
            "depot_name",
            "depot_layer",
            "start_source",
            "start_mode",
            "crow_flies_km",
        ):
            if col in fb.columns:
                located.loc[need_fb, col] = fb[col].to_numpy()

        # Restore alarm-box dest when we had one; recompute crow-flies + tag mode.
        fb_idx = located.index[need_fb]
        for idx in fb_idx:
            if pd.notna(saved_dest.at[idx, "box_lat"]):
                located.at[idx, "dest_lat"] = saved_dest.at[idx, "dest_lat"]
                located.at[idx, "dest_lon"] = saved_dest.at[idx, "dest_lon"]
                located.at[idx, "dest_source"] = "alarm_box"
            r = located.loc[idx]
            if pd.notna(r.get("start_lat")) and pd.notna(r.get("dest_lat")):
                located.at[idx, "crow_flies_km"] = float(
                    _haversine_km(
                        float(r["start_lon"]),
                        float(r["start_lat"]),
                        float(r["dest_lon"]),
                        float(r["dest_lat"]),
                    )
                )
            if not str(r.get("start_mode", "")).startswith("first_due"):
                located.at[idx, "start_mode"] = f"fallback_{fallback_mode}"

    located = qc_first_due_origins(located, firehouses)
    return located


def first_due_summary(od: pd.DataFrame) -> dict[str, Any]:
    usable = od.dropna(subset=["start_lat", "dest_lat"])
    return {
        "rows": int(len(od)),
        "usable": int(len(usable)),
        "alarm_box_dest_share": float((od.get("dest_source") == "alarm_box").mean())
        if "dest_source" in od.columns and len(od)
        else 0.0,
        "first_due_share": float((od.get("start_mode") == "first_due").mean())
        if "start_mode" in od.columns and len(od)
        else 0.0,
        "start_mode_counts": usable["start_mode"].value_counts().to_dict()
        if len(usable) and "start_mode" in usable.columns
        else {},
        "qc_speed_counts": usable["qc_speed_flag"].value_counts().to_dict()
        if len(usable) and "qc_speed_flag" in usable.columns
        else {},
        "qc_not_nearest_share": float(usable["qc_not_nearest_house"].mean())
        if len(usable) and "qc_not_nearest_house" in usable.columns
        else 0.0,
        "qc_keep_share": float(usable["qc_keep"].mean())
        if len(usable) and "qc_keep" in usable.columns
        else 0.0,
        "median_implied_speed_kph": float(
            pd.to_numeric(usable.get("implied_speed_kph"), errors="coerce").median()
        )
        if len(usable)
        else None,
    }


def write_first_due_meta(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n")
