"""Infer EMV starting locations (depots) missing from public EMS CAD extracts."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

# First letter of INCIDENT_DISPATCH_AREA → borough used by FDNY CAD sectors.
DISPATCH_PREFIX_BOROUGH = {
    "B": "BRONX",
    "X": "BRONX",
    "K": "BROOKLYN",
    "M": "MANHATTAN",
    "Q": "QUEENS",
    "S": "STATEN ISLAND",
    "R": "STATEN ISLAND",
}

BOROUGH_ALIASES = {
    "BRONX": "BRONX",
    "BROOKLYN": "BROOKLYN",
    "MANHATTAN": "MANHATTAN",
    "QUEENS": "QUEENS",
    "STATEN ISLAND": "STATEN ISLAND",
    "RICHMOND": "STATEN ISLAND",
    "RICHMOND / STATEN ISLAND": "STATEN ISLAND",
}

# Preference among static depot layers (lower = higher priority when distances tie).
LAYER_PRIORITY = {
    "ems_station": 0,
    "hospital_bay": 1,
    "fdny_firehouse": 2,
    "csl": 3,
}

# Specialty facilities that typically do not stage 911 EMS tours.
_NON_911_HOSPITAL_RE = re.compile(
    r"PSYCH|PSYCHIAT|REHAB|NURSING|CANCER|SPECIAL SURGERY|EYE AND EAR|ORTHOPED",
    re.I,
)


def filter_hospital_bays(df: pd.DataFrame) -> pd.DataFrame:
    """Keep acute-care receiving hospitals; drop psych/rehab/specialty non-deployers."""
    if df is None or not len(df):
        return df
    name_col = "facname" if "facname" in df.columns else None
    if not name_col:
        return df
    mask = ~df[name_col].astype(str).str.contains(_NON_911_HOSPITAL_RE, na=False)
    return df.loc[mask].copy()


def normalize_borough(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    key = str(value).strip().upper()
    return BOROUGH_ALIASES.get(key, key)


def dispatch_area_borough(dispatch_area: Any) -> str | None:
    if dispatch_area is None or (isinstance(dispatch_area, float) and math.isnan(dispatch_area)):
        return None
    code = str(dispatch_area).strip().upper()
    if not code:
        return None
    return DISPATCH_PREFIX_BOROUGH.get(code[0])


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def prepare_depots(df: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    """Normalize heterogeneous facility tables into a common depot schema."""
    colmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    lat_col = pick("latitude", "lat", "start_lat")
    lon_col = pick("longitude", "lon", "lng", "start_lon")
    boro_col = pick("borough", "boro")
    name_col = pick("facname", "facilityname", "facility_name", "name", "depot_name", "location")
    addr_col = pick("facilityaddress", "address", "depot_address")
    type_col = pick("factype", "facility_type", "type", "depot_type", "box_type")

    out = pd.DataFrame(
        {
            "depot_name": df[name_col] if name_col else None,
            "depot_address": df[addr_col] if addr_col else None,
            "depot_type": df[type_col] if type_col else source_label,
            "borough": df[boro_col].map(normalize_borough) if boro_col else None,
            "start_lon": pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan,
            "start_lat": pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan,
            "depot_layer": source_label,
        }
    )

    # Hospital bay subtype tags (H+H vs voluntary) for start_source auditing.
    if source_label == "hospital_bay":
        op_col = pick("opname", "overagency")
        ops = df[op_col].astype(str) if op_col else pd.Series([""] * len(df))
        tags = []
        for op in ops:
            low = op.lower()
            if "health and hospitals" in low or "nyc health + hospitals" in low:
                tags.append("nyc_hh_hospital_bay")
            else:
                tags.append("voluntary_hospital_bay")
        out["bay_tag"] = tags
    elif "bay_tag" in df.columns:
        out["bay_tag"] = df["bay_tag"]
    else:
        out["bay_tag"] = source_label

    if "incident_volume" in df.columns:
        out["incident_volume"] = pd.to_numeric(df["incident_volume"], errors="coerce").fillna(0)
    else:
        out["incident_volume"] = 0.0

    out = out.dropna(subset=["start_lon", "start_lat"]).reset_index(drop=True)
    out.insert(0, "depot_id", np.arange(len(out)))
    return out


def prepare_firehouses(firehouses: pd.DataFrame) -> pd.DataFrame:
    return prepare_depots(firehouses, source_label="fdny_firehouse")


def combine_depot_layers(
    layers: Iterable[tuple[pd.DataFrame | None, str]] | None = None,
    *,
    preferred: pd.DataFrame | None = None,
    fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a unified depot table.

    Preferred call style:
        combine_depot_layers([
            (ems_stations, "ems_station"),
            (hospital_bays, "hospital_bay"),
            (firehouses, "fdny_firehouse"),
            (csls, "csl"),
        ])
    """
    frames = []
    if layers is not None:
        for df, label in layers:
            if df is not None and len(df):
                frames.append(prepare_depots(df, source_label=label))
    else:
        # Back-compat with earlier signature.
        if preferred is not None and len(preferred):
            frames.append(prepare_depots(preferred, source_label="ems_station"))
        if fallback is not None and len(fallback):
            frames.append(prepare_depots(fallback, source_label="fdny_firehouse"))

    if not frames:
        raise ValueError("No depot layers provided")
    out = pd.concat(frames, ignore_index=True)
    out["depot_id"] = np.arange(len(out))
    return out


def _nearest_depot(depots: pd.DataFrame, dest_lon: float, dest_lat: float):
    dists = _haversine_km(
        depots["start_lon"].to_numpy(),
        depots["start_lat"].to_numpy(),
        dest_lon,
        dest_lat,
    )
    j = int(np.argmin(dists))
    return depots.iloc[j], float(dists[j])


def _source_label(chosen: pd.Series, prefer_b: str | None, scope: str) -> str:
    layer = str(chosen.get("depot_layer") or "depot")
    tag = str(chosen.get("bay_tag") or layer)
    boro = (prefer_b or "citywide").lower().replace(" ", "_")
    if layer == "hospital_bay":
        base = tag  # voluntary_hospital_bay / nyc_hh_hospital_bay
    elif layer == "csl":
        base = "synthetic_csl"
    elif layer == "ems_station":
        base = "ems_station"
    elif layer == "fdny_firehouse":
        base = "firehouse"
    else:
        base = layer
    if scope == "borough" and prefer_b:
        return f"nearest_{base}_in_{boro}"
    return f"nearest_{base}_citywide"


def _pick_static(
    prefer_b: str | None,
    preferred_by_boro: dict[str, pd.DataFrame],
    preferred_all: pd.DataFrame,
    fallback_by_boro: dict[str, pd.DataFrame],
    fallback_all: pd.DataFrame,
    dest_lon: float,
    dest_lat: float,
):
    """Prefer EMS/hospital layers over firehouses."""
    cand = preferred_by_boro.get(prefer_b) if prefer_b else None
    if cand is not None and len(cand):
        chosen, dist = _nearest_depot(cand, dest_lon, dest_lat)
        return chosen, dist, "borough"
    cand = fallback_by_boro.get(prefer_b) if prefer_b else None
    if cand is not None and len(cand):
        chosen, dist = _nearest_depot(cand, dest_lon, dest_lat)
        return chosen, dist, "borough"
    if len(preferred_all):
        chosen, dist = _nearest_depot(preferred_all, dest_lon, dest_lat)
        return chosen, dist, "citywide"
    chosen, dist = _nearest_depot(fallback_all if len(fallback_all) else preferred_all, dest_lon, dest_lat)
    return chosen, dist, "citywide"


def infer_start_locations(
    incidents: pd.DataFrame,
    firehouses: pd.DataFrame,
    zip_centroids: pd.DataFrame,
    ems_stations: pd.DataFrame | None = None,
    hospital_bays: pd.DataFrame | None = None,
    csl_points: pd.DataFrame | None = None,
    *,
    start_mode: str = "hybrid",
) -> pd.DataFrame:
    """
    Attach inferred start (depot) and destination (ZIP centroid) coordinates.

    start_mode:
      - static: EMS station / hospital bay / firehouse only
      - csl: synthetic Cross Street Locations only
      - hybrid: use CSL when it is closer to the destination than the best static
        depot (proxy for an already-on-the-road unit); otherwise static
    """
    start_mode = (start_mode or "hybrid").lower().strip()
    if start_mode not in {"static", "csl", "hybrid"}:
        raise ValueError(f"Unknown start_mode={start_mode!r}")

    layers = [
        (ems_stations, "ems_station"),
        (hospital_bays, "hospital_bay"),
        (firehouses, "fdny_firehouse"),
    ]
    if start_mode in {"csl", "hybrid"} and csl_points is not None and len(csl_points):
        layers.append((csl_points, "csl"))

    depots = combine_depot_layers(layers)
    static = depots[depots["depot_layer"] != "csl"].copy()
    preferred_static = static[static["depot_layer"].isin(["ems_station", "hospital_bay"])].copy()
    fallback_static = static[static["depot_layer"] == "fdny_firehouse"].copy()
    if preferred_static.empty:
        preferred_static = static
        fallback_static = static.iloc[0:0]
    csls = depots[depots["depot_layer"] == "csl"].copy()
    if start_mode != "csl" and static.empty:
        raise ValueError("No static depots with valid coordinates")
    if start_mode == "csl" and csls.empty:
        raise ValueError("No CSL points available for start_mode=csl")

    zips = zip_centroids.copy()
    zips["zipcode"] = zips["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)

    inc = incidents.copy()
    rename = {}
    for src, dst in [
        ("cad_incident_id", "incident_id"),
        ("incident_travel_tm_seconds_qy", "travel_seconds"),
        ("incident_response_seconds_qy", "response_seconds"),
        ("incident_dispatch_area", "dispatch_area"),
    ]:
        if src in inc.columns and dst not in inc.columns:
            rename[src] = dst
    inc = inc.rename(columns=rename)

    inc["zipcode"] = inc["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    inc["borough_norm"] = inc.get("borough", pd.Series(index=inc.index)).map(normalize_borough)
    if "dispatch_area" in inc.columns:
        inc["dispatch_borough"] = inc["dispatch_area"].map(dispatch_area_borough)
    else:
        inc["dispatch_borough"] = None
    inc["prefer_borough"] = inc["dispatch_borough"].fillna(inc["borough_norm"])

    inc = inc.merge(zips, on="zipcode", how="left", suffixes=("", "_zip"))

    start_lon = np.full(len(inc), np.nan)
    start_lat = np.full(len(inc), np.nan)
    depot_id = np.full(len(inc), -1, dtype=int)
    depot_name = np.array([None] * len(inc), dtype=object)
    depot_layer = np.array([None] * len(inc), dtype=object)
    start_source = np.array(["missing"] * len(inc), dtype=object)
    start_dist_km = np.full(len(inc), np.nan)
    start_mode_used = np.array([None] * len(inc), dtype=object)

    preferred_by_boro = {b: g for b, g in preferred_static.groupby("borough") if b is not None}
    fallback_by_boro = {b: g for b, g in fallback_static.groupby("borough") if b is not None}
    csl_by_boro = {b: g for b, g in csls.groupby("borough") if b is not None}

    for i, row in enumerate(inc.itertuples(index=False)):
        dest_lon = getattr(row, "dest_lon", np.nan)
        dest_lat = getattr(row, "dest_lat", np.nan)
        if dest_lon is None or dest_lat is None or (isinstance(dest_lon, float) and math.isnan(dest_lon)):
            start_source[i] = "no_destination_zip"
            continue

        prefer_b = getattr(row, "prefer_borough", None)
        dest_lon_f = float(dest_lon)
        dest_lat_f = float(dest_lat)

        static_choice = None
        csl_choice = None

        if start_mode in {"static", "hybrid"} and len(static):
            chosen, dist, scope = _pick_static(
                prefer_b,
                preferred_by_boro,
                preferred_static,
                fallback_by_boro,
                fallback_static if len(fallback_static) else preferred_static,
                dest_lon_f,
                dest_lat_f,
            )
            static_choice = (chosen, dist, scope)

        if start_mode in {"csl", "hybrid"} and len(csls):
            cand = csl_by_boro.get(prefer_b) if prefer_b else None
            if cand is not None and len(cand):
                chosen, dist = _nearest_depot(cand, dest_lon_f, dest_lat_f)
                csl_choice = (chosen, dist, "borough")
            else:
                chosen, dist = _nearest_depot(csls, dest_lon_f, dest_lat_f)
                csl_choice = (chosen, dist, "citywide")

        if start_mode == "static":
            chosen, dist, scope = static_choice
            mode_used = "static"
        elif start_mode == "csl":
            chosen, dist, scope = csl_choice
            mode_used = "csl"
        else:
            # hybrid: prefer on-road CSL when closer than static depot
            if csl_choice and static_choice and csl_choice[1] + 1e-9 < static_choice[1]:
                chosen, dist, scope = csl_choice
                mode_used = "hybrid_csl"
            elif static_choice:
                chosen, dist, scope = static_choice
                mode_used = "hybrid_static"
            else:
                chosen, dist, scope = csl_choice
                mode_used = "hybrid_csl"

        start_lon[i] = chosen["start_lon"]
        start_lat[i] = chosen["start_lat"]
        depot_id[i] = int(chosen["depot_id"])
        depot_name[i] = chosen["depot_name"]
        depot_layer[i] = chosen["depot_layer"]
        start_dist_km[i] = dist
        start_source[i] = _source_label(chosen, prefer_b, scope)
        start_mode_used[i] = mode_used

    return inc.assign(
        start_lon=start_lon,
        start_lat=start_lat,
        depot_id=depot_id,
        depot_name=depot_name,
        depot_layer=depot_layer,
        start_source=start_source,
        crow_flies_km=start_dist_km,
        start_mode=start_mode_used,
    )
