"""Travel-time prediction features and training helpers (slides Step 4).

Even without known ambulance GPS origins, we can supervise on the real
`incident_travel_tm_seconds_qy` label using:

1. Destination features (ZIP centroid, borough, demand)
2. Temporal / call-type features from CAD
3. Multi-candidate origin distances (EMS station, hospital bay, CSL)
4. A primary inferred origin (hybrid rule) for crow-flies / bearing

This matches the slide plan: segment-aggregated gradient boosting that scores
routes — here the “route summary” starts with geometry + context features;
LION/OSM segment aggregates plug into the same table later.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .depots import (
    _haversine_km,
    _nearest_depot,
    combine_depot_layers,
    dispatch_area_borough,
    filter_hospital_bays,
    normalize_borough,
    prepare_depots,
)


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _nearest_km(depots: pd.DataFrame | None, prefer_b, dest_lon, dest_lat):
    if depots is None or not len(depots):
        return np.nan, None
    if prefer_b is not None:
        cand = depots[depots["borough"] == prefer_b]
        if len(cand):
            chosen, dist = _nearest_depot(cand, dest_lon, dest_lat)
            return dist, chosen
    chosen, dist = _nearest_depot(depots, dest_lon, dest_lat)
    return dist, chosen


def build_ambulance_depot_layers(
    ems_stations: pd.DataFrame | None,
    hospital_bays: pd.DataFrame | None,
    csl_points: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Ambulance-only staging layers (no fire trucks / firehouses)."""
    layers: dict[str, pd.DataFrame] = {}
    if ems_stations is not None and len(ems_stations):
        layers["ems_station"] = prepare_depots(ems_stations, source_label="ems_station")
    if hospital_bays is not None and len(hospital_bays):
        hb = filter_hospital_bays(hospital_bays)
        if len(hb):
            layers["hospital_bay"] = prepare_depots(hb, source_label="hospital_bay")
    if csl_points is not None and len(csl_points):
        layers["csl"] = prepare_depots(csl_points, source_label="csl")
    return layers


def build_travel_time_feature_table(
    incidents: pd.DataFrame,
    zip_centroids: pd.DataFrame,
    *,
    ems_stations: pd.DataFrame | None = None,
    hospital_bays: pd.DataFrame | None = None,
    csl_points: pd.DataFrame | None = None,
    od_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One row per EMS incident with travel-time label + features.

    Label: travel_seconds (ambulance assignment → on-scene).
    Origins are uncertain, so we include distances to multiple ambulance
    staging hypotheses instead of pretending we know the true start.
    """
    layers = build_ambulance_depot_layers(ems_stations, hospital_bays, csl_points)

    inc = incidents.copy()
    rename = {
        "incident_travel_tm_seconds_qy": "travel_seconds",
        "incident_response_seconds_qy": "response_seconds",
        "incident_dispatch_area": "dispatch_area",
        "dispatch_response_seconds_qy": "dispatch_wait_seconds",
    }
    for src, dst in rename.items():
        if src in inc.columns and dst not in inc.columns:
            inc = inc.rename(columns={src: dst})

    for col in ("travel_seconds", "response_seconds", "dispatch_wait_seconds"):
        if col in inc.columns:
            inc[col] = pd.to_numeric(inc[col], errors="coerce")

    inc["zipcode"] = inc["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    inc["borough_norm"] = inc.get("borough", pd.Series(index=inc.index)).map(normalize_borough)
    if "dispatch_area" in inc.columns:
        inc["dispatch_borough"] = inc["dispatch_area"].map(dispatch_area_borough)
    else:
        inc["dispatch_borough"] = None
    inc["prefer_borough"] = inc["dispatch_borough"].fillna(inc["borough_norm"])

    zips = zip_centroids.copy()
    zips["zipcode"] = zips["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    # ZIP demand as a simple congestion / workload proxy
    zip_volume = (
        inc.dropna(subset=["zipcode"]).groupby("zipcode").size().rename("zip_incident_volume")
    )
    zips = zips.merge(zip_volume, on="zipcode", how="left")
    zips["zip_incident_volume"] = zips["zip_incident_volume"].fillna(0)

    inc = inc.merge(zips, on="zipcode", how="left", suffixes=("", "_zip"))

    # Optional: reuse hybrid primary start from od_pairs
    primary = None
    if od_pairs is not None and len(od_pairs):
        keep = [
            c
            for c in [
                "incident_id",
                "start_lat",
                "start_lon",
                "crow_flies_km",
                "depot_layer",
                "start_mode",
                "start_source",
            ]
            if c in od_pairs.columns
        ]
        primary = od_pairs[keep].drop_duplicates(subset=["incident_id"])

    rows: list[dict[str, Any]] = []
    for rec in inc.itertuples(index=False):
        travel = getattr(rec, "travel_seconds", np.nan)
        dest_lat = getattr(rec, "dest_lat", np.nan)
        dest_lon = getattr(rec, "dest_lon", np.nan)
        if travel is None or travel != travel or travel <= 0:
            continue
        if dest_lat != dest_lat or dest_lon != dest_lon:
            continue
        # Drop absurd outliers (> 1 hour travel)
        if travel > 3600:
            continue

        prefer_b = getattr(rec, "prefer_borough", None)
        dest_lat_f = float(dest_lat)
        dest_lon_f = float(dest_lon)

        station_km, station = _nearest_km(layers.get("ems_station"), prefer_b, dest_lon_f, dest_lat_f)
        hospital_km, hospital = _nearest_km(layers.get("hospital_bay"), prefer_b, dest_lon_f, dest_lat_f)
        csl_km, csl = _nearest_km(layers.get("csl"), prefer_b, dest_lon_f, dest_lat_f)

        candidates = {
            "ems_station": station_km,
            "hospital_bay": hospital_km,
            "csl": csl_km,
        }
        finite = {k: v for k, v in candidates.items() if v == v}
        if not finite:
            continue
        nearest_layer = min(finite, key=finite.get)
        nearest_km = finite[nearest_layer]
        farthest_km = max(finite.values())
        spread_km = farthest_km - nearest_km

        # Primary geometry: prefer od_pairs hybrid start if present, else nearest candidate
        start_lat = np.nan
        start_lon = np.nan
        primary_layer = nearest_layer
        crow = nearest_km
        if primary is not None:
            iid = getattr(rec, "incident_id", None)
            hit = primary[primary["incident_id"].astype(str) == str(iid)]
            # Only keep ambulance layers from od_pairs
            if len(hit):
                layer = str(hit.iloc[0].get("depot_layer") or "")
                if layer in {"ems_station", "hospital_bay", "csl"}:
                    start_lat = float(hit.iloc[0]["start_lat"])
                    start_lon = float(hit.iloc[0]["start_lon"])
                    primary_layer = layer
                    crow = float(hit.iloc[0].get("crow_flies_km") or nearest_km)

        if start_lat != start_lat:
            chosen = {"ems_station": station, "hospital_bay": hospital, "csl": csl}.get(nearest_layer)
            if chosen is not None:
                start_lat = float(chosen["start_lat"])
                start_lon = float(chosen["start_lon"])

        bearing = (
            _bearing_deg(start_lat, start_lon, dest_lat_f, dest_lon_f)
            if start_lat == start_lat
            else np.nan
        )

        dt = pd.to_datetime(getattr(rec, "incident_datetime", None), errors="coerce")
        hour = int(dt.hour) if pd.notna(dt) else -1
        dow = int(dt.dayofweek) if pd.notna(dt) else -1
        month = int(dt.month) if pd.notna(dt) else -1
        is_weekend = int(dow >= 5) if dow >= 0 else -1
        is_rush = int(hour in (7, 8, 9, 16, 17, 18, 19)) if hour >= 0 else -1
        is_night = int(hour >= 22 or hour < 6) if hour >= 0 else -1

        severity = pd.to_numeric(getattr(rec, "final_severity_level_code", np.nan), errors="coerce")
        if severity != severity:
            severity = pd.to_numeric(getattr(rec, "initial_severity_level_code", np.nan), errors="coerce")

        held = str(getattr(rec, "held_indicator", "N") or "N").upper()
        valid = str(getattr(rec, "valid_incident_rspns_time_indc", "Y") or "Y").upper()

        rows.append(
            {
                "incident_id": getattr(rec, "incident_id", None),
                "incident_datetime": getattr(rec, "incident_datetime", None),
                "travel_seconds": float(travel),
                "response_seconds": float(getattr(rec, "response_seconds", np.nan) or np.nan),
                "dispatch_wait_seconds": float(getattr(rec, "dispatch_wait_seconds", np.nan) or np.nan),
                "borough": getattr(rec, "borough_norm", None) or getattr(rec, "borough", None),
                "dispatch_area": getattr(rec, "dispatch_area", None),
                "zipcode": getattr(rec, "zipcode", None),
                "dest_lat": dest_lat_f,
                "dest_lon": dest_lon_f,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "primary_origin_layer": primary_layer,
                "crow_flies_km": crow,
                "bearing_deg": bearing,
                "station_km": station_km,
                "hospital_km": hospital_km,
                "csl_km": csl_km,
                "nearest_origin_km": nearest_km,
                "origin_spread_km": spread_km,
                "zip_incident_volume": float(getattr(rec, "zip_incident_volume", 0) or 0),
                "hour": hour,
                "dow": dow,
                "month": month,
                "is_weekend": is_weekend,
                "is_rush": is_rush,
                "is_night": is_night,
                "severity": float(severity) if severity == severity else -1,
                "initial_call_type": str(getattr(rec, "initial_call_type", "") or ""),
                "final_call_type": str(getattr(rec, "final_call_type", "") or ""),
                "held_indicator": 1 if held == "Y" else 0,
                "valid_travel": 1 if valid == "Y" else 0,
                # Ambiguity: how much do station vs CSL disagree? High = origin uncertain
                "station_minus_csl_km": (
                    station_km - csl_km if station_km == station_km and csl_km == csl_km else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "crow_flies_km",
    "bearing_deg",
    "station_km",
    "hospital_km",
    "csl_km",
    "nearest_origin_km",
    "origin_spread_km",
    "station_minus_csl_km",
    "zip_incident_volume",
    "dispatch_wait_seconds",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_rush",
    "is_night",
    "severity",
    "held_indicator",
    "dest_lat",
    "dest_lon",
    # Weather (Open-Meteo)
    "wx_temp_c",
    "wx_humidity",
    "wx_precip_mm",
    "wx_rain_mm",
    "wx_snow_mm",
    "wx_cloudcover",
    "wx_wind_kmh",
    "wx_visibility_m",
    "wx_is_precip",
    "wx_is_snow",
    # OSM path aggregates (LION swap-in later)
    "osm_path_km",
    "osm_n_edges",
    "gmaps_duration_s",
    "gmaps_traffic_s",
    "gmaps_distance_m",
    "gmaps_vs_osm_ratio",
    "gmaps_ok",
    "osm_n_nodes",
    "osm_mean_speed_kmh",
    "osm_min_speed_kmh",
    "osm_primary_share",
    "osm_motorway_share",
    "osm_residential_share",
    "osm_circuity",
    "osm_route_ok",
]

# Route-scoring feature set: no CAD dispatch wait (system load ≠ street physics)
ROUTE_ONLY_DROP = {
    "dispatch_wait_seconds",
    "held_indicator",
}

CATEGORICAL_COLUMNS = [
    "borough",
    "primary_origin_layer",
    "initial_call_type",
    "final_call_type",
]


def feature_list(feature_set: str = "full") -> list[str]:
    """Return numeric feature columns for ``full`` or ``route`` models."""
    if feature_set == "full":
        return list(FEATURE_COLUMNS)
    if feature_set == "route":
        return [c for c in FEATURE_COLUMNS if c not in ROUTE_ONLY_DROP]
    raise ValueError(f"Unknown feature_set={feature_set!r}; use 'full' or 'route'")


def prepare_matrix(df: pd.DataFrame, feature_set: str = "full"):
    """Return X, y, and categorical feature names for LightGBM."""
    data = df.copy()
    data = data[data["valid_travel"] == 1] if "valid_travel" in data.columns else data
    y = data["travel_seconds"].astype(float)
    feats = feature_list(feature_set) + CATEGORICAL_COLUMNS
    missing = [c for c in feats if c not in data.columns]
    for c in missing:
        data[c] = np.nan
    X = data[feats].copy()
    for c in CATEGORICAL_COLUMNS:
        X[c] = X[c].astype("category")
    return X, y, CATEGORICAL_COLUMNS
