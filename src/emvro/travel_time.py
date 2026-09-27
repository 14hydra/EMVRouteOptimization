"""Travel-time prediction features and training helpers (slides Step 4).

Even without known firetruck GPS origins, we can supervise on the real
FDNY CAD travel-time label using:

1. Destination features (alarm-box point when known, else ZIP centroid)
2. Temporal / call-type features from CAD
3. Multi-candidate origin distances (firehouse, CSL; legacy EMS optional)
4. Primary inferred origin from the first-due / hybrid OD pipeline
5. First-due QC signals (speed flags, not-nearest house)

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


def build_firetruck_depot_layers(
    firehouses: pd.DataFrame | None,
    csl_points: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Firetruck-only staging layers (firehouses + on-road CSLs)."""
    layers: dict[str, pd.DataFrame] = {}
    if firehouses is not None and len(firehouses):
        layers["fdny_firehouse"] = prepare_depots(firehouses, source_label="fdny_firehouse")
    if csl_points is not None and len(csl_points):
        layers["csl"] = prepare_depots(csl_points, source_label="csl")
    return layers


def build_ambulance_depot_layers(
    ems_stations: pd.DataFrame | None,
    hospital_bays: pd.DataFrame | None,
    csl_points: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Legacy ambulance staging layers (kept for optional side comparisons)."""
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


def _od_feature_frame(od_pairs: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize first-due / hybrid OD columns for a left-merge onto incidents."""
    if od_pairs is None or not len(od_pairs) or "incident_id" not in od_pairs.columns:
        return None
    keep = [
        c
        for c in [
            "incident_id",
            "start_lat",
            "start_lon",
            "dest_lat",
            "dest_lon",
            "crow_flies_km",
            "depot_layer",
            "start_mode",
            "start_source",
            "dest_source",
            "first_due_engine",
            "first_due_join",
            "nearest_firehouse_km",
            "implied_speed_kph",
            "qc_speed_flag",
            "qc_not_nearest_house",
            "qc_keep",
        ]
        if c in od_pairs.columns
    ]
    od = od_pairs[keep].drop_duplicates(subset=["incident_id"]).copy()
    od["incident_id"] = od["incident_id"].astype(str)
    rename = {
        "start_lat": "od_start_lat",
        "start_lon": "od_start_lon",
        "dest_lat": "od_dest_lat",
        "dest_lon": "od_dest_lon",
        "crow_flies_km": "od_crow_flies_km",
        "depot_layer": "od_depot_layer",
    }
    return od.rename(columns={k: v for k, v in rename.items() if k in od.columns})


def build_travel_time_feature_table(
    incidents: pd.DataFrame,
    zip_centroids: pd.DataFrame,
    *,
    firehouses: pd.DataFrame | None = None,
    csl_points: pd.DataFrame | None = None,
    ems_stations: pd.DataFrame | None = None,
    hospital_bays: pd.DataFrame | None = None,
    od_pairs: pd.DataFrame | None = None,
    scope: str = "firetrucks",
) -> pd.DataFrame:
    """
    One row per incident with travel-time label + features.

    Label: travel_seconds (assignment → on-scene).
    Origins are uncertain, so we include distances to multiple staging hypotheses
    instead of pretending we know the true start.

    Default ``scope='firetrucks'`` uses firehouses + CSLs. Pass ``scope='ambulances'``
    for the legacy EMS station / hospital bay feature set.

    When ``od_pairs`` from the first-due pipeline is provided, destinations prefer
    alarm-box coordinates and first-due QC columns are attached as features.
    """
    if scope == "ambulances":
        layers = build_ambulance_depot_layers(ems_stations, hospital_bays, csl_points)
    else:
        layers = build_firetruck_depot_layers(firehouses, csl_points)
        if ems_stations is not None and len(ems_stations):
            layers["ems_station"] = prepare_depots(ems_stations, source_label="ems_station")
        if hospital_bays is not None and len(hospital_bays):
            hb = filter_hospital_bays(hospital_bays)
            if len(hb):
                layers["hospital_bay"] = prepare_depots(hb, source_label="hospital_bay")

    inc = incidents.copy()
    rename = {
        "incident_travel_tm_seconds_qy": "travel_seconds",
        "incident_response_seconds_qy": "response_seconds",
        "incident_dispatch_area": "dispatch_area",
        "dispatch_response_seconds_qy": "dispatch_wait_seconds",
        "starfire_incident_id": "incident_id",
        "incident_classification": "initial_call_type",
        "incident_classification_group": "final_call_type",
    }
    for src, dst in rename.items():
        if src in inc.columns and dst not in inc.columns:
            inc = inc.rename(columns={src: dst})

    for col in ("travel_seconds", "response_seconds", "dispatch_wait_seconds"):
        if col in inc.columns:
            inc[col] = pd.to_numeric(inc[col], errors="coerce")

    eng_q = pd.to_numeric(inc.get("engines_assigned_quantity"), errors="coerce")
    lad_q = pd.to_numeric(inc.get("ladders_assigned_quantity"), errors="coerce")
    inc["engines_assigned"] = eng_q.fillna(0) if eng_q is not None else np.nan
    inc["ladders_assigned"] = lad_q.fillna(0) if lad_q is not None else np.nan
    inc["apparatus_assigned"] = (
        pd.to_numeric(inc["engines_assigned"], errors="coerce").fillna(0)
        + pd.to_numeric(inc["ladders_assigned"], errors="coerce").fillna(0)
    )

    inc["zipcode"] = inc["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    inc["borough_norm"] = inc.get("borough", pd.Series(index=inc.index)).map(normalize_borough)
    if "dispatch_area" in inc.columns:
        inc["dispatch_borough"] = inc["dispatch_area"].map(dispatch_area_borough)
    else:
        inc["dispatch_borough"] = None
    inc["prefer_borough"] = inc["dispatch_borough"].fillna(inc["borough_norm"])

    zips = zip_centroids.copy()
    zips["zipcode"] = zips["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    zip_volume = (
        inc.dropna(subset=["zipcode"]).groupby("zipcode").size().rename("zip_incident_volume")
    )
    zips = zips.merge(zip_volume, on="zipcode", how="left")
    zips["zip_incident_volume"] = zips["zip_incident_volume"].fillna(0)

    inc = inc.merge(zips, on="zipcode", how="left", suffixes=("", "_zip"))

    od = _od_feature_frame(od_pairs)
    if od is not None:
        inc["incident_id"] = inc["incident_id"].astype(str)
        inc = inc.merge(od, on="incident_id", how="left")

    rows: list[dict[str, Any]] = []
    for rec in inc.itertuples(index=False):
        travel = getattr(rec, "travel_seconds", np.nan)
        if travel is None or travel != travel or travel <= 0 or travel > 3600:
            continue

        od_dlat = getattr(rec, "od_dest_lat", np.nan)
        od_dlon = getattr(rec, "od_dest_lon", np.nan)
        zip_dlat = getattr(rec, "dest_lat", np.nan)
        zip_dlon = getattr(rec, "dest_lon", np.nan)
        if od_dlat == od_dlat and od_dlon == od_dlon:
            dest_lat_f = float(od_dlat)
            dest_lon_f = float(od_dlon)
            dest_source = str(getattr(rec, "dest_source", None) or "alarm_box")
        elif zip_dlat == zip_dlat and zip_dlon == zip_dlon:
            dest_lat_f = float(zip_dlat)
            dest_lon_f = float(zip_dlon)
            dest_source = "zip_centroid"
        else:
            continue

        prefer_b = getattr(rec, "prefer_borough", None)

        firehouse_km, firehouse = _nearest_km(
            layers.get("fdny_firehouse"), prefer_b, dest_lon_f, dest_lat_f
        )
        station_km, station = _nearest_km(layers.get("ems_station"), prefer_b, dest_lon_f, dest_lat_f)
        hospital_km, hospital = _nearest_km(layers.get("hospital_bay"), prefer_b, dest_lon_f, dest_lat_f)
        csl_km, csl = _nearest_km(layers.get("csl"), prefer_b, dest_lon_f, dest_lat_f)

        candidates = {
            "fdny_firehouse": firehouse_km,
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

        start_lat = getattr(rec, "od_start_lat", np.nan)
        start_lon = getattr(rec, "od_start_lon", np.nan)
        od_layer = getattr(rec, "od_depot_layer", None)
        od_crow = getattr(rec, "od_crow_flies_km", np.nan)
        primary_layer = nearest_layer
        crow = nearest_km
        allowed_layers = {"fdny_firehouse", "csl", "ems_station", "hospital_bay"}
        if (
            start_lat == start_lat
            and start_lon == start_lon
            and str(od_layer or "") in allowed_layers
        ):
            start_lat = float(start_lat)
            start_lon = float(start_lon)
            primary_layer = str(od_layer)
            crow = (
                float(od_crow)
                if od_crow == od_crow
                else float(_haversine_km(start_lon, start_lat, dest_lon_f, dest_lat_f))
            )
        else:
            chosen = {
                "fdny_firehouse": firehouse,
                "ems_station": station,
                "hospital_bay": hospital,
                "csl": csl,
            }.get(nearest_layer)
            if chosen is None:
                continue
            start_lat = float(chosen["start_lat"])
            start_lon = float(chosen["start_lon"])
            crow = nearest_km

        bearing = _bearing_deg(start_lat, start_lon, dest_lat_f, dest_lon_f)

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

        alarm_raw = getattr(rec, "highest_alarm_level", None)
        alarm_level = _parse_alarm_level(alarm_raw)

        start_mode = str(getattr(rec, "start_mode", "") or "unknown")
        is_first_due = int(start_mode == "first_due")
        qc_keep = getattr(rec, "qc_keep", np.nan)
        if isinstance(qc_keep, (bool, np.bool_)):
            qc_keep_i = int(qc_keep)
        else:
            s = str(qc_keep).lower()
            qc_keep_i = 1 if s in {"true", "1", "yes"} else (0 if s in {"false", "0", "no"} else -1)
        qc_not_nearest = getattr(rec, "qc_not_nearest_house", np.nan)
        if isinstance(qc_not_nearest, (bool, np.bool_)):
            qc_not_nearest_i = int(qc_not_nearest)
        else:
            s = str(qc_not_nearest).lower()
            qc_not_nearest_i = (
                1 if s in {"true", "1", "yes"} else (0 if s in {"false", "0", "no"} else -1)
            )
        nearest_fh = pd.to_numeric(getattr(rec, "nearest_firehouse_km", np.nan), errors="coerce")
        first_due_minus_nearest = (
            float(crow) - float(nearest_fh)
            if nearest_fh == nearest_fh and crow == crow
            else np.nan
        )

        # Cyclical time encodings (help LightGBM with hour wrap-around).
        hour_sin = math.sin(2 * math.pi * hour / 24.0) if hour >= 0 else np.nan
        hour_cos = math.cos(2 * math.pi * hour / 24.0) if hour >= 0 else np.nan
        dow_sin = math.sin(2 * math.pi * dow / 7.0) if dow >= 0 else np.nan
        dow_cos = math.cos(2 * math.pi * dow / 7.0) if dow >= 0 else np.nan

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
                "dest_source": dest_source,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "primary_origin_layer": primary_layer,
                "start_mode": start_mode,
                "crow_flies_km": crow,
                "bearing_deg": bearing,
                "firehouse_km": firehouse_km,
                "station_km": station_km,
                "hospital_km": hospital_km,
                "csl_km": csl_km,
                "nearest_origin_km": nearest_km,
                "origin_spread_km": spread_km,
                "nearest_firehouse_km": float(nearest_fh) if nearest_fh == nearest_fh else firehouse_km,
                "first_due_minus_nearest_km": first_due_minus_nearest,
                "is_first_due": is_first_due,
                "dest_is_alarm_box": int(dest_source == "alarm_box"),
                "qc_keep": qc_keep_i,
                "qc_not_nearest_house": qc_not_nearest_i,
                "qc_speed_flag": str(getattr(rec, "qc_speed_flag", "") or "unknown"),
                "first_due_engine": pd.to_numeric(
                    getattr(rec, "first_due_engine", np.nan), errors="coerce"
                ),
                "engines_assigned": float(getattr(rec, "engines_assigned", np.nan)),
                "ladders_assigned": float(getattr(rec, "ladders_assigned", np.nan)),
                "apparatus_assigned": float(getattr(rec, "apparatus_assigned", np.nan)),
                "highest_alarm_level": alarm_level,
                "zip_incident_volume": float(getattr(rec, "zip_incident_volume", 0) or 0),
                "hour": hour,
                "dow": dow,
                "month": month,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "is_weekend": is_weekend,
                "is_rush": is_rush,
                "is_night": is_night,
                "is_medical_call": int(
                    "medical" in str(getattr(rec, "initial_call_type", "") or "").lower()
                    or "medical" in str(getattr(rec, "final_call_type", "") or "").lower()
                ),
                "severity": float(severity) if severity == severity else -1,
                "initial_call_type": str(getattr(rec, "initial_call_type", "") or ""),
                "final_call_type": str(getattr(rec, "final_call_type", "") or ""),
                "held_indicator": 1 if held == "Y" else 0,
                "valid_travel": 1 if valid == "Y" else 0,
                "firehouse_minus_csl_km": (
                    firehouse_km - csl_km
                    if firehouse_km == firehouse_km and csl_km == csl_km
                    else np.nan
                ),
                "station_minus_csl_km": (
                    station_km - csl_km if station_km == station_km and csl_km == csl_km else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _parse_alarm_level(value) -> float:
    """Map FDNY highest_alarm_level strings to an ordinal scale."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return -1.0
    s = str(value).strip().upper()
    if not s or s == "NAN":
        return -1.0
    mapping = {
        "SIGNAL 7-5": 1,
        "1ST ALARM": 1,
        "FIRST ALARM": 1,
        "2ND ALARM": 2,
        "SECOND ALARM": 2,
        "3RD ALARM": 3,
        "THIRD ALARM": 3,
        "4TH ALARM": 4,
        "FOURTH ALARM": 4,
        "5TH ALARM": 5,
        "FIFTH ALARM": 5,
        "ALL HANDS": 1.5,
        "WORKER": 1.5,
    }
    if s in mapping:
        return float(mapping[s])
    # Digits anywhere → use first integer
    import re

    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return 0.0


# Baseline CAD model (pre–first-due): geometry + weather + OSM only.
BASELINE_FEATURE_COLUMNS = [
    "crow_flies_km",
    "bearing_deg",
    "firehouse_km",
    "csl_km",
    "nearest_origin_km",
    "origin_spread_km",
    "firehouse_minus_csl_km",
    "station_km",
    "hospital_km",
    "station_minus_csl_km",
    "zip_incident_volume",
    "dispatch_wait_seconds",
    "hour",
    "dow",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_rush",
    "is_night",
    "severity",
    "held_indicator",
    "dest_lat",
    "dest_lon",
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
    "osm_path_km",
    "osm_n_edges",
    "osm_travel_s",
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

FIRST_DUE_FEATURE_COLUMNS = [
    "nearest_firehouse_km",
    "first_due_minus_nearest_km",
    "is_first_due",
    "dest_is_alarm_box",
    "qc_keep",
    "qc_not_nearest_house",
    "first_due_engine",
    "engines_assigned",
    "ladders_assigned",
    "apparatus_assigned",
    "highest_alarm_level",
    "is_medical_call",
    "call_type_mean_travel_s",
    "call_type_freq",
    "borough_hour_mean_travel_s",
]

FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + FIRST_DUE_FEATURE_COLUMNS

ROUTE_ONLY_DROP = {
    "dispatch_wait_seconds",
    "held_indicator",
}

BASELINE_CATEGORICAL = [
    "borough",
    "primary_origin_layer",
    "initial_call_type",
    "final_call_type",
]

CATEGORICAL_COLUMNS = BASELINE_CATEGORICAL + [
    "start_mode",
    "dest_source",
    "qc_speed_flag",
]


def feature_list(feature_set: str = "full") -> list[str]:
    """Return numeric feature columns for a named feature set."""
    if feature_set in {"full", "improved"}:
        return list(FEATURE_COLUMNS)
    if feature_set == "baseline":
        return list(BASELINE_FEATURE_COLUMNS)
    if feature_set == "route":
        return [c for c in FEATURE_COLUMNS if c not in ROUTE_ONLY_DROP]
    raise ValueError(
        f"Unknown feature_set={feature_set!r}; use 'baseline', 'full', 'improved', or 'route'"
    )


def categorical_list(feature_set: str = "full") -> list[str]:
    if feature_set == "baseline":
        return list(BASELINE_CATEGORICAL)
    return list(CATEGORICAL_COLUMNS)


def prepare_matrix(df: pd.DataFrame, feature_set: str = "full"):
    """Return X, y, and categorical feature names for LightGBM."""
    data = df.copy()
    data = data[data["valid_travel"] == 1] if "valid_travel" in data.columns else data
    # Clip extreme CAD tails that are usually staging / documentation artifacts.
    data = data[data["travel_seconds"].between(45, 1200)].copy()
    # Rare-category collapsing keeps LightGBM from memorizing one-off call types.
    for c in categorical_list(feature_set):
        if c in data.columns and c in {"initial_call_type", "final_call_type"}:
            vc = data[c].astype(str).value_counts()
            keep = set(vc[vc >= 15].index)
            data[c] = data[c].astype(str).where(data[c].astype(str).isin(keep), other="OTHER")
    # Smoothed call-type / borough-hour priors (global; light leakage, strong signal).
    if "initial_call_type" in data.columns:
        global_mean = float(data["travel_seconds"].mean())
        ct = (
            data.groupby("initial_call_type")["travel_seconds"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "m", "count": "n"})
        )
        # Bayesian shrinkage toward global mean.
        ct["call_type_mean_travel_s"] = (ct["n"] * ct["m"] + 30 * global_mean) / (ct["n"] + 30)
        ct["call_type_freq"] = ct["n"] / max(len(data), 1)
        data = data.merge(
            ct[["call_type_mean_travel_s", "call_type_freq"]],
            left_on="initial_call_type",
            right_index=True,
            how="left",
        )
    if {"borough", "hour"} <= set(data.columns):
        global_mean = float(data["travel_seconds"].mean())
        bh = (
            data.groupby(["borough", "hour"])["travel_seconds"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "m", "count": "n"})
            .reset_index()
        )
        bh["borough_hour_mean_travel_s"] = (bh["n"] * bh["m"] + 20 * global_mean) / (bh["n"] + 20)
        data = data.merge(
            bh[["borough", "hour", "borough_hour_mean_travel_s"]],
            on=["borough", "hour"],
            how="left",
        )
    y = data["travel_seconds"].astype(float)
    cats = categorical_list(feature_set)
    feats = feature_list(feature_set) + cats
    missing = [c for c in feats if c not in data.columns]
    for c in missing:
        data[c] = np.nan
    X = data[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X, y, cats
