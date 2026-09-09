"""Synthetic Cross Street Location (CSL) staging points for on-road EMV starts."""

from __future__ import annotations

import pandas as pd

from .depots import normalize_borough


def incident_volume_by_zip(incidents: pd.DataFrame) -> pd.DataFrame:
    """Count EMS incidents per ZIP (volume weights for CSL selection)."""
    df = incidents.copy()
    zip_col = "zipcode" if "zipcode" in df.columns else None
    if zip_col is None:
        raise ValueError("incidents must include zipcode")
    df["zipcode"] = df[zip_col].astype(str).str.extract(r"(\d{5})", expand=False)
    boro_col = "borough" if "borough" in df.columns else None
    if boro_col:
        df["borough"] = df[boro_col].map(normalize_borough)
    else:
        df["borough"] = None
    g = (
        df.dropna(subset=["zipcode"])
        .groupby(["zipcode", "borough"], dropna=False)
        .size()
        .reset_index(name="incident_volume")
    )
    return g.sort_values("incident_volume", ascending=False).reset_index(drop=True)


def build_csl_points(
    alarm_boxes: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    top_zips_per_borough: int = 25,
    max_boxes_per_zip: int = 5,
) -> pd.DataFrame:
    """
    Build synthetic CSLs from FDNY alarm-box intersections, volume-weighted by EMS demand.

    Alarm boxes sit at street corners (intersection proxies). We keep the highest-volume
    ZIPs per borough, then up to `max_boxes_per_zip` alarm boxes in those ZIPs.
    """
    vol = incident_volume_by_zip(incidents)
    # Keep top ZIPs within each borough by incident volume.
    top = (
        vol.sort_values(["borough", "incident_volume"], ascending=[True, False])
        .groupby("borough", dropna=False)
        .head(top_zips_per_borough)
        .reset_index(drop=True)
    )
    top_zips = set(top["zipcode"].astype(str))

    boxes = alarm_boxes.copy()
    colmap = {c.lower(): c for c in boxes.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    zip_col = pick("zip", "zipcode", "postcode")
    lat_col = pick("latitude", "lat")
    lon_col = pick("longitude", "lon", "lng")
    boro_col = pick("borough", "boro")
    loc_col = pick("location", "name", "borobox")
    type_col = pick("box_type", "type")

    if not lat_col or not lon_col or not zip_col:
        raise ValueError("alarm_boxes need latitude, longitude, and zip columns")

    boxes["zipcode"] = boxes[zip_col].astype(str).str.extract(r"(\d{5})", expand=False)
    boxes["borough"] = boxes[boro_col].map(normalize_borough) if boro_col else None
    boxes["latitude"] = pd.to_numeric(boxes[lat_col], errors="coerce")
    boxes["longitude"] = pd.to_numeric(boxes[lon_col], errors="coerce")
    boxes["location"] = boxes[loc_col] if loc_col else boxes["zipcode"].map(lambda z: f"CSL-{z}")
    boxes["box_type"] = boxes[type_col] if type_col else "ALARM_BOX"
    boxes = boxes.dropna(subset=["latitude", "longitude", "zipcode"])
    boxes = boxes[boxes["zipcode"].isin(top_zips)]

    boxes = boxes.merge(
        vol[["zipcode", "incident_volume"]],
        on="zipcode",
        how="left",
    )
    boxes["incident_volume"] = boxes["incident_volume"].fillna(0)

    # Diversify: take a few boxes per ZIP, preferring denser ZIPs overall.
    boxes = boxes.sort_values(["incident_volume", "zipcode"], ascending=[False, True])
    selected = boxes.groupby("zipcode", dropna=False).head(max_boxes_per_zip).copy()
    selected["facname"] = selected["location"].map(lambda s: f"CSL {s}")
    selected["factype"] = "SYNTHETIC_CSL"
    selected["address"] = selected["location"]
    selected["boro"] = selected["borough"]
    return selected.reset_index(drop=True)


def build_csl_from_zip_centroids(
    zip_centroids: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    top_n: int = 80,
) -> pd.DataFrame:
    """Fallback CSLs when alarm boxes are unavailable: top-volume ZIP centroids."""
    vol = incident_volume_by_zip(incidents)
    z = zip_centroids.copy()
    z["zipcode"] = z["zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    merged = vol.merge(z, on="zipcode", how="inner")
    merged = merged.sort_values("incident_volume", ascending=False).head(top_n).copy()
    merged["facname"] = merged["zipcode"].map(lambda z_: f"CSL ZIP {z_}")
    merged["factype"] = "SYNTHETIC_CSL_ZIP_CENTROID"
    merged["address"] = merged["facname"]
    merged["boro"] = merged.get("borough")
    merged["latitude"] = merged["dest_lat"]
    merged["longitude"] = merged["dest_lon"]
    return merged.reset_index(drop=True)
