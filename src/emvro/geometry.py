"""Geometry helpers for ZIP centroids from MODZCTA polygons."""

from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union


def _geom_from_row(row: dict[str, Any] | pd.Series):
    geom = row.get("the_geom") if not isinstance(row, pd.Series) else row.get("the_geom")
    if geom is None or (isinstance(geom, float) and pd.isna(geom)):
        return None
    if isinstance(geom, str):
        text = geom.strip()
        try:
            geom = json.loads(text)
        except json.JSONDecodeError:
            # pandas may round-trip SODA dicts with single quotes
            geom = ast.literal_eval(text)
    if isinstance(geom, dict) and "type" in geom:
        return shape(geom)
    return None


def zip_centroids_from_modzcta(modzcta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per ZIP with centroid lon/lat.

    MODZCTA rows may cover multiple ZCTAs in the `zcta` field; we explode those
    so EMS `zipcode` values join cleanly.
    """
    records: list[dict[str, Any]] = []
    for _, row in modzcta_df.iterrows():
        geom = _geom_from_row(row)
        if geom is None or geom.is_empty:
            continue
        # Representative point is guaranteed inside the polygon (better than centroid for odd shapes)
        pt = geom.representative_point()
        zips = []
        for field in ("modzcta", "zcta", "label"):
            raw = row.get(field)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            for part in str(raw).replace(" ", "").split(","):
                part = "".join(ch for ch in part if ch.isdigit())
                if len(part) == 5:
                    zips.append(part)
        for z in sorted(set(zips)):
            records.append(
                {
                    "zipcode": z,
                    "dest_lon": float(pt.x),
                    "dest_lat": float(pt.y),
                    "modzcta": str(row.get("modzcta") or z),
                }
            )

    out = pd.DataFrame(records).drop_duplicates(subset=["zipcode"])
    return out.reset_index(drop=True)


def multipolygon_bounds_union(modzcta_df: pd.DataFrame):
    geoms = []
    for _, row in modzcta_df.iterrows():
        g = _geom_from_row(row)
        if g is not None and not g.is_empty:
            geoms.append(g)
    if not geoms:
        return None
    return unary_union(geoms)
