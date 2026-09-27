"""Geocode FDNY alarm-box intersection strings with NYC Centerline.

About a third of CAD rows carry an alarm-box number that is missing from the
current in-service listing (``v57i-gtxb``) — typically retired boxes — so they
used to fall back to the ZIP centroid. CAD still records the box location as an
intersection string ("3 AVE & E 44 ST"), and NYC Centerline (``inkn-q76z``)
uses the same street-name style, so the intersection is the shared endpoint of
the two streets' centerline segments. When a name pair occurs in more than one
place (e.g. "BROADWAY & 1 AVE"), the candidate nearest the incident ZIP
centroid wins.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .depots import _haversine_km

# Token-level canonical forms (applied to both CAD and Centerline names).
_TOKEN_MAP = {
    "AV": "AVE", "AVENUE": "AVE",
    "STREET": "ST",
    "PWAY": "PKWY", "PY": "PKWY", "PARKWAY": "PKWY", "PKY": "PKWY",
    "BOULEVARD": "BLVD", "BL": "BLVD",
    "ROAD": "RD", "PLACE": "PL", "DRIVE": "DR", "LANE": "LN", "COURT": "CT",
    "TERRACE": "TER", "TERR": "TER",
    "EXPRESSWAY": "EXPY", "EXPWY": "EXPY", "EXWY": "EXPY", "EXPW": "EXPY",
    "TURNPIKE": "TPKE", "TPK": "TPKE", "HIGHWAY": "HWY",
    "POINT": "PT", "BEACH": "BCH", "SAINT": "ST",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "SQUARE": "SQ", "PLAZA": "PLZ", "CIRCLE": "CIR", "BRIDGE": "BRG",
    "BWAY": "BROADWAY", "PENNA": "PENNSYLVANIA", "EX": "EXPY", "EXP": "EXPY",
}
# Whole-name aliases CAD uses for streets Centerline names differently.
_ALIASES = {
    "6 AVE": ["AVE OF THE AMERICAS"],
    "AVE OF THE AMERICAS": ["6 AVE"],
    "LENOX AVE": ["MALCOLM X BLVD"],
    "7 AVE": ["ADAM CLAYTON POWELL JR BLVD"],
    "8 AVE": ["FREDERICK DOUGLASS BLVD"],
}
_TRANSIT_PREFIX = re.compile(r"^(?:IND|IRT|BMT|SIR|PATH)\b.*?\s-\s*")
# Streets that cross without sharing a segment endpoint (offset intersections,
# slivers): accept the closest pair of endpoints within this distance.
NEAR_MATCH_M = 40.0
_ORD = re.compile(r"\b(\d+)(?:ST|ND|RD|TH)\b")
_SPLIT = re.compile(r"\s+&\s+|\s+AND\s+")
_OFFSET = re.compile(r"^(.*?)\s+\d+\s*(?:FT\s+)?[NSEW]\s+OF\s+(.*)$")
_BETWEEN = re.compile(r"^(.*?)\s+BET(?:WEEN)?\s+(.*?)\s*&\s*(.*)$")


def normalize_street(name: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    s = _ORD.sub(r"\1", s)
    toks = [_TOKEN_MAP.get(t, t) for t in s.split()]
    return " ".join(toks)


def _options(raw: str) -> list[str]:
    """All candidate names for one side ("LENOX AVE/MALCOLM X BLVD" → both + aliases)."""
    out: list[str] = []
    for part in raw.split("/"):
        n = normalize_street(part)
        if n and n not in out:
            out.append(n)
        for a in _ALIASES.get(n, []):
            if a not in out:
                out.append(a)
    return out


def parse_intersection(loc: str) -> tuple[list[str], list[str]] | None:
    """Return candidate names for the two streets of a CAD box location, or None."""
    if not isinstance(loc, str) or not loc.strip():
        return None
    s = _TRANSIT_PREFIX.sub("", loc.upper().strip())
    m = _BETWEEN.match(s)
    if m:  # "A BET B & C" → A & B (box sits on A near B)
        a, b = m.group(1), m.group(2)
    else:
        m = _OFFSET.match(s)
        if m:  # "A 200 W OF B" → A & B (offset ignored, ~60 m)
            a, b = m.group(1), m.group(2)
        else:
            parts = _SPLIT.split(s)
            if len(parts) < 2:
                return None
            a, b = parts[0], parts[1]
    oa, ob = _options(a), _options(b)
    return (oa, ob) if oa and ob else None


class IntersectionIndex:
    """Street name → set of centerline endpoint coordinates."""

    def __init__(self, centerline) -> None:
        pts_by_name: dict[str, set[tuple[float, float]]] = defaultdict(set)
        geoms = centerline.geometry.values
        names = centerline["full_street_name"].astype(str).map(normalize_street).values
        for geom, name in zip(geoms, names):
            if geom is None or not name:
                continue
            parts = getattr(geom, "geoms", [geom])
            for line in parts:
                cs = line.coords
                for x, y in (cs[0], cs[-1]):
                    pts_by_name[name].add((round(y, 5), round(x, 5)))
        self.pts = dict(pts_by_name)

    def locate(self, a: str, b: str, near_lat: float, near_lon: float, max_km: float) -> tuple[float, float] | None:
        pa, pb = self.pts.get(a), self.pts.get(b)
        if not pa or not pb or near_lat != near_lat or near_lon != near_lon:
            return None
        common = pa & pb
        if common:
            c = np.array(list(common))
            d = _haversine_km(c[:, 1], c[:, 0], near_lon, near_lat)
            j = int(np.argmin(d))
            return (float(c[j, 0]), float(c[j, 1])) if d[j] <= max_km else None

        # No shared endpoint: closest endpoint pair near the ZIP, if tight enough.
        A = np.array(list(pa))
        B = np.array(list(pb))
        A = A[_haversine_km(A[:, 1], A[:, 0], near_lon, near_lat) <= max_km]
        B = B[_haversine_km(B[:, 1], B[:, 0], near_lon, near_lat) <= max_km]
        if not len(A) or not len(B):
            return None
        k = np.cos(np.radians(40.7))
        dy = (A[:, None, 0] - B[None, :, 0]) * 111_320
        dx = (A[:, None, 1] - B[None, :, 1]) * 111_320 * k
        dist = np.hypot(dx, dy)
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if dist[i, j] > NEAR_MATCH_M:
            return None
        return float((A[i, 0] + B[j, 0]) / 2), float((A[i, 1] + B[j, 1]) / 2)


def geocode_box_locations(
    locs: pd.DataFrame,
    centerline,
    *,
    max_km_from_zip: float = 4.0,
) -> pd.DataFrame:
    """Geocode unique (alarm_box_location, zip_lat, zip_lon) rows.

    ``zip_lat`` / ``zip_lon`` anchor the search (nearest candidate wins); an
    optional ``max_km`` column overrides the search radius per row.
    Returns ``geo_lat`` / ``geo_lon`` aligned to ``locs.index`` (NaN when not found).
    """
    idx = IntersectionIndex(centerline)
    lat = np.full(len(locs), np.nan)
    lon = np.full(len(locs), np.nan)
    radius = (
        locs["max_km"].to_numpy(float)
        if "max_km" in locs.columns
        else np.full(len(locs), max_km_from_zip)
    )
    for i, (loc, zlat, zlon) in enumerate(
        zip(locs["alarm_box_location"], locs["zip_lat"], locs["zip_lon"])
    ):
        pair = parse_intersection(loc)
        if pair is None:
            continue
        hit = None
        for a in pair[0]:
            for b in pair[1]:
                hit = idx.locate(a, b, zlat, zlon, radius[i])
                if hit is not None:
                    break
            if hit is not None:
                break
        if hit is not None:
            lat[i], lon[i] = hit
    return pd.DataFrame({"geo_lat": lat, "geo_lon": lon}, index=locs.index)
