"""Shared helpers for London (LFB) closure validation / ranking / training scripts.

Crow-fly (haversine) distances, nearest-station lookups, and loading of the
planner incident table joined to firehouse start coordinates.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .cities import get_city, load_firehouses
from .kolesar import KolesarModel
from .lfb_standards import DEFAULT_TURNOUT_S

EARTH_R_KM = 6371.0088
DRIVE_CLIP_S = (30.0, 1200.0)
MAX_CROW_KM = 25.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_station_km(
    inc_lat: np.ndarray, inc_lon: np.ndarray, st_lat: np.ndarray, st_lon: np.ndarray, chunk: int = 5000
) -> np.ndarray:
    """Crow-fly km from each incident to its nearest station."""
    inc_lat, inc_lon = np.asarray(inc_lat, float), np.asarray(inc_lon, float)
    out = np.empty(inc_lat.size)
    for i in range(0, inc_lat.size, chunk):
        d = haversine_km(
            inc_lat[i : i + chunk, None], inc_lon[i : i + chunk, None], st_lat[None, :], st_lon[None, :]
        )
        out[i : i + chunk] = d.min(axis=1)
    return out


def predicted_attendance_s(model: KolesarModel, crow_km: np.ndarray, turnout_s: float = DEFAULT_TURNOUT_S):
    return model.predict(crow_km) + turnout_s


def load_london(
    years: Optional[Sequence[int]] = None, max_rows: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (firehouses, planner incidents with start_lat/start_lon/crow_km).

    Start coords come from ``deployed_from_station`` (fallback ``station_ground``).
    Incidents with no matching station or crow_km outside (0, MAX_CROW_KM] are dropped.
    """
    spec = get_city("london")
    fh = load_firehouses(spec)
    inc = pd.read_csv(spec.incidents_path)
    if years is not None:
        inc = inc[inc["cal_year"].isin([int(y) for y in years])]
    lookup = fh.drop_duplicates("facilityname").set_index("facilityname")[["lat", "lon"]]
    dep, gr = inc["deployed_from_station"], inc["station_ground"]
    start = dep.where(dep.isin(lookup.index), gr)
    inc = inc.assign(start_station=start)
    inc = inc[inc["start_station"].isin(lookup.index)].copy()
    inc["start_lat"] = inc["start_station"].map(lookup["lat"]).to_numpy()
    inc["start_lon"] = inc["start_station"].map(lookup["lon"]).to_numpy()
    inc["crow_km"] = haversine_km(inc["start_lat"], inc["start_lon"], inc["dest_lat"], inc["dest_lon"])
    # drop bad geocodes (e.g. 0,0 lat/lon gave crow_km in the thousands) and implausible hops
    inc = inc[(inc["crow_km"] > 0) & (inc["crow_km"] <= MAX_CROW_KM)]
    inc["drive_s"] = (inc["travel_seconds"] - DEFAULT_TURNOUT_S).clip(*DRIVE_CLIP_S)
    if max_rows is not None and len(inc) > max_rows:
        inc = inc.sample(max_rows, random_state=42)
    return fh, inc.reset_index(drop=True)
