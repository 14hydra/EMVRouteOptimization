"""Hourly traffic congestion from NYC DOT Traffic Speeds (``i4gi-tjb9``).

The feed reports speed every ~5 min on ~150 monitored links (mostly highways
and major arterials). We aggregate server-side to link × hour, then express
congestion as speed / the link's own free-flow speed (85th percentile of its
hourly speeds), so a highway and an avenue are comparable.

Features use the **previous full hour**, which is always known at dispatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

URL = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"

TRAFFIC_FEATURE_COLUMNS = [
    "traffic_city_index",
    "traffic_borough_index",
    "traffic_near_index",
    "traffic_near_km",
    "traffic_city_speed_mph",
]


def _get(params: dict) -> list[dict]:
    r = requests.get(URL, params=params, timeout=600)
    r.raise_for_status()
    return r.json()


def download_hourly_speeds(start: str, end: str, out_dir: Path | str, *, page: int = 50_000) -> tuple[Path, Path]:
    """Link × hour mean speeds for [start, end] (month chunks) + link geometry."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    months = pd.date_range(pd.Timestamp(start).to_period("M").start_time, end, freq="MS")
    for m0 in tqdm(months, desc="traffic months"):
        m1 = min(m0 + pd.offsets.MonthBegin(1), pd.Timestamp(end) + pd.Timedelta(seconds=1))
        where = (
            f"data_as_of >= '{m0:%Y-%m-%dT%H:%M:%S}' and data_as_of < '{m1:%Y-%m-%dT%H:%M:%S}' "
            "and speed::number > 0"
        )
        offset = 0
        while True:
            batch = _get(
                {
                    "$select": "link_id,date_trunc_ymd(data_as_of) as d,date_extract_hh(data_as_of) as h,"
                    "avg(speed::number) as spd,count(*) as n",
                    "$where": where,
                    "$group": "link_id,d,h",
                    "$order": "link_id,d,h",
                    "$limit": page,
                    "$offset": offset,
                }
            )
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page
    hourly = pd.DataFrame(rows)
    hourly["hour_ts"] = pd.to_datetime(hourly["d"]) + pd.to_timedelta(hourly["h"].astype(int), unit="h")
    hourly = hourly.assign(
        link_id=hourly["link_id"].astype(str),
        spd=pd.to_numeric(hourly["spd"]),
        n=pd.to_numeric(hourly["n"]),
    )[["link_id", "hour_ts", "spd", "n"]]
    h_path = out_dir / "traffic_hourly.parquet"
    hourly.to_parquet(h_path, index=False)

    links = link_geometry(end)
    l_path = out_dir / "traffic_links.csv"
    links.to_csv(l_path, index=False)
    return h_path, l_path


def link_geometry(as_of: str) -> pd.DataFrame:
    """Latest geometry per link from one day of readings (grouping the whole
    112M-row table over the long ``link_points`` text is too slow)."""
    t1 = pd.Timestamp(as_of)
    t0 = t1 - pd.Timedelta(days=1)
    rows = _get(
        {
            "$select": "link_id,borough,link_name,link_points",
            "$where": f"data_as_of >= '{t0:%Y-%m-%dT%H:%M:%S}' and data_as_of <= '{t1:%Y-%m-%dT%H:%M:%S}'",
            "$order": "data_as_of DESC",
            "$limit": 50_000,
        }
    )
    return pd.DataFrame(rows).drop_duplicates("link_id")


def _link_points(s: str) -> np.ndarray:
    pts = []
    for tok in str(s).split():
        try:
            lat, lon = tok.split(",")
            pts.append((float(lat), float(lon)))
        except ValueError:  # the feed truncates some point strings
            continue
    return np.array(pts) if pts else np.empty((0, 2))


def congestion_tables(hourly: pd.DataFrame, links: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(link×hour index, borough×hour index, city×hour index/speed)."""
    h = hourly.merge(links[["link_id", "borough"]].astype({"link_id": str}), on="link_id", how="left")
    ff = h.groupby("link_id")["spd"].quantile(0.85).rename("ff")
    h = h.join(ff, on="link_id")
    h["idx"] = (h["spd"] / h["ff"]).clip(0, 1.5)
    link_hr = h[["link_id", "hour_ts", "idx"]]
    boro_hr = h.groupby(["borough", "hour_ts"])["idx"].mean().rename("traffic_borough_index").reset_index()
    city_hr = h.groupby("hour_ts").agg(traffic_city_index=("idx", "mean"), traffic_city_speed_mph=("spd", "mean")).reset_index()
    return link_hr, boro_hr, city_hr


_BORO = {"MANHATTAN": "Manhattan", "BROOKLYN": "Brooklyn", "QUEENS": "Queens", "BRONX": "Bronx", "STATEN ISLAND": "Staten Island"}


def attach_traffic_features(df: pd.DataFrame, hourly: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Add previous-hour congestion features to an incident table.

    ``df`` needs incident_datetime, borough (upper-case), dest_lat, dest_lon.
    """
    from scipy.spatial import cKDTree

    link_hr, boro_hr, city_hr = congestion_tables(hourly, links)
    out = df.copy()
    key = pd.to_datetime(out["incident_datetime"]).dt.floor("h") - pd.Timedelta(hours=1)

    city = city_hr.set_index("hour_ts")
    out["traffic_city_index"] = key.map(city["traffic_city_index"]).to_numpy()
    out["traffic_city_speed_mph"] = key.map(city["traffic_city_speed_mph"]).to_numpy()

    b = boro_hr.set_index(["borough", "hour_ts"])["traffic_borough_index"]
    bk = pd.MultiIndex.from_arrays([out["borough"].map(_BORO), key])
    out["traffic_borough_index"] = b.reindex(bk).to_numpy()

    # Nearest monitored link to the incident (by any point on its polyline).
    k = np.cos(np.radians(40.7))
    pts, ids = [], []
    for lid, lp in zip(links["link_id"].astype(str), links["link_points"]):
        p = _link_points(lp)
        pts.append(p)
        ids.extend([lid] * len(p))
    pts = np.vstack([p for p in pts if len(p)])
    tree = cKDTree(np.c_[pts[:, 1] * k, pts[:, 0]] * 111.32)
    q = np.c_[out["dest_lon"].to_numpy() * k, out["dest_lat"].to_numpy()] * 111.32
    ok = np.isfinite(q).all(axis=1)
    dist = np.full(len(out), np.nan)
    near = np.array([None] * len(out), dtype=object)
    d_, i_ = tree.query(q[ok])
    dist[ok] = d_
    near[ok] = np.array(ids, dtype=object)[i_]
    out["traffic_near_km"] = dist
    li = link_hr.set_index(["link_id", "hour_ts"])["idx"]
    out["traffic_near_index"] = li.reindex(pd.MultiIndex.from_arrays([near, key])).to_numpy()
    return out
