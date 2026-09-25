"""NYC Open Data SODA helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://data.cityofnewyork.us/resource"

DATASETS: dict[str, dict[str, Any]] = {
    # Primary CAD extract — fire apparatus / firetrucks.
    "fdny_incidents": {
        "id": "8m42-w767",
        "description": "FDNY Fire Incident Dispatch Data (no unit start GPS)",
        "default_select": (
            "starfire_incident_id,incident_datetime,alarm_box_borough,alarm_box_number,"
            "alarm_box_location,incident_borough,zipcode,"
            "incident_classification,incident_classification_group,"
            "dispatch_response_seconds_qy,incident_response_seconds_qy,"
            "incident_travel_tm_seconds_qy,valid_incident_rspns_time_indc,"
            "engines_assigned_quantity,ladders_assigned_quantity,"
            "other_units_assigned_quantity,highest_alarm_level"
        ),
        "datetime_col": "incident_datetime",
    },
    "fdny_firehouses": {
        "id": "hc8x-tcnd",
        "description": "FDNY Firehouse Listing (preferred firetruck depots)",
    },
    "alarm_boxes": {
        "id": "v57i-gtxb",
        "description": "In-Service Alarm Box Locations (intersection proxies for on-road posts / CSLs)",
    },
    "modzcta": {
        "id": "pri4-ifjk",
        "description": "Modified Zip Code Tabulation Areas (incident ZIP geometry)",
    },
    "lion": {
        "id": "2v4z-66xt",
        "description": "NYC LION street centerline database",
        "note": "Large geospatial asset; prefer DCP download or GeoJSON export when available.",
    },
    # Legacy EMS layers kept for optional side comparisons only — not used by default.
    "ems_incidents": {
        "id": "76xm-jjuj",
        "description": "EMS Incident Dispatch Data (legacy; project scope is firetrucks)",
        "default_select": (
            "incident_id,incident_datetime,initial_call_type,final_call_type,"
            "initial_severity_level_code,final_severity_level_code,"
            "dispatch_response_seconds_qy,incident_response_seconds_qy,"
            "incident_travel_tm_seconds_qy,borough,incident_dispatch_area,"
            "zipcode,held_indicator,valid_incident_rspns_time_indc"
        ),
        "datetime_col": "incident_datetime",
    },
    "ems_stations": {
        "id": "ji82-xba5",
        "description": "City Facilities Database — EMS/ambulance stations (legacy)",
        "default_where": (
            "upper(factype) in ('AMBULANCE STATION','EMERGENCY MEDICAL STATION','EMERGENCY MEDICL STN') "
            "OR upper(facname) like '%EMS STATION%'"
        ),
        "default_select": (
            "uid,facname,factype,facsubgrp,address,boro,borocode,zipcode,"
            "latitude,longitude,opname,optype"
        ),
    },
    "hospital_bays": {
        "id": "ji82-xba5",
        "description": "City Facilities — HOSPITAL / ACUTE CARE HOSPITAL (legacy EMS staging)",
        "default_where": "upper(factype) in ('HOSPITAL','ACUTE CARE HOSPITAL')",
        "default_select": (
            "uid,facname,factype,facsubgrp,address,boro,borocode,zipcode,"
            "latitude,longitude,opname,optype,overagency"
        ),
    },
}


def sodaclient_get(
    dataset_id: str,
    *,
    limit: int = 1000,
    offset: int = 0,
    where: str | None = None,
    select: str | None = None,
    order: str | None = None,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"$limit": limit, "$offset": offset}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if order:
        params["$order"] = order
    url = f"{BASE}/{dataset_id}.json"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_dataset(
    key: str,
    out_dir: Path | str,
    *,
    limit: int | None = 5000,
    page_size: int = 1000,
    where: str | None = None,
    years: Iterable[int] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Download a named dataset to CSV under out_dir. Returns output path.

    For incident CAD extracts, prefer ``start``/``end`` ISO datetimes (full calendar
    days) over a bare ``limit`` on newest rows — the latter truncates overnight hours.
    """
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset key {key!r}. Known: {sorted(DATASETS)}")
    meta = DATASETS[key]
    dataset_id = meta["id"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    select = meta.get("default_select")
    rows: list[dict[str, Any]] = []
    offset = 0
    target = limit if limit is not None else 10**12

    dt_col = meta.get("datetime_col", "incident_datetime")
    incident_keys = {"fdny_incidents", "ems_incidents"}
    where_parts = []
    if meta.get("default_where"):
        where_parts.append(f"({meta['default_where']})")
    if where:
        where_parts.append(f"({where})")
    if start and end and key in incident_keys:
        where_parts.append(f"({dt_col} between '{start}' and '{end}')")
    elif years and key in incident_keys:
        year_clauses = [
            f"{dt_col} between '{y}-01-01T00:00:00' and '{y}-12-31T23:59:59'"
            for y in years
        ]
        where_parts.append("(" + " OR ".join(year_clauses) + ")")
    where_q = " AND ".join(where_parts) if where_parts else None

    order = None
    if key in incident_keys:
        order = f"{dt_col} ASC" if (start and end) else f"{dt_col} DESC"

    pbar = tqdm(total=min(target, 50_000) if limit else None, desc=f"download:{key}", unit="row")
    while len(rows) < target:
        batch_n = min(page_size, target - len(rows))
        batch = sodaclient_get(
            dataset_id,
            limit=batch_n,
            offset=offset,
            where=where_q,
            select=select,
            order=order,
        )
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        pbar.update(len(batch))
        if len(batch) < batch_n:
            break
    pbar.close()

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{key}.csv"
    df.to_csv(out_path, index=False)

    meta_path = out_dir / f"{key}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "key": key,
                "dataset_id": dataset_id,
                "rows": len(df),
                "columns": list(df.columns),
                "description": meta.get("description"),
                "where": where_q,
                "start": start,
                "end": end,
            },
            indent=2,
        )
        + "\n"
    )
    return out_path
