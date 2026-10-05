# Datasets: London primary, NYC / SF secondary

The CSEF/ISEF study is **London** (LFB). NYC and SF are legacy / secondary sources that
supplied scaffolding code. Prefer sources with **coordinates + a mobilise→arrive
(or dispatch→arrive) interval**.

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.

## Primary: London

| Layer | Source | Status |
|---|---|---|
| Incidents + attendance | **LFB incident records** (London Datastore): `FirstPumpArriving_AttendanceTime` (mobilise→arrive), `FirstPumpArriving_DeployedFromStation`, `IncidentStationGround`, second pump fields, BNG `Easting_m`/`Northing_m` and lat/lon | **EXISTS**: `data/raw/london/lfb_incidents_2024_onwards.csv` (+ xlsx). Calendar years 2024 → partial 2026. **2023 and earlier PLANNED** (training/demand use 2023–24, eval 2025) |
| Planner extract | `scripts/build_lfb_planner_incidents.py` → `lfb_incidents_planner.csv` (~129k rows after coordinate + 30–1800 s attendance filter) | **EXISTS** |
| Stations | `data/raw/london/london_firehouses.csv` (name, lat, lon; real station buildings from OpenStreetMap) | **EXISTS** |
| Mobilisation records (turnout vs drive split) | LFB, when available | **PLANNED / not obtained** |
| Street network, primary | **Ordnance Survey NGD** road links (class, width, speed limit, directionality) | **PLANNED** — not downloaded, no loader |
| Street network, fallback | **OpenStreetMap** (OSMnx) | Tooling **EXISTS** (`scripts/build_osm_graph.py`); `london_drive.graphml` not built |
| Weather | Open-Meteo archive hourly | Tooling **EXISTS** (`emvro.weather`; NYC cache only); London pull **PLANNED** |
| 2014 closures | `emvro.london_closures_2014` (10 stations, 9 Jan 2014) | **EXISTS** |
| GPS traces (route-model calibration) | FOI request | **PLANNED / not obtained** (see `docs/route_models.md`) |

Key London facts for modelling:

- The public CSV gives **attendance only** (turnout + drive).
- London records the **deployed station** for the first and second pump, so starting
  locations are known (unlike FDNY; see `docs/starting_locations.md`). `busy_flag`
  (`deployed ≠ ground`) is derived from this.
- Standards: first engine mean ≤ 6 min, > 90 % within 10 min (`emvro.lfb_standards`).

## Secondary: NYC (legacy scaffolding)

FDNY Fire Incident Dispatch Data (`8m42-w767`), firehouses, fire company polygons, alarm
boxes, LION, DOT traffic speeds. Missing apparatus GPS at assignment. Used for the
bag4 hybrid and router demos. Details: root `README.md` (Legacy NYC), 
`docs/starting_locations.md`, `docs/travel_time_training.md` (appendix).

## Secondary: other public CAD sources

| City | Dataset | Why it might help | Status in repo |
|---|---|---|---|
| **San Francisco** | [Fire/EMS Calls for Service](https://data.sf.gov/resource/nuek-vuh3.json) (`nuek-vuh3`) | `response_dttm`→`on_scene_dttm`, `case_location` lon/lat | `scripts/ingest_sf_ems.py` → `data/processed/od_pairs_sf_ems.csv` (**EXISTS**, EMS-oriented, secondary) |
| **Dublin** | [DFB Ambulance Incidents](https://data.smartdublin.ie/dataset/fire-brigade-and-ambulance) | TOC/ORD/MOB/IA timestamps + geometry (has a mobilisation stamp, as LFB internally does) | Planned: `scripts/ingest_dublin_ems.py` (not written) |
| **Chicago** | EMS Calls | Large US CAD volume | Candidate |
| **Seattle** | SPD/SFD 911 | Network diversity | Candidate |
| **Virginia Beach** | [EMS Calls for Service](https://data.virginia.gov/dataset/ems-calls-for-service) | Full CAD lifecycle | Candidate |

Paris (BSPP) was evaluated and **discarded** (no public files in the repo). The
replacement for it is **London** (LFB publishes incident-level data), not NYC.

## Target schema (generic OD table)

- `city` (e.g. `london`, `nyc`, `san_francisco`)
- `start_lat`, `start_lon`, `dest_lat`, `dest_lon`
- `travel_seconds` (London: attendance; separate turnout column once available)
- `hour`, `dow`, optional `borough`/neighbourhood
- `depot_layer`, `depot_name`, `crow_flies_km`

The London planner CSV is a subset: `dest_lat`, `dest_lon`, `travel_seconds`, `cal_year`,
`date_of_call`, `hour_of_call`, `borough`, `station_ground`, `deployed_from_station`,
`second_attendance_s`, `second_deployed_from`, `busy_flag`. It has **no** `start_lat/lon`
yet — joining `deployed_from_station` to station coordinates is **PLANNED**.

## How a new city plugs in (generic, NYC/SF-era pipeline)

1. Ingest → `data/processed/od_pairs_<city>.csv`
2. Build an OSM drive graph for the bbox (`street_features.build_drive_graph`)
3. Append to `route_eta_training.csv` with `city` categorical
4. Optional: `enrich_training_with_gmaps.py`
5. Retrain: `scripts/train_route_eta_model.py`

## SF multi-city quick start (legacy)

```bash
PYTHONPATH=src python scripts/ingest_sf_ems.py --limit 4000
PYTHONPATH=src python scripts/build_multicity_route_eta.py
PYTHONPATH=src python scripts/train_route_eta_model.py \
  --data data/processed/route_eta_multicity_training.csv --target-r2 0.9
```

Raw SF CAD travel times alone do **not** support R² ≥ 0.9. The multi-city route-ETA labels
are OSM→EMV mapped (network-derived), so that R² is not real-world accuracy and is not a
London result.
