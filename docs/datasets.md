# Datasets (London)

All raw files live in `data/raw/london/` and are not in git.
`scripts/download_datasets.py` fetches them.

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.

## Sources

| Layer | Source | File(s) | Status |
|---|---|---|---|
| Incidents | **LFB incident records** (London Datastore, OGL): location (BNG `Easting_m`/`Northing_m`, always `Easting_rounded`/`Northing_rounded` to 50 m), `IncidentStationGround`, `FirstPumpArriving_AttendanceTime`, deployed station for 1st/2nd pump | `lfb_incidents_2018_2023.csv`, `lfb_incidents_2024_onwards.csv` | **EXISTS** |
| Engine trips | **LFB mobilisation records** (London Datastore, OGL): one row per engine sent; `DeployedFromStation_Name`, `DeployedFromLocation` (Home Station / Other Station), `TurnoutTimeSeconds`, `TravelTimeSeconds`, `DelayCode_Description`, `PumpOrder` | `lfb_mobilisations_2021_2024.csv`, `lfb_mobilisations_2025.csv` | **EXISTS** |
| Station locations | OpenStreetMap `amenity=fire_station`, name-matched to the 102 LFB stations; 8 pinned to specific OSM features (`OSM_OVERRIDES`) | `london_firehouses.csv` (built by `scripts/build_london_firehouses.py`) | **EXISTS** |
| Street network | OpenStreetMap, Geofabrik Greater London extract, filtered to drivable roads (osmnx "drive" rules) | `greater-london-latest.osm.pbf` → `london_drive.graphml` (built by `emvro.road_distance`) | **EXISTS** |
| Street detail | **Ordnance Survey NGD** road links (width, speed limit, class) | — | **PLANNED** |
| Pre-2014 incidents | LFB incident records 2009–2017 (London Datastore) | — | **PLANNED** (2014 closure test) |
| GPS traces | FOI request to LFB | — | **PLANNED / not obtained** |

## Derived tables

| Table | Built by | Contents |
|---|---|---|
| `data/raw/london/lfb_incidents_planner.csv` | `scripts/build_lfb_planner_incidents.py` | One row per incident (2024+): lat/lon, attendance, station ground, deployed station, `busy_flag`. Demand for the planner. |
| `data/processed/lfb_travel_training.csv` | `scripts/build_lfb_travel_training.py` | One row per engine trip (1,024,833 rows, 2021 → 2026): real station origin, incident location, `crow_km`, `road_km`, `drive_s` = `TravelTimeSeconds`. Training data for the travel-time model. |

The training set keeps trips where the engine left from an LFB station, was not already
out on the road ("On outside duty when mobilised"), had a correct address, and drove
30–1200 s. Trips are joined to incident locations on `IncidentNumber`.

## Key London facts for modelling

- The mobilisation records split turnout from driving time and name the station each
  engine actually left from, so every trip's origin is known.
- `busy_flag` (`deployed ≠ station ground`) marks engines sent from outside the
  incident's own station area.
- Standards: first engine mean ≤ 6 min, > 90 % within 10 min (`emvro.lfb_standards`).
