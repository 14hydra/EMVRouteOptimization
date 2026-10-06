# EMV Route Optimization — London Fire Brigade station placement (CSEF / ISEF)

**Optimizing London Fire Brigade station placement with a street-characteristic
travel-time model** (gradient-boosted decision trees, LightGBM),
scored on held-out 2025 LFB calls.

Team: **Ricky Zhao** and **Sarp Akalin** (ASI), CSEF/ISEF.

> **Status legend used in these docs:** **EXISTS** = code or data in this repo today ·
> **PLANNED** = in the project plan, not built / not run yet. Please do not read
> PLANNED items as results.

## Research question and hypothesis

**Research question.** Does placing fire stations with *street-aware* driving-time
predictions beat placement by straight-line distance or the classic Kolesar
distance→time curve, when scored on real 2025 LFB calls?

**Hypothesis.** A travel-time model that sees street characteristics (road class,
width, speed limits, one-ways, junction density, …) will
rank candidate station layouts differently from distance-only or Kolesar-only
placement, and the street-aware layout will do better on held-out calls
(mean first-engine attendance and share of calls within 10 min).

**Not our claim:** that we have beaten LFB's real operational planning, or that
public data alone recovers the true turnout/drive split (see
[Limits](#limits-and-honesty)).

## Standards we score against

London Fire Brigade (London Safety Plan 2022) — **not** NFPA 1710 / the US 4-minute rule:

| Standard | Target |
|---|---|
| First engine | average **≤ 6 min**; **> 90 %** of calls within **10 min** |
| Second engine | average ≤ 8 min; 95th percentile within 12 min |

Encoded in `src/emvro/lfb_standards.py` (**EXISTS**).

## What we score

A candidate placement is scored with

```
attendance  =  predicted driving time (station → incident)  +  that station's recorded turnout
```

- The **public LFB incident CSV only has attendance** (`FirstPumpArriving_AttendanceTime`,
  mobilise → arrive). Turnout and drive are not separable there.
- When LFB **mobilisation** records are available we split turnout from drive and learn
  per-station turnout. Until then, `DEFAULT_TURNOUT_S` (60 s) is a fallback prior for
  hypothetical sites — a stated assumption, not a measurement.

## Three models

| # | Model | Role | Status |
|---|---|---|---|
| 1 | **Travel-time LightGBM, no station IDs** — gradient-boosted trees, 4-seed average | Predict driving time to any point from any site | `train_travel_time_lfb.py` **EXISTS** on mobilisation `TravelTimeSeconds` (train 2021–24 / test 2025, MAE ≈ **72 s** vs Kolesar ≈ 76 s); full street attrs **PLANNED** |
| 2 | **Route model** with learned segment speeds | Dijkstra under learned EMV edge speeds; FOI GPS when available | API scaffold **EXISTS** (`segment_speeds.py`); London graph + FOI **PLANNED** |
| 3 | **Placement on a directed street graph** | redesign / replace / expand station sets | All three modes **EXISTS**; London OSM/OS NGD graph **PLANNED** |

Details: [`docs/travel_time_training.md`](docs/travel_time_training.md),
[`docs/route_models.md`](docs/route_models.md),
[`docs/firehouse_location.md`](docs/firehouse_location.md).

## Validation plan

1. **2014 ten-station closures** — on 9 Jan 2014 LFB closed Belsize, Bow, Clerkenwell,
   Downham, Kingsland, Knightsbridge, Silvertown, Southwark, Westminster, Woolwich.
   Counterfactual + ranking scripts **EXIST** on today's network with approximate
   closed-station coords; the true pre-2014 train → post-2014 test needs older LFB
   extracts (**PLANNED** ingest).
2. **Leave-one-firehouse-out** — hide one station's calls and predict its area from the rest. **PLANNED**
3. **Multi-scorer** — score layouts under Kolesar / network / ML; planner supports
   `--scorer crow|kolesar|network` today (**EXISTS**); full cross-table comparison script **PLANNED**

Temporal split: demand from **2023–24**, evaluation on held-out **2025**.
The repo currently has LFB data from 2024 onward only, so demand is **2024-only until
2023 is ingested**. The planner enforces `--demand-years` / `--eval-years` when
`cal_year` is present.

## Datasets

| Role | Source | Status |
|---|---|---|
| Incidents, attendance, deployed station, 2nd pump | **LFB incident records** (London Datastore). `data/raw/london/lfb_incidents_2024_onwards.csv` (+ xlsx) | **EXISTS** (2024 → 2026 partial); 2023 and earlier **PLANNED** |
| Planner extract | `scripts/build_lfb_planner_incidents.py` → `data/raw/london/lfb_incidents_planner.csv` (lat/lon from BNG, `travel_seconds` = attendance, `busy_flag`) | **EXISTS** (~129k rows) |
| Station list | `data/raw/london/london_firehouses.csv` (real station buildings from OpenStreetMap; `scripts/build_london_firehouses.py`) | **EXISTS** |
| Street network (primary goal) | **Ordnance Survey NGD** (road links with width, speed limit, class, directionality) | **PLANNED** — not downloaded, no loader yet |
| Street network (fallback) | **OpenStreetMap** via OSMnx (`scripts/build_osm_graph.py`, `emvro.street_features`) | Tooling **EXISTS**; London graph (`london_drive.graphml`) not built yet |
| Weather | Open-Meteo archive hourly (`emvro.weather`) | Tooling **EXISTS** (cached for NYC); London pull **PLANNED** |
| 2014 closures | Hard-coded list in `emvro.london_closures_2014` | **EXISTS** |

## Code that exists for the London study

| Piece | Path |
|---|---|
| City registry (`london`, `nyc`, `sf`) | `src/emvro/cities.py` |
| Firehouse planner (redesign, replace, expand) | `src/emvro/firehouse_location.py`, `scripts/plan_firehouse_locations.py` |
| Kolesar piecewise `T(d)` (fit + predict) | `src/emvro/kolesar.py` |
| LFB 6 / 10-min standards scoring, turnout + drive | `src/emvro/lfb_standards.py` |
| Busy-engine rates (`deployed ≠ ground`) | `src/emvro/busy_engines.py` |
| 2014 closure list + fuzzy name matching | `src/emvro/london_closures_2014.py` |
| LFB planner CSV builder (with `busy_flag`) | `scripts/build_lfb_planner_incidents.py` |
| Placement-ready LFB travel-time trainer | `scripts/train_travel_time_lfb.py` |
| 2014 closure counterfactual + station ranking | `scripts/validate_london_2014_closures.py`, `scripts/rank_closed_stations_2014.py` |
| Segment-speed route scaffold | `src/emvro/segment_speeds.py`, `scripts/fit_segment_speeds.py` |
| Incident density map | `scripts/visualize_data.py` |

Still **PLANNED**: OS NGD download, London OSM graph, full street-feature LFB model,
SHAP/ablations, leave-one-firehouse-out CV, mobilisation turnout split, pre-2014 LFB ingest.

## Quick start (London first)

```bash
cd EMVRouteOptimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Put the LFB incident file at data/raw/london/lfb_incidents_2024_onwards.csv
#    (or LFB_Incident_data_from_2024_onwards.xlsx from the London Datastore).

# 2. Build the planner extract (BNG -> WGS84, attendance, busy_flag)
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py

# 3. Incident density map -> data/figures/london_incident_density.html
PYTHONPATH=src python scripts/visualize_data.py

# 4. Placement planner on London (Kolesar crow TT; no London graph yet)
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode redesign --scorer kolesar --no-graph \
  --demand-years 2024 --eval-years 2025
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode replace --replace 10 --scorer kolesar --no-graph
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode expand --add-stations 2 --scorer kolesar --no-graph

# 5. Placement-ready travel-time model + 2014 closure scaffolds
PYTHONPATH=src python scripts/build_lfb_travel_training.py --crow-only   # or without --crow-only once london_drive.graphml exists
PYTHONPATH=src python scripts/train_travel_time_lfb.py
PYTHONPATH=src python scripts/validate_london_2014_closures.py
PYTHONPATH=src python scripts/rank_closed_stations_2014.py

# 6. 3D race: newly placed station vs existing (firetruck routing)
PYTHONPATH=src python scripts/build_london_drive_graph.py                 # central London OSM graph
PYTHONPATH=src python scripts/build_firehouse_race_3d.py
open data/figures/firehouse_race_3d.html

# Synthetic smoke test (no real data)
PYTHONPATH=src python scripts/plan_firehouse_locations.py --demo
```

```python
# Kolesar prior, LFB standards, busy engines, 2014 closures (all importable today)
from emvro.kolesar import KolesarModel, fit_kolesar
from emvro.lfb_standards import score_first_engine, attendance_seconds
from emvro.busy_engines import estimate_busy_rates
from emvro.london_closures_2014 import CLOSED_2014, flag_closed
```

Defaults: `--city london`, `--threshold-min 10`, LFB `score_first_engine` in the
summary, drive + turnout attendance scoring, `--demand-years` / `--eval-years` when
`cal_year` is present. Modelled cover rates are optimistic vs observed attendance
(nearest engine always free unless `--use-busy`).

## Docs

| Doc | Content |
|---|---|
| [`docs/firehouse_location.md`](docs/firehouse_location.md) | Placement: modes, KPIs, scorers, closures, busy engines |
| [`docs/travel_time_training.md`](docs/travel_time_training.md) | LFB driving-time model; baselines; ablations; legacy FDNY appendix |
| [`docs/route_models.md`](docs/route_models.md) | Route model plan; NYC router demos as scaffolding |
| [`docs/multi_city_datasets.md`](docs/multi_city_datasets.md) | London primary; NYC / SF secondary |
| [`docs/model_implementations_template.md`](docs/model_implementations_template.md) | Catalog of model implementations + blank entry |
| [`docs/starting_locations.md`](docs/starting_locations.md) | NYC missing-start CAD (legacy); London has known stations |
| [`docs/patrol_routes.md`](docs/patrol_routes.md) | **Out of scope** sandbox |
| [`docs/gmaps_control.md`](docs/gmaps_control.md) | Google Maps control (NYC-era) |

## Limits and honesty

- Public LFB attendance = turnout + drive; we cannot separate them without mobilisation data.
- Station sites in `london_firehouses.csv` come from OpenStreetMap building footprints, name-matched to LFB stations (8 pinned by hand, see the script).
- `busy_flag` (first pump deployed from a station other than the incident's ground station)
  is a proxy for "home engine unavailable".
- Candidate sites are discrete; no land cost, staffing, or planning constraints.
- Street-feature gains over a good distance prior may be small; the NYC experiments saw
  street features add little to incident-level R². The London claim is about **placement
  ranking**, and we will report that honestly whichever way it comes out.

## Out of the main story

- **Patrol posts / loops** (`scripts/build_patrol_routes.py`) — sandbox only, see `docs/patrol_routes.md`.
- **3D ambulance race visualization** (`scripts/build_ambulance_race_3d.py`) — demo only.

## Legacy NYC / FDNY

Everything below is **legacy scaffolding** from the original NYC project. It still runs and
is where the LightGBM bag-of-4 hybrid and the route-model demos were prototyped; it is not
the CSEF/ISEF headline.

**Problem.** Public FDNY Fire Incident Dispatch Data (`8m42-w767`) has incident geography
(borough, ZIP, alarm box) and travel times, but not apparatus GPS at assignment. We
reconstruct starts by alarm box → first-due engine polygon → firehouse, with a nearest
firehouse / CSL hybrid fallback and QC flags (see `docs/starting_locations.md`).
London does not have this problem: LFB records `DeployedFromStation`.

| Role | Source | Open Data ID |
|------|--------|--------------|
| Incidents + travel times | FDNY Fire Incident Dispatch Data | `8m42-w767` |
| Firehouses | FDNY Firehouse Listing | `hc8x-tcnd` |
| First-due areas | Fire Companies | `bst7-5464` |
| Incident / CSL geometry | In-Service Alarm Box Locations | `v57i-gtxb` |
| ZIP geometry | MODZCTA | `pri4-ifjk` |
| Streets | NYC LION / OSM | `2v4z-66xt` |
| Weather | Open-Meteo (`data/processed/weather_hourly_nyc.csv`) | — |

```bash
python scripts/download_datasets.py --limit 5000
python scripts/build_od_pairs.py --start-mode hybrid

# Travel-time models (FDNY)
PYTHONPATH=src python scripts/build_osm_graph.py   # once
PYTHONPATH=src python scripts/build_travel_time_training_set.py
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full
PYTHONPATH=src python scripts/build_travel_time_network_dataset.py --raw data/raw/big
PYTHONPATH=src python scripts/train_travel_time_network.py
PYTHONPATH=src python scripts/train_travel_time_hybrid.py

# NYC firehouse planner
PYTHONPATH=src python scripts/plan_firehouse_locations.py --city nyc --mode replace --replace 10 --no-graph

# NYC route-model demos (see docs/route_models.md)
PYTHONPATH=src python scripts/run_route_models.py --origin-lat 40.758 --origin-lon -73.985 \
  --dest-lat 40.712 --dest-lon -74.006 --hour 17

# Sandbox, out of scope: patrol loops
PYTHONPATH=src python scripts/build_patrol_routes.py
```
