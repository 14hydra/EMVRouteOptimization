# Fire Station Placement — London Fire Brigade (CSEF / ISEF)

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

**Not our claim:** that we have beaten LFB's real operational planning (see
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

LFB mobilisation records give driving time (`TravelTimeSeconds`) and turnout
(`TurnoutTimeSeconds`) separately, so the travel-time model is trained on driving time
alone. The planner still uses `DEFAULT_TURNOUT_S` (60 s) for hypothetical sites.

## Three models

| # | Model | Role | Status |
|---|---|---|---|
| 1 | **Travel-time LightGBM, no station IDs** — gradient-boosted trees, 4-seed average | Predict driving time to any point from any site | **EXISTS** (`scripts/train_travel_time_lfb.py`): real station origins, shortest legal road distance; 2025 MAE 64.2 s. Street characteristics **PLANNED** |
| 2 | **Route model** with learned segment speeds | Dijkstra under learned edge speeds; FOI GPS when available | Shortest-distance routing on the directed London graph **EXISTS** (`emvro.road_distance`); learned speeds scaffold **EXISTS** (`segment_speeds.py`); FOI **PLANNED** |
| 3 | **Placement** | redesign / replace / expand station sets | All three modes **EXISTS** (Kolesar / crow scorers); scoring with model 1 **PLANNED** |

Details: [`docs/travel_time_training.md`](docs/travel_time_training.md),
[`docs/firehouse_location.md`](docs/firehouse_location.md).

## Validation plan

1. **2014 ten-station closures** — on 9 Jan 2014 LFB closed Belsize, Bow, Clerkenwell,
   Downham, Kingsland, Knightsbridge, Silvertown, Southwark, Westminster, Woolwich.
   Counterfactual + ranking scripts **EXIST** on today's network with approximate
   closed-station coords; the true pre-2014 train → post-2014 test needs the 2009–2017
   LFB extracts (**PLANNED** ingest).
2. **Leave-one-firehouse-out** — hide one station's calls and predict its area from the rest. **PLANNED**
3. **Multi-scorer** — score layouts under Kolesar / crow / ML; planner supports
   `--scorer kolesar|crow` today (**EXISTS**); full cross-table comparison script **PLANNED**

Temporal split: the travel-time model trains on **2021–2024** and tests on **2025**.
The planner takes demand and evaluation years from `--demand-years` / `--eval-years`.

## Datasets

All raw files live in `data/raw/london/` (not in git). `scripts/download_datasets.py`
fetches them; see [`docs/datasets.md`](docs/datasets.md).

| Role | Source | Status |
|---|---|---|
| Incidents (location, station ground, attendance) | **LFB incident records** 2018–2023 and 2024 onwards (London Datastore) | **EXISTS** |
| Engine trips (deployed-from station, turnout, driving time) | **LFB mobilisation records** 2021–2024 and 2025 onwards (London Datastore) | **EXISTS** |
| Station locations | OpenStreetMap fire stations, name-matched to LFB (`scripts/build_london_firehouses.py`) | **EXISTS** |
| Street network | OpenStreetMap, Geofabrik Greater London extract → directed drive graph (`emvro.road_distance`) | **EXISTS** |
| Street network (detail) | **Ordnance Survey NGD** (width, speed limit, class) | **PLANNED** |
| 2014 closures | Hard-coded list in `emvro.london_closures_2014` | **EXISTS** |

## Code

| Piece | Path |
|---|---|
| London data download | `scripts/download_datasets.py` |
| Station locations (OSM) | `scripts/build_london_firehouses.py` |
| Engine-trip training set (real origin, road distance) | `scripts/build_lfb_travel_training.py` |
| Shortest legal road distance (one-way streets respected) | `src/emvro/road_distance.py` |
| LightGBM travel-time trainer | `scripts/train_travel_time_lfb.py` |
| Kolesar piecewise `T(d)` (fit + predict) | `src/emvro/kolesar.py` |
| LFB planner CSV builder (with `busy_flag`) | `scripts/build_lfb_planner_incidents.py` |
| Firehouse planner (redesign, replace, expand) | `src/emvro/firehouse_location.py`, `src/emvro/facility.py`, `scripts/plan_firehouse_locations.py` |
| LFB 6 / 10-min standards scoring, turnout + drive | `src/emvro/lfb_standards.py` |
| Busy-engine rates (`deployed ≠ ground`) | `src/emvro/busy_engines.py` |
| 2014 closure list + counterfactual + ranking | `src/emvro/london_closures_2014.py`, `scripts/validate_london_2014_closures.py`, `scripts/rank_closed_stations_2014.py` |
| Segment-speed route scaffold | `src/emvro/segment_speeds.py`, `scripts/fit_segment_speeds.py` |
| Incident density map | `scripts/visualize_data.py` |
| Street features along each route | `src/emvro/route_features.py` |
| Initial data analysis + street feature impact | `scripts/analyze_london.py` → `data/figures/london_eda/` |

Still **PLANNED**: street characteristics in the main trainer (measured in `docs/london_eda.md`), OS NGD download,
SHAP/ablations, leave-one-firehouse-out CV, pre-2014 LFB ingest, planner scoring with
the LightGBM model.

## Quick start

```bash
cd FireStationPlacement
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
brew install libomp            # macOS: LightGBM needs it

# 1. Download LFB incidents + mobilisations and the London OSM extract (~600 MB)
python scripts/download_datasets.py

# 2. Derived tables
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py
python scripts/build_london_firehouses.py
python scripts/build_lfb_travel_training.py      # builds the road graph on first run

# 3. Travel-time model
python scripts/train_travel_time_lfb.py

# 4. Incident density map -> data/figures/london_incident_density.html
PYTHONPATH=src python scripts/visualize_data.py

# 4b. Initial analysis + street feature impact -> data/figures/london_eda/ (~15 min)
python scripts/analyze_london.py

# 5. Placement planner
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode redesign --demand-years 2024 --eval-years 2025
PYTHONPATH=src python scripts/plan_firehouse_locations.py --city london --mode replace --replace 10
PYTHONPATH=src python scripts/plan_firehouse_locations.py --city london --mode expand --add-stations 2

# 6. 2014 closure scaffolds
PYTHONPATH=src python scripts/validate_london_2014_closures.py
PYTHONPATH=src python scripts/rank_closed_stations_2014.py

# Synthetic smoke test (no real data)
PYTHONPATH=src python scripts/plan_firehouse_locations.py --demo
```

Defaults: `--city london`, `--scorer kolesar`, `--threshold-min 10`, LFB
`score_first_engine` in the summary, drive + turnout attendance scoring. Modelled cover
rates are optimistic vs observed attendance (nearest engine always free unless `--use-busy`).

## Docs

| Doc | Content |
|---|---|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Research plan, prior work, validation, references |
| [`docs/travel_time_training.md`](docs/travel_time_training.md) | LFB driving-time model; trip origin; road distance; baselines |
| [`docs/firehouse_location.md`](docs/firehouse_location.md) | Placement: modes, KPIs, scorers, closures, busy engines |
| [`docs/datasets.md`](docs/datasets.md) | London data sources and how they are joined |
| [`docs/london_eda.md`](docs/london_eda.md) | Initial analysis; which street features affect driving time |
| [`docs/model_implementations_template.md`](docs/model_implementations_template.md) | Catalog of model implementations + blank entry |

## Limits and honesty

- Station sites in `london_firehouses.csv` come from OpenStreetMap building footprints,
  name-matched to LFB stations (8 pinned by hand, see the script).
- Engines are assumed to take the shortest-distance legal route; emergency exemptions
  (contraflow, bus gates) are not modelled.
- `busy_flag` (engine deployed from a station other than the incident's ground station)
  is a proxy for "home engine unavailable".
- Candidate sites are discrete; no land cost, staffing, or planning constraints.
- Street-feature gains over a good distance model may be small. The claim is about
  **placement ranking**, and we will report that honestly whichever way it comes out.
