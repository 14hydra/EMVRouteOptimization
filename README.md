# EMV Route Optimization (ASI 2026–27)

Emergency vehicle (EMV) travel-time prediction and route optimization for NYC — Ricky Zhao & Sarp Akalin (ASI).

## The starting-location problem

Public **EMS Incident Dispatch Data** (`76xm-jjuj`) has:

- approximate **incident** geography (borough, ZIP, dispatch area)
- response / travel times

It does **not** publish the GPS of the unit at assignment time (HIPAA / privacy). Without a start point you cannot reconstruct the true dispatched route.

### Fix in this repo

**Scope: ambulances only** (EMS CAD). Fire trucks / firehouses are out of scope.

We reconstruct usable OD pairs by:

1. Treating each incident ZIP’s **MODZCTA centroid** as the destination.
2. Mapping `incident_dispatch_area` (e.g. `K6`, `B1`) to a borough sector.
3. Choosing the nearest **EMS/ambulance station** or **hospital bay** (HOSPITAL / ACUTE CARE HOSPITAL from City Facilities — covers NYC H+H and voluntary hospitals) in that borough.
4. In **hybrid** mode, using a **synthetic CSL** (FDNY alarm-box intersection in a high-volume ZIP) when it is closer than the best static depot — a proxy for an already-on-the-road unit.
5. Flagging each row with `start_source` / `depot_layer` / `start_mode`.

For the **travel-time model**, we do not commit to one true start: we feed
`station_km` / `hospital_km` / `csl_km` together so LightGBM can learn times
under origin uncertainty (see `docs/travel_time_training.md`).

## Chosen datasets

| Role | Source | Open Data ID |
|------|--------|--------------|
| Incidents + travel times | NYC EMS Incident Dispatch Data | `76xm-jjuj` |
| Preferred EMV starts | City Facilities — EMS/ambulance stations | `ji82-xba5` |
| Hospital bay starts | City Facilities — HOSPITAL / ACUTE CARE HOSPITAL | `ji82-xba5` |
| CSL intersection proxies | In-Service Alarm Box Locations | `v57i-gtxb` |
| Incident ZIP geometry | Modified Zip Code Tabulation Areas (MODZCTA) | `pri4-ifjk` |
| Street network / widths | NYC LION | `2v4z-66xt` |
| Street attributes | OpenStreetMap (NYC extract) | OSM |
| Hourly weather | Open-Meteo archive (NYC) | no key; cached under `data/processed/weather_hourly_nyc.csv` |
| Congestion prior | Hour-of-day BPR-style multiplier (TomTom/GMaps traffic optional later) | in `emvro.routing.conditions` |

## Quick start

```bash
cd EMVRouteOptimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download samples + build inferred OD pairs
python scripts/download_datasets.py --limit 5000
python scripts/build_od_pairs.py --start-mode hybrid

# Optional: Google Maps civilian control → label station-plausible vs on-road
export GOOGLE_MAPS_API_KEY=your_key_here
PYTHONPATH=src python scripts/classify_origins_with_gmaps.py --limit 200
PYTHONPATH=src python scripts/visualize_data.py

# Travel-time model (slides Step 4) — ambulances only
PYTHONPATH=src python scripts/build_osm_graph.py   # once
# Enrich training with Google Maps civilian ETAs, then retrain
PYTHONPATH=src python scripts/enrich_training_with_gmaps.py --limit 600 --remap-route-eta-labels
PYTHONPATH=src python scripts/train_route_eta_model.py
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full
# CAD context model (real EMS labels; R² noise-limited ≈ 0.2)
PYTHONPATH=src python scripts/build_travel_time_training_set.py
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full
PYTHONPATH=src python scripts/analyze_travel_time_model.py --feature-set full
```

See `docs/travel_time_training.md` and `docs/gmaps_control.md`.
Route-ETA eval figures (R²≈0.96): `data/figures/model_eval_route_eta/` (also mirrored at `data/figures/model_eval/`).

## Route optimization models (slides Step 3)

```bash
PYTHONPATH=src python scripts/run_route_models.py
```

| Model | Code |
|---|---|
| MIPSSTW + MCS | `src/emvro/routing/mipsstw_mcs.py` |
| Composite DRL | `src/emvro/routing/composite_drl.py` |
| GBDT router | `src/emvro/routing/gbdt_router.py` |

See `docs/route_models.md`. Figures: `data/figures/route_models/`.

**Master map** (NYC + SF: dataset / examples / ROW wins):

```bash
PYTHONPATH=src python scripts/build_master_map.py
# open data/figures/master_map.html
# City buttons: NYC | SF | Both · Case buttons: Examples | Dataset | ROW wins
```

## Patrol posts & loops (pre-positioning)

The inferred-start work answers *where a unit probably started*. This answers the mentor
question *where should units wait so the next call is closer* — historical incident density
becomes patrol posts plus one short loop per unit, drawn from the **same** inferred layers
(EMS stations, hospital bays, synthetic CSLs) used as OD starts.

```bash
PYTHONPATH=src python scripts/build_patrol_routes.py                       # NYC + OSM graph (~40 s)
PYTHONPATH=src python scripts/build_patrol_routes.py --objective coverage  # max share within 8 min
PYTHONPATH=src python scripts/build_patrol_routes.py --no-graph            # crow-flies only (~2 s)
PYTHONPATH=src python scripts/build_patrol_routes.py --demo                # synthetic demand, no data needed
# open data/figures/patrol/patrol_map.html
```

| Objective | Meaning |
|---|---|
| `response_time` (default) | p-median: minimize demand-weighted expected EMV travel time |
| `coverage` | maximal covering: maximize demand reachable within `--threshold-min` |

Code: `src/emvro/patrol.py`. Outputs in `data/figures/patrol/`: `patrol_posts.csv`,
`patrol_segments.csv`, `patrol_demand_cells.csv`, `patrol_map.html`, `summary.json`.
See `docs/patrol_routes.md`.

**Multi-city training data** (SF EMS CAD + NYC):

```bash
PYTHONPATH=src python scripts/ingest_sf_ems.py --limit 4000
PYTHONPATH=src python scripts/build_multicity_route_eta.py
PYTHONPATH=src python scripts/train_route_eta_model.py \
  --data data/processed/route_eta_multicity_training.csv --target-r2 0.9
```

See `docs/multi_city_datasets.md`.

Traffic × weather condition grid (ASI eval — clear / rush / rain / snow / night):

```bash
PYTHONPATH=src python scripts/run_condition_scenarios.py --with-open-meteo
```

Figures: `data/figures/route_models/conditions/` (civilian vs EMV ROW payoff by regime).
Conditions map with Google-untakeable corridors: `data/figures/route_models/route_conditions_map.html`.

## Layout

```
data/raw/          downloaded Open Data extracts
data/processed/    cleaned tables + inferred starts
data/samples/      small CSVs safe to share / upload to Gemini
src/emvro/         Python package (download, geometry, depot inference)
scripts/           CLI entrypoints
docs/              dataset notes + methodology
```

## Bridge / Gemini ASI chat

This project is coordinated with the Gemini **ASI** chat via `cursor-gemini-bridge`.
Sample CSVs under `data/samples/` are meant for upload into that chat.
