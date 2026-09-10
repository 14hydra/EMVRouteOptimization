# Training the travel-time model without known ambulance GPS origins

Aligned with slides **Step 4: Travel Time Prediction Model**
(segment-aggregated LightGBM / XGBoost; inputs = route + traffic/context;
validate on real EMS travel times).

## Scope

- **Ambulances only** — use NYC **EMS Incident Dispatch Data** (not Fire Incident Dispatch).
- Do **not** use FDNY firehouses as staging origins for this model.

## The trick

Public CAD gives a strong **label**:

`y = incident_travel_tm_seconds_qy`  (assignment → first on-scene)

but not the unit’s GPS at assignment. So we **do not claim** to reconstruct the
true historical path. We train a model that predicts ambulance travel time from
**observable context + multi-hypothesis origins**.

## Feature design

| Feature group | Examples | Why |
|---|---|---|
| Destination | ZIP centroid lat/lon, borough, ZIP demand volume | End of trip (approximate) |
| Temporal | hour, dow, rush/night/weekend, month | Congestion regimes |
| Call context | severity, call type, held flag, dispatch wait | Urgency / system load |
| Multi-origin geometry | `station_km`, `hospital_km`, `csl_km`, spread | Origin uncertainty made explicit |
| Primary inferred OD | crow-flies km, bearing, primary layer | One working geometry for routing |
| Weather (Open-Meteo) | temp, humidity, wind, precip, visibility | Weather regimes |
| OSM path aggregates | `osm_path_km`, circuity, speed, highway mix | Street-network summary (LION swap-in later) |

## Two model variants

| `--feature-set` | Use for |
|---|---|
| `full` | Best offline fit; includes `dispatch_wait_seconds` |
| `route` | Route scoring / optimizers — drops CAD wait + held (not available as street physics) |

## Pipeline

```bash
# Larger full-day EMS pull (overnight hours included)
PYTHONPATH=src python scripts/download_datasets.py --ems-only \
  --start '2024-06-01T00:00:00' --end '2024-06-05T23:59:59' --limit 50000

python scripts/build_od_pairs.py --start-mode hybrid

# One-time OSM drive graph (~130MB)
PYTHONPATH=src python scripts/build_osm_graph.py

PYTHONPATH=src python scripts/build_travel_time_training_set.py

# macOS: brew install libomp && export DYLD_LIBRARY_PATH=...
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set route

PYTHONPATH=src python scripts/analyze_travel_time_model.py --feature-set full
PYTHONPATH=src python scripts/analyze_travel_time_model.py --feature-set route
```

Outputs:

- `data/processed/travel_time_training.csv`
- `data/processed/models/travel_time_lightgbm_{full,route}.joblib`
- `data/processed/models/travel_time_metrics_{full,route}.json`
- `data/figures/model_eval_{full,route}/`

## Honest limits

- Destinations are **ZIP centroids**, not exact blocks.
- Starts are **inferred**; multi-origin features reduce that risk.
- OSM path is along the **inferred** OD, not the true AVL path.
- Fire apparatus are out of scope.
