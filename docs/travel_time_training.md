# Travel-time models (firetrucks only)

Aligned with slides **Step 4: Travel Time Prediction Model**.

## Important: two different models

Public FDNY CAD **cannot** reach high incident-level R². Destinations are ZIP
centroids and unit GPS at assignment is missing — inferred path length is
essentially uncorrelated with observed travel seconds (CAD holdout R² ≈ 0.2).

| Model | Label | Typical holdout R² | Use for |
|---|---|---|---|
| **CAD context model** | real FDNY CAD travel seconds | ~0.2 | understanding dispatch/context drivers |
| **Route ETA model** | network EMV time on **known OD** | **≥ 0.8** (currently ~0.96) | scoring candidate routes in optimizers |

## Route ETA model (R² ≥ 0.8) — use this for optimizers

1. Sample firetruck origins (FDNY firehouses) → ZIP destinations.
2. Shortest-path **civilian** time on the cached OSM drive graph.
3. Map to EMV travel time with hour congestion, EMV speedup, severity, weather + noise.
4. Train LightGBM on path + context features.

```bash
PYTHONPATH=src python scripts/build_osm_graph.py          # once
PYTHONPATH=src python scripts/build_route_eta_training_set.py --n-pairs 9000
PYTHONPATH=src python scripts/train_route_eta_model.py
```

Outputs:

- `data/processed/route_eta_training.csv`
- `data/processed/models/travel_time_route_eta_lightgbm.joblib`
- `data/processed/models/travel_time_route_eta_metrics.json`

At inference time for a candidate route: compute the same path features (+
`civilian_network_s` from the graph) and predict EMV seconds.

## CAD context model (honest, noise-limited)

Multi-origin distances + weather + OSM aggregates on *inferred* OD; label =
real EMS travel seconds.

```bash
PYTHONPATH=src python scripts/build_travel_time_training_set.py
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set route
PYTHONPATH=src python scripts/analyze_travel_time_model.py --feature-set full
```

| `--feature-set` | Notes |
|---|---|
| `full` | includes `dispatch_wait_seconds` |
| `route` | drops CAD wait / held (still CAD-label limited) |

## Why CAD R² is capped

- Same rounded inferred OD has huge within-trip travel variance.
- Path length vs travel time correlation ≈ 0 (sometimes slightly negative).
- Hitting R² 0.8 on individual CAD rows would require true AVL start/end GPS
  (not public). Ask mentors about a data-use agreement if needed.

## Scope

- Ambulances only (EMS CAD). No fire trucks / firehouses as origins.
