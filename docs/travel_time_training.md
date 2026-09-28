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

## CAD model v2 — network origins, 1 year of data (recommended)

Real FDNY CAD label (`incident_travel_tm_seconds_qy`: first unit assigned →
first unit on scene). Replaces the single guessed start with label-free
descriptions of every plausible origin:

| Fix | What changed |
|---|---|
| More data | 552k incidents (Jan 2024 – Mar 2025) instead of 3.4k |
| Destinations | Unlisted alarm boxes geocoded from the CAD intersection string on NYC Centerline (`emvro.intersections`, median error 1 m on known boxes). ZIP-centroid destinations 30% → 5% |
| Origins | Street-network free-flow time/km from the first-due **engine** and **ladder** houses, nearest 1st/2nd/3rd houses, houses within 3/5 min (`emvro.network_origins`, one multi-source Dijkstra) |
| Unit availability | Past-only incident counts in the first-due area (15/30/60 min) |
| Route street features | Along the fastest path from the first-due house: width, lanes, parking lanes, posted speed (NYC Centerline), bus/bike-lane share (DOT + OSM), road class, one-ways, signals and sharp turns per km (`emvro.route_street_features`) |
| Traffic | Previous-hour congestion from NYC DOT Traffic Speeds (`i4gi-tjb9`): citywide, borough and nearest monitored link, as speed ÷ free-flow speed (`emvro.traffic`) |
| Location priors | Out-of-fold / past-only mean travel time per box, engine, ZIP |
| Leak removed | `qc_speed_flag` / `qc_keep` are computed from the label; no longer features |
| Honest test | Time split: train 2024, test Jan–Mar 2025 |

```bash
PYTHONPATH=src python scripts/download_datasets.py --out data/raw/big \
    --start 2024-01-01T00:00:00 --end 2025-03-31T23:59:59 --limit 2000000   # ~10 min
PYTHONPATH=src python -c "from emvro.traffic import download_hourly_speeds as d; d('2023-12-31','2025-03-31T23:59:59','data/raw/big')"   # ~5 min
PYTHONPATH=src python scripts/build_osm_graph.py --rich-tags --out data/raw/nyc_drive_rich.graphml
PYTHONPATH=src python scripts/analyze_street_characteristics.py --no-plots   # builds street_edge_table.pkl
PYTHONPATH=src python scripts/build_travel_time_network_dataset.py --raw data/raw/big
PYTHONPATH=src python scripts/train_travel_time_network.py
PYTHONPATH=src python scripts/travel_time_learning_curve.py
```

Test results (107,232 incidents, Jan–Mar 2025):

| Model | MAE | R² |
|---|---|---|
| Predict median | 96.7 s | 0.00 |
| Previous design, old data size (2,692 rows) | 80.4 s | 0.18 |
| Previous design, retrained on 2024 | 71.4 s | 0.31 |
| + geocoded destinations | 69.1 s | 0.33 |
| + network origins | 68.0 s | 0.34 |
| + location priors | 67.9 s | 0.34 |
| + unit-availability load | 67.8 s | 0.34 |
| + route street features | 67.7 s | 0.34 |
| + traffic congestion (final) | **67.7 s** | **0.34** |

Figures: `data/figures/model_eval_network/`. The earlier CAD model's reported
R² 0.24 used a random split and the leaking QC flags, so it is not comparable.

Street features and traffic add little (about 0.1 s each, ~5% of model gain):
two incidents at the same box, hour block and call type still differ by ~75 s,
so most remaining error is not about the road.

## Hybrid (network features + residual bag4 + HGB)

Merges the network-origin table with the residual LightGBM seed bag +
HistGradientBoosting blend used on the smaller CAD training set.

```bash
# On the 60k extract in data/raw (or --raw data/raw/big for the year table):
PYTHONPATH=src python scripts/build_travel_time_network_dataset.py --raw data/raw
PYTHONPATH=src python scripts/train_travel_time_hybrid.py
```

Chronological holdout on the Dec-2024 60k extract (~40k usable rows):

| Model | MAE | R² |
|---|---|---|
| Median | 92.3 s | ~0 |
| Single residual LGBM | 68.6 s | 0.37 |
| Bag4 | 68.4 s | 0.37 |
| **Bag4 + HGB (hybrid)** | **68.0 s** | **0.38** |

Artifacts: `data/processed/models/travel_time_hybrid.joblib`,
`data/figures/model_eval_hybrid/`. For a year-scale time split, build
`data/raw/big` as above and point `--data` at that parquet.


The ceiling is the label: the first-arriving unit is often not at its house
(returning, relocated, already out), and turnout time varies. Unit AVL/GPS
would be needed to go much further.

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

- Firetrucks (FDNY Fire Incident Dispatch). Legacy EMS layers are optional.
