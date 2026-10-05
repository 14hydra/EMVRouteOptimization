# Travel-time model: LFB driving time (London primary)

Model 1 of the CSEF/ISEF study: predict **driving time from a station to an
incident** with **no station IDs**, so the same model can score stations that
do not exist yet. Used by the placement scorer in `docs/firehouse_location.md`.

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.
> **`scripts/train_travel_time_lfb.py` EXISTS**: placement-ready (no station IDs), gradient-boosted
> decision trees (LightGBM) predicting drive time directly, averaged over 4 seeds, features
> `road_km` (shortest legal route, one-way streets respected) + `crow_km` + hour / dow / month /
> rush / night / `busy_flag` / borough; trained on **1,024,833
> engine trips** from LFB mobilisation records, each from the **real station building the engine
> left from** to the incident; chronological train **2021–2024**, test **2025**; held-out MAE
> **64.2 s** on recorded driving time (`TravelTimeSeconds`). Numbers in the Legacy NYC appendix are FDNY.

## Target and trip origin

- **Target:** LFB mobilisation records (London Datastore) give `TravelTimeSeconds`
  (leaving the station → arriving), recorded separately from `TurnoutTimeSeconds`.
  The model is trained on that driving time directly. **EXISTS**
- **Origin:** each mobilisation names the station the engine left from
  (`DeployedFromStation_Name`) and whether it was at its home station or standing in at
  another (`DeployedFromLocation`). The origin is that station's building, from
  OpenStreetMap (`scripts/build_london_firehouses.py` → `data/raw/london/london_firehouses.csv`).
  Engines already out on the road ("On outside duty when mobilised") and trips with a wrong
  address are dropped. **EXISTS**
- **Effect of the real origin:** on the same trips, measuring distance from the real station
  building instead of the centre of its station area lowers held-out 2025 MAE from
  **72.8 s to 67.9 s** (Kolesar: 77.3 s → 71.4 s). Station buildings sit a median 0.49 km from
  their area's centre.
- **Road distance:** `road_km` is the shortest-distance route on London's directed OSM drive
  graph (`emvro.road_distance`; Geofabrik Greater London extract, osmnx "drive" filter, largest
  strongly connected component: 131,549 junctions, 306,131 directed segments, 30,145 one-way).
  One-way streets are a single directed edge, so Dijkstra only uses them in their legal
  direction. Station and incident are snapped to the nearest junction, and the straight-line
  hop to it is added. Engines are assumed to take the shortest-distance route (no route model
  yet; emergency exemptions such as contraflow or bus gates are not modelled). Road distance is
  a median 1.37× straight-line. Adding it lowered 2025 MAE from **67.9 s to 64.2 s**. **EXISTS**
- Placement is scored as `predicted_drive + station's recorded turnout`.

## Design

| Piece | Choice | Status |
|---|---|---|
| Station IDs | **Not features** (avoids memorising stations; allows hypothetical sites) | **EXISTS** (`train_travel_time_lfb.py`) |
| Learner | **LightGBM** (L1 objective) predicting `drive_s` directly; 4 seeds (42–45) averaged; no parametric prior | **EXISTS** (`scripts/train_travel_time_lfb.py`) |
| Kolesar | Piecewise `T(d)`: `a + b√d` short, `c + e·d` long (`emvro.kolesar`), refit on train `crow_km` → `drive_s` | **EXISTS** as a comparison baseline only, not a model input |
| Train / test | Train **2024** (+ **2023** when present in the incident file), test **held-out 2025** (chronological) | **EXISTS** |
| Street features | Width, lanes, A-road / dual-carriageway flags, signals per km, sharp turns, … along origin→destination | **PLANNED** until **OS NGD** (primary; not downloaded) and/or **London drive graph** + route street pipeline (`emvro.route_street_features` is NYC-tuned today) |
| Context features | `crow_km`, hour, dow, month, rush, night, `busy_flag`, borough (label-encoded area context) | **EXISTS** in LFB trainer |
| Extra context | Weather (Open-Meteo, `emvro.weather`), incident type | tooling **EXISTS**; use in London model **PLANNED** |

London records the station every engine left from (unlike FDNY; see
`docs/starting_locations.md`), so each trip's origin is known; see *Target and trip origin*.

## Baselines (all on the same 2025 test calls)

| Baseline | Description | Status |
|---|---|---|
| Median | Predict train median `drive_s` | **EXISTS** (printed by `train_travel_time_lfb.py`) |
| Crow-flies | Fixed 32 km/h on `crow_km` | **EXISTS** (same script) |
| Road | Shortest legal route distance at one fitted speed (35 km/h) | **EXISTS** (80.8 s MAE) |
| Kolesar (road) | Kolesar curve on `road_km` | **EXISTS** (68.7 s MAE; 71.4 s on `crow_km`) |
| **Kolesar** | `fit_kolesar(crow_km, drive_s)` on train | **EXISTS** |
| Linear regression | Same features as the ML model, no trees | **PLANNED** |
| Neural net | Small MLP on same features | **PLANNED** |
| **LightGBM** | Proposed placement model | **EXISTS** (64.2 s MAE on 2025 recorded driving time vs Kolesar on road distance 68.7 s) |

Report MAE, RMSE, R², and for placement-relevant scoring the share of calls predicted
within 6 / 10 min versus observed (`lfb_standards.score_first_engine`).

## Ablations and interpretability (planned)

- Ablations: remove street features / weather / busy one at a time. **PLANNED**
- **SHAP** on the LightGBM model to show which street characteristics matter. **PLANNED — not run;**
  `shap` is not in `requirements.txt`.
- Learning curve on training-set size: `scripts/travel_time_learning_curve.py` **EXISTS** (NYC).
- Honest expectation: in the NYC experiments street features added only ≈0.1 s MAE each
  (see appendix). London may be similar at the incident level; the claim to test is
  about placement ranking, not R².

## Extensions (planned)

- **Quantile LightGBM** (interval predictions for placement risk). **PLANNED**
- **Leave-one-firehouse-out** holdout (generalisation across stations). **PLANNED**

## Scripts that exist and are reusable

| Script / module | Reuse for London |
|---|---|
| `scripts/build_lfb_planner_incidents.py` | Builds planner CSV (coords, attendance, deployed station, busy flag) |
| `emvro/london_eval.py` | Joins stations → start coords, `crow_km`, `drive_s`; feeds the trainer |
| `scripts/train_travel_time_lfb.py` | **Primary London trainer** (LightGBM, placement-safe features) |
| `emvro/kolesar.py` | `fit_kolesar(distance_km, travel_s)` on London trips |
| `scripts/train_travel_time_hybrid.py` | NYC bag4 residual template (full street-feature table) |
| `scripts/build_osm_graph.py` | Build `london_drive.graphml` (OSM fallback) — graph **PLANNED** |
| `scripts/compare_travel_time_models.py`, `compare_hybrid_to_cad.py` | Figure templates for baseline tables (NYC) |

## Run (what works today for London)

```bash
# Inputs in data/raw/london/ (London Datastore, OGL): lfb_mobilisations_2021_2024.csv,
# lfb_mobilisations_2025.csv, lfb_incidents_2018_2023.csv, lfb_incidents_2024_onwards.csv
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py
python scripts/build_london_firehouses.py      # real station locations (OpenStreetMap)
python scripts/build_lfb_travel_training.py    # one row per engine trip; builds the road graph on first run
python scripts/train_travel_time_lfb.py
```

Writes `data/processed/models/travel_time_lfb.joblib` and
`travel_time_lfb.metrics.json` (test-year metrics, Kolesar params, feature list).

---

# Appendix — Legacy NYC / FDNY

> Scaffolding from the original NYC project. Not the CSEF/ISEF headline. Kept because
> the residual-bag4 recipe and the street-feature pipeline were developed here.

Aligned with the original slides **Step 4: Travel Time Prediction Model**.
Firetrucks only (FDNY Fire Incident Dispatch, `8m42-w767`).

## Two different models (don't conflate)

Public FDNY CAD **cannot** reach high incident-level R²: destinations are ZIP/alarm-box
approximations and unit GPS at assignment is missing.

| Model | Label | Typical holdout R² | Use for |
|---|---|---|---|
| **CAD context model** | real FDNY CAD travel seconds | ~0.2–0.4 | understanding dispatch/context drivers |
| **Route ETA model** | network EMV time on **known OD** | ≥ 0.8 (~0.96–0.98) | scoring candidate routes in optimizers |

## Route ETA model (R² ≥ 0.8)

1. Sample firetruck origins (FDNY firehouses) → ZIP destinations.
2. Shortest-path **civilian** time on the cached OSM drive graph.
3. Map to EMV time with hour congestion, EMV speedup, severity, weather + noise.
4. Train LightGBM on path + context features.

```bash
PYTHONPATH=src python scripts/build_osm_graph.py          # once
PYTHONPATH=src python scripts/build_route_eta_training_set.py --n-pairs 9000
PYTHONPATH=src python scripts/train_route_eta_model.py
```

Outputs: `data/processed/route_eta_training.csv`,
`data/processed/models/travel_time_route_eta_lightgbm.joblib`,
`data/processed/models/travel_time_route_eta_metrics.json`.

The high R² is because the label is itself derived from the network path, so it should
**not** be compared with CAD R² or presented as real-world accuracy.

## CAD model v2 — network origins, 1 year of data

Real FDNY CAD label (`incident_travel_tm_seconds_qy`: first unit assigned → first unit on scene).

| Fix | What changed |
|---|---|
| More data | 552k incidents (Jan 2024 – Mar 2025) instead of 3.4k |
| Destinations | Unlisted alarm boxes geocoded from the CAD intersection string on NYC Centerline (`emvro.intersections`); ZIP-centroid destinations 30% → 5% |
| Origins | Street-network free-flow time/km from first-due **engine** and **ladder** houses, nearest 1st/2nd/3rd houses, houses within 3/5 min (`emvro.network_origins`) |
| Unit availability | Past-only incident counts in the first-due area (15/30/60 min) |
| Route street features | Width, lanes, parking lanes, posted speed, bus/bike-lane share, road class, one-ways, signals and sharp turns per km (`emvro.route_street_features`) |
| Traffic | Previous-hour NYC DOT Traffic Speeds (`i4gi-tjb9`) (`emvro.traffic`) |
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

Figures: `data/figures/model_eval_network/`. The earlier CAD model's R² 0.24 used a random
split and leaking QC flags, so it is not comparable.

Street features and traffic add little (~0.1 s each, ~5% of model gain): two incidents at
the same box, hour block and call type still differ by ~75 s, so most remaining error is
not about the road. This is the main caution for the London study.

## Hybrid (network features + residual bag4 + HGB)

Merges the network-origin table with the residual LightGBM seed bag + HistGradientBoosting
blend. This is the template for the London bag-of-4.

```bash
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

Later year-scale hybrid (see `model_implementations_template.md` [TT-04]): MAE ≈ 64.5 s,
R² ≈ 0.42 on 2025-Q1, with the HGB blend weight settling at 0 (bag-only).

Artifacts: `data/processed/models/travel_time_hybrid.joblib`, `data/figures/model_eval_hybrid/`.

The ceiling is the label: the first-arriving unit is often not at its house and turnout
varies. FDNY AVL/GPS would be needed to go much further (see FOIL in
`docs/route_models.md`).

## CAD context model (noise-limited)

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

## Why FDNY CAD R² is capped

- Same rounded inferred OD has huge within-trip travel variance.
- Path length vs travel time correlation ≈ 0 (sometimes slightly negative).
- Hitting R² 0.8 on individual CAD rows would require true AVL start/end GPS (not public).
