# Travel-time model: LFB driving time

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
> **64.2 s** on recorded driving time (`TravelTimeSeconds`).

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
| Kolesar | Piecewise `T(d)`: `a + b√d` short, `c + e·d` long (`emvro.kolesar`), refit on train → `drive_s` | **EXISTS** as a comparison baseline only, not a model input |
| Train / test | Train **2021–2024**, test **held-out 2025** (chronological) | **EXISTS** |
| Distance features | `road_km` (shortest legal route on the directed drive graph), `crow_km` | **EXISTS** |
| Street features | Road class shares, one-way / 20 mph / bus-lane shares, lanes, signals, traffic calming, crossings, junctions, turns, roundabouts along each route (`emvro.route_features`) | **EXISTS** in the training set; tested in `scripts/analyze_london.py` (MAE 64.2 → 61.7 s, `docs/london_eda.md`); not yet in `train_travel_time_lfb.py`. OS NGD width **PLANNED** |
| Context features | hour, dow, month, rush, night, `busy_flag`, borough (label-encoded area context) | **EXISTS** in LFB trainer |
| Extra context | Weather (Open-Meteo), incident type | **PLANNED** |

London records the station every engine left from, so each trip's origin is known; see
*Target and trip origin*.

## Baselines (all on the same 2025 test calls)

| Baseline | Description | Status |
|---|---|---|
| Median | Predict train median `drive_s` | **EXISTS** (printed by `train_travel_time_lfb.py`) |
| Crow-flies | Fixed 32 km/h on `crow_km` | **EXISTS** (same script) |
| Road | Shortest legal route distance at one fitted speed (35 km/h) | **EXISTS** (80.8 s MAE) |
| Kolesar (road) | Kolesar curve on `road_km` | **EXISTS** (68.7 s MAE; 71.4 s on `crow_km`) |
| Kolesar (crow) | `fit_kolesar(crow_km, drive_s)` on train | **EXISTS** (71.4 s MAE) |
| Linear regression | Same features as the ML model, no trees | **PLANNED** |
| Neural net | Small MLP on same features | **PLANNED** |
| **LightGBM** | Proposed placement model | **EXISTS** (64.2 s MAE on 2025 recorded driving time vs Kolesar on road distance 68.7 s) |

Report MAE, RMSE, R², and for placement-relevant scoring the share of calls predicted
within 6 / 10 min versus observed (`lfb_standards.score_first_engine`).

## Ablations and interpretability (planned)

- Ablations: remove street features / weather / busy one at a time. **PLANNED**
- **SHAP** on the LightGBM model to show which street characteristics matter. **PLANNED — not run;**
  `shap` is not in `requirements.txt`.
- Honest expectation: street features may add little at the incident level; the claim to
  test is about placement ranking, not R².

## Extensions (planned)

- **Quantile LightGBM** (interval predictions for placement risk). **PLANNED**
- **Leave-one-firehouse-out** holdout (generalisation across stations). **PLANNED**

## Code

| Script / module | Role |
|---|---|
| `scripts/build_london_firehouses.py` | Real station locations (OpenStreetMap) |
| `scripts/build_lfb_travel_training.py` | One row per engine trip: origin, destination, `crow_km`, `road_km`, `drive_s` |
| `emvro/road_distance.py` | Directed London drive graph; shortest legal road distance |
| `scripts/train_travel_time_lfb.py` | LightGBM trainer + baselines |
| `emvro/kolesar.py` | `fit_kolesar(distance_km, travel_s)` |

## Run

```bash
python scripts/download_datasets.py            # LFB incidents + mobilisations, London OSM extract
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py
python scripts/build_london_firehouses.py      # real station locations (OpenStreetMap)
python scripts/build_lfb_travel_training.py    # one row per engine trip; builds the road graph on first run
python scripts/train_travel_time_lfb.py
```

Writes `data/processed/models/travel_time_lfb.joblib` and
`travel_time_lfb.metrics.json` (test-year metrics, Kolesar params, feature list).
