# Route models (Model 2 of the London study)

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.
> **Segment-speed route scaffold EXISTS** (`src/emvro/segment_speeds.py`,
> `scripts/fit_segment_speeds.py`). **London drive graph** (`london_drive.graphml`) and
> **FOI GPS traces** are **PLANNED**. NYC router demos are scaffolding only (see
> [NYC router demos](#legacy-nyc-router-demos-scaffolding-only)).

## Role in the study

Model 2 answers: *given an origin station and an incident, which path do engines take
and how long does it take?* — so that placement can be scored on a **directed street
graph** (Model 3) with edge speeds that reflect real fire-engine driving, not posted
limits or civilian GPS speeds.

The main London scorer is the street-aware travel-time model
(`docs/travel_time_training.md`). The route model is the network scorer in the
multi-scorer comparison (`docs/firehouse_location.md`).

## Plan (PLANNED)

1. **Learned segment speeds.** Predict an EMV speed (or seconds) per directed street
   segment from segment characteristics (road class, speed limit, width, lanes, junction
   density, one-way, hour, weather), instead of a fixed free-flow or civilian speed.
   - Street data: **OS NGD** (primary goal; not downloaded yet) → **OSM fallback**
     (`scripts/build_osm_graph.py`, `emvro.street_features`).
   - Training signal: LFB attendance is whole-route, not per-segment, so segment speeds
     must be fit by path-level regression (sum of segment times ≈ observed drive), with
     turnout removed using the per-station turnout once mobilisation data is available.
2. **Dijkstra under EMV speeds.** Run shortest-time paths on the directed graph with the
   learned speeds; this produces station→cell travel-time matrices for the planner.
3. **GPS traces via FOI.** File an early Freedom of Information request with
   the London Fire Brigade for engine GPS/AVL logs (and optionally compare with
   published London Ambulance blue-light findings). If needed for NYC transfer
   experiments, FOIL is the New York analogue. Request traces to
   calibrate or validate segment speeds with real driven paths. **PLANNED / not obtained.**
4. **Fallback claim.** If GPS is not obtained, the claim is narrowed to **travel-time
   accuracy only**: the route/graph model is validated by how well its predicted
   station→incident times match held-out 2025 LFB attendance (minus turnout), **not** by
   route fidelity. We do not claim our paths are the paths engines actually drove.

Validation shared with the rest of the study: 2014 ten-station closures,
leave-one-firehouse-out, multi-scorer.

## What exists that this builds on

| Piece | Path | Notes |
|---|---|---|
| Edge-cost GBDT + Dijkstra | `src/emvro/routing/gbdt_router.py` | Predicts edge EMV costs from `length_m`, `speed_kph`, road-class flags, bus lane, etc. Closest existing thing to "learned segment speeds"; its labels are graph-derived `emv_s` × hour congestion prior (synthetic, NYC), **not** observed speeds and **not** LFB |
| Graph + Dijkstra utilities | `src/emvro/routing/graph.py` | |
| Firetruck edge weights | `src/emvro/routing/firetruck.py` | Hand-set heuristics (contraflow, residential penalty); used by the planner when a graph is supplied |
| OSM graph build | `scripts/build_osm_graph.py` | Works for any bbox; no London graph built yet |
| Segment-speed fit scaffold | `src/emvro/segment_speeds.py`, `scripts/fit_segment_speeds.py` | Zhan-style iterative path fit; **EXISTS** as API + demo/selftest; not trained on LFB |
| City registry | `src/emvro/cities.py` | `london` expects `data/raw/london/london_drive.graphml` |

Nothing here has been fit on London incident data, and there is no OS NGD loader.

---

## Legacy NYC router demos (scaffolding only)

Three demo routers from the original project slides (Step 3), plus civilian controls.
These are **not** part of the CSEF/ISEF headline claim. Their "EMV advantage" comes from
hand-written ROW / contraflow / bus-lane rules for NYC and says nothing about London.

| Model | Module | Idea (first draft) |
|---|---|---|
| **MIPSSTW + MCS** | `emvro.routing.mipsstw_mcs` | Minimize EMV time with a semi-soft time-window penalty; Modified Cuckoo Search |
| **Composite DRL** | `emvro.routing.composite_drl` | Tabular Q-learning with composite reward |
| **GBDT router** | `emvro.routing.gbdt_router` | LightGBM predicts edge EMV costs → Dijkstra (the one worth carrying forward) |
| Control | `emvro.routing.graph` | Shortest distance / civilian congested time |

Route scoring in these demos uses the NYC route-ETA model (`docs/travel_time_training.md`,
legacy appendix), whose labels are network-derived, so its high R² is not real-world accuracy.

### Run the demo (NYC)

```bash
# needs data/raw/nyc_drive.graphml
PYTHONPATH=src python scripts/run_route_models.py \
  --origin-lat 40.758 --origin-lon -73.985 \
  --dest-lat 40.712 --dest-lon -74.006 \
  --hour 17
```

Output: `data/processed/route_models_demo.json`

### Visualizations (NYC)

```bash
PYTHONPATH=src python scripts/visualize_route_models.py
# faster (charts only):  PYTHONPATH=src python scripts/visualize_route_models.py --skip-map
```

Figures in `data/figures/route_models/`: `00_dashboard.png`, `01_travel_time_comparison.png`,
`02_distance_comparison.png`, `03_main_models_focus.png`, `04_time_vs_distance.png`,
`05_mcs_fitness.png`, `06_drl_learning_curve.png`, `route_models_map.html`,
`route_conditions_map.html` (time slider + weather), `row_wins_map.html` (verified
EMV-vs-civilian ROW wins; `row_wins.csv`), `multi_route_times.csv`.

### Traffic & weather conditions (NYC demo)

| Regime | How it enters the graph |
|---|---|
| Traffic amount | Hour congestion prior (`clear_offpeak`, `clear_rush`, `night_clear`); TomTom live flow could replace it |
| Weather | Open-Meteo factors from presets or `weather_hourly_nyc.csv` |
| Bus lanes / ROW | EMV discounts on primary/trunk + `busway` tags; discount grows with congestion |
| Contraflow (EMV-only) | Short reverse edges on primary/trunk one-ways (≤120 m; no motorways), with a cost penalty |

```bash
PYTHONPATH=src python scripts/run_condition_scenarios.py
PYTHONPATH=src python scripts/run_condition_scenarios.py --with-open-meteo
```

Figures: `data/figures/route_models/conditions/`. Presets:
`emvro.routing.conditions.CONDITION_PRESETS`.

### Old upgrade path (NYC demos, deprioritised)

1. MIPSSTW: true MILP on a corridor subgraph (PuLP/CBC).
2. Composite DRL: DQN / graph neural policy.
3. GBDT: train edge costs on **real** labeled paths — this is what the London plan above supersedes.
4. Batch-eval on held-out OD samples.
