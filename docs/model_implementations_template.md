# Model implementations & changes — template

Living catalog of **major model work** on EMV Route Optimization, plus a
blank form to copy for future changes.

**Project frame (CSEF/ISEF):** London Fire Brigade station placement with a
street-characteristic travel-time model (gradient-boosted decision trees, LightGBM),
scored on held-out 2025 LFB calls. London entries are **primary**. NYC/FDNY entries
below are **legacy scaffolding**. Patrol is **out of the main story**.

**Status words:** `active` = exists and used · `planned` = in the plan, not built or run ·
`legacy` = NYC scaffolding · `out-of-story` = kept as sandbox.

**How to use**

1. Copy the [Blank entry](#blank-entry) block into [Changelog](#changelog--filled-entries).
2. Fill every field you can; leave unknowns as `TBD`.
3. Link scripts, modules, metrics JSON, and figures.
4. Keep CAD travel-time and route-ETA entries **separate** — they answer different questions.
5. Do not log a `planned` item as done. No London model metrics exist yet.

**Related docs:** [`travel_time_training.md`](travel_time_training.md) ·
[`route_models.md`](route_models.md) · [`starting_locations.md`](starting_locations.md) ·
[`patrol_routes.md`](patrol_routes.md) · [`gmaps_control.md`](gmaps_control.md) ·
[`multi_city_datasets.md`](multi_city_datasets.md) ·
[`firehouse_location.md`](firehouse_location.md)

---

## Blank entry

```markdown
### [ID] Short title
- **Date:** YYYY-MM-DD
- **Status:** planned | draft | active | superseded | discarded | legacy | out-of-story
- **Family:** LFB travel-time | placement | standards/scoring | CAD travel-time (legacy) | route-ETA (legacy) | path router (legacy) | patrol (out-of-story) | data/infra | other
- **Problem:** one sentence
- **Label / target:** what is predicted
- **Holdout:** split rule (chronological / random / OD) + dates / n
- **Features (added/changed):** bullet list
- **Model / algorithm:** e.g. LightGBM bag4 residual + HGB
- **Train script:** `scripts/...`
- **Build / data script:** `scripts/...`
- **Core modules:** `src/emvro/...`
- **Artifacts:** joblib / metrics json / figures dir
- **Headline metrics:** MAE=… · R²=… · (slice: …)
- **vs previous:** ΔMAE / ΔR² vs named baseline
- **What worked:** …
- **What failed / discarded:** …
- **Ceiling / honesty notes:** …
- **Next lever:** …
```

---

## Map of model families

| Family | Question it answers | Honesty constraint |
|---|---|---|
| **LFB travel-time** (primary) | Predict drive (+turnout) from any station to any London incident, no station IDs | Public CSV = attendance only; no London model has been trained yet |
| **Placement** (primary) | Where should London stations be (redesign / replace / expand)? | Scored vs LFB 6 / 10-min standards on held-out 2025; discrete candidates; travel-time scorers must add turnout |
| **Standards / busy / closures** (primary) | KPIs, engine unavailability, 2014 natural experiment | `busy_flag` is a proxy; closures list is fixed at 10 names |
| **CAD travel-time** (legacy NYC) | Predict FDNY assign→on-scene seconds from public CAD | No unit GPS at assignment; R² capped (~0.3–0.4) |
| **Route ETA** (legacy NYC) | Predict EMV seconds on a **known** OD / path | Synthetic or measured OD; R² can be ≥0.9 |
| **Path routers** (legacy demos) | Given O→D, which path? | Uses graph + EMV edge costs; scored by route-ETA |
| **Patrol** (out of story) | Where to stage units *before* a call | Prescriptive posts/loops, not dispatch; sandbox only |
| **Data / origins** | Reconstruct usable starts & destinations | NYC only: first-due inference ≠ true AVL. London records `DeployedFromStation` |

Do **not** compare route-ETA R² (~0.96) to CAD hybrid R² (~0.42) as if they were the same task.

---

## Changelog — filled entries

## London / LFB entries (primary)

### [LD-01] Kolesar piecewise travel-time module
- **Date:** 2026-10 (module present in repo)
- **Status:** active (module) · London fit/evaluation **planned**
- **Family:** LFB travel-time (prior)
- **Problem:** Distance-only parametric prior and baseline: `T = a + b√d` (short), `c + e·d` (long).
- **Label / target:** travel seconds from distance (km)
- **Holdout:** **planned** — fit on 2023–24, test on 2025 LFB
- **Features:** distance only
- **Model / algorithm:** Kolesar, Walker & Hausner (1975); grid-searched breakpoint, optional Huber IRLS (`fit_kolesar`)
- **Core modules:** `src/emvro/kolesar.py` (`KolesarModel`, `fit_kolesar`, `fit_kolesar_from_frame`)
- **Headline metrics:** none yet
- **Ceiling / honesty notes:** `NYC_DEFAULT_MILES` coefficients are approximate and flagged in the module as unverified against the paper; refit on London
- **Next lever:** distance column (deployed station → incident) in the planner CSV; use as prior for LightGBM residual and as the "Kolesar" scorer for placement

---

### [LD-02] LFB attendance standards + turnout/drive scoring
- **Date:** 2026-10
- **Status:** active (module) · not yet in the planner
- **Family:** standards/scoring
- **Problem:** Score first-engine response against LFB 2022: mean ≤ 6 min, > 90 % within 10 min (second engine 8 min mean / 95 % within 12). Not NFPA 4-min.
- **Label / target:** attendance = drive + turnout (`attendance_seconds`; fallback turnout 60 s, clip 0–300 s)
- **Core modules:** `src/emvro/lfb_standards.py` (`score_first_engine`, `attendance_seconds`)
- **Headline metrics:** none yet
- **Ceiling / honesty notes:** public LFB CSV has attendance only (mobilise→arrive); turnout split needs mobilisation records
- **Next lever:** call `score_first_engine` from the planner summary; per-station turnout from mobilisation data

---

### [LD-03] Busy engines
- **Date:** 2026-10
- **Status:** active (module + `busy_flag` column) · not yet in objective/model
- **Family:** standards/scoring
- **Problem:** First pump often comes from a station other than the ground station; estimate how often.
- **Label / target:** `busy = (deployed_from_station != station_ground)`; `busy_prob` per station (and station × hour)
- **Core modules:** `src/emvro/busy_engines.py` (`estimate_busy_rates`, `expected_first_arrival`); `scripts/build_lfb_planner_incidents.py` writes `busy_flag` (≈29 % of 2024–26 planner rows)
- **Ceiling / honesty notes:** proxy; includes cross-border and nearest-appliance mobilisation, not only "home engine out"
- **Next lever:** use `expected_first_arrival` in the placement objective; busy features in the LightGBM

---

### [LD-04] 2014 ten-station closures
- **Date:** 2026-10
- **Status:** active (list + matching) · experiment **planned**
- **Family:** validation
- **Problem:** Natural experiment: LFB closed Belsize, Bow, Clerkenwell, Downham, Kingsland, Knightsbridge, Silvertown, Southwark, Westminster, Woolwich on 9 Jan 2014.
- **Core modules:** `src/emvro/london_closures_2014.py` (`CLOSED_2014`, `match_closed`, `flag_closed`)
- **Planned test:** planner `replace --replace 10` vs the real closed set; layout KPIs and overlap
- **Next lever:** confirm every closed name resolves in `london_firehouses.csv`

---

### [LD-05] LFB planner extract
- **Date:** 2026-10-04
- **Status:** active
- **Family:** data/infra
- **Problem:** Turn the raw LFB incident CSV into the planner schema.
- **Method:** BNG→WGS84 where lat/lon is missing; `travel_seconds` = `FirstPumpArriving_AttendanceTime` (kept 30–1800 s); `deployed_from_station`, second pump fields, `busy_flag`
- **Build / data script:** `scripts/build_lfb_planner_incidents.py` → `data/raw/london/lfb_incidents_planner.csv` (~129k rows: 2024, 2025, partial 2026)
- **Honesty notes:** attendance, not drive. Demand should be 2023–24 and eval 2025; **2023 not ingested** (2024-only demand until then). `cities.load_incidents(..., years=)` filters `cal_year` when present.
- **Next lever:** ingest 2023; mobilisation turnout split; surveyed station coordinates

---

### [LD-06] London firehouse planner (redesign / replace / expand)
- **Date:** 2026-10-04
- **Status:** active (all three modes; Kolesar/crow/network scorers; LFB KPIs)
- **Family:** placement
- **Problem:** Place or relocate stations; compare layouts under Kolesar / network / ML scorers.
- **Method today:** `--city london` default; `--scorer crow|kolesar|network`; attendance = drive + turnout; `--demand-years` / `--eval-years`; `--use-busy` blends reporting; Dijkstra when a graph is supplied
- **Build / data script:** `scripts/plan_firehouse_locations.py`
- **Core modules:** `emvro.firehouse_location`, `emvro.cities`, `emvro.kolesar`, `emvro.lfb_standards`, `emvro.busy_engines`, `emvro.routing.firetruck`
- **Headline metrics:** smoke expand+2 Kolesar (20k): demand E[T] −4.2 s vs current; modelled 10-min share optimistic vs observed ~94%
- **Next lever:** ML street-aware scorer; street-node candidates; London OSM/OS NGD graph; multi-scorer ranking table

---

### [LD-07] LFB driving-time model (LightGBM, no station IDs)
- **Date:** 2026-10-04
- **Status:** active (crow/context features); full street attrs **planned**
- **Family:** LFB travel-time
- **Label / target:** approx drive = attendance − DEFAULT_TURNOUT_S (60 s)
- **Holdout:** train 2024 (2023 when present), test 2025
- **Features:** crow_km, hour/dow/month, rush/night, busy_flag, borough (label); **no station IDs**
- **Model / algorithm:** LightGBM (L1) predicting drive time directly, 4 seeds averaged; no Kolesar prior (changed 2026-10-04)
- **Train script:** `scripts/train_travel_time_lfb.py` → `data/processed/models/travel_time_lfb.joblib`
- **Baselines:** median, crow@32kph, Kolesar (**EXISTS**); road / linear / NN **planned**
- **Headline metrics:** 2025 MAE ≈ **78.4 s** (LightGBM) vs Kolesar 81.8 s / median 106.5 s
- **Ablations / interpretability:** ablations and SHAP **planned**
- **Docs:** `docs/travel_time_training.md`

---

### [LD-08] Route model (learned segment speeds) and street data
- **Date:** 2026-10-04
- **Status:** scaffold **active**; London graph + FOI GPS **planned**
- **Family:** LFB route model
- **Method:** `emvro.segment_speeds.SegmentSpeedModel`; Dijkstra under EMV speeds; FOI to LFB for GPS; fallback claim = travel-time accuracy only
- **Build / data script:** `scripts/fit_segment_speeds.py` (exits with plan JSON if no `london_drive.graphml`)
- **Street data:** OS NGD (primary; **not downloaded**), OSM (fallback; London graph not built)
- **Docs:** `docs/route_models.md`

---

## Legacy NYC / FDNY entries

### [TT-01] CAD context model (crow / multi-origin LGBM)
- **Date:** early project
- **Status:** superseded · legacy NYC (kept for comparisons)
- **Family:** CAD travel-time
- **Problem:** Predict real CAD travel seconds with inferred OD.
- **Label / target:** `travel_seconds` (assign → on scene)
- **Holdout:** historically random / small extracts; later superseded by chronological
- **Features:** crow-flies from guessed house, context (hour/dow/call type), optional weather/OSM aggregates; early versions leaked `qc_*` flags derived from the label
- **Model / algorithm:** LightGBM (`--feature-set baseline|full|route`); later seed bag4 + HGB
- **Train script:** `scripts/train_travel_time_model.py`
- **Build / data script:** `scripts/build_od_pairs.py`, `scripts/build_travel_time_training_set.py`
- **Core modules:** `emvro.travel_time`, `emvro.first_due`, `emvro.depots`, `emvro.weather`
- **Artifacts:** `data/processed/models/travel_time_metrics*.json`, `data/figures/model_eval*`
- **Headline metrics (full bag4+HGB, older split):** MAE ≈ 72.9 s · R² ≈ 0.34 (n_test ≈ 8k)
- **vs previous:** beat naive median / crow prior
- **What worked:** context features; bagging
- **What failed / discarded:** treating random-split R² 0.24 with QC leakage as comparable; expecting high R² from CAD alone
- **Ceiling / honesty notes:** path length ≈ uncorrelated with CAD travel seconds without true starts
- **Next lever:** network origins + year CAD → [TT-03]

---

### [TT-02] Route ETA (known-OD EMV seconds)
- **Date:** mid project
- **Status:** legacy NYC (optimizer scoring in NYC demos)
- **Family:** route-ETA
- **Problem:** Score candidate routes / OD pairs when origin and destination are known.
- **Label / target:** network EMV seconds (civilian shortest path → EMV mapping + hour/weather/noise); optional GMaps-enriched variant
- **Holdout:** held-out OD pairs (not CAD chronological)
- **Features:** path geometry, `civilian_network_s`, congestion/hour, severity, weather; GMaps control features when enriched
- **Model / algorithm:** LightGBM
- **Train script:** `scripts/train_route_eta_model.py`
- **Build / data script:** `scripts/build_osm_graph.py`, `scripts/build_route_eta_training_set.py`, `scripts/build_multicity_route_eta.py`, `scripts/enrich_training_with_gmaps.py`
- **Core modules:** `emvro.route_eta`, `emvro.street_features`, `emvro.gmaps`
- **Artifacts:** `travel_time_route_eta_lightgbm.joblib`, `travel_time_route_eta_metrics.json`, `route_eta_nyc_gmaps_metrics.json`
- **Headline metrics:** MAE ≈ 37.6 s · R² ≈ 0.979 (base); GMaps-enriched R² ≈ 0.974 · MAE ≈ 56.6 s
- **vs previous:** meets target R² ≥ 0.8 / 0.9 for optimizer use
- **What worked:** known OD + network path features
- **What failed / discarded:** using this R² to claim CAD travel-time is “solved”
- **Ceiling / honesty notes:** different task from CAD hybrid
- **Next lever:** multi-city OD diversity (`docs/multi_city_datasets.md`); live traffic priors

---

### [TT-03] Network CAD model (“network + traffic”)
- **Date:** year-scale rebuild
- **Status:** legacy NYC baseline *model type*
- **Family:** CAD travel-time
- **Problem:** Beat crow-prior CAD models with label-free origin/network features + year of FDNY data.
- **Label / target:** real FDNY CAD travel seconds
- **Holdout:** chronological — train through 2024, test 2025-Q1 (~107k)
- **Features (ladder of gains):**
  - Year CAD (~550k–970k usable after filters)
  - Geocoded destinations (alarm box / intersection; ZIP centroid residual)
  - Network origins: first-due engine/ladder free-flow times, nearest 1–3 houses (`emvro.network_origins`)
  - Unit-load counts (15/30/60 min, past-only)
  - Route street features along first-due path (`emvro.route_street_features`)
  - DOT traffic speeds `i4gi-tjb9` (`emvro.traffic`)
  - OOF location priors (box / engine / ZIP)
  - Removed label-leaking `qc_*` features
- **Model / algorithm:** LightGBM on network feature table
- **Train script:** `scripts/train_travel_time_network.py`
- **Build / data script:** `scripts/build_travel_time_network_dataset.py`, `scripts/download_datasets.py`, traffic download helpers
- **Core modules:** `emvro.network_origins`, `emvro.intersections`, `emvro.route_street_features`, `emvro.traffic`
- **Artifacts:** `travel_time_network.parquet`, `data/figures/model_eval_network/`
- **Headline metrics (2025-Q1):** MAE ≈ **67.7 s** · R² ≈ **0.34**
- **vs previous:** ~71–80 s designs → 67.7 s; street+traffic ≈ +0.1 s only
- **What worked:** more data, geocoded dests, network origins (largest CAD gains)
- **What failed / discarded:** expecting street/traffic to close the gap to 50 s MAE
- **Ceiling / honesty notes:** first-due often ≠ first-arriving unit; turnout/CAD noise dominates
- **Next lever:** residual bag hybrid → [TT-04]

---

### [TT-04] Hybrid CAD (network features + residual bag4)
- **Date:** continued after TT-03
- **Status:** legacy NYC — best FDNY CAD model; template for London bag4 ([LD-07])
- **Family:** CAD travel-time
- **Problem:** Same CAD task as TT-03; improve MAE with residual modeling + denser features.
- **Label / target:** real FDNY CAD travel seconds (raw labels on test)
- **Holdout:** chronological 2025-Q1 on `travel_time_network.parquet` (n_test ≈ 106,844)
- **Features (added/changed vs TT-03):**
  - Residual prior: free-flow network blend + dest/engine OOF priors
  - **OD key** + **engine×hour** priors
  - GMaps civilian/EMV duration features (~82% row coverage; *feature only*, not residual prior blend)
  - Dense traffic history (2023-01 → 2025-03)
  - Sample weights: alarm_box ≫ intersection ≫ zip; boost GMaps/traffic rows
  - Optional soft cell-median labels — **discarded** (hurt holdout; off by default)
- **Model / algorithm:** residual LightGBM bag4 (L2 + L1 seeds); HGB blend weight chosen on valid (settled at **0.0** → bag-only)
- **Train script:** `scripts/train_travel_time_hybrid.py`
- **Build / data script:** `scripts/build_travel_time_network_dataset.py` (+ GMaps enrich / traffic attach)
- **Core modules:** same as TT-03 + `emvro.gmaps`
- **Artifacts:** `travel_time_hybrid.joblib`, `travel_time_hybrid_metrics.json`, `data/figures/model_eval_hybrid/`, `data/figures/model_eval_comparison/`
- **Headline metrics:**
  - Full test: MAE ≈ **64.5 s** · R² ≈ **0.42**
  - Alarm-box slice: MAE ≈ **63.2 s**
- **vs previous:** vs network+traffic **67.7 s** → **−3.2 s MAE**; vs prior hybrid ~65.0 s → **−0.5 s**
- **What worked:** OD/engine-hour priors; bag-only (no HGB); alarm/GMaps/traffic weights
- **What failed / discarded:**
  - Soft-label denoising (α=0.35 → MAE 66.5 s)
  - Blending GMaps into residual prior (civilian ≠ CAD assign→scene)
  - Framing “Network + traffic” as a rival person’s model — it is a **model type**
  - Paris BSPP ingest (demo-only, no public files) — **discarded**; pivot to **London LFB**, not NYC-as-product
- **Ceiling / honesty notes:** CAD noise floor; engines often not at house; MAE ~50–60 s stretch without AVL
- **Next lever:** cleaner alarm-box-only production slice; more GMaps ODs; AVL/GPS if available

---

### [RT-01] Path routers — MIPSSTW + MCS
- **Date:** slide Step 3
- **Status:** legacy NYC demo
- **Family:** path router
- **Problem:** Fastest EMV path with soft time-window style objective.
- **Label / target:** path cost (EMV seconds + penalties)
- **Holdout:** scenario OD demos / condition presets
- **Features:** EMV edge costs (congestion, weather, busway/ROW, contraflow risk)
- **Model / algorithm:** Modified Cuckoo Search (Lévy perturbations + nest abandonment) on MIPSSTW-style objective
- **Train script:** n/a (search)
- **Build / data script:** `scripts/run_route_models.py`, `scripts/visualize_route_models.py`, `scripts/run_condition_scenarios.py`
- **Core modules:** `emvro.routing.mipsstw_mcs`, `emvro.routing.graph`, `emvro.routing.conditions`
- **Artifacts:** `data/processed/route_models_demo.json`, `data/figures/route_models/`
- **Headline metrics:** scenario travel time / distance charts (not CAD MAE)
- **vs previous:** vs civilian shortest path / distance
- **What worked:** condition presets (weather × congestion × ROW)
- **What failed / discarded:** treating MCS demo fitness as CAD accuracy
- **Ceiling / honesty notes:** demo scale; upgrade path = MILP on corridor subgraph
- **Next lever:** PuLP/CBC MILP; BPR + intersection delay

---

### [RT-02] Path routers — Composite DRL
- **Date:** slide Step 3
- **Status:** legacy NYC demo
- **Family:** path router
- **Problem:** Learn path policy with composite reward (−EMV time + lane bonuses − backtrack).
- **Label / target:** cumulative reward / resulting path time
- **Holdout:** scenario ODs
- **Features:** graph state; primary/bus-lane bonuses
- **Model / algorithm:** tabular Q-learning
- **Train script:** via `scripts/run_route_models.py`
- **Core modules:** `emvro.routing.composite_drl`
- **Artifacts:** `06_drl_learning_curve.png`, multi-route maps
- **Headline metrics:** learning curve / scenario times
- **What worked:** interpretable reward shaping for EMV ROW
- **Next lever:** DQN / GNN policy; multi-agent preemption

---

### [RT-03] Path routers — GBDT edge costs
- **Date:** slide Step 3
- **Status:** legacy NYC demo
- **Family:** path router
- **Problem:** Predict per-edge EMV cost → Dijkstra path.
- **Label / target:** edge EMV seconds
- **Holdout:** scenario ODs
- **Features:** street + hour features on edges
- **Model / algorithm:** LightGBM edge costs + Dijkstra
- **Core modules:** `emvro.routing.gbdt_router`
- **Artifacts:** route comparison figures / maps
- **Next lever:** train edge costs from route-ETA labeled paths + LION attributes

---

### [PT-01] Patrol posts & loops
- **Date:** mentor “optimal patrol routes” ask
- **Status:** out-of-story (sandbox, not part of the CSEF/ISEF claim)
- **Family:** patrol
- **Problem:** Where to stage firetrucks *before* calls to cut expected response time.
- **Label / target:** minimize E[T] (p-median) or maximize coverage within T
- **Holdout:** historical demand grid; baselines on same score matrix
- **Features:** demand cells from CAD destinations; candidate posts = firehouses + CSL + demand centroids
- **Model / algorithm:** greedy + k-medoids swaps; NN+2-opt loops; crow screen then OSM EMV Dijkstra score
- **Train script:** n/a
- **Build / data script:** `scripts/build_patrol_routes.py`
- **Core modules:** `emvro.patrol`, `emvro.routing`
- **Artifacts:** `data/figures/patrol/`, `summary.json`
- **Headline metrics:** E[T] / coverage vs `best_k_facilities_only` (equal-cost) and `all_facilities_stationary` (context)
- **What worked:** equal-cost facility-only baseline as honest headline
- **What failed / discarded:** calibrating crow speed from inferred OD (≈6 kph artifact) — default 28 kph prior
- **Ceiling / honesty notes:** travel only (no chute/queue); inferred demand geography
- **Next lever:** severity weighting; live demand; more posts/units sweeps

---

### [FH-01] Firehouse location planner — NYC origin (redesign / replace)
- **Date:** 2026-09-30
- **Status:** active; NYC origin of the planner now used for London ([LD-06])
- **Family:** facility location (prescriptive)
- **Problem:** Place or relocate `k` / `r` firehouses so trucks reach demand fastest.
- **Label / target:** minimize demand-weighted E[T] (or max coverage within threshold)
- **Holdout:** vs current firehouse posture (+ random-replace baseline)
- **Features / method:** city registry; firetruck-weighted Dijkstra matrix; affine CAD calibration; greedy+swap (redesign) or joint close/open swaps (replace)
- **Train script:** n/a
- **Build / data script:** `scripts/plan_firehouse_locations.py`
- **Core modules:** `emvro.firehouse_location`, `emvro.cities`, `emvro.routing.firetruck`
- **Artifacts:** `data/figures/firehouse_plan/`, `docs/firehouse_location.md`
- **Headline metrics (NYC replace r=8, crow screen):** E[T] 115.2s → **111.2s** vs current
- **vs previous:** patrol posts are on-road staging; this redesigns the facility network
- **What worked:** both CLI modes; firetruck cost overlay; city input
- **What failed / discarded:** n/a
- **Ceiling / honesty notes:** London path scores drive+turnout vs LFB 6/10; NYC origin was travel-only. Discrete candidates; no zoning/land cost
- **Next lever:** graph-scored runs; parcel-feasible candidates; borough constraints

---

### [DF-01] First-due starting locations
- **Date:** foundational
- **Status:** legacy NYC (default OD builder for FDNY only; London records deployed station)
- **Family:** data/infra
- **Problem:** Public CAD has no apparatus GPS at assignment.
- **Label / target:** reconstructed `(start, dest)` for OD tables
- **Holdout:** QC flags (`qc_speed_flag`, `qc_not_nearest_house`, `qc_keep`)
- **Features / method:** alarm-box dest → engine polygon → firehouse; hybrid CSL/nearest fallback
- **Build / data script:** `scripts/build_od_pairs.py`, `scripts/download_datasets.py`
- **Core modules:** `emvro.first_due`, `emvro.depots`, `emvro.csl`
- **Artifacts:** OD CSVs; `docs/starting_locations.md`; QC figures
- **What worked:** first-due chain; speed QC
- **What failed / discarded:** assuming first-due = first-arriving always
- **Next lever:** Geosupport for unmatched box strings; AVL under DUA

---

### [DF-02] Destination geocoding & street/traffic layers
- **Date:** with TT-03
- **Status:** legacy NYC
- **Family:** data/infra
- **Problem:** ZIP centroids are too coarse for CAD travel models.
- **Method:** Centerline intersection geocode for unlisted boxes; street edge table; DOT hourly speeds
- **Core modules:** `emvro.intersections`, `emvro.route_street_features`, `emvro.traffic`, `emvro.street_features`
- **Build scripts:** `analyze_street_characteristics.py`, traffic download, `build_travel_time_network_dataset.py`
- **Headline effect:** ZIP-only dests ~30% → ~5%; enables network+traffic ladder

---

### [DF-03] Google Maps control / enrich
- **Date:** ongoing
- **Status:** legacy NYC (feature / control; not CAD prior blend)
- **Family:** data/infra
- **Problem:** Civilian ETA as control and as CAD *features*.
- **Method:** Routes API cache (`emvro.gmaps`); OD enrich on first-due→dest pairs
- **Scripts:** `enrich_training_with_gmaps.py`, `classify_origins_with_gmaps.py`
- **Docs:** `docs/gmaps_control.md`
- **What worked:** ~82% GMaps coverage on year network table as features
- **What failed:** blending GMaps EMV seconds into residual prior for CAD

---

### [DF-04] Multi-city (SF secondary) / Paris discarded → London
- **Date:** late
- **Status:** SF draft (secondary); **Paris discarded**; replaced by **London** as the primary city
- **Family:** data/infra (route-ETA expansion)
- **Problem:** Diversify route-ETA geometry beyond NYC.
- **SF:** `scripts/ingest_sf_ems.py` → `build_multicity_route_eta.py`
- **Paris BSPP:** ENS challenge GPS start + δ departure-presentation — draft ingest removed; not trained. Direction is Paris → **London** (LFB incident data), not Paris → NYC
- **Docs:** `docs/multi_city_datasets.md`

---

## Current recommended stack (London)

| Need | Use | Script / artifact | Status |
|---|---|---|---|
| Build LFB demand/eval extract | Planner CSV [LD-05] | `build_lfb_planner_incidents.py` | exists |
| Distance→time prior / baseline | Kolesar [LD-01] | `emvro.kolesar` | module exists, London fit planned |
| Predict driving time (no station IDs) | LightGBM [LD-07] | `train_travel_time_lfb.py` | exists |
| KPIs (6 min mean, 10 min > 90 %) | LFB standards [LD-02] | `emvro.lfb_standards` | exists, not yet in planner |
| Engine unavailability | Busy engines [LD-03] | `emvro.busy_engines`, `busy_flag` | exists |
| 2014 validation | Closures [LD-04] | `emvro.london_closures_2014` | list exists, experiment planned |
| Redesign / replace / expand stations | Planner [LD-06] | `plan_firehouse_locations.py --city london` | redesign + replace exist; expand planned |
| Route model, learned segment speeds | [LD-08] | — | **planned** |
| Street data | OS NGD (primary), OSM (fallback) | — | NGD **not downloaded**; OSM tooling exists |

## Legacy NYC stack (scaffolding)

| Need | Use | Script / artifact |
|---|---|---|
| Score known OD / optimizer edges | Route ETA [TT-02] | `train_route_eta_model.py` |
| Predict FDNY CAD travel | Hybrid bag4 [TT-04] | `train_travel_time_hybrid.py` → `travel_time_hybrid.joblib` |
| CAD baseline model type | Network + traffic [TT-03] | `train_travel_time_network.py` |
| Choose a path O→D (demo) | Routers [RT-01..03] | `run_route_models.py` |
| NYC firehouse planner | [FH-01] | `plan_firehouse_locations.py --city nyc` |
| Stage units before calls | Patrol [PT-01] — **out of story** | `build_patrol_routes.py` |

Holdout figure of record for NYC CAD types: `data/figures/model_eval_comparison/`
(`comparison_summary.json`: median · network+traffic · hybrid).

---

## Metrics cheat sheet (latest recorded)

**London: no metrics recorded yet.** The rows below are NYC/FDNY legacy and must not be quoted as the London result.

| Entry | Holdout | MAE | R² |
|---|---|---|---|
| Hybrid CAD [TT-04] | 2025-Q1 CAD | **64.5 s** (alarm-box **63.2 s**) | ~0.42 |
| Network + traffic [TT-03] | 2025-Q1 CAD | 67.7 s | ~0.34 |
| Older CAD bag4+HGB [TT-01] | smaller / older | ~72.9 s | ~0.34 |
| Route ETA [TT-02] | known OD | ~37.6 s | ~0.98 |

---

## Discarded / anti-patterns (do not revive without new evidence)

| Idea | Why not |
|---|---|
| Soft-label toward cell medians (α≈0.35) | Raised CAD MAE to ~66.5 s |
| GMaps in residual prior | Civilian ETA ≠ CAD assign→scene |
| Compare route-ETA R² to CAD R² | Different labels / tasks |
| Paris BSPP ingest | Discarded; no public ENS files in-repo; replaced by London (LFB) |
| NFPA 1710 / 4-minute US standard for London | LFB uses 6 min mean / 10 min > 90 % |
| Treating LFB `AttendanceTime` as pure drive | It is mobilise→arrive (turnout + drive) |
| Reporting planner numbers that include 2025 in demand | Eval year must be held out |
| Claiming SHAP / OS NGD done | Neither has been done |
| Label-leaking `qc_*` as features | Inflates metrics dishonestly |
| Crow-speed calibrated from inferred OD | ~6 kph artifact |

---

## Appendix — script ↔ family index

| Script | Family |
|---|---|
| `train_travel_time_model.py` | CAD (legacy) |
| `train_travel_time_network.py` | CAD network+traffic |
| `train_travel_time_hybrid.py` | CAD hybrid (best) |
| `compare_hybrid_to_cad.py` / `compare_travel_time_models.py` | CAD eval figures |
| `train_route_eta_model.py` | Route ETA |
| `run_route_models.py` / `visualize_route_models.py` | Path routers |
| `run_condition_scenarios.py` / `find_row_wins.py` | Conditions / ROW |
| `build_patrol_routes.py` | Patrol (out of story) |
| `plan_firehouse_locations.py` | Firehouse facility location (London primary; NYC legacy) |
| `build_lfb_planner_incidents.py` | London data (LFB planner CSV) |
| `emvro/kolesar.py`, `lfb_standards.py`, `busy_engines.py`, `london_closures_2014.py` | London modules (no scripts yet) |
| `build_od_pairs.py` / `build_travel_time_*` / `build_route_eta_*` | Data |
| `ingest_sf_ems.py` / `build_multicity_route_eta.py` | Multi-city ETA |
