# Model implementations & changes — template

Living catalog of **major model work** on the fire station placement project, plus a
blank form to copy for future changes.

**Project frame (CSEF/ISEF):** London Fire Brigade station placement with a
street-characteristic travel-time model (gradient-boosted decision trees, LightGBM),
scored on held-out 2025 LFB calls.

**Status words:** `active` = exists and used · `planned` = in the plan, not built or run.

**How to use**

1. Copy the [Blank entry](#blank-entry) block into [Changelog](#changelog--filled-entries).
2. Fill every field you can; leave unknowns as `TBD`.
3. Link scripts, modules, metrics JSON, and figures.
4. Do not log a `planned` item as done.

**Related docs:** [`travel_time_training.md`](travel_time_training.md) ·
[`datasets.md`](datasets.md) · [`firehouse_location.md`](firehouse_location.md)

---

## Blank entry

```markdown
### [ID] Short title
- **Date:** YYYY-MM-DD
- **Status:** planned | draft | active | superseded | discarded
- **Family:** LFB travel-time | route model | placement | standards/scoring | data/infra | other
- **Problem:** one sentence
- **Label / target:** what is predicted
- **Holdout:** split rule (chronological / random / OD) + dates / n
- **Features (added/changed):** bullet list
- **Model / algorithm:** e.g. LightGBM, 4 seeds averaged
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
| **LFB travel-time** | Predict driving time from any station to any London incident, no station IDs | Shortest-distance routes assumed; no street characteristics yet |
| **Route model** | Which streets does an engine drive? | No GPS tracks yet (FOI planned) |
| **Placement** | Where should London stations be (redesign / replace / expand)? | Scored vs LFB 6 / 10-min standards on held-out 2025; discrete candidates; travel-time scorers must add turnout |
| **Standards / busy / closures** | KPIs, engine unavailability, 2014 natural experiment | `busy_flag` is a proxy; closures list is fixed at 10 names |

---

## Changelog — filled entries

## Entries

### [LD-01] Kolesar piecewise travel-time module
- **Date:** 2026-10 (module present in repo)
- **Status:** active (baseline and planner scorer)
- **Family:** LFB travel-time (baseline)
- **Problem:** Distance-only baseline: `T = a + b√d` (short), `c + e·d` (long).
- **Label / target:** driving seconds from distance (km)
- **Holdout:** fit on 2021–24 LFB trips, test on 2025
- **Features:** distance only
- **Model / algorithm:** Kolesar, Walker & Hausner (1975); grid-searched breakpoint, optional Huber IRLS (`fit_kolesar`)
- **Core modules:** `src/emvro/kolesar.py` (`KolesarModel`, `fit_kolesar`, `fit_kolesar_from_frame`)
- **Headline metrics:** 2025 MAE 71.4 s on straight-line distance, 68.7 s on road distance
- **Ceiling / honesty notes:** `KolesarModel.london_default` (the 2021–24 straight-line fit) is only a fallback when a caller has too few trips

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
- **Status:** active (all three modes; Kolesar/crow scorers; LFB KPIs)
- **Family:** placement
- **Problem:** Place or relocate stations; compare layouts under Kolesar / road / ML scorers.
- **Method today:** `--city london` default; `--scorer kolesar|crow`; attendance = drive + turnout; `--demand-years` / `--eval-years`; `--use-busy` blends reporting
- **Build / data script:** `scripts/plan_firehouse_locations.py`
- **Core modules:** `emvro.firehouse_location`, `emvro.cities`, `emvro.kolesar`, `emvro.lfb_standards`, `emvro.busy_engines`, `emvro.facility`
- **Headline metrics:** smoke expand+2 Kolesar (20k): demand E[T] −4.2 s vs current; modelled 10-min share optimistic vs observed ~94%
- **Next lever:** LightGBM and road-distance scorers; street-node candidates; multi-scorer ranking table

---

### [LD-07] LFB driving-time model (LightGBM, no station IDs)
- **Date:** 2026-10-04
- **Status:** active (road distance + context features); street characteristics **planned**
- **Family:** LFB travel-time
- **Label / target:** recorded driving time (`TravelTimeSeconds`, LFB mobilisation records); origin = real station building the engine left from (OSM)
- **Holdout:** train 2021–2024 (701k trips), test 2025 (201k trips)
- **Features:** road_km (shortest legal route, one-ways respected), crow_km, hour/dow/month, rush/night, busy_flag, borough (label); **no station IDs**
- **Model / algorithm:** LightGBM (L1) predicting drive time directly, 4 seeds averaged; no Kolesar prior (changed 2026-10-04)
- **Train script:** `scripts/train_travel_time_lfb.py` → `data/processed/models/travel_time_lfb.joblib`
- **Baselines:** median, crow@32kph, road@35kph, Kolesar on crow and road distance (**EXISTS**); linear / NN **planned**
- **Headline metrics:** 2025 MAE **64.2 s** (LightGBM with road distance) vs Kolesar on road distance 68.7 s / median 111.9 s. Steps (2026-10-05): real station origin 72.8 → 67.9 s; + shortest legal road distance (one-way streets respected) 67.9 → 64.2 s
- **Ablations / interpretability:** ablations and SHAP **planned**
- **Docs:** `docs/travel_time_training.md`

---

### [LD-08] Route model (learned segment speeds) and street data
- **Date:** 2026-10-04
- **Status:** shortest-distance routing **active** (`emvro.road_distance`); learned speeds scaffold **active**; FOI GPS **planned**
- **Family:** LFB route model
- **Method:** `emvro.segment_speeds.SegmentSpeedModel`; Dijkstra under EMV speeds; FOI to LFB for GPS; fallback claim = travel-time accuracy only
- **Build / data script:** `scripts/fit_segment_speeds.py`
- **Street data:** OSM directed drive graph **built** (Geofabrik extract; 131,549 junctions, 30,145 one-way segments); OS NGD **not downloaded**

---

## Current recommended stack (London)

| Need | Use | Script / artifact | Status |
|---|---|---|---|
| Build LFB demand/eval extract | Planner CSV [LD-05] | `build_lfb_planner_incidents.py` | exists |
| Engine-trip training set (real origin, road distance) | [LD-07] | `build_lfb_travel_training.py` | exists |
| Distance→time baseline | Kolesar [LD-01] | `emvro.kolesar` | exists |
| Predict driving time (no station IDs) | LightGBM [LD-07] | `train_travel_time_lfb.py` | exists |
| KPIs (6 min mean, 10 min > 90 %) | LFB standards [LD-02] | `emvro.lfb_standards` | exists |
| Engine unavailability | Busy engines [LD-03] | `emvro.busy_engines`, `busy_flag` | exists |
| 2014 validation | Closures [LD-04] | `emvro.london_closures_2014` | list + scaffolds exist; pre-2014 test planned |
| Redesign / replace / expand stations | Planner [LD-06] | `plan_firehouse_locations.py --city london` | exists |
| Routes | Shortest legal route [LD-08] | `emvro.road_distance` | exists; learned speeds planned |
| Street data | OSM (built), OS NGD | `emvro.road_distance` | NGD **not downloaded** |

---

## Metrics cheat sheet (latest recorded)

| Entry | Holdout | MAE |
|---|---|---|
| LightGBM, real origin + road distance [LD-07] | 2025 (201k trips) | **64.2 s** |
| Kolesar on road distance [LD-01] | 2025 | 68.7 s |
| Kolesar on straight-line distance [LD-01] | 2025 | 71.4 s |
| Constant guess (training median) | 2025 | 111.9 s |

---

## Discarded / anti-patterns (do not revive without new evidence)

| Idea | Why not |
|---|---|
| Paris BSPP ingest | Discarded; no public ENS files in-repo; replaced by London (LFB) |
| NFPA 1710 / 4-minute US standard for London | LFB uses 6 min mean / 10 min > 90 % |
| Treating LFB `AttendanceTime` as pure drive | It is mobilise→arrive (turnout + drive); use mobilisation `TravelTimeSeconds` |
| Reporting planner numbers that include 2025 in demand | Eval year must be held out |
| Claiming SHAP / OS NGD done | Neither has been done |
| Station IDs as features | Model could not score sites with no station |

---

## Appendix — script ↔ family index

| Script | Family |
|---|---|
| `download_datasets.py` | Data (LFB + OSM download) |
| `build_lfb_planner_incidents.py` | Data (LFB planner CSV) |
| `build_london_firehouses.py` | Data (station locations) |
| `build_lfb_travel_training.py` | Data (engine-trip training set) |
| `train_travel_time_lfb.py` | LFB travel-time |
| `fit_segment_speeds.py` | Route model (scaffold) |
| `plan_firehouse_locations.py` | Placement |
| `validate_london_2014_closures.py` / `rank_closed_stations_2014.py` | 2014 closures |
| `visualize_data.py` | Incident density map |
