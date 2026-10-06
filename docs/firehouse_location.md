# Firehouse location planner

Plan **where fire stations should sit** so the first engine meets LFB attendance
standards, and compare drive scorers (**Kolesar / crow**; the LightGBM model is **PLANNED**).

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.

## Research question

Does placing stations with street-aware (or Kolesar) drive times beat crow-flies
placement on held-out **2025** LFB calls, once attendance = drive + turnout?

## KPIs: LFB standards (not NFPA / US 4-minute)

| KPI | Target | Code |
|---|---|---|
| First engine, mean attendance | **≤ 6 min** | `emvro.lfb_standards.FIRST_ENGINE_AVG_S` |
| First engine, share within 10 min | **> 90 %** | `FIRST_ENGINE_P90_S`, `FIRST_ENGINE_TARGET_SHARE` |
| Second engine (secondary) | mean ≤ 8 min; 95 % within 12 min | `SECOND_ENGINE_*` |

`score_first_engine(response_s, weights)` returns mean, p90, shares within 6 / 10 min,
and pass/fail (**EXISTS**). The planner summary includes `lfb_first_engine` on the
demand grid and, when `--eval-years` is set, held-out eval plus optional
`observed_attendance` from CSV **`travel_seconds`** (**EXISTS**). Coverage bars use
`--threshold-min` (default **10** min, LFB tail target).

## What is scored: turnout + drive

```
attendance(station s → incident i) = predicted_drive(s, i) + turnout(s)
```

- Public LFB incident CSV has only `FirstPumpArriving_AttendanceTime` (mobilise → arrive).
  Turnout and drive are **not separable** in that file; per-station turnout from
  mobilisation data is **PLANNED** when ingested.
- The planner fits/scores **drive** via the chosen scorer, then adds turnout:
  default `DEFAULT_TURNOUT_S = 60 s` for new sites; existing houses get a
  Kolesar-residual estimate when data allow (**EXISTS** in `run_firehouse_plan`).
- `lfb_standards.attendance_seconds(drive_s, turnout_s)` implements the sum (**EXISTS**).

## Modes

| Mode | Meaning | Status |
|---|---|---|
| `redesign` | Place exactly `k` stations from the pool (`--n-houses`; default = current count) | **EXISTS** |
| `replace` | Close `r` / open `r` (`--replace r`) — e.g. `r = 10` mirrors 2014 closures | **EXISTS** |
| `expand` | Keep all current stations, add `m` new ones (`--add-stations m`) | **EXISTS** |

Candidates (**EXISTS**): existing stations plus top demand-cell
centroids (`--demand-candidates`). **PLANNED**: additional candidates on street-graph
nodes. Objectives: p-median `response_time` or max-cover `coverage`.

## Scorers (multi-scorer comparison)

| Scorer | Travel time from… | Status |
|---|---|---|
| **crow** | Haversine surrogate at a flat speed with a 1.3 detour factor | **EXISTS** (`--scorer crow`) |
| **Kolesar** | Piecewise `T = a + b√d` / `c + e·d` on distance; refit on London trips (`emvro.kolesar`) | **EXISTS** (`--scorer kolesar`) |
| **Road** | Shortest legal route on the directed London drive graph (`emvro.road_distance`) | Routing **EXISTS**; planner scorer **PLANNED** |
| **ML (street-aware)** | LightGBM travel-time model (`docs/travel_time_training.md`) | Model **EXISTS**; planner scorer **PLANNED** |

Default scorer: **kolesar**.

## Demand and evaluation split

| Use | Years | Status |
|---|---|---|
| Demand (where calls happen) | **2023–24** target | **2024 only** until 2023 is ingested |
| Evaluation (held-out) | **2025** | **EXISTS** via `--demand-years` / `--eval-years` |

CLI defaults: `--demand-years 2024`, `--eval-years 2025`. When `cal_year` is present
in the planner CSV, `run_firehouse_plan` builds the demand grid from demand years and
reports eval KPIs on eval-year incidents (**EXISTS**). Non-LFB cities still use
`--max-incidents` subsampling without year split.

## Busy engines

LFB's first pump is often not from the home station. `emvro.busy_engines` marks
**busy** when `FirstPumpArriving_DeployedFromStation != IncidentStationGround`
(**EXISTS**). `build_lfb_planner_incidents.py` writes `busy_flag` and deploy columns.

- **`--use-busy`** (**EXISTS**): blends nearest and next-nearest attendance in
  **reported** response via `expected_first_arrival`; **optimisation still uses
  nearest engine only** (`posture_response` vs `plan_*` on matrix `A`).
- Without `--use-busy`, modelled cover is **optimistic** vs observed attendance
  (always-free nearest engine).

Caveat: busy ≠ "home engine physically unavailable" — also cross-border and
closest-appliance mobilisation.

## Validation

1. **2014 ten-station closures** — list in `emvro.london_closures_2014` (**EXISTS**).
   `scripts/validate_london_2014_closures.py` and `scripts/rank_closed_stations_2014.py`
   run counterfactuals on **today's network** with approximate closed-station coords
   (**EXISTS**). True pre-2014 train → post-2014 test needs older LFB extracts
   (**PLANNED** ingest).
2. **Leave-one-firehouse-out** — **PLANNED**.
3. **Multi-scorer table** — planner runs one scorer per invocation (**EXISTS**);
   batch comparison script **PLANNED**.

## Run (London)

Defaults: `--city london`, `--threshold-min 10`, year split as above.

```bash
# Planner extract from raw LFB CSV
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py

# Redesign / replace / expand (Kolesar scorer)
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode redesign --demand-years 2024 --eval-years 2025
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode replace --replace 10
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode expand --add-stations 2

# Optional busy blend in reported metrics only
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city london --mode expand --add-stations 3 --use-busy

# Synthetic smoke test
PYTHONPATH=src python scripts/plan_firehouse_locations.py --demo

# 2014 closure scaffolds (today's-data approximation)
PYTHONPATH=src python scripts/validate_london_2014_closures.py
PYTHONPATH=src python scripts/rank_closed_stations_2014.py

# Custom city
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city custom --firehouses path/houses.csv --incidents path/incidents.csv \
  --mode redesign --n-houses 20
```

`london` in `emvro.cities` uses `data/raw/london/london_firehouses.csv` and
`lfb_incidents_planner.csv`.

## Outputs (`data/figures/firehouse_plan/`)

| File | Content |
|---|---|
| `01_metrics_comparison.png` | E[T] + coverage bars vs current / random |
| `02_response_cdf.png` | Demand-weighted response CDF |
| `03_response_surface_map.png` | Before/after response surfaces |
| `04_relocations.png` | Closed → opened moves |
| `05_site_demand_share.png` | Top sites by assigned demand |
| `firehouse_plan_map.html` | Interactive Folium map |
| `planned_firehouses.csv` | Selected sites + assigned demand share |
| `closed_firehouses.csv` / `opened_firehouses.csv` | Replace-mode moves |
| `demand_cells.csv` | Grid cells with planned / current response |
| `candidates.csv` | Full candidate pool |
| `summary.json` | Metrics, `lfb_first_engine`, `demand_years`, `eval_years`, `eval`, honesty text |

## Honesty limits

- **Modelled vs observed:** predicted attendance uses nearest-engine drive + turnout;
  observed `travel_seconds` reflects real dispatch (busy engines, cross-border pumps).
  Compare via eval block `observed_attendance` — expect modelled cover to look **better**
  unless `--use-busy` (partial correction, not full dispatch simulation).
- Station sites are OpenStreetMap building locations, name-matched to LFB stations.
- Discrete candidates only; no land cost, staffing, borough politics, or planning rules.

## Code

| Piece | Path |
|---|---|
| City registry (London + custom) | `src/emvro/cities.py` |
| Planner (redesign, replace, expand) | `src/emvro/firehouse_location.py` |
| Demand grid, p-median / max-cover selection | `src/emvro/facility.py` |
| Kolesar model | `src/emvro/kolesar.py` |
| LFB standards + attendance sum | `src/emvro/lfb_standards.py` |
| Busy-engine rates | `src/emvro/busy_engines.py` |
| 2014 closures | `src/emvro/london_closures_2014.py` |
| LFB planner CSV | `scripts/build_lfb_planner_incidents.py` |
| 2014 validation scaffolds | `scripts/validate_london_2014_closures.py`, `scripts/rank_closed_stations_2014.py` |
| CLI | `scripts/plan_firehouse_locations.py` |
