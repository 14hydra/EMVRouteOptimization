# Firehouse location planner

Plan **where firehouses should sit** so trucks reach emergencies as fast as
possible. Distinct from patrol posts (`docs/patrol_routes.md`): this relocates
or redesigns the **facility network**, not on-road staging loops.

## Inputs

| Input | Flag | Notes |
|---|---|---|
| City | `--city nyc` | NYC wired; `sf` stub; `custom` + paths |
| Mode | `--mode redesign\|replace` | Both supported |
| Fleet size | `--n-houses k` | Redesign: place exactly `k` sites |
| Relocations | `--replace r` | Replace: close `r` / open `r` (same total) |
| Hour | `--hour` | Congestion for firetruck graph costs |

## What it uses

1. **Graph theory** — OSM drive graph with **firetruck** edge weights
   (`emvro.routing.firetruck`: stricter contraflow, residential penalty,
   arterial preference) → Dijkstra OD matrix.
2. **Travel-time model** — hybrid CAD model does not score brand-new sites
   feature-complete; instead we **affine-calibrate** graph seconds to observed
   CAD travel (`y ≈ a + b · graph_s`) so the objective is on a CAD-like scale.
3. **Facility location** — p-median (`response_time`) or max-cover (`coverage`),
   reusing the greedy + swap engine from `emvro.patrol.select_posts`.
4. **Candidates** — existing firehouses + top demand-cell centroids (discrete
   sites only — not continuous zoning).

```mermaid
flowchart LR
  city[City registry] --> demand[Demand grid]
  city --> houses[Existing firehouses]
  city --> graph[Firetruck graph]
  hybrid[CAD travel calibration] --> T[Travel matrix]
  graph --> T
  demand --> opt[redesign / replace]
  houses --> opt
  T --> opt
  opt --> out[CSV + map + summary]
```

## Run

```bash
# Synthetic smoke test (no NYC data)
PYTHONPATH=src python scripts/plan_firehouse_locations.py --demo

# NYC redesign: optimally place the same number of houses
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city nyc --mode redesign --n-houses 48 --no-graph

# NYC replace: relocate 10 houses (keep total constant)
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city nyc --mode replace --replace 10 --hour 17

# Custom city
PYTHONPATH=src python scripts/plan_firehouse_locations.py \
  --city custom --firehouses path/houses.csv --incidents path/incidents.csv \
  --graph path/drive.graphml --mode redesign --n-houses 20
```

Omit `--no-graph` to score with firetruck Dijkstra (slower, more accurate).

## Outputs (`data/figures/firehouse_plan/`)

| File | Content |
|---|---|
| `01_metrics_comparison.png` | E[T] + coverage bars vs current / random |
| `02_response_cdf.png` | Demand-weighted response CDF |
| `03_response_surface_map.png` | Before/after response surfaces |
| `04_relocations.png` | Closed → opened moves |
| `05_site_demand_share.png` | Top sites by assigned demand |
| `firehouse_plan_map.html` | Interactive Folium (heat, Δ response, close/open) |
| `planned_firehouses.csv` | Selected sites + assigned demand share |
| `closed_firehouses.csv` | Existing sites dropped |
| `opened_firehouses.csv` | New sites opened |
| `demand_cells.csv` | Grid cells with planned / current response |
| `candidates.csv` | Full candidate pool |
| `summary.json` | Metrics + baselines + calibrator meta |

## Honesty limits

- **Travel only** — no chute/turnout, staffing, land cost, or politics.
- **Discrete candidates** — not a continuous land-use optimizer.
- Hybrid joblib calibrates scale; it does not invent full feature rows for
  hypothetical houses.
- Public CAD demand uses inferred destinations (alarm box / geocode / ZIP).

## Code

| Piece | Path |
|---|---|
| City registry | `src/emvro/cities.py` |
| Firetruck routing | `src/emvro/routing/firetruck.py` |
| Planner | `src/emvro/firehouse_location.py` |
| CLI | `scripts/plan_firehouse_locations.py` |
