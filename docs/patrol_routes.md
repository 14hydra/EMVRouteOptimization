# Optimal EMV patrol posts and patrol loops

## Scope

**Pre-positioning, not dispatch.** The route models (`docs/route_models.md`) answer *"given a
call, what is the fastest path?"*. This answers *"before any call comes in, where should
firetrucks be?"* — the question the industry mentor asked as "show optimal patrol routes so
EMVs can respond faster".

Code: `src/emvro/patrol.py` + `scripts/build_patrol_routes.py`.

## How it pairs with inferred starting locations

`docs/starting_locations.md` infers a *descriptive* start per incident (nearest FDNY firehouse,
or a synthetic alarm-box CSL when it is closer — `start_mode=hybrid`). Patrol
posts are the *prescriptive* version of the same idea:

| Inferred starts (`depots.py` / `csl.py`) | Patrol plan (`patrol.py`) |
|---|---|
| assumes a unit *happened* to be at a CSL | says *which* CSL-like posts to hold |
| one start per historical incident | k posts for the whole demand surface |
| descriptive (rebuild OD pairs) | prescriptive (where to stage next shift) |

Concretely the same layers are reused: `fdny_firehouse` are the candidate **home
bases / anchors** (a crew can actually be relieved there), `csl` points are on-road **post**
candidates, and the top demand-cell centroids are added so the optimizer is not limited to
legacy station real estate.

## Objective

Historical FDNY incident destinations are binned into a demand grid:

- `w_c` — incident weight of cell `c` (count; `--severity-weighted` up-weights acute calls)
- `t(p, c)` — EMV travel seconds from post `p` to cell `c`
- `P` — the `--posts k` locations chosen from the candidate set

```
--objective response_time   minimize  E[T] = Σ w_c · min_{p∈P} t(p,c) / Σ w_c      (p-median)
--objective coverage        maximize  C(T) = Σ w_c · 1[min_p t(p,c) ≤ T] / Σ w_c   (max covering)
```

Both are NP-hard, so each is solved with a greedy seed plus a bounded k-medoids swap pass.
Units are then anchored (p-median over post demand, facility layers only), posts are assigned
to units demand-balanced, and each unit's loop is ordered with nearest-neighbour + 2-opt.

Only **travel** time is modelled: no call-processing, no chute/turnout, no queueing.

## Travel time: two stages

1. **Screening** — calibrated crow-flies surrogate (`crow_km × detour / speed`). Runs with no
   OSM graph at all, so the whole pipeline degrades gracefully.
2. **Scoring + geometry** — Dijkstra on the EMV edge costs from `emvro.routing`
   (`prepare_routing_graph`, `--hour` congestion). Posts are **re-picked** on these exact times,
   and every posture (plan and baselines) is scored on the same matrix so comparisons are fair.
   Pairs beyond `--threshold-min × --cutoff-factor` fall back to the surrogate; `summary.json`
   reports what share of the matrix the graph resolved.

Note on calibration: fitting the effective speed from the OD table is **opt-in**
(`--calibrate-speed`) and usually rejected. Inferred starts sit a median ~0.3 km from a
ZIP-centroid destination while observed travel time is ~8 min, so the implied ~6 kph is an
artifact of the start inference, not a driving speed. The default is a 28 kph prior.

## Baselines

| Posture | Meaning |
|---|---|
| `patrol_plan` | the selected posts |
| `best_k_facilities_only` | **equal-cost**: same unit count, existing facilities only |
| `all_facilities_stationary` | context only: a staffed unit at every FDNY firehouse |

`best_k_facilities_only` is the honest headline comparison — it isolates the value of going
on-road rather than the value of adding trucks.

## Run

```bash
PYTHONPATH=src python scripts/build_patrol_routes.py                       # NYC + OSM graph (~40 s)
PYTHONPATH=src python scripts/build_patrol_routes.py --posts 30 --units 8
PYTHONPATH=src python scripts/build_patrol_routes.py --objective coverage
PYTHONPATH=src python scripts/build_patrol_routes.py --no-graph            # crow-flies only (~2 s)
PYTHONPATH=src python scripts/build_patrol_routes.py --demo                # synthetic demand
```

Outputs in `data/figures/patrol/`:

| File | Contents |
|---|---|
| `patrol_posts.csv` | post, unit, loop position, layer, assigned demand share, mean response, coverage |
| `patrol_segments.csv` | one row per loop leg: travel seconds, km, geometry source, `path_latlons` |
| `patrol_demand_cells.csv` | demand grid with nearest post, response seconds, coverage flag |
| `patrol_map.html` | Folium map: demand heat, response surface, posts, loops, home bases |
| `summary.json` | objective, parameters, search trace, metrics vs baselines, limits |

## Current result (24 posts, 6 units, 8-min threshold, hour 17)

Expected response 4.08 min (p90 7.11 min), 92.1% of demand within 8 min, versus 4.26 min for
the best 24 existing facilities — about 4% faster for the same unit count. The gain is modest
because destinations are ZIP centroids: there are only ~174 distinct demand points citywide,
and 123 facilities already sit among them. `--objective coverage` trades E[T] for reach and
gets 97.2% within 8 minutes.

## Limits / next upgrades

- Demand resolution is ZIP-centroid limited (HIPAA). Finer `--cell-km` adds nothing until
  block-level incident geography is available.
- Static demand: one plan per run. Next: hour-of-day and day-of-week re-posting (the
  `conditions` module already has the congestion schedule).
- No queueing: response assumes the nearest post has a free unit. Next: a busy-probability
  (hypercube / Larson-style) correction.
- Response is measured from a post; a unit caught mid-leg is slower than reported. Next: score
  the loop as a distribution of positions rather than the post alone.
