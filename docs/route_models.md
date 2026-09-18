# Route optimization models (slides Step 3)

Three main testing models from the project slides, plus civilian controls.

| Model | Module | Idea (first draft) |
|---|---|---|
| **1. MIPSSTW + MCS** | `emvro.routing.mipsstw_mcs` | Minimize EMV time with a **semi-soft time window** penalty; solve with **Modified Cuckoo Search** (Lévy path perturbations + nest abandonment) |
| **2. Composite DRL** | `emvro.routing.composite_drl` | Tabular Q-learning with a **composite reward**: −EMV seconds + primary/bus-lane bonuses − backtrack penalty |
| **3. GBDT router** | `emvro.routing.gbdt_router` | LightGBM predicts **edge EMV costs** from street + hour features → Dijkstra |
| Control | `emvro.routing.graph` | Shortest distance / civilian congested time |

Travel-time scoring for evaluation still uses the separate route-ETA model
(`docs/travel_time_training.md`).

## Run the demo

```bash
# needs data/raw/nyc_drive.graphml
PYTHONPATH=src python scripts/run_route_models.py \
  --origin-lat 40.758 --origin-lon -73.985 \
  --dest-lat 40.712 --dest-lon -74.006 \
  --hour 17
```

Output: `data/processed/route_models_demo.json`

## Visualizations

```bash
PYTHONPATH=src python scripts/visualize_route_models.py
# faster (charts only):  PYTHONPATH=src python scripts/visualize_route_models.py --skip-map
```

Figures land in `data/figures/route_models/`:

- `00_dashboard.png` — overview panel
- `01_travel_time_comparison.png`
- `02_distance_comparison.png`
- `03_main_models_focus.png` — three main models vs civilian
- `04_time_vs_distance.png`
- `05_mcs_fitness.png`
- `06_drl_learning_curve.png`
- `route_models_map.html` — **multiple OD trips** across NYC; toggle models in the layer control
- `route_conditions_map.html` — **time slider + weather buttons** swap the full route set
  (civilian vs EMV) for that hour/weather; gold = Google-untakeable EMV corridors
- `row_wins_map.html` — **verified EMV ROW wins**: OD cases where EMV beats civilian GPS
  *because* of Google-untakeable corridors (contraflow/busway); see `row_wins.csv`
- `multi_route_times.csv` — per-trip travel times for the map scenarios

## Traffic & weather conditions (ASI eval variables)

ASI Gemini evaluation called out **clear vs rain/snow**, **congestion**, and
**bus-lane / ROW** contrasts. Those are wired into routing edge costs:

| Regime | How it enters the graph |
|---|---|
| Traffic amount | Hour congestion prior (`clear_offpeak`, `clear_rush`, `night_clear`) — TomTom live flow can replace the prior later |
| Weather | Open-Meteo factors (precip / snow / visibility / wind) from presets or `weather_hourly_nyc.csv` |
| Bus lanes / ROW | EMV discounts on primary/trunk + real `busway` tags; discount **grows** with congestion |
| Contraflow (EMV-only) | Short reverse edges on primary/trunk one-ways (≤120 m; no motorways). **Cost penalty** (~+25% + 8 s risk) so models only use it when the save is worth it — not free wrong-way speedups |

Civilian GPS takes the full congestion × weather hit. EMV models take a reduced
weather penalty on legal arterials/busways. Contraflow is harder in bad weather
and is never given the congestion ROW bonus.

```bash
PYTHONPATH=src python scripts/run_condition_scenarios.py
# also pull harshest hours from cached Open-Meteo:
PYTHONPATH=src python scripts/run_condition_scenarios.py --with-open-meteo
```

Figures: `data/figures/route_models/conditions/`

Presets live in `emvro.routing.conditions.CONDITION_PRESETS`.

## Upgrade path

1. **MIPSSTW** — replace MCS with a true MILP on a corridor subgraph (PuLP/CBC); wire BPR + intersection delay models from the paper.
2. **Composite DRL** — swap Q-table for DQN / graph neural policy; multi-agent signal preemption (EMVLight-style).
3. **GBDT** — train edge costs on labeled paths from the route-ETA table / LION attributes; add weather features.
4. Batch-eval all three + controls on held-out EMS OD samples; report % time saved (slides Step 5).
