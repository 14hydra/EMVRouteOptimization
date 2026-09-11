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

- `01_travel_time_comparison.png`
- `02_distance_comparison.png`
- `03_pct_vs_civilian.png`
- `04_time_vs_distance.png`
- `05_mcs_fitness.png`
- `route_models_map.html` (paths overlaid)

## Upgrade path

1. **MIPSSTW** — replace MCS with a true MILP on a corridor subgraph (PuLP/CBC); wire BPR + intersection delay models from the paper.
2. **Composite DRL** — swap Q-table for DQN / graph neural policy; multi-agent signal preemption (EMVLight-style).
3. **GBDT** — train edge costs on labeled paths from the route-ETA table / LION attributes; add weather features.
4. Batch-eval all three + controls on held-out EMS OD samples; report % time saved (slides Step 5).
