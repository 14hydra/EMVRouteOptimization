# EMV Route Optimization (ASI 2026–27)

Emergency vehicle (EMV) travel-time prediction and route optimization for NYC — Ricky Zhao & Sarp Akalin (ASI).

## The starting-location problem

Public **FDNY Fire Incident Dispatch Data** (`8m42-w767`) has:

- approximate **incident** geography (borough, ZIP, alarm box)
- response / travel times

It does **not** publish the GPS of the apparatus at assignment time (privacy). Without a start point you cannot reconstruct the true dispatched route.

### Fix in this repo

**Scope: firetrucks only** (FDNY CAD). Ambulances / EMS stations / hospital bays are out of scope by default (legacy loaders remain behind `--legacy-ems`).

We reconstruct usable OD pairs by:

1. Treating each incident ZIP’s **MODZCTA centroid** as the destination.
2. Using `incident_borough` (and alarm-box borough when present) as the borough prior.
3. Choosing the nearest **FDNY firehouse** (`hc8x-tcnd`) in that borough.
4. In **hybrid** mode, using a **synthetic CSL** (FDNY alarm-box intersection in a high-volume ZIP) when it is closer than the best firehouse — a proxy for an already-on-the-road unit.
5. Flagging each row with `start_source` / `depot_layer` / `start_mode`.

For the **travel-time model**, we do not commit to one true start: we feed
`firehouse_km` / `csl_km` together so the model can learn times under origin
uncertainty (see `docs/travel_time_training.md`).

## Chosen datasets

| Role | Source | Open Data ID |
|------|--------|--------------|
| Incidents + travel times | FDNY Fire Incident Dispatch Data | `8m42-w767` |
| Preferred EMV starts | FDNY Firehouse Listing | `hc8x-tcnd` |
| CSL intersection proxies | In-Service Alarm Box Locations | `v57i-gtxb` |
| Incident ZIP geometry | Modified Zip Code Tabulation Areas (MODZCTA) | `pri4-ifjk` |
| Street network / widths | NYC LION | `2v4z-66xt` |
| Street attributes | OpenStreetMap (NYC extract) | OSM |
| Hourly weather | Open-Meteo archive (NYC) | no key; cached under `data/processed/weather_hourly_nyc.csv` |
| Congestion prior | Hour-of-day BPR-style multiplier | in routing conditions |

## Quick start

```bash
cd EMVRouteOptimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download FDNY samples + build inferred OD pairs
python scripts/download_datasets.py --limit 5000
python scripts/build_od_pairs.py --start-mode hybrid

# Travel-time model (slides Step 4) — firetrucks only
PYTHONPATH=src python scripts/build_osm_graph.py   # once
PYTHONPATH=src python scripts/build_travel_time_training_set.py
PYTHONPATH=src python scripts/train_travel_time_model.py --feature-set full

# Optimal patrol posts / loops (mentor ask)
PYTHONPATH=src python scripts/build_patrol_routes.py
```

See `docs/starting_locations.md`, `docs/travel_time_training.md`, and `docs/patrol_routes.md`.

## What differentiates this from prior EMV projects

1. **Missing-start CAD** — hybrid firehouse + on-road CSL inference for real NYC open data (no unit GPS).
2. **Civilian control vs EMV privileges** — Google Maps / civilian graph as control; EMV ROW (busways, limited contraflow) on the same clock.
3. **Prescriptive patrol** — same layers as inferred starts, flipped forward: where *should* apparatus sit before the next call.
4. **Honest limits** — ZIP-centroid destinations, weak street/weather correlations documented, not oversold.
