# Multi-city fire / EMV datasets

Public CAD sources that can expand training beyond NYC. Prefer cities with
**coordinates + response→on-scene timestamps** (or clear dispatch→arrival).

## Priority sources

| City | Dataset | Why it helps | Status in repo |
|---|---|---|---|
| **San Francisco** | [Fire/EMS Calls for Service](https://data.sf.gov/resource/nuek-vuh3.json) (`nuek-vuh3`) | Medical incidents, `response_dttm`→`on_scene_dttm`, `case_location` lon/lat | `scripts/ingest_sf_ems.py` → `data/processed/od_pairs_sf_ems.csv` |
| **Dublin** | [DFB Ambulance Incidents](https://data.smartdublin.ie/dataset/fire-brigade-and-ambulance) GeoJSON | TOC/ORD/MOB/IA timestamps + geometry | Planned: `scripts/ingest_dublin_ems.py` |
| **Chicago** | [EMS Calls](https://data.cityofchicago.org/) (search EMS / ambulance) | Large US CAD volume | Candidate |
| **Seattle** | SPD/SFD 911 (data.seattle.gov) | West-coast network diversity | Candidate |
| **Virginia Beach** | [EMS Calls for Service](https://data.virginia.gov/dataset/ems-calls-for-service) | Full CAD lifecycle; geocode addresses | Candidate |

## Target schema (compatible with NYC OD table)

Required for route-ETA / multi-city merge:

- `city` (e.g. `nyc`, `san_francisco`)
- `start_lat`, `start_lon`, `dest_lat`, `dest_lon`
- `travel_seconds` (on-road response travel when available)
- `hour`, `dow` (optional but useful)
- `borough` / neighborhood (optional categorical)
- `depot_layer`, `depot_name`, `crow_flies_km`

## How new cities plug into training

1. Ingest → `data/processed/od_pairs_<city>.csv`
2. Build / cache an OSM drive graph for that city’s bbox (`street_features.build_drive_graph`)
3. Append rows into `route_eta_training.csv` with `city` categorical
4. Optional: `enrich_training_with_gmaps.py` on the new OD sample
5. Retrain: `scripts/train_route_eta_model.py`

CAD-level R² will stay noisy without unit GPS; the payoff is **more diverse route geometry + traffic regimes** for the route-ETA / GBDT edge models.

## Quick start (SF → multi-city R² ≥ 0.9)

```bash
PYTHONPATH=src python scripts/ingest_sf_ems.py --limit 4000
# Builds SF OSM graph, network features, EMV-mapped labels (not raw CAD alone)
PYTHONPATH=src python scripts/build_multicity_route_eta.py
PYTHONPATH=src python scripts/train_route_eta_model.py \
  --data data/processed/route_eta_multicity_training.csv --target-r2 0.9
```

Raw SF CAD travel times alone do **not** support R² ≥ 0.9 (distance explains ~0% of
observed travel variance). Multi-city training uses the same OSM→EMV label mapping
as NYC, with a light blend toward observed EMS times only when the ratio is plausible.
