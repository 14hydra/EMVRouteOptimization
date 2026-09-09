# EMV Route Optimization (ASI 2026–27)

Emergency vehicle (EMV) travel-time prediction and route optimization for NYC — Ricky Zhao & Sarp Akalin (ASI).

## The starting-location problem

Public **EMS Incident Dispatch Data** (`76xm-jjuj`) has:

- approximate **incident** geography (borough, ZIP, dispatch area)
- response / travel times

It does **not** publish the GPS of the unit at assignment time (HIPAA / privacy). Without a start point you cannot reconstruct the true dispatched route.

### Fix in this repo

We reconstruct usable OD pairs by:

1. Treating each incident ZIP’s **MODZCTA centroid** as the destination.
2. Mapping `incident_dispatch_area` (e.g. `K6`, `B1`) to a borough sector.
3. Choosing the nearest **EMS/ambulance station** or **hospital bay** (HOSPITAL / ACUTE CARE HOSPITAL from City Facilities — covers NYC H+H and voluntary hospitals) in that borough.
4. Falling back to the nearest **FDNY firehouse** (`hc8x-tcnd`).
5. In **hybrid** mode, using a **synthetic CSL** (FDNY alarm-box intersection in a high-volume ZIP) when it is closer than the best static depot — a proxy for an already-on-the-road unit.
6. Flagging each row with `start_source` / `depot_layer` / `start_mode`.

## Chosen datasets

| Role | Source | Open Data ID |
|------|--------|--------------|
| Incidents + travel times | NYC EMS Incident Dispatch Data | `76xm-jjuj` |
| Preferred EMV starts | City Facilities — EMS/ambulance stations | `ji82-xba5` |
| Hospital bay starts | City Facilities — HOSPITAL / ACUTE CARE HOSPITAL | `ji82-xba5` |
| CSL intersection proxies | In-Service Alarm Box Locations | `v57i-gtxb` |
| Fallback starts | FDNY Firehouse Listing | `hc8x-tcnd` |
| Incident ZIP geometry | Modified Zip Code Tabulation Areas (MODZCTA) | `pri4-ifjk` |
| Street network / widths | NYC LION | `2v4z-66xt` |
| Street attributes | OpenStreetMap (NYC extract) | OSM |

## Quick start

```bash
cd EMVRouteOptimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download samples + build inferred OD pairs
python scripts/download_datasets.py --limit 5000
python scripts/build_od_pairs.py --start-mode hybrid

# Optional: Google Maps civilian control → label station-plausible vs on-road
export GOOGLE_MAPS_API_KEY=your_key_here
PYTHONPATH=src python scripts/classify_origins_with_gmaps.py --limit 200
PYTHONPATH=src python scripts/visualize_data.py
```

Figures land in `data/figures/` (PNGs + interactive `emv_nyc_map.html`). See `docs/gmaps_control.md` for the station vs on-road classification rule.

## Layout

```
data/raw/          downloaded Open Data extracts
data/processed/    cleaned tables + inferred starts
data/samples/      small CSVs safe to share / upload to Gemini
src/emvro/         Python package (download, geometry, depot inference)
scripts/           CLI entrypoints
docs/              dataset notes + methodology
```

## Bridge / Gemini ASI chat

This project is coordinated with the Gemini **ASI** chat via `cursor-gemini-bridge`.
Sample CSVs under `data/samples/` are meant for upload into that chat.
