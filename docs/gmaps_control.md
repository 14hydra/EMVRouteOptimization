# Google Maps as civilian control model

## Idea

Public EMS CAD has travel times but no unit start GPS. Units are often already
**on the road** (CSL staging), not at a station.

Use **Google Maps driving ETAs** as the stand-in **control model** (civilian GPS):

1. Nearest official depot (EMS station / hospital bay, else firehouse) → incident ZIP centroid  
2. Ask Google Maps for civilian `duration` / `duration_in_traffic`  
3. Scale by an EMV speedup factor (default `0.75` = ~25% faster than cars)  
4. Compare to observed `incident_travel_tm_seconds_qy`

### Classification rule

```
expected_emv_from_station = gmaps_station_seconds * emv_speed_factor

if expected_emv_from_station > actual_travel_seconds * slack:
    origin_class = on_road_required   # even an EMV from the station couldn't make it
else:
    origin_class = station_plausible  # station origin is compatible with the control
```

Optionally also query GMaps from the nearest synthetic CSL and set
`best_match_origin` to whichever EMV-adjusted ETA is closer to the observed time.

This gives a **data-driven** split between official-depot vs on-road starts, instead of
only using crow-flies hybrid heuristics.

## Setup

1. Google Cloud project → enable **Directions API**  
2. Create an API key  
3. Export it (do not commit the key):

```bash
export GOOGLE_MAPS_API_KEY=your_key_here
# or copy .env.example → .env and load it yourself
```

## Run

```bash
# Needs existing OD pairs from scripts/build_od_pairs.py
PYTHONPATH=src python scripts/classify_origins_with_gmaps.py --limit 200
```

Outputs:

- `data/processed/origin_class_gmaps_control.csv`
- `data/processed/origin_class_gmaps_summary.json`
- `data/samples/origin_class_gmaps_sample.csv`
- `data/processed/gmaps_cache.json` (cached Directions responses)

## Tunables

| Flag | Default | Meaning |
|------|---------|---------|
| `--emv-speed-factor` | `0.75` | EMV vs civilian (`<1` = faster) |
| `--slack` | `1.15` | Tolerance on observed time |
| `--limit` | `200` | Cap API calls while testing |
| `--no-csl` | off | Skip CSL GMaps queries |

## Limits

- Google Maps is **civilian** routing (no bus lanes / contraflow) — that is intentional for the control.
- EMV speedup is a constant factor, not street-level yet.
- Destinations are ZIP centroids, not exact addresses (HIPAA aggregation).
- API usage costs money; keep `--limit` low until you’re ready to scale.
