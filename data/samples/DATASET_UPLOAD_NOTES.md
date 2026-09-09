# Sample upload pack for Gemini ASI chat

- `od_pairs_sample.csv`: 150 OD pairs (`start_mode=hybrid`)
- `ems_stations_sample.csv`: dedicated EMS/ambulance stations
- `hospital_bays_sample.csv`: HOSPITAL / ACUTE CARE HOSPITAL (H+H + voluntary)
- `synthetic_csls_sample.csv`: volume-weighted alarm-box intersection CSLs
- `fdny_firehouses_sample.csv`: firehouse fallback

## Starting-location fix

Public EMS CAD has no unit GPS at assignment. We infer starts as:
1. nearest EMS station or hospital bay in the dispatch-area borough
2. else nearest FDNY firehouse
3. in `hybrid` mode, a synthetic CSL (alarm-box intersection in a high-volume ZIP) wins when closer to the destination than the best static depot (proxy for an already-on-the-road unit)

`start_source` / `depot_layer` / `start_mode` make the rule auditable per row.
