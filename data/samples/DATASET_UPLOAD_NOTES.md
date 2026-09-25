# Sample upload pack (firetruck scope)

- `od_pairs_sample.csv`: 200 OD pairs (`start_mode=hybrid`)
- `fdny_firehouses_sample.csv`: FDNY firehouse depots
- `synthetic_csls_sample.csv`: volume-weighted alarm-box intersection CSLs

## Starting-location fix

Public FDNY CAD has no unit GPS at assignment. We infer starts as:
1. nearest FDNY firehouse in the incident borough
2. in `hybrid` mode, a synthetic CSL (alarm-box intersection in a high-volume ZIP) wins when closer to the destination than the firehouse (proxy for an already-on-the-road apparatus)

`start_source` / `depot_layer` / `start_mode` make the rule auditable per row.
