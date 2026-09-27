# Sample upload pack (firetruck scope)

- `od_pairs_sample.csv`: 200 OD pairs (`start_mode=first_due`)
- `fdny_firehouses_sample.csv`: FDNY firehouse depots
- `synthetic_csls_sample.csv`: volume-weighted alarm-box intersection CSLs

## Starting-location fix

Public FDNY CAD has no unit GPS at assignment. Default inference:
1. locate the incident at its in-service alarm-box coordinates
2. spatial-join to the first-due **engine** company polygon
3. map that engine number to its firehouse listing coordinates
4. fall back to nearest firehouse / CSL hybrid when the chain misses
5. QC via crow-flies implied speed + nearest-house check

`start_source` / `depot_layer` / `start_mode` / `qc_*` make the rule auditable per row.
