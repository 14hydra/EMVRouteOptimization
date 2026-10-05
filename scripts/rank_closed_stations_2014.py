#!/usr/bin/env python3
"""
Rank the ten stations closed in January 2014 by how much response time each
would save if put back, ONE AT A TIME, onto today's London network.

Response = Kolesar(crow km to nearest station) + DEFAULT_TURNOUT_S, evaluated on
a 2025 incident sample; Kolesar is fit on 2024 non-busy trips.
Coordinates are APPROXIMATE (`CLOSED_2014_APPROX_COORDS`).

HONESTY: this is a today's-worth ranking approximation. The experiment in the
planning doc needs a pre-2014 model (pre-2014 LFB extracts + the 2013 station
list), which is not in this repo; today's demand/network differ from 2013.

Run:  .venv/bin/python scripts/rank_closed_stations_2014.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.kolesar import fit_kolesar  # noqa: E402
from emvro.lfb_standards import DEFAULT_TURNOUT_S, FIRST_ENGINE_P90_S, FIRST_ENGINE_AVG_S  # noqa: E402
from emvro.london_closures_2014 import CLOSED_2014_APPROX_COORDS  # noqa: E402
from emvro.london_eval import haversine_km, load_london, nearest_station_km  # noqa: E402

OUT = ROOT / "data" / "figures" / "london_2014_closure_validation" / "closed_station_ranking.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-year", type=int, default=2025)
    ap.add_argument("--train-year", type=int, default=2024)
    ap.add_argument("--max-eval", type=int, default=40_000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    fh, inc = load_london(years=[args.train_year, args.eval_year])
    tr = inc[(inc.cal_year == args.train_year) & (inc.busy_flag == 0)]
    ev_all = inc[inc.cal_year == args.eval_year]
    ev = ev_all.sample(min(args.max_eval, len(ev_all)), random_state=42).reset_index(drop=True)
    kol = fit_kolesar(tr["crow_km"], tr["drive_s"])

    ilat, ilon = ev["dest_lat"].to_numpy(), ev["dest_lon"].to_numpy()
    slat, slon = fh["lat"].to_numpy(), fh["lon"].to_numpy()
    base_km = nearest_station_km(ilat, ilon, slat, slon)
    base = kol.predict(base_km) + DEFAULT_TURNOUT_S

    rows = []
    for name, (la, lo) in CLOSED_2014_APPROX_COORDS.items():
        d_new = haversine_km(ilat, ilon, la, lo)
        new_km = np.minimum(base_km, d_new)
        new = kol.predict(new_km) + DEFAULT_TURNOUT_S
        saved = base - new
        scale = len(ev_all) / len(ev)
        rows.append(
            {
                "station": name,
                "approx_lat": la,
                "approx_lon": lo,
                "mean_seconds_saved": float(saved.mean()),
                "total_hours_saved_2025_est": float(saved.sum() * scale / 3600),
                "incidents_improved_pct": float((saved > 1e-6).mean() * 100),
                "mean_saved_when_improved_s": float(saved[saved > 1e-6].mean()) if (saved > 1e-6).any() else 0.0,
                "p90_s_after": float(np.quantile(new, 0.9)),
                "within_6min_gain_pp": float(((new <= FIRST_ENGINE_AVG_S).mean() - (base <= FIRST_ENGINE_AVG_S).mean()) * 100),
                "nearest_current_station_km": float(haversine_km(la, lo, slat, slon).min()),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_seconds_saved", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["note"] = "approx coords; today's network+demand; Kolesar crow + 60s turnout; not a pre-2014 model"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Baseline current network (n={len(ev):,}): mean {base.mean():.1f}s, p90 {np.quantile(base, .9):.1f}s")
    print(out.drop(columns=["approx_lat", "approx_lon", "note"]).round(2).to_string(index=False))
    print(f"\nWrote {args.out}")
    print("HONESTY: today's-worth ranking approximation; the doc experiment needs a pre-2014 model "
          "(pre-2014 LFB extracts not in repo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
