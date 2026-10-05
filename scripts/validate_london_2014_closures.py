#!/usr/bin/env python3
"""
Validation-plan scaffold: London 2014 closures, using TODAY's data only.

What this does
  1. Loads current London firehouses + planner incidents; reports which of the
     ten CLOSED_2014 stations still appear by name (most will not).
  2. Counterfactual closure test on 2025 eval incidents (<= 40k sample):
       * CURRENT network (post-closure stations) vs
       * CURRENT minus simulated closures: stations that fuzzy-match a closed
         name are removed; plus (always) the 10 current stations nearest to the
         APPROXIMATE hand-entered closed-station coordinates are removed
         (`CLOSED_2014_APPROX_COORDS`, ~300-500 m accuracy, not surveyed).
       * RESTORED network = current + the 10 closed stations put back at
         their approximate coords (the 'pre-2014-like' side of the contrast).
     Response = Kolesar(crow km to nearest station) + DEFAULT_TURNOUT_S,
     scored with LFB first-engine standards.
  3. Held-out check: Kolesar fit on 2024 non-busy trips (crow km from the
     deploying station) -> MAE vs observed 2025 attendance (drive+default turnout).

HONESTY: the real 2014 pre/post train test needs pre-2014 LFB incident extracts
(and the 2013 station list), which are NOT in this repo. Everything here is a
today's-data approximation of that experiment.

Run:  .venv/bin/python scripts/validate_london_2014_closures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.kolesar import fit_kolesar  # noqa: E402
from emvro.lfb_standards import DEFAULT_TURNOUT_S, score_first_engine  # noqa: E402
from emvro.london_closures_2014 import (  # noqa: E402
    CLOSED_2014,
    CLOSED_2014_APPROX_COORDS,
    match_closed,
)
from emvro.london_eval import haversine_km, load_london, nearest_station_km  # noqa: E402

OUT_DIR = ROOT / "data" / "figures" / "london_2014_closure_validation"


def _pred(model, inc, lat, lon):
    km = nearest_station_km(inc["dest_lat"].to_numpy(), inc["dest_lon"].to_numpy(), lat, lon)
    return model.predict(km) + DEFAULT_TURNOUT_S


def _score(x):
    return {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in score_first_engine(x).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-year", type=int, default=2025)
    ap.add_argument("--train-year", type=int, default=2024)
    ap.add_argument("--max-eval", type=int, default=40_000)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "summary.json")
    args = ap.parse_args()

    fh, inc = load_london(years=[args.train_year, args.eval_year])
    tr = inc[(inc.cal_year == args.train_year) & (inc.busy_flag == 0)]
    ev_all = inc[inc.cal_year == args.eval_year]
    ev = ev_all.sample(min(args.max_eval, len(ev_all)), random_state=42).reset_index(drop=True)
    print(f"firehouses={len(fh)}  train(non-busy {args.train_year})={len(tr):,}  "
          f"eval {args.eval_year}: {len(ev_all):,} (sampled {len(ev):,})")

    # 1) name match status
    matches = match_closed(fh["facilityname"].tolist())
    print("\n[1] CLOSED_2014 presence in today's firehouse list (match_closed):")
    for name in CLOSED_2014:
        hit = [k for k, v in matches.items() if v == name]
        print(f"    {name:<14} {'STILL LISTED as ' + ', '.join(hit) if hit else 'not in current list (closed)'}")

    # 3) Kolesar fit on train
    kol = fit_kolesar(tr["crow_km"], tr["drive_s"])
    print("\n    Kolesar (km, s) fit on 2024 non-busy:", {k: round(v, 3) for k, v in kol.to_dict().items()})

    # Held-out MAE on 2025 (observed attendance = drive + default turnout)
    def mae_block(df):
        obs = df["drive_s"].to_numpy()
        base = {
            "median_train": np.full(len(df), float(np.median(tr["drive_s"]))),
            "crow_32kph": np.clip(df["crow_km"].to_numpy() / 32.0 * 3600, *(30, 1200)),
            "kolesar": kol.predict(df["crow_km"].to_numpy()),
        }
        return {
            "n": int(len(df)),
            **{f"mae_s_{k}": float(np.mean(np.abs(obs - v))) for k, v in base.items()},
            **{f"bias_s_{k}": float(np.mean(v - obs)) for k, v in base.items()},
        }

    heldout = {"all_2025": mae_block(ev_all), "non_busy_2025": mae_block(ev_all[ev_all.busy_flag == 0])}
    print("\n[3] Held-out MAE vs observed drive (= attendance - 60 s turnout), 2025:")
    for k, v in heldout.items():
        print(f"    {k:<14} n={v['n']:>6,}  MAE median={v['mae_s_median_train']:.1f}s  "
              f"crow@32kph={v['mae_s_crow_32kph']:.1f}s  Kolesar={v['mae_s_kolesar']:.1f}s")

    # 2) counterfactual networks
    st_lat, st_lon = fh["lat"].to_numpy(), fh["lon"].to_numpy()
    drop = set(fh.index[fh["facilityname"].isin(matches)])
    name_removed = fh.loc[sorted(drop), "facilityname"].tolist()
    # nearest current station to each approximate closed coord (greedy, unique)
    near_removed = []
    taken = set(drop)
    for name, (la, lo) in CLOSED_2014_APPROX_COORDS.items():
        d = haversine_km(la, lo, st_lat, st_lon)
        for j in np.argsort(d):
            if int(j) not in taken:
                taken.add(int(j))
                near_removed.append({"closed": name, "removed_station": fh.loc[j, "facilityname"], "km": float(d[j])})
                break
    keep = np.ones(len(fh), bool)
    keep[list(taken)] = False
    rest_lat = np.append(st_lat, [c[0] for c in CLOSED_2014_APPROX_COORDS.values()])
    rest_lon = np.append(st_lon, [c[1] for c in CLOSED_2014_APPROX_COORDS.values()])

    nets = {
        "current": _pred(kol, ev, st_lat, st_lon),
        "current_minus_10_nearest": _pred(kol, ev, st_lat[keep], st_lon[keep]),
        "restored_plus_10_closed": _pred(kol, ev, rest_lat, rest_lon),
    }
    observed = ev["travel_seconds"].to_numpy()
    scores = {k: _score(v) for k, v in nets.items()}
    scores["observed_2025"] = _score(observed)

    print("\n[2] LFB first-engine score (Kolesar crow + 60 s turnout), 2025 sample:")
    print(f"    {'network':<26}{'mean_s':>9}{'p90_s':>9}{'<=6min':>9}{'<=10min':>9}  passes")
    for k, s in scores.items():
        print(f"    {k:<26}{s['mean_s']:>9.1f}{s['p90_s']:>9.1f}{s['share_within_6min']:>9.3f}"
              f"{s['share_within_10min']:>9.3f}  {s['passes']}")
    d_close = scores["current_minus_10_nearest"]["mean_s"] - scores["current"]["mean_s"]
    d_restore = scores["current"]["mean_s"] - scores["restored_plus_10_closed"]["mean_s"]
    print(f"\n    Removing 10 more nearest stations: mean +{d_close:.1f} s; "
          f"restoring the 10 closed: mean {-d_restore:+.1f} s")
    print("    Removed (nearest to approx closed coords):",
          "; ".join(f"{r['closed']}->{r['removed_station']} ({r['km']:.1f} km)" for r in near_removed))

    summary = {
        "n_firehouses_current": int(len(fh)),
        "closed_2014_name_matches_in_current_list": {k: v for k, v in matches.items()},
        "stations_removed_by_name": name_removed,
        "stations_removed_by_proximity_to_approx_coords": near_removed,
        "kolesar_fit_2024_non_busy_km_s": kol.to_dict(),
        "heldout_2025_drive_mae": heldout,
        "n_eval_sample": int(len(ev)),
        "turnout_s_assumed": DEFAULT_TURNOUT_S,
        "lfb_scores_2025_sample": scores,
        "delta_mean_s_remove_10_nearest": float(d_close),
        "delta_mean_s_restore_10_closed": float(-d_restore),
        "closed_station_coords_approx": {k: list(v) for k, v in CLOSED_2014_APPROX_COORDS.items()},
        "honesty": (
            "Scaffold on TODAY's network only. Closed-station coords are approximate hand-entered values. "
            "Crow-fly Kolesar ignores the road network and real dispatch (nearest-station assumption, no "
            "busy/availability). The true 2014 pre/post train test requires pre-2014 LFB incident extracts "
            "(and the 2013 station list) that are not yet in this repo."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    print("\nHONESTY: full 2014 pre/post train test needs pre-2014 LFB extracts not yet in repo; "
          "this is a today's-data counterfactual with approximate closed-station coords.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
