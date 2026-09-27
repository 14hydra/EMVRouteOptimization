#!/usr/bin/env python3
"""
How much does training-set size matter? Train the previous design and the new
feature set on random subsamples of 2024 (down to the 2,692 rows the old model
had) and score each on the same Jan–Mar 2025 test window.

  PYTHONPATH=src python scripts/travel_time_learning_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_travel_time_network import STEPS, add_priors, fit, matrix  # noqa: E402

SIZES = [2_692, 10_000, 30_000, 100_000, 200_000, None]  # None = all of 2024
PAIRS = ["1 Previous design", "7 + traffic congestion"]
OUT = ROOT / "data" / "figures" / "model_eval_network"


def main():
    d = pd.read_parquet(ROOT / "data" / "processed" / "travel_time_network.parquet")
    d["dest_key"] = d["dest_lat"].round(5).astype(str) + "," + d["dest_lon"].round(5).astype(str)
    full = d[d["incident_datetime"] < "2025-01-01"].reset_index(drop=True)
    te = d[d["incident_datetime"] >= "2025-01-01"].reset_index(drop=True)
    summary = json.loads((OUT / "summary.json").read_text())
    y_te = te["travel_seconds"].to_numpy()
    rows = []
    for n in SIZES:
        sub = full if n is None else full.sample(n=n, random_state=0).reset_index(drop=True)
        for step in PAIRS:
            spec = STEPS[step]
            ch = summary["chosen"][step]
            # Rounds tuned on the full set over-fit small samples; scale them down.
            rounds = ch["rounds"] if n is None else max(60, int(ch["rounds"] * min(1.0, (n / len(full)) ** 0.5)))
            sp, (tp,) = add_priors(sub, [te], spec["prior_keys"]) if spec["prior_keys"] else (sub, [te])
            cats = {c: pd.Categorical(full[c].astype(str)).categories for c in spec["cat"]}
            m, inv = fit(
                matrix(sp, spec["num"], spec["cat"], spec["prior_keys"], cats),
                sub["travel_seconds"].to_numpy(),
                objective=ch["objective"],
                rounds=rounds,
            )
            p = inv(m.predict(matrix(tp, spec["num"], spec["cat"], spec["prior_keys"], cats)))
            mae = float(np.abs(p - y_te).mean())
            r2 = float(1 - np.sum((p - y_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2))
            rows.append({"n_train": len(sub), "model": step, "mae_s": mae, "r2": r2})
            print(f"n={len(sub):>7,}  {step:28s}  MAE {mae:6.1f}s  R² {r2:.3f}", flush=True)
    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "learning_curve.csv", index=False)

    from travel_time_network_plots import learning_curve

    learning_curve(lc, OUT)


if __name__ == "__main__":
    main()
