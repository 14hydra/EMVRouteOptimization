#!/usr/bin/env python3
"""
Train and evaluate the FDNY travel-time model on the network-origin table.

Honest evaluation: a *time* split — train on Jan–Nov 2024, tune on Dec 2024,
then refit on all of 2024 and test once on Jan–Mar 2025 (future incidents).
No feature uses the label of its own row: location priors are out-of-fold
inside the training window and strictly past data for the test window.

An ablation ladder on the same test rows shows what each fix buys:

  0  predict the training median
  1  previous design — crow-flies from one guessed house, ZIP-centroid
     destination when the box is unlisted (retrained on the same big data)
  2  + geocoded destinations (unlisted boxes located on NYC Centerline)
  3  + street-network origin features (first-due engine/ladder, nearest houses)
  4  + historical location priors (past travel times at the same box / company)

  PYTHONPATH=src python scripts/train_travel_time_network.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402

from emvro.network_origins import NETWORK_FEATURE_COLUMNS  # noqa: E402
from emvro.weather import WEATHER_FEATURE_COLUMNS  # noqa: E402

CONTEXT = [
    "hour", "dow", "month", "minute_of_day", "is_weekend", "is_rush", "is_night",
    "dispatch_wait_seconds", "engines_assigned", "ladders_assigned", "other_units_assigned",
] + WEATHER_FEATURE_COLUMNS
CATS_CONTEXT = ["borough", "call_type", "call_group", "alarm_level"]

STEPS = {
    "1 Previous design": {
        "num": ["prev_crow_km_primary", "prev_crow_km_nearest", "prev_dest_lat", "prev_dest_lon", "prev_dest_is_box"] + CONTEXT,
        "cat": CATS_CONTEXT,
        "prior_keys": [],
    },
    "2 + geocoded destinations": {
        "num": ["crow_km_engine", "crow_km_nearest", "dest_lat", "dest_lon"] + CONTEXT,
        "cat": CATS_CONTEXT + ["dest_source"],
        "prior_keys": [],
    },
    "3 + network origins": {
        "num": NETWORK_FEATURE_COLUMNS + ["dest_lat", "dest_lon"] + CONTEXT,
        "cat": CATS_CONTEXT + ["dest_source", "first_due_engine", "first_due_ladder"],
        "prior_keys": [],
    },
    "4 + location priors": {
        "num": NETWORK_FEATURE_COLUMNS + ["dest_lat", "dest_lon"] + CONTEXT,
        "cat": CATS_CONTEXT + ["dest_source", "first_due_engine", "first_due_ladder"],
        "prior_keys": ["dest_key", "first_due_engine", "zipcode"],
    },
}

LOAD = [
    f"{area}_{c}"
    for area in ("engine", "ladder", "borough")
    for c in ("n_prev_15m", "n_prev_30m", "n_prev_60m", "min_since_prev")
]
STEPS["5 + unit-availability load"] = {
    "num": NETWORK_FEATURE_COLUMNS + ["dest_lat", "dest_lon"] + CONTEXT + LOAD,
    "cat": CATS_CONTEXT + ["dest_source", "first_due_engine", "first_due_ladder"],
    "prior_keys": ["dest_key", "first_due_engine", "zipcode"],
}

from emvro.route_street_features import ROUTE_STREET_COLUMNS  # noqa: E402
from emvro.traffic import TRAFFIC_FEATURE_COLUMNS  # noqa: E402

_prev = STEPS["5 + unit-availability load"]
STEPS["6 + route street features"] = dict(_prev, num=_prev["num"] + ROUTE_STREET_COLUMNS)
STEPS["7 + traffic congestion"] = dict(_prev, num=_prev["num"] + ROUTE_STREET_COLUMNS + TRAFFIC_FEATURE_COLUMNS)

PARAMS = dict(
    learning_rate=0.1,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    cat_smooth=20,
    verbose=-1,
    seed=42,
)
SMOOTH = 20.0  # prior strength (incidents) for location priors


# --------------------------------------------------------------------------
# Location priors (target encodings that never see their own row / the future)
# --------------------------------------------------------------------------
def _encode(train_keys, train_y, apply_keys, glob: float) -> tuple[np.ndarray, np.ndarray]:
    g = pd.DataFrame({"k": train_keys, "y": train_y}).groupby("k")["y"].agg(["sum", "count"])
    s = pd.Series(apply_keys).map(g["sum"]).fillna(0).to_numpy()
    n = pd.Series(apply_keys).map(g["count"]).fillna(0).to_numpy()
    return (s + SMOOTH * glob) / (n + SMOOTH), n


def add_priors(train: pd.DataFrame, other: list[pd.DataFrame], keys: list[str], n_folds: int = 5):
    """Out-of-fold priors for ``train``; full-train priors for each frame in ``other``."""
    y = np.log(train["travel_seconds"].to_numpy())
    glob = float(y.mean())
    train = train.copy()
    other = [o.copy() for o in other]
    fold = np.random.default_rng(0).integers(0, n_folds, len(train))
    for k in keys:
        kt = train[k].astype(str).to_numpy()
        oof = np.empty(len(train))
        cnt = np.empty(len(train))
        for f in range(n_folds):
            m = fold == f
            oof[m], cnt[m] = _encode(kt[~m], y[~m], kt[m], glob)
        train[f"prior_{k}"] = oof
        train[f"prior_n_{k}"] = cnt
        for o in other:
            o[f"prior_{k}"], o[f"prior_n_{k}"] = _encode(kt, y, o[k].astype(str).to_numpy(), glob)
    return train, other


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def matrix(df: pd.DataFrame, num: list[str], cat: list[str], prior_keys: list[str], cats_ref=None):
    cols = num + [f"prior_{k}" for k in prior_keys] + [f"prior_n_{k}" for k in prior_keys]
    X = df[cols].astype(float).copy()
    for c in cat:
        vals = df[c].astype(str)
        if cats_ref is not None:
            X[c] = pd.Categorical(vals, categories=cats_ref[c])
        else:
            X[c] = pd.Categorical(vals)
    return X


def fit(Xtr, ytr, Xva=None, yva=None, *, objective: str, rounds: int | None = None):
    target = np.log if objective == "l2_log" else (lambda v: v)
    params = dict(PARAMS, objective="l2" if objective == "l2_log" else objective, metric="l1")
    dtr = lgb.Dataset(Xtr, target(ytr), free_raw_data=False)
    if rounds is None:
        dva = lgb.Dataset(Xva, target(yva), reference=dtr)
        m = lgb.train(
            params, dtr, num_boost_round=2000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        m = lgb.train(params, dtr, num_boost_round=rounds)
    inv = np.exp if objective == "l2_log" else (lambda v: v)
    return m, inv


def metrics(y, p, y_train_median: float) -> dict:
    e = p - y
    ae = np.abs(e)
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae_s": float(ae.mean()),
        "median_ae_s": float(np.median(ae)),
        "rmse_s": float(np.sqrt(np.mean(e**2))),
        "r2": 1 - ss_res / ss_tot,
        "bias_s": float(e.mean()),
        "pct_within_60s": float(100 * np.mean(ae <= 60)),
        "pct_within_120s": float(100 * np.mean(ae <= 120)),
        "mae_vs_median_pct": float(100 * (1 - ae.mean() / np.abs(y - y_train_median).mean())),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "travel_time_network.parquet")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "figures" / "model_eval_network")
    p.add_argument("--model-out", type=Path, default=ROOT / "data" / "processed" / "models" / "travel_time_network.joblib")
    p.add_argument("--train-end", default="2024-12-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--objectives", nargs="+", default=["l2_log", "l1"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = pd.read_parquet(args.data)
    d["dest_key"] = d["dest_lat"].round(5).astype(str) + "," + d["dest_lon"].round(5).astype(str)
    t = d["incident_datetime"]
    tr = d[t < args.train_end].reset_index(drop=True)
    va = d[(t >= args.train_end) & (t < args.test_start)].reset_index(drop=True)
    full = d[t < args.test_start].reset_index(drop=True)
    te = d[t >= args.test_start].reset_index(drop=True)
    print(f"train {len(tr):,} (to {args.train_end}) | valid {len(va):,} | refit {len(full):,} | test {len(te):,}")

    all_keys = sorted({k for s in STEPS.values() for k in s["prior_keys"]})
    tr_p, (va_p,) = add_priors(tr, [va], all_keys)
    full_p, (te_p,) = add_priors(full, [te], all_keys)

    med = float(full["travel_seconds"].median())
    y_te = te["travel_seconds"].to_numpy()
    results = {"0 Predict median": metrics(y_te, np.full(len(te), med), med)}
    preds = {"0 Predict median": np.full(len(te), med)}
    chosen = {}
    models = {}

    for step, spec in STEPS.items():
        cats_ref = {c: pd.Categorical(full[c].astype(str)).categories for c in spec["cat"]}
        Xtr = matrix(tr_p, spec["num"], spec["cat"], spec["prior_keys"], cats_ref)
        Xva = matrix(va_p, spec["num"], spec["cat"], spec["prior_keys"], cats_ref)
        # Every step picks its objective on Dec-2024 MAE, so the ladder
        # measures features rather than loss functions.
        objs = args.objectives
        best = None
        for obj in objs:
            m, inv = fit(Xtr, tr["travel_seconds"].to_numpy(), Xva, va["travel_seconds"].to_numpy(), objective=obj)
            mae = float(np.abs(inv(m.predict(Xva, num_iteration=m.best_iteration)) - va["travel_seconds"]).mean())
            print(f"  {step:28s} {obj:7s} valid MAE {mae:6.1f}s  ({m.best_iteration} rounds)")
            if best is None or mae < best[0]:
                best = (mae, obj, m.best_iteration)
        _, obj, rounds = best
        chosen[step] = {"objective": obj, "rounds": int(rounds), "valid_mae_s": best[0]}

        Xfull = matrix(full_p, spec["num"], spec["cat"], spec["prior_keys"], cats_ref)
        Xte = matrix(te_p, spec["num"], spec["cat"], spec["prior_keys"], cats_ref)
        m, inv = fit(Xfull, full["travel_seconds"].to_numpy(), objective=obj, rounds=int(rounds * 1.08))
        pte = inv(m.predict(Xte))
        preds[step] = pte
        results[step] = metrics(y_te, pte, med)
        models[step] = (m, obj, spec, cats_ref)
        print(f"  -> TEST {step}: MAE {results[step]['mae_s']:.1f}s  R² {results[step]['r2']:.3f}")

    final = list(STEPS)[-1]
    m, obj, spec, cats_ref = models[final]
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)

    # Breakdowns on the test window
    ev = te[["borough", "dest_source", "hour", "travel_seconds"]].copy()
    ev["prev"] = preds["1 Previous design"]
    ev["new"] = preds[final]
    ev.to_csv(args.out / "test_predictions.csv", index=False)

    summary = {
        "data": str(args.data),
        "rows": {"train": len(tr), "valid": len(va), "refit": len(full), "test": len(te)},
        "split": {"train_end": args.train_end, "test_start": args.test_start, "test_end": str(te["incident_datetime"].max())},
        "label": "incident_travel_tm_seconds_qy: first unit assigned → first unit on scene",
        "test_metrics": results,
        "chosen": chosen,
        "top_features_gain": imp.head(20).round(0).to_dict(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    import joblib

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": m, "objective": obj, "numeric": spec["num"], "categorical": spec["cat"],
            "prior_keys": spec["prior_keys"], "categories": cats_ref, "smooth": SMOOTH,
            "trained_through": args.test_start, "note": "Priors must be rebuilt from training history (see add_priors).",
        },
        args.model_out,
    )

    from travel_time_network_plots import make_all  # noqa: WPS433

    make_all(results, preds, te, imp, args.out)
    print(json.dumps({k: {m_: round(v, 3) for m_, v in r.items()} for k, r in results.items()}, indent=1))
    print("figures ->", args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
