#!/usr/bin/env python3
"""
Hybrid CAD travel-time trainer: Ricky's network / load / prior features
+ residual geometry prior + 4-seed LightGBM bag + HistGradientBoosting.

Uses ``travel_time_network.parquet``. Prefer a chronological holdout when the
table has no Jan–Mar 2025 test window (e.g. a Dec-only 60k extract).

  PYTHONPATH=src .venv/bin/python scripts/train_travel_time_hybrid.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.network_origins import NETWORK_FEATURE_COLUMNS  # noqa: E402
from emvro.route_street_features import ROUTE_STREET_COLUMNS  # noqa: E402
from emvro.traffic import TRAFFIC_FEATURE_COLUMNS  # noqa: E402
from emvro.weather import WEATHER_FEATURE_COLUMNS  # noqa: E402

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

LOAD = [
    f"{area}_{c}"
    for area in ("engine", "ladder", "borough")
    for c in ("n_prev_15m", "n_prev_30m", "n_prev_60m", "min_since_prev")
]
GMAPS_COLS = [
    "gmaps_duration_s",
    "gmaps_traffic_s",
    "gmaps_distance_m",
    "gmaps_ok",
    "gmaps_emv_s",
    "gmaps_vs_net",
]
CONTEXT = [
    "hour",
    "dow",
    "month",
    "minute_of_day",
    "is_weekend",
    "is_rush",
    "is_night",
    "dispatch_wait_seconds",
    "engines_assigned",
    "ladders_assigned",
    "other_units_assigned",
] + WEATHER_FEATURE_COLUMNS
CATS = [
    "borough",
    "call_type",
    "call_group",
    "alarm_level",
    "dest_source",
    "first_due_engine",
    "first_due_ladder",
]
PRIOR_KEYS = ["dest_key", "first_due_engine", "zipcode"]
SMOOTH = 20.0


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true, y_pred, y_train) -> dict:
    baseline = float(mean_absolute_error(y_true, np.full_like(y_true, float(np.median(y_train)))))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "n_test": int(len(y_true)),
        "mae_seconds": mae,
        "rmse_seconds": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_true - y_pred))),
        "baseline_mae_predict_median": baseline,
        "mae_improvement_vs_median_pct": float(100.0 * (1.0 - mae / baseline)) if baseline else None,
        "pct_within_60s": float(np.mean(np.abs(y_true - y_pred) <= 60) * 100),
        "pct_within_3_min": float(np.mean(np.abs(y_true - y_pred) <= 180) * 100),
        "pct_within_5_min": float(np.mean(np.abs(y_true - y_pred) <= 300) * 100),
        "bias_seconds": float(np.mean(y_pred - y_true)),
    }


def _encode(train_keys, train_y, apply_keys, glob: float):
    g = pd.DataFrame({"k": train_keys, "y": train_y}).groupby("k")["y"].agg(["sum", "count"])
    s = pd.Series(apply_keys).map(g["sum"]).fillna(0).to_numpy()
    n = pd.Series(apply_keys).map(g["count"]).fillna(0).to_numpy()
    return (s + SMOOTH * glob) / (n + SMOOTH), n


def add_priors(train: pd.DataFrame, other: list[pd.DataFrame], keys: list[str], n_folds: int = 5):
    y = np.log(np.clip(train["travel_seconds"].to_numpy(), 30, None))
    glob = float(y.mean())
    train = train.copy()
    other = [o.copy() for o in other]
    fold = np.random.default_rng(0).integers(0, n_folds, len(train))
    for k in keys:
        if k not in train.columns:
            continue
        kt = train[k].astype(str).to_numpy()
        oof = np.empty(len(train))
        cnt = np.empty(len(train))
        for f in range(n_folds):
            m = fold == f
            oof[m], cnt[m] = _encode(kt[~m], y[~m], kt[m], glob)
        train[f"prior_{k}"] = oof
        train[f"prior_n_{k}"] = cnt
        for o in other:
            if k not in o.columns:
                o[f"prior_{k}"] = glob
                o[f"prior_n_{k}"] = 0.0
                continue
            o[f"prior_{k}"], o[f"prior_n_{k}"] = _encode(kt, y, o[k].astype(str).to_numpy(), glob)
    return train, other


def network_prior(X: pd.DataFrame) -> np.ndarray:
    """Free-flow network seconds as residual base (EMV factor ~0.75), blended with GMaps EMV prior when present."""
    base = pd.to_numeric(X.get("net_s_first_due_min"), errors="coerce")
    nearest = pd.to_numeric(X.get("net_s_nearest1"), errors="coerce")
    prior = (0.6 * base.fillna(nearest) + 0.4 * nearest.fillna(base)).fillna(300) * 0.75
    prior = prior.clip(45, 900)
    if "gmaps_emv_s" in X.columns:
        # Keep GMaps as a *feature* only — blending into the residual prior
        # hurts when civilian ETA ≠ CAD assign→on-scene.
        pass
    if "prior_dest_key" in X.columns:
        dest = np.exp(pd.to_numeric(X["prior_dest_key"], errors="coerce").fillna(np.log(300)))
        prior = 0.55 * prior + 0.45 * dest
    if "prior_first_due_engine" in X.columns:
        eng = np.exp(pd.to_numeric(X["prior_first_due_engine"], errors="coerce").fillna(np.log(300)))
        prior = 0.75 * prior + 0.25 * eng
    return np.asarray(prior, dtype=float)


def feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    num = (
        [c for c in NETWORK_FEATURE_COLUMNS if c in df.columns]
        + [c for c in LOAD if c in df.columns]
        + [c for c in CONTEXT if c in df.columns]
        + [c for c in ROUTE_STREET_COLUMNS if c in df.columns]
        + [c for c in TRAFFIC_FEATURE_COLUMNS if c in df.columns]
        + [c for c in GMAPS_COLS if c in df.columns]
        + [c for c in df.columns if c.startswith("prior_")]
        + ["dest_lat", "dest_lon"]
    )
    num = list(dict.fromkeys(num))  # stable unique
    cats = [c for c in CATS if c in df.columns]
    X = df[num + cats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X, cats


def fit_lgbm(X_train, y_train, X_val, y_val, cats, seed: int, sample_weight=None, objective: str = "regression"):
    model = lgb.LGBMRegressor(
        n_estimators=3500,
        learning_rate=0.01,
        num_leaves=255,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=seed,
        verbose=-1,
        objective=objective,
    )
    prior_tr = network_prior(X_train)
    prior_va = network_prior(X_val)
    fit_kw = {
        "categorical_feature": cats,
        "eval_set": [(X_val, np.asarray(y_val) - prior_va)],
        "callbacks": [lgb.early_stopping(180, verbose=False)],
    }
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    model.fit(X_train, np.asarray(y_train) - prior_tr, **fit_kw)
    return model


def predict_lgbm(model, X):
    return np.clip(network_prior(X) + model.predict(X), 30, 1800)


def numify(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    Xn = X.copy()
    for c in cats:
        if c in Xn.columns:
            Xn[c] = Xn[c].astype("category").cat.codes
    return Xn.apply(pd.to_numeric, errors="coerce")


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("incident_datetime").reset_index(drop=True)
    cut = int(len(df) * (1.0 - test_frac))
    cut = max(1, min(cut, len(df) - 1))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "travel_time_network.parquet")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "models")
    p.add_argument("--figures", type=Path, default=ROOT / "data" / "figures" / "model_eval_hybrid")
    p.add_argument("--test-frac", type=float, default=0.2, help="Used only when --train-end/--test-start unset")
    p.add_argument("--train-end", type=str, default=None, help="ISO cutoff; rows before this are train/valid")
    p.add_argument("--test-start", type=str, default=None, help="ISO; rows on/after this are test")
    p.add_argument("--valid-start", type=str, default=None, help="ISO; default = 1 month before test-start")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = pd.read_parquet(args.data)
    d = d.dropna(subset=["travel_seconds"]).copy()
    d = d[d["travel_seconds"].between(45, 1200)].reset_index(drop=True)
    d["dest_key"] = d["dest_lat"].round(5).astype(str) + "," + d["dest_lon"].round(5).astype(str)
    d["incident_datetime"] = pd.to_datetime(d["incident_datetime"], errors="coerce")
    d = d.sort_values("incident_datetime").reset_index(drop=True)

    if args.train_end and args.test_start:
        train_end = pd.Timestamp(args.train_end)
        test_start = pd.Timestamp(args.test_start)
        valid_start = pd.Timestamp(args.valid_start) if args.valid_start else (test_start - pd.DateOffset(months=1))
        train_fit = d[d["incident_datetime"] < valid_start].reset_index(drop=True)
        valid = d[(d["incident_datetime"] >= valid_start) & (d["incident_datetime"] < test_start)].reset_index(drop=True)
        test = d[d["incident_datetime"] >= test_start].reset_index(drop=True)
        if len(valid) < 500:
            # fall back: last 10% of pre-test as valid
            pre = d[d["incident_datetime"] < test_start].reset_index(drop=True)
            val_n = max(500, int(0.1 * len(pre)))
            train_fit = pre.iloc[:-val_n].reset_index(drop=True)
            valid = pre.iloc[-val_n:].reset_index(drop=True)
        print(
            f"time split: train_fit {len(train_fit):,} | valid {len(valid):,} | "
            f"test {len(test):,} (from {args.test_start})"
        )
    else:
        train, test = chronological_split(d, test_frac=args.test_frac)
        val_n = max(500, int(0.1 * len(train)))
        valid = train.iloc[-val_n:].reset_index(drop=True)
        train_fit = train.iloc[:-val_n].reset_index(drop=True)

    train_p, (valid_p, test_p) = add_priors(train_fit, [valid, test], PRIOR_KEYS)
    # Rebuild priors on full train (fit+valid) for final bag, apply to test
    full_train = pd.concat([train_fit, valid], ignore_index=True)
    full_p, (test_final,) = add_priors(full_train, [test], PRIOR_KEYS)

    X_tr, cats = feature_frame(train_p)
    X_va, _ = feature_frame(valid_p)
    y_tr = train_p["travel_seconds"].to_numpy()
    y_va = valid_p["travel_seconds"].to_numpy()

    X_full, _ = feature_frame(full_p)
    X_te, _ = feature_frame(test_final)
    y_full = full_p["travel_seconds"].to_numpy()
    y_te = test_final["travel_seconds"].to_numpy()

    # Align columns
    cols = list(X_full.columns)
    X_tr = X_tr.reindex(columns=cols)
    X_va = X_va.reindex(columns=cols)
    X_te = X_te.reindex(columns=cols)
    for c in cats:
        for frame in (X_tr, X_va, X_full, X_te):
            if c in frame.columns:
                frame[c] = frame[c].astype("category")

    bag_models = []
    bag_preds = []
    # Up-weight precise destinations and GMaps-covered rows (cleaner signal).
    # Alarm-box coords are the most accurate public destinations; zip centroids
    # are noisy — down-weight them harder so year-scale data doesn't dilute.
    sw = np.ones(len(train_p))
    if "dest_source" in train_p.columns:
        ds = train_p["dest_source"].astype(str)
        sw = np.where(ds == "alarm_box", 1.5, np.where(ds == "intersection_geocode", 1.15, 0.65))
    if "gmaps_ok" in train_p.columns:
        sw = sw * np.where(pd.to_numeric(train_p["gmaps_ok"], errors="coerce").fillna(0) == 1, 1.15, 1.0)
    # Mild up-weight when traffic sensors cover the hour (more accurate context).
    if "traffic_city_index" in train_p.columns:
        sw = sw * np.where(pd.to_numeric(train_p["traffic_city_index"], errors="coerce").notna(), 1.08, 1.0)
    bag_specs = [
        (args.seed, "regression"),
        (args.seed + 1, "regression"),
        (args.seed + 2, "regression"),
        (args.seed + 3, "regression_l1"),
    ]
    for seed, obj in bag_specs:
        m = fit_lgbm(X_tr, y_tr, X_va, y_va, cats, seed, sample_weight=sw, objective=obj)
        bag_models.append(m)
        bag_preds.append(predict_lgbm(m, X_te))
    pred_bag = np.mean(np.vstack(bag_preds), axis=0)

    Xn_tr = numify(X_full, cats)
    Xn_te = numify(X_te, cats)
    prior_tr = network_prior(X_full)
    prior_te = network_prior(X_te)
    hgb = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.025,
        max_depth=12,
        min_samples_leaf=8,
        l2_regularization=1.5,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )
    hgb.fit(Xn_tr, y_full - prior_tr)
    pred_h = np.clip(prior_te + hgb.predict(Xn_te), 30, 1800)
    # Bag-only has been beating bag+HGB on chronological CAD holdouts; keep a
    # light HGB blend for variance, not equal weight.
    hgb_w = 0.15
    pred = (1.0 - hgb_w) * pred_bag + hgb_w * pred_h

    # Baselines for comparison on the same holdout
    crow = pd.to_numeric(test_final.get("prev_crow_km_primary"), errors="coerce").fillna(0.8)
    prev_design = np.clip(crow * 1.25 / 32.0 * 3600, 60, 900)
    single = predict_lgbm(bag_models[0], X_te)

    metrics = {
        "approach": "network_features+residual_bag4+hgb",
        "n_train": int(len(full_train)),
        "n_test": int(len(test)),
        "n_features": int(X_full.shape[1]),
        "date_train_min": str(full_train["incident_datetime"].min()),
        "date_train_max": str(full_train["incident_datetime"].max()),
        "date_test_min": str(test["incident_datetime"].min()),
        "date_test_max": str(test["incident_datetime"].max()),
        "hybrid": _metrics(y_te, pred, y_full),
        "bag_only": _metrics(y_te, pred_bag, y_full),
        "single_lgbm": _metrics(y_te, single, y_full),
        "prev_crow_prior": _metrics(y_te, prev_design, y_full),
        "median_baseline": _metrics(y_te, np.full_like(y_te, float(np.median(y_full))), y_full),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": bag_models[0],
        "bag_models": bag_models[1:],
        "hgb_model": hgb,
        "ensemble_hgb_weight": hgb_w,
        "features": cols,
        "categorical": cats,
        "prior_keys": PRIOR_KEYS,
        "approach": metrics["approach"],
        "residual_prior": True,
        "network_prior": True,
    }
    joblib.dump(bundle, args.out_dir / "travel_time_hybrid.joblib")
    (args.out_dir / "travel_time_hybrid_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    slim = {
        "feature_set": "hybrid_network",
        "approach": metrics["approach"],
        **metrics["hybrid"],
        "n_train": metrics["n_train"],
        "n_features": metrics["n_features"],
    }
    (args.out_dir / "travel_time_hybrid_metrics_slim.json").write_text(json.dumps(slim, indent=2) + "\n")

    # Simple figures
    import matplotlib.pyplot as plt

    args.figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(y_te / 60, pred / 60, s=12, alpha=0.3, c="#c0392b", edgecolors="none")
    lim = max(y_te.max(), pred.max()) / 60
    ax.plot([0, lim], [0, lim], color="#222", lw=1.2)
    ax.set_xlabel("Actual (min)")
    ax.set_ylabel("Predicted (min)")
    ax.set_title(
        f"Hybrid network+bag4+HGB\nR²={metrics['hybrid']['r2']:.3f} · "
        f"MAE={metrics['hybrid']['mae_seconds']/60:.2f} min"
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(args.figures / "01_pred_vs_actual.png", dpi=160)
    plt.close(fig)

    # MAE ladder bars
    labels = ["Median", "Crow prior", "Single LGBM", "Bag4", "Bag4+HGB"]
    maes = [
        metrics["median_baseline"]["mae_seconds"] / 60,
        metrics["prev_crow_prior"]["mae_seconds"] / 60,
        metrics["single_lgbm"]["mae_seconds"] / 60,
        metrics["bag_only"]["mae_seconds"] / 60,
        metrics["hybrid"]["mae_seconds"] / 60,
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(labels, maes, color=["#7f8c8d", "#95a5a6", "#2980b9", "#1f4e79", "#c0392b"])
    ax.set_ylabel("MAE (min)")
    ax.set_title("Holdout MAE ladder (chronological)")
    for i, v in enumerate(maes):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.figures / "02_mae_ladder.png", dpi=160)
    plt.close(fig)

    (args.figures / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print("Wrote", args.out_dir / "travel_time_hybrid.joblib")
    print("Figures ->", args.figures)


if __name__ == "__main__":
    main()
