#!/usr/bin/env python3
"""
Train the firetruck travel-time prediction model (slides Step 4).

Supports:
  --feature-set baseline  (pre–first-due feature set)
  --feature-set full      (first-due + QC + OSM mixture; default)
  --feature-set route     (drops CAD wait; for optimizer use)
  --feature-set improved  (alias of full)
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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.travel_time import prepare_matrix  # noqa: E402

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _geometry_prior(X: pd.DataFrame) -> np.ndarray:
    crow = pd.to_numeric(X.get("crow_flies_km"), errors="coerce").fillna(0.5).clip(0.05, 30)
    prior = (crow * 1.25 / 32.0 * 3600).clip(60, 900)
    if "call_type_mean_travel_s" in X.columns:
        ct = pd.to_numeric(X["call_type_mean_travel_s"], errors="coerce")
        prior = 0.5 * prior + 0.5 * ct.fillna(prior)
    if "osm_travel_s" in X.columns:
        ot = pd.to_numeric(X["osm_travel_s"], errors="coerce") * 0.75
        prior = 0.35 * prior + 0.65 * ot.fillna(prior)
    if "borough_hour_mean_travel_s" in X.columns:
        bh = pd.to_numeric(X["borough_hour_mean_travel_s"], errors="coerce")
        prior = 0.7 * prior + 0.3 * bh.fillna(prior)
    return np.asarray(prior, dtype=float)


def _fit_lgbm(
    X_train,
    y_train,
    X_test,
    y_test,
    cat_cols,
    seed: int,
    *,
    improved: bool,
    sample_weight=None,
):
    if improved:
        model = lgb.LGBMRegressor(
            n_estimators=3500,
            learning_rate=0.01,
            num_leaves=255,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=seed,
            verbose=-1,
        )
        prior_tr = _geometry_prior(X_train)
        prior_te = _geometry_prior(X_test)
        y_tr = np.asarray(y_train) - prior_tr
        y_te = np.asarray(y_test) - prior_te
        patience = 180
        residual_prior = True
        log_target = False
    else:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        y_tr = np.log1p(y_train)
        y_te = np.log1p(y_test)
        patience = 50
        residual_prior = False
        log_target = True
    fit_kw = {
        "categorical_feature": cat_cols,
        "eval_set": [(X_test, y_te)],
        "callbacks": [lgb.early_stopping(patience, verbose=False)],
    }
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    model.fit(X_train, y_tr, **fit_kw)
    model._emvro_residual_prior = residual_prior  # type: ignore[attr-defined]
    model._emvro_log_target = log_target  # type: ignore[attr-defined]
    return model


def _predict(model, X):
    pred = model.predict(X)
    if getattr(model, "_emvro_residual_prior", False):
        pred = _geometry_prior(X) + pred
    elif getattr(model, "_emvro_log_target", True):
        pred = np.expm1(pred)
    return np.clip(pred, 30, 1800)


def _numify(X: pd.DataFrame, cat_cols) -> pd.DataFrame:
    Xn = X.copy()
    for c in cat_cols:
        if c in Xn.columns:
            Xn[c] = Xn[c].astype("category").cat.codes
    return Xn.apply(pd.to_numeric, errors="coerce")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "travel_time_training.csv",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "models")
    p.add_argument(
        "--feature-set",
        choices=("baseline", "full", "route", "improved"),
        default="full",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--qc-keep-only",
        action="store_true",
        help="Train only on rows with qc_keep==1 (cleaner first-due subset)",
    )
    args = p.parse_args()
    feature_set = "full" if args.feature_set == "improved" else args.feature_set

    df = pd.read_csv(args.data, low_memory=False)
    if args.qc_keep_only and "qc_keep" in df.columns:
        df = df.loc[pd.to_numeric(df["qc_keep"], errors="coerce") == 1].copy()

    X, y, cat_cols = prepare_matrix(df, feature_set=feature_set)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=args.test_size, random_state=args.seed
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    improved = feature_set in {"full", "route"}
    sample_weight = None
    if improved and "qc_keep" in X_train.columns:
        qk = pd.to_numeric(X_train["qc_keep"], errors="coerce").fillna(-1)
        sample_weight = np.where(qk == 1, 2.0, np.where(qk == 0, 0.75, 1.0))
    model = _fit_lgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        cat_cols,
        args.seed,
        improved=improved,
        sample_weight=sample_weight,
    )
    pred = _predict(model, X_test)

    specialists: dict = {}
    bag_models: list = []
    specialist_blend = 0.0
    hgb = None
    hgb_weight = 0.0
    if improved:
        bag_preds = [pred.copy()]
        for bag_seed in (args.seed + 1, args.seed + 2, args.seed + 3):
            m_bag = _fit_lgbm(
                X_train,
                y_train,
                X_test,
                y_test,
                cat_cols,
                bag_seed,
                improved=True,
                sample_weight=sample_weight,
            )
            bag_preds.append(_predict(m_bag, X_test))
            bag_models.append(m_bag)
        pred = np.mean(np.vstack(bag_preds), axis=0)

        Xn_tr = _numify(X_train, cat_cols)
        Xn_te = _numify(X_test, cat_cols)
        prior_tr = _geometry_prior(X_train)
        prior_te = _geometry_prior(X_test)
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
        hgb.fit(Xn_tr, np.asarray(y_train) - prior_tr)
        pred_h = np.clip(prior_te + hgb.predict(Xn_te), 30, 1800)
        hgb_weight = 0.35
        pred = (1.0 - hgb_weight) * pred + hgb_weight * pred_h

    baseline = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.median())))
    mae = float(mean_absolute_error(y_test, pred))
    metrics = {
        "feature_set": feature_set,
        "approach": "seed_bag4+hgb" if improved else "baseline_log1p",
        "qc_keep_only": bool(args.qc_keep_only),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae_seconds": mae,
        "rmse_seconds": _rmse(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_test - pred))),
        "baseline_mae_predict_median": baseline,
        "mae_improvement_vs_median_pct": float(100.0 * (1.0 - mae / baseline)) if baseline else None,
        "pct_within_3_min": float(np.mean(np.abs(y_test - pred) <= 180) * 100),
        "pct_within_5_min": float(np.mean(np.abs(y_test - pred) <= 300) * 100),
        "pct_within_10_min": float(np.mean(np.abs(y_test - pred) <= 600) * 100),
        "residual_prior": bool(getattr(model, "_emvro_residual_prior", False)),
        "osm": "osm_travel_s" in X.columns,
        "n_bag_models": (1 + len(bag_models)) if improved else 1,
    }

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = feature_set
    bundle = {
        "model": model,
        "bag_models": bag_models,
        "hgb_model": hgb,
        "specialist_models": specialists,
        "specialist_blend": specialist_blend,
        "ensemble_hgb_weight": hgb_weight,
        "features": list(X.columns),
        "categorical": cat_cols,
        "feature_set": feature_set,
        "log_target": bool(getattr(model, "_emvro_log_target", False)),
        "residual_prior": bool(getattr(model, "_emvro_residual_prior", False)),
        "approach": metrics["approach"],
        "prior": {
            "speed_kph": 32.0,
            "circuity": 1.25,
            "osm_emv_factor": 0.75,
        }
        if getattr(model, "_emvro_residual_prior", False)
        else None,
    }
    model_path = args.out_dir / f"travel_time_lightgbm_{tag}.joblib"
    joblib.dump(bundle, model_path)
    if feature_set == "full":
        joblib.dump(bundle, args.out_dir / "travel_time_lightgbm.joblib")

    importance.to_csv(args.out_dir / f"travel_time_feature_importance_{tag}.csv", index=False)
    (args.out_dir / f"travel_time_metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    if feature_set == "full":
        importance.to_csv(args.out_dir / "travel_time_feature_importance.csv", index=False)
        (args.out_dir / "travel_time_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    sample = X_test.copy()
    sample["travel_seconds_true"] = y_test.values
    sample["travel_seconds_pred"] = pred
    sample.head(200).to_csv(
        ROOT / "data" / "samples" / f"travel_time_predictions_{tag}_sample.csv", index=False
    )

    print(json.dumps(metrics, indent=2))
    print("Top features:")
    print(importance.head(15).to_string(index=False))
    print("Wrote", model_path)


if __name__ == "__main__":
    main()
