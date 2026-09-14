#!/usr/bin/env python3
"""Train the route-level EMV travel-time model (target R² ≥ 0.8)."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.route_eta import prepare_route_eta_matrix  # noqa: E402

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Training CSV (defaults to route_eta_gmaps_training.csv if present)",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "models")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-r2", type=float, default=0.8)
    args = p.parse_args()

    if args.data is None:
        gmaps_data = ROOT / "data" / "processed" / "route_eta_gmaps_training.csv"
        legacy = ROOT / "data" / "processed" / "route_eta_training.csv"
        args.data = gmaps_data if gmaps_data.exists() else legacy
    print("Training on", args.data)

    df = pd.read_csv(args.data, low_memory=False)
    X, y, cat_cols = prepare_route_eta_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=96,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.5,
        random_state=args.seed,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )

    pred = model.predict(X_test)
    metrics = {
        "model": "route_eta",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "mae_seconds": float(mean_absolute_error(y_test, pred)),
        "rmse_seconds": _rmse(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_test - pred))),
        "pct_within_5_min": float(np.mean(np.abs(y_test - pred) <= 300) * 100),
        "target_r2": args.target_r2,
        "meets_target_r2": bool(r2_score(y_test, pred) >= args.target_r2),
    }

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "travel_time_route_eta_lightgbm.joblib"
    joblib.dump(
        {"model": model, "features": list(X.columns), "categorical": cat_cols, "feature_set": "route_eta"},
        model_path,
    )
    importance.to_csv(args.out_dir / "travel_time_route_eta_feature_importance.csv", index=False)
    (args.out_dir / "travel_time_route_eta_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    sample = X_test.copy()
    sample["travel_seconds_true"] = y_test.values
    sample["travel_seconds_pred"] = pred
    sample.head(200).to_csv(ROOT / "data" / "samples" / "route_eta_predictions_sample.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("Top features:")
    print(importance.head(12).to_string(index=False))
    print("Wrote", model_path)
    if not metrics["meets_target_r2"]:
        raise SystemExit(
            f"R²={metrics['r2']:.3f} below target {args.target_r2}. "
            "Try more pairs or lower label noise."
        )


if __name__ == "__main__":
    main()
