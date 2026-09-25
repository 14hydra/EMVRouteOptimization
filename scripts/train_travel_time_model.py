#!/usr/bin/env python3
"""
Train the firetruck travel-time prediction model (slides Step 4).

Supports:
  --feature-set full   (includes dispatch_wait / held)
  --feature-set route  (route-scoring set: drops CAD wait; for optimizer use)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "travel_time_training.csv",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "models")
    p.add_argument("--feature-set", choices=("full", "route"), default="full")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    X, y, cat_cols = prepare_matrix(df, feature_set=args.feature_set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = model.predict(X_test)
    baseline = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.median())))
    mae = float(mean_absolute_error(y_test, pred))
    metrics = {
        "feature_set": args.feature_set,
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae_seconds": mae,
        "rmse_seconds": _rmse(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_test - pred))),
        "baseline_mae_predict_median": baseline,
        "mae_improvement_vs_median_pct": float(100.0 * (1.0 - mae / baseline)) if baseline else None,
        "pct_within_5_min": float(np.mean(np.abs(y_test - pred) <= 300) * 100),
    }

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.feature_set
    model_path = args.out_dir / f"travel_time_lightgbm_{tag}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "categorical": cat_cols,
            "feature_set": args.feature_set,
        },
        model_path,
    )
    # Keep legacy filename for full model
    if args.feature_set == "full":
        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "categorical": cat_cols,
                "feature_set": args.feature_set,
            },
            args.out_dir / "travel_time_lightgbm.joblib",
        )

    importance.to_csv(args.out_dir / f"travel_time_feature_importance_{tag}.csv", index=False)
    (args.out_dir / f"travel_time_metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    if args.feature_set == "full":
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
