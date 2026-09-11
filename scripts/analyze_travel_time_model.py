#!/usr/bin/env python3
"""
Visualize and analyze ambulance travel-time model performance.

Reloads the saved LightGBM model, rebuilds the same holdout split, and writes
figures + a metrics JSON under data/figures/model_eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.travel_time import prepare_matrix  # noqa: E402
from emvro.route_eta import prepare_route_eta_matrix  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape_capped(y_true, y_pred, floor=30.0) -> float:
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_actual(eval_df: pd.DataFrame, out: Path, title: str | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(
        eval_df["y_true"] / 60,
        eval_df["y_pred"] / 60,
        s=18,
        alpha=0.35,
        c="#1f4e79",
        edgecolors="none",
    )
    lim = max(eval_df["y_true"].max(), eval_df["y_pred"].max()) / 60
    ax.plot([0, lim], [0, lim], color="#c0392b", lw=1.5, label="perfect")
    ax.set_xlabel("Actual travel time (min)")
    ax.set_ylabel("Predicted travel time (min)")
    ax.set_title(title or "Holdout: predicted vs actual")
    ax.legend(loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _save(fig, out)


def plot_residuals(eval_df: pd.DataFrame, out: Path):
    resid_min = eval_df["residual"] / 60
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axes[0].hist(resid_min, bins=40, color="#2c7a7b", edgecolor="white", alpha=0.9)
    axes[0].axvline(0, color="#c0392b", lw=1.4)
    axes[0].set_xlabel("Residual (pred − actual), minutes")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual distribution")

    axes[1].scatter(
        eval_df["y_pred"] / 60,
        resid_min,
        s=16,
        alpha=0.35,
        c="#1f4e79",
        edgecolors="none",
    )
    axes[1].axhline(0, color="#c0392b", lw=1.4)
    axes[1].set_xlabel("Predicted (min)")
    axes[1].set_ylabel("Residual (min)")
    axes[1].set_title("Residuals vs predicted")
    _save(fig, out)


def plot_error_by_group(eval_df: pd.DataFrame, group: str, out: Path, title: str, order=None):
    g = (
        eval_df.groupby(group, dropna=False)
        .agg(
            n=("abs_error", "size"),
            mae_s=("abs_error", "mean"),
            medae_s=("abs_error", "median"),
            mean_true_s=("y_true", "mean"),
        )
        .reset_index()
    )
    g["mae_min"] = g["mae_s"] / 60
    if order is not None:
        g[group] = pd.Categorical(g[group], categories=order, ordered=True)
        g = g.sort_values(group)
    else:
        g = g.sort_values("mae_min", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=g, x=group, y="mae_min", ax=ax, color="#1f4e79")
    for i, row in g.reset_index(drop=True).iterrows():
        ax.text(i, row["mae_min"] + 0.05, f"n={int(row['n'])}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE (minutes)")
    ax.set_xlabel(group)
    ax.set_title(title)
    if g[group].astype(str).str.len().max() > 12:
        ax.tick_params(axis="x", rotation=30)
    _save(fig, out)
    return g


def plot_error_by_distance(eval_df: pd.DataFrame, out: Path):
    bins = [0, 0.5, 1, 2, 4, 8, 50]
    labels = ["0–0.5", "0.5–1", "1–2", "2–4", "4–8", "8+"]
    tmp = eval_df.copy()
    tmp["dist_bin"] = pd.cut(tmp["crow_flies_km"], bins=bins, labels=labels, include_lowest=True)
    g = (
        tmp.groupby("dist_bin", observed=False)
        .agg(n=("abs_error", "size"), mae_s=("abs_error", "mean"), mean_true_s=("y_true", "mean"))
        .reset_index()
    )
    g["mae_min"] = g["mae_s"] / 60
    g["mean_true_min"] = g["mean_true_s"] / 60

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(g))
    ax.bar(x - 0.18, g["mae_min"], width=0.36, color="#1f4e79", label="MAE")
    ax.bar(x + 0.18, g["mean_true_min"], width=0.36, color="#7f8c8d", alpha=0.75, label="Mean actual")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(g["dist_bin"], g["n"])])
    ax.set_ylabel("Minutes")
    ax.set_xlabel("Crow-flies distance bin (km)")
    ax.set_title("Error vs trip distance")
    ax.legend()
    _save(fig, out)
    return g


def plot_feature_importance(imp: pd.DataFrame, out: Path, top_n: int = 15):
    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.barh(top["feature"], top["importance"], color="#1f4e79")
    ax.set_xlabel("LightGBM split importance")
    ax.set_title(f"Top {top_n} features")
    _save(fig, out)


def plot_calibration(eval_df: pd.DataFrame, out: Path, n_bins: int = 10):
    tmp = eval_df.copy()
    tmp["pred_bin"] = pd.qcut(tmp["y_pred"], q=n_bins, duplicates="drop")
    cal = (
        tmp.groupby("pred_bin", observed=False)
        .agg(pred_mean=("y_pred", "mean"), true_mean=("y_true", "mean"), n=("y_true", "size"))
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot(cal["pred_mean"] / 60, cal["true_mean"] / 60, "o-", color="#1f4e79", label="binned means")
    lo = min(cal["pred_mean"].min(), cal["true_mean"].min()) / 60
    hi = max(cal["pred_mean"].max(), cal["true_mean"].max()) / 60
    ax.plot([lo, hi], [lo, hi], color="#c0392b", lw=1.4, label="perfect")
    ax.set_xlabel("Mean predicted (min)")
    ax.set_ylabel("Mean actual (min)")
    ax.set_title("Calibration by prediction decile")
    ax.legend()
    _save(fig, out)
    return cal


def plot_abs_error_cdf(eval_df: pd.DataFrame, out: Path):
    errs = np.sort(eval_df["abs_error"].values) / 60
    cdf = np.arange(1, len(errs) + 1) / len(errs)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(errs, cdf, color="#1f4e79", lw=2)
    for q, c in [(0.5, "#2c7a7b"), (0.8, "#b7791f"), (0.9, "#c0392b")]:
        v = float(np.quantile(errs, q))
        ax.axvline(v, color=c, ls="--", lw=1.2, label=f"P{int(q*100)} = {v:.1f} min")
    ax.set_xlabel("|Error| (minutes)")
    ax.set_ylabel("Fraction of holdout ≤ error")
    ax.set_title("Absolute error CDF")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    _save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--feature-set",
        choices=("full", "route", "route_eta"),
        default="route_eta",
        help="route_eta = known-OD optimizer model (default); full/route = CAD context models",
    )
    p.add_argument(
        "--importance",
        type=Path,
        default=None,
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    models_dir = ROOT / "data" / "processed" / "models"
    figures_root = ROOT / "data" / "figures"

    if args.feature_set == "route_eta":
        args.data = args.data or (ROOT / "data" / "processed" / "route_eta_training.csv")
        args.model = args.model or (models_dir / "travel_time_route_eta_lightgbm.joblib")
        args.importance = args.importance or (
            models_dir / "travel_time_route_eta_feature_importance.csv"
        )
        args.out_dir = args.out_dir or (figures_root / "model_eval_route_eta")
        prepare = prepare_route_eta_matrix
        origin_col = "origin_layer"
        origin_title = "MAE by origin layer"
    else:
        args.data = args.data or (ROOT / "data" / "processed" / "travel_time_training.csv")
        tagged_model = models_dir / f"travel_time_lightgbm_{args.feature_set}.joblib"
        args.model = args.model or (
            tagged_model if tagged_model.exists() else models_dir / "travel_time_lightgbm.joblib"
        )
        tagged_imp = models_dir / f"travel_time_feature_importance_{args.feature_set}.csv"
        args.importance = args.importance or (
            tagged_imp if tagged_imp.exists() else models_dir / "travel_time_feature_importance.csv"
        )
        args.out_dir = args.out_dir or (figures_root / f"model_eval_{args.feature_set}")
        prepare = lambda df: prepare_matrix(df, feature_set=args.feature_set)
        origin_col = "primary_origin_layer"
        origin_title = "MAE by primary inferred origin"

    # Ensure OpenMP is findable on macOS Homebrew installs
    omp = "/opt/homebrew/opt/libomp/lib"
    if Path(omp).exists():
        os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

    df = pd.read_csv(args.data, low_memory=False)
    bundle = joblib.load(args.model)
    model = bundle["model"]

    X, y, _ = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    pred = model.predict(X_test)

    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = pred
    eval_df["residual"] = eval_df["y_pred"] - eval_df["y_true"]
    eval_df["abs_error"] = np.abs(eval_df["residual"])
    eval_df["hour_bin"] = pd.cut(
        eval_df["hour"].astype(float),
        bins=[-0.1, 6, 10, 16, 20, 24],
        labels=["night 0–6", "morning 6–10", "midday 10–16", "evening 16–20", "late 20–24"],
    )

    baseline_pred = np.full_like(y_test, float(y_train.median()), dtype=float)
    metrics = {
        "feature_set": args.feature_set,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae_seconds": float(mean_absolute_error(y_test, pred)),
        "rmse_seconds": _rmse(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_test - pred))),
        "mape_capped_pct": _mape_capped(y_test.values, pred),
        "baseline_mae_predict_median": float(mean_absolute_error(y_test, baseline_pred)),
        "mae_improvement_vs_median_pct": float(
            100.0
            * (1.0 - mean_absolute_error(y_test, pred) / mean_absolute_error(y_test, baseline_pred))
        ),
        "pct_within_3_min": float(np.mean(np.abs(y_test - pred) <= 180) * 100),
        "pct_within_5_min": float(np.mean(np.abs(y_test - pred) <= 300) * 100),
        "pct_within_10_min": float(np.mean(np.abs(y_test - pred) <= 600) * 100),
        "mean_actual_seconds": float(y_test.mean()),
        "median_actual_seconds": float(y_test.median()),
        "bias_seconds": float(np.mean(pred - y_test)),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    r2 = metrics["r2"]
    plot_pred_vs_actual(
        eval_df,
        args.out_dir / "01_pred_vs_actual.png",
        title=f"Holdout: predicted vs actual  (R²={r2:.3f})",
    )
    plot_residuals(eval_df, args.out_dir / "02_residuals.png")
    by_boro = plot_error_by_group(
        eval_df, "borough", args.out_dir / "03_mae_by_borough.png", "MAE by borough"
    )
    by_layer = None
    if origin_col in eval_df.columns:
        by_layer = plot_error_by_group(
            eval_df,
            origin_col,
            args.out_dir / "04_mae_by_origin_layer.png",
            origin_title,
        )
    by_hour = plot_error_by_group(
        eval_df,
        "hour_bin",
        args.out_dir / "05_mae_by_hour.png",
        "MAE by time of day",
        order=["night 0–6", "morning 6–10", "midday 10–16", "evening 16–20", "late 20–24"],
    )
    by_dist = plot_error_by_distance(eval_df, args.out_dir / "06_mae_by_distance.png")
    plot_abs_error_cdf(eval_df, args.out_dir / "07_abs_error_cdf.png")
    plot_calibration(eval_df, args.out_dir / "08_calibration.png")

    if args.importance and Path(args.importance).exists():
        imp = pd.read_csv(args.importance)
        plot_feature_importance(imp, args.out_dir / "09_feature_importance.png")

    # Persist full holdout preds for further inspection
    keep_cols = [
        c
        for c in [
            "borough",
            "primary_origin_layer",
            "origin_layer",
            "crow_flies_km",
            "civilian_network_s",
            "osm_path_km",
            "station_km",
            "hospital_km",
            "csl_km",
            "hour",
            "severity",
            "dispatch_wait_seconds",
            "y_true",
            "y_pred",
            "residual",
            "abs_error",
        ]
        if c in eval_df.columns
    ]
    eval_df[keep_cols].to_csv(args.out_dir / "holdout_predictions.csv", index=False)

    summary = {
        "metrics": metrics,
        "mae_by_borough_min": {
            str(r["borough"]): round(float(r["mae_min"]), 2) for _, r in by_boro.iterrows()
        },
        "mae_by_hour_min": {
            str(r["hour_bin"]): round(float(r["mae_min"]), 2) for _, r in by_hour.iterrows()
        },
        "mae_by_distance_min": {
            str(r["dist_bin"]): round(float(r["mae_min"]), 2) for _, r in by_dist.iterrows()
        },
        "figures": sorted(str(p.name) for p in args.out_dir.glob("*.png")),
        "out_dir": str(args.out_dir),
    }
    if by_layer is not None:
        summary["mae_by_origin_layer_min"] = {
            str(r[origin_col]): round(float(r["mae_min"]), 2) for _, r in by_layer.iterrows()
        }
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
