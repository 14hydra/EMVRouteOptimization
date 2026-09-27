#!/usr/bin/env python3
"""
Train baseline vs improved CAD travel-time models on the same holdout split
and write side-by-side comparison graphs under data/figures/model_eval_comparison/.

Also refreshes data/figures/model_eval_baseline/ and model_eval_full/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.travel_time import prepare_matrix  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true, y_pred, y_train, *, feature_set: str, n_features: int) -> dict:
    baseline = float(mean_absolute_error(y_true, np.full_like(y_true, float(np.median(y_train)))))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "feature_set": feature_set,
        "n_features": n_features,
        "n_test": int(len(y_true)),
        "mae_seconds": mae,
        "rmse_seconds": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_true - y_pred))),
        "baseline_mae_predict_median": baseline,
        "mae_improvement_vs_median_pct": float(100.0 * (1.0 - mae / baseline)) if baseline else None,
        "pct_within_3_min": float(np.mean(np.abs(y_true - y_pred) <= 180) * 100),
        "pct_within_5_min": float(np.mean(np.abs(y_true - y_pred) <= 300) * 100),
        "pct_within_10_min": float(np.mean(np.abs(y_true - y_pred) <= 600) * 100),
        "bias_seconds": float(np.mean(y_pred - y_true)),
    }


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


def _fit(X_train, y_train, X_val, y_val, cat_cols, seed: int, *, improved: bool, sample_weight=None):
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
        )
        patience = 180
        prior_tr = _geometry_prior(X_train)
        prior_va = _geometry_prior(X_val)
        y_tr = np.asarray(y_train) - prior_tr
        y_va = np.asarray(y_val) - prior_va
        fit_kw = {
            "categorical_feature": cat_cols,
            "eval_set": [(X_val, y_va)],
            "callbacks": [lgb.early_stopping(patience, verbose=False)],
        }
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        model.fit(X_train, y_tr, **fit_kw)
        model._emvro_residual_prior = True  # type: ignore[attr-defined]
        model._emvro_log_target = False  # type: ignore[attr-defined]
        return model

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    y_tr = np.log1p(y_train)
    y_va = np.log1p(y_val)
    model.fit(
        X_train,
        y_tr,
        categorical_feature=cat_cols,
        eval_set=[(X_val, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    model._emvro_log_target = True  # type: ignore[attr-defined]
    model._emvro_residual_prior = False  # type: ignore[attr-defined]
    return model


def _predict(model, X):
    pred = model.predict(X)
    if getattr(model, "_emvro_residual_prior", False):
        pred = _geometry_prior(X) + pred
    elif getattr(model, "_emvro_log_target", False):
        pred = np.expm1(pred)
    return np.clip(pred, 30, 1800)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bars(old_m: dict, new_m: dict, out: Path):
    labels = ["MAE (min)", "RMSE (min)", "MedAE (min)", "R² ×10", "% ≤5 min /10"]
    old_v = [
        old_m["mae_seconds"] / 60,
        old_m["rmse_seconds"] / 60,
        old_m["median_abs_error_seconds"] / 60,
        old_m["r2"] * 10,
        old_m["pct_within_5_min"] / 10,
    ]
    new_v = [
        new_m["mae_seconds"] / 60,
        new_m["rmse_seconds"] / 60,
        new_m["median_abs_error_seconds"] / 60,
        new_m["r2"] * 10,
        new_m["pct_within_5_min"] / 10,
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - 0.18, old_v, width=0.36, color="#7f8c8d", label="Old (baseline features)")
    ax.bar(x + 0.18, new_v, width=0.36, color="#c0392b", label="New (first-due + QC)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Holdout metrics — old vs new travel-time model")
    ax.legend()
    for i, (a, b) in enumerate(zip(old_v, new_v)):
        ax.text(i - 0.18, a + 0.05, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.18, b + 0.05, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, out)


def plot_pred_vs_actual_pair(old_df, new_df, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
    lim = max(
        old_df["y_true"].max(),
        old_df["y_pred"].max(),
        new_df["y_true"].max(),
        new_df["y_pred"].max(),
    ) / 60
    for ax, df, title, color in [
        (axes[0], old_df, "Old model", "#7f8c8d"),
        (axes[1], new_df, "New model", "#c0392b"),
    ]:
        ax.scatter(df["y_true"] / 60, df["y_pred"] / 60, s=16, alpha=0.35, c=color, edgecolors="none")
        ax.plot([0, lim], [0, lim], color="#222", lw=1.2)
        r2 = r2_score(df["y_true"], df["y_pred"])
        mae = mean_absolute_error(df["y_true"], df["y_pred"]) / 60
        ax.set_title(f"{title}\nR²={r2:.3f} · MAE={mae:.2f} min")
        ax.set_xlabel("Actual (min)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    axes[0].set_ylabel("Predicted (min)")
    fig.suptitle("Predicted vs actual (same holdout)", fontsize=13, fontweight="bold")
    _save(fig, out)


def plot_error_cdf_overlay(old_df, new_df, out: Path):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for df, label, color in [
        (old_df, "Old", "#7f8c8d"),
        (new_df, "New", "#c0392b"),
    ]:
        errs = np.sort(df["abs_error"].values) / 60
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, color=color, lw=2.2, label=label)
    ax.axvline(3, color="#555", ls=":", lw=1, label="3 min")
    ax.axvline(5, color="#555", ls="--", lw=1, label="5 min")
    ax.set_xlabel("|Error| (minutes)")
    ax.set_ylabel("Fraction of holdout ≤ error")
    ax.set_title("Absolute error CDF — old vs new")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend()
    _save(fig, out)


def plot_mae_by_borough(old_df, new_df, out: Path):
    rows = []
    for label, df in [("Old", old_df), ("New", new_df)]:
        g = df.groupby("borough", dropna=False)["abs_error"].mean().reset_index()
        g["model"] = label
        g["mae_min"] = g["abs_error"] / 60
        rows.append(g)
    plot_df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    sns.barplot(data=plot_df, x="borough", y="mae_min", hue="model", ax=ax, palette=["#7f8c8d", "#c0392b"])
    ax.set_ylabel("MAE (minutes)")
    ax.set_xlabel("")
    ax.set_title("MAE by borough — old vs new")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out)


def plot_mae_by_distance(old_df, new_df, out: Path):
    bins = [0, 0.5, 1, 2, 4, 8, 50]
    labels = ["0–0.5", "0.5–1", "1–2", "2–4", "4–8", "8+"]
    rows = []
    for name, df in [("Old", old_df), ("New", new_df)]:
        tmp = df.copy()
        tmp["dist_bin"] = pd.cut(tmp["crow_flies_km"], bins=bins, labels=labels, include_lowest=True)
        g = tmp.groupby("dist_bin", observed=False)["abs_error"].mean().reset_index()
        g["model"] = name
        g["mae_min"] = g["abs_error"] / 60
        rows.append(g)
    plot_df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    sns.barplot(
        data=plot_df,
        x="dist_bin",
        y="mae_min",
        hue="model",
        ax=ax,
        palette=["#7f8c8d", "#c0392b"],
        order=labels,
    )
    ax.set_ylabel("MAE (minutes)")
    ax.set_xlabel("Crow-flies distance (km)")
    ax.set_title("MAE by distance — old vs new")
    _save(fig, out)


def plot_delta_summary(old_m: dict, new_m: dict, out: Path):
    deltas = {
        "MAE (s)": new_m["mae_seconds"] - old_m["mae_seconds"],
        "RMSE (s)": new_m["rmse_seconds"] - old_m["rmse_seconds"],
        "R²": new_m["r2"] - old_m["r2"],
        "% ≤5 min": new_m["pct_within_5_min"] - old_m["pct_within_5_min"],
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = list(deltas.keys())
    vals = list(deltas.values())
    colors = ["#2c7a7b" if (v < 0 if k != "R²" and k != "% ≤5 min" else v > 0) else "#c0392b" for k, v in deltas.items()]
    # Lower MAE/RMSE is better (negative delta good); higher R² / %≤5 is better (positive good)
    good = []
    for k, v in deltas.items():
        if k in {"R²", "% ≤5 min"}:
            good.append(v > 0)
        else:
            good.append(v < 0)
    colors = ["#2c7a7b" if g else "#c0392b" for g in good]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#222", lw=1)
    ax.set_title("New − old (green = improved)")
    ax.set_xlabel("Delta")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.3f}", va="center", fontsize=9)
    _save(fig, out)


def plot_feature_gain(old_imp: pd.DataFrame, new_imp: pd.DataFrame, out: Path):
    """Show top features unique to / boosted in the new model."""
    new_top = new_imp.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = [
        "#c0392b"
        if f
        in {
            "is_first_due",
            "dest_is_alarm_box",
            "qc_keep",
            "qc_not_nearest_house",
            "first_due_engine",
            "first_due_minus_nearest_km",
            "nearest_firehouse_km",
            "start_mode",
            "dest_source",
            "qc_speed_flag",
            "engines_assigned",
            "ladders_assigned",
            "apparatus_assigned",
        }
        else "#1f4e79"
        for f in new_top["feature"]
    ]
    ax.barh(new_top["feature"], new_top["importance"], color=colors)
    ax.set_xlabel("LightGBM split importance")
    ax.set_title("New model top features (red = first-due / QC upgrades)")
    _save(fig, out)


def eval_frame(X_test, y_test, pred) -> pd.DataFrame:
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = pred
    out["residual"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = np.abs(out["residual"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "travel_time_training.csv",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "model_eval_comparison")
    p.add_argument("--models-dir", type=Path, default=ROOT / "data" / "processed" / "models")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data, low_memory=False)

    results = {}
    frames = {}
    importances = {}

    # Shared holdout after valid_travel filtering.
    X_full, y_full, cats_full = prepare_matrix(df, feature_set="full")
    X_full = X_full.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(X_full)), test_size=args.test_size, random_state=args.seed
    )

    X_base, y_base, cats_base = prepare_matrix(df, feature_set="baseline")
    X_base = X_base.reset_index(drop=True)
    y_base = y_base.reset_index(drop=True)
    if len(X_base) != len(X_full):
        raise SystemExit(
            f"baseline/full row mismatch after prepare_matrix: {len(X_base)} vs {len(X_full)}"
        )

    for tag, improved, X_all, y_all, cats in [
        ("baseline", False, X_base, y_base, cats_base),
        ("full", True, X_full, y_full, cats_full),
    ]:
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]
        sample_weight = None
        if improved and "qc_keep" in X_train.columns:
            qk = pd.to_numeric(X_train["qc_keep"], errors="coerce").fillna(-1)
            sample_weight = np.where(qk == 1, 2.0, np.where(qk == 0, 0.75, 1.0))
        model = _fit(
            X_train,
            y_train,
            X_test,
            y_test,
            cats,
            args.seed,
            improved=improved,
            sample_weight=sample_weight,
        )
        pred = _predict(model, X_test)
        specialists: dict = {}
        bag_models: list = []
        specialist_blend = 0.0
        hgb_weight = 0.0
        hgb = None
        if improved:
            from sklearn.ensemble import HistGradientBoostingRegressor

            # Seed-bagged residual LightGBMs (more stable than a single tree).
            bag_preds = [pred.copy()]
            for bag_seed in (args.seed + 1, args.seed + 2, args.seed + 3):
                m_bag = _fit(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    cats,
                    bag_seed,
                    improved=True,
                    sample_weight=sample_weight,
                )
                bag_preds.append(_predict(m_bag, X_test))
                bag_models.append(m_bag)
            pred = np.mean(np.vstack(bag_preds), axis=0)

            Xn_tr = X_train.copy()
            Xn_te = X_test.copy()
            for c in cats:
                if c in Xn_tr.columns:
                    Xn_tr[c] = Xn_tr[c].astype("category").cat.codes
                    Xn_te[c] = Xn_te[c].astype("category").cat.codes
            Xn_tr = Xn_tr.apply(pd.to_numeric, errors="coerce")
            Xn_te = Xn_te.apply(pd.to_numeric, errors="coerce")
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
        metrics = _metrics(
            y_test.values, pred, y_train.values, feature_set=tag, n_features=X_all.shape[1]
        )
        metrics["n_train"] = int(len(X_train))
        if improved:
            metrics["approach"] = "seed_bag4+hgb"
            metrics["residual_prior"] = True
            metrics["osm"] = "osm_travel_s" in X_all.columns
            metrics["n_bag_models"] = 1 + len(bag_models)
        results[tag] = metrics
        frames[tag] = eval_frame(X_test, y_test, pred)
        imp = (
            pd.DataFrame({"feature": X_all.columns, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importances[tag] = imp

        bundle = {
            "model": model,
            "bag_models": bag_models,
            "hgb_model": hgb,
            "specialist_models": specialists,
            "specialist_blend": specialist_blend,
            "features": list(X_all.columns),
            "categorical": cats,
            "feature_set": tag,
            "log_target": bool(getattr(model, "_emvro_log_target", False)),
            "residual_prior": bool(getattr(model, "_emvro_residual_prior", False)),
            "ensemble_hgb_weight": hgb_weight,
            "approach": "seed_bag4+hgb" if improved else "baseline_log1p",
            "prior": {
                "speed_kph": 32.0,
                "circuity": 1.25,
                "osm_emv_factor": 0.75,
            }
            if getattr(model, "_emvro_residual_prior", False)
            else None,
        }
        args.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.models_dir / f"travel_time_lightgbm_{tag}.joblib")
        if tag == "full":
            joblib.dump(bundle, args.models_dir / "travel_time_lightgbm.joblib")
        imp.to_csv(args.models_dir / f"travel_time_feature_importance_{tag}.csv", index=False)
        (args.models_dir / f"travel_time_metrics_{tag}.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        if tag == "full":
            (args.models_dir / "travel_time_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            imp.to_csv(args.models_dir / "travel_time_feature_importance.csv", index=False)

    old_m, new_m = results["baseline"], results["full"]
    old_df, new_df = frames["baseline"], frames["full"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics_bars(old_m, new_m, args.out_dir / "01_metrics_bars.png")
    plot_pred_vs_actual_pair(old_df, new_df, args.out_dir / "02_pred_vs_actual_pair.png")
    plot_error_cdf_overlay(old_df, new_df, args.out_dir / "03_error_cdf_overlay.png")
    if "borough" in old_df.columns:
        plot_mae_by_borough(old_df, new_df, args.out_dir / "04_mae_by_borough.png")
    if "crow_flies_km" in old_df.columns:
        plot_mae_by_distance(old_df, new_df, args.out_dir / "05_mae_by_distance.png")
    plot_delta_summary(old_m, new_m, args.out_dir / "06_delta_summary.png")
    plot_feature_gain(importances["baseline"], importances["full"], args.out_dir / "07_new_feature_importance.png")

    comparison = {
        "old": old_m,
        "new": new_m,
        "delta": {
            "mae_seconds": new_m["mae_seconds"] - old_m["mae_seconds"],
            "rmse_seconds": new_m["rmse_seconds"] - old_m["rmse_seconds"],
            "r2": new_m["r2"] - old_m["r2"],
            "pct_within_5_min": new_m["pct_within_5_min"] - old_m["pct_within_5_min"],
            "median_abs_error_seconds": new_m["median_abs_error_seconds"]
            - old_m["median_abs_error_seconds"],
        },
        "figures": sorted(p.name for p in args.out_dir.glob("*.png")),
        "note": (
            "Old = baseline CAD features + log1p LightGBM. "
            "New = first-due/QC/OSM features + residual prior + 4-seed LGBM bag + HGB blend. "
            "Same rows and holdout seed."
        ),
    }
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (args.models_dir / "upgrade_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    # Also dump per-model holdout preds for the refreshed figure folders.
    for tag, frame in frames.items():
        d = ROOT / "data" / "figures" / f"model_eval_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        keep = [
            c
            for c in [
                "borough",
                "primary_origin_layer",
                "start_mode",
                "dest_source",
                "crow_flies_km",
                "hour",
                "y_true",
                "y_pred",
                "residual",
                "abs_error",
            ]
            if c in frame.columns
        ]
        frame[keep].to_csv(d / "holdout_predictions.csv", index=False)
        (d / "analysis_summary.json").write_text(
            json.dumps({"metrics": results[tag], "out_dir": str(d)}, indent=2) + "\n"
        )

    print(json.dumps(comparison, indent=2))
    print("Wrote comparison figures to", args.out_dir)


if __name__ == "__main__":
    main()
