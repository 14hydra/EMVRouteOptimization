#!/usr/bin/env python3
"""
Side-by-side comparison for data/figures/model_eval_comparison/:

  Old  = previous CAD design (crow-flies from guessed house + context LGBM)
  New  = hybrid (network origins + OOF priors + residual bag4 + HGB)

Same chronological holdout on travel_time_network.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_travel_time_hybrid import (  # noqa: E402
    PRIOR_KEYS,
    add_priors,
    chronological_split,
    feature_frame,
    fit_lgbm,
    network_prior,
    numify,
    predict_lgbm,
)

sns.set_theme(style="whitegrid", context="notebook")
omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true, y_pred, y_train, *, label: str) -> dict:
    baseline = float(mean_absolute_error(y_true, np.full_like(y_true, float(np.median(y_train)))))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "label": label,
        "n_test": int(len(y_true)),
        "n_train": int(len(y_train)),
        "mae_seconds": mae,
        "rmse_seconds": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_error_seconds": float(np.median(np.abs(y_true - y_pred))),
        "baseline_mae_predict_median": baseline,
        "mae_improvement_vs_median_pct": float(100.0 * (1.0 - mae / baseline)) if baseline else None,
        "pct_within_3_min": float(np.mean(np.abs(y_true - y_pred) <= 180) * 100),
        "pct_within_5_min": float(np.mean(np.abs(y_true - y_pred) <= 300) * 100),
        "pct_within_60s": float(np.mean(np.abs(y_true - y_pred) <= 60) * 100),
        "bias_seconds": float(np.mean(y_pred - y_true)),
    }


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bars(old_m, new_m, out: Path):
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
    ax.bar(x - 0.18, old_v, width=0.36, color="#7f8c8d", label="Old (CAD crow + LGBM)")
    ax.bar(x + 0.18, new_v, width=0.36, color="#c0392b", label="New (network hybrid)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Holdout metrics — old CAD vs network hybrid")
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
        (axes[0], old_df, "Old CAD model", "#7f8c8d"),
        (axes[1], new_df, "Network hybrid", "#c0392b"),
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
    fig.suptitle("Predicted vs actual (same chronological holdout)", fontsize=13, fontweight="bold")
    _save(fig, out)


def plot_error_cdf_overlay(old_df, new_df, out: Path):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for df, label, color in [(old_df, "Old CAD", "#7f8c8d"), (new_df, "Network hybrid", "#c0392b")]:
        errs = np.sort(df["abs_error"].values) / 60
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, color=color, lw=2.2, label=label)
    ax.axvline(3, color="#555", ls=":", lw=1, label="3 min")
    ax.axvline(5, color="#555", ls="--", lw=1, label="5 min")
    ax.set_xlabel("|Error| (minutes)")
    ax.set_ylabel("Fraction of holdout ≤ error")
    ax.set_title("Absolute error CDF — old vs hybrid")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend()
    _save(fig, out)


def plot_mae_by_borough(old_df, new_df, out: Path):
    rows = []
    for label, df in [("Old CAD", old_df), ("Hybrid", new_df)]:
        g = df.groupby("borough", dropna=False)["abs_error"].mean().reset_index()
        g["model"] = label
        g["mae_min"] = g["abs_error"] / 60
        rows.append(g)
    plot_df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    sns.barplot(data=plot_df, x="borough", y="mae_min", hue="model", ax=ax, palette=["#7f8c8d", "#c0392b"])
    ax.set_ylabel("MAE (minutes)")
    ax.set_xlabel("")
    ax.set_title("MAE by borough — old vs hybrid")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out)


def plot_delta_summary(old_m, new_m, out: Path):
    deltas = {
        "MAE (s)": new_m["mae_seconds"] - old_m["mae_seconds"],
        "RMSE (s)": new_m["rmse_seconds"] - old_m["rmse_seconds"],
        "R²": new_m["r2"] - old_m["r2"],
        "% ≤5 min": new_m["pct_within_5_min"] - old_m["pct_within_5_min"],
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = list(deltas.keys())
    vals = list(deltas.values())
    good = []
    for k, v in deltas.items():
        good.append(v > 0 if k in {"R²", "% ≤5 min"} else v < 0)
    colors = ["#2c7a7b" if g else "#c0392b" for g in good]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#222", lw=1)
    ax.set_title("Hybrid − old CAD (green = improved)")
    ax.set_xlabel("Delta")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.3f}", va="center", fontsize=9)
    _save(fig, out)


def plot_three_way_mae(base_m, cad_m, hyb_m, out: Path):
    labels = ["Old CAD\n(crow LGBM)", "CAD bag4+HGB\n(first-due/OSM)", "Network hybrid\n(bag4+HGB)"]
    maes = [base_m["mae_seconds"] / 60, cad_m["mae_seconds"] / 60, hyb_m["mae_seconds"] / 60]
    r2s = [base_m["r2"], cad_m["r2"], hyb_m["r2"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = ["#7f8c8d", "#2980b9", "#c0392b"]
    axes[0].bar(labels, maes, color=colors)
    axes[0].set_ylabel("MAE (min)")
    axes[0].set_title("MAE (same chronological holdout)")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].bar(labels, r2s, color=colors)
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² (same chronological holdout)")
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    _save(fig, out)


def _old_cad_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    num = [
        c
        for c in [
            "prev_crow_km_primary",
            "prev_crow_km_nearest",
            "prev_dest_lat",
            "prev_dest_lon",
            "prev_dest_is_box",
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
            "wx_temp_c",
            "wx_humidity",
            "wx_precip_mm",
            "wx_wind_kmh",
            "wx_is_precip",
            "wx_is_snow",
        ]
        if c in df.columns
    ]
    cats = [c for c in ["borough", "call_type", "call_group", "alarm_level"] if c in df.columns]
    X = df[num + cats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X, cats


def _fit_old(X_tr, y_tr, X_va, y_va, cats, seed: int):
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=12,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_tr,
        np.log1p(y_tr),
        categorical_feature=cats,
        eval_set=[(X_va, np.log1p(y_va))],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "travel_time_network.parquet")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "model_eval_comparison")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--train-end", type=str, default=None)
    p.add_argument("--test-start", type=str, default=None)
    p.add_argument("--valid-start", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = pd.read_parquet(args.data)
    d = d.dropna(subset=["travel_seconds"]).copy()
    d = d[d["travel_seconds"].between(45, 1200)].reset_index(drop=True)
    d["dest_key"] = d["dest_lat"].round(5).astype(str) + "," + d["dest_lon"].round(5).astype(str)
    d["incident_datetime"] = pd.to_datetime(d["incident_datetime"], errors="coerce")
    d = d.sort_values("incident_datetime").reset_index(drop=True)

    if args.train_end and args.test_start:
        test_start = pd.Timestamp(args.test_start)
        valid_start = pd.Timestamp(args.valid_start) if args.valid_start else (test_start - pd.DateOffset(months=1))
        train_fit = d[d["incident_datetime"] < valid_start].reset_index(drop=True)
        valid = d[(d["incident_datetime"] >= valid_start) & (d["incident_datetime"] < test_start)].reset_index(drop=True)
        test = d[d["incident_datetime"] >= test_start].reset_index(drop=True)
        if len(valid) < 500:
            pre = d[d["incident_datetime"] < test_start].reset_index(drop=True)
            val_n = max(500, int(0.1 * len(pre)))
            train_fit = pre.iloc[:-val_n].reset_index(drop=True)
            valid = pre.iloc[-val_n:].reset_index(drop=True)
        print(f"time split: train_fit {len(train_fit):,} | valid {len(valid):,} | test {len(test):,}")
    else:
        train, test = chronological_split(d, test_frac=args.test_frac)
        val_n = max(500, int(0.1 * len(train)))
        valid = train.iloc[-val_n:].reset_index(drop=True)
        train_fit = train.iloc[:-val_n].reset_index(drop=True)
    full_train = pd.concat([train_fit, valid], ignore_index=True)

    # --- Old CAD-style model ---
    Xo_tr, cats_o = _old_cad_features(train_fit)
    Xo_va, _ = _old_cad_features(valid)
    Xo_te, _ = _old_cad_features(test)
    cols_o = list(Xo_tr.columns)
    Xo_va = Xo_va.reindex(columns=cols_o)
    Xo_te = Xo_te.reindex(columns=cols_o)
    for c in cats_o:
        for fr in (Xo_tr, Xo_va, Xo_te):
            fr[c] = fr[c].astype("category")
    old_model = _fit_old(
        Xo_tr, train_fit["travel_seconds"].to_numpy(), Xo_va, valid["travel_seconds"].to_numpy(), cats_o, args.seed
    )
    pred_old = np.clip(np.expm1(old_model.predict(Xo_te)), 30, 1800)

    # --- Hybrid ---
    train_p, (valid_p, test_p) = add_priors(train_fit, [valid, test], PRIOR_KEYS)
    full_p, (test_final,) = add_priors(full_train, [test], PRIOR_KEYS)
    X_tr, cats = feature_frame(train_p)
    X_va, _ = feature_frame(valid_p)
    X_full, _ = feature_frame(full_p)
    X_te, _ = feature_frame(test_final)
    cols = list(X_full.columns)
    X_tr = X_tr.reindex(columns=cols)
    X_va = X_va.reindex(columns=cols)
    X_te = X_te.reindex(columns=cols)
    for c in cats:
        for fr in (X_tr, X_va, X_full, X_te):
            if c in fr.columns:
                fr[c] = fr[c].astype("category")

    bag_preds = []
    bag_models = []
    for seed in (args.seed, args.seed + 1, args.seed + 2, args.seed + 3):
        m = fit_lgbm(
            X_tr,
            train_p["travel_seconds"].to_numpy(),
            X_va,
            valid_p["travel_seconds"].to_numpy(),
            cats,
            seed,
        )
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
    hgb.fit(Xn_tr, full_p["travel_seconds"].to_numpy() - prior_tr)
    pred_h = np.clip(prior_te + hgb.predict(Xn_te), 30, 1800)
    pred_new = 0.65 * pred_bag + 0.35 * pred_h

    # CAD bag4 on first-due/OSM training table metrics (from disk) as middle reference
    cad_path = ROOT / "data" / "processed" / "models" / "travel_time_metrics_full.json"
    cad_m = json.loads(cad_path.read_text()) if cad_path.exists() else None

    y_te = test["travel_seconds"].to_numpy()
    y_tr = full_train["travel_seconds"].to_numpy()
    old_m = _metrics(y_te, pred_old, y_tr, label="old_cad_crow_lgbm")
    new_m = _metrics(y_te, pred_new, y_tr, label="network_hybrid_bag4_hgb")
    new_m["approach"] = "network_features+residual_bag4+hgb"
    new_m["n_features"] = int(X_full.shape[1])

    old_df = pd.DataFrame(
        {
            "borough": test["borough"].astype(str).values,
            "y_true": y_te,
            "y_pred": pred_old,
            "abs_error": np.abs(pred_old - y_te),
        }
    )
    new_df = pd.DataFrame(
        {
            "borough": test["borough"].astype(str).values,
            "y_true": y_te,
            "y_pred": pred_new,
            "abs_error": np.abs(pred_new - y_te),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics_bars(old_m, new_m, args.out_dir / "01_metrics_bars.png")
    plot_pred_vs_actual_pair(old_df, new_df, args.out_dir / "02_pred_vs_actual_pair.png")
    plot_error_cdf_overlay(old_df, new_df, args.out_dir / "03_error_cdf_overlay.png")
    plot_mae_by_borough(old_df, new_df, args.out_dir / "04_mae_by_borough.png")
    plot_delta_summary(old_m, new_m, args.out_dir / "06_delta_summary.png")

    # Three-way panel if CAD bag metrics exist (different split — labeled as reference)
    if cad_m is not None:
        # Re-score a single LGBM bag member as "mid" on THIS holdout for fair 3-way
        mid_pred = pred_bag  # bag without HGB on same holdout
        mid_m = _metrics(y_te, mid_pred, y_tr, label="network_bag4_no_hgb")
        plot_three_way_mae(old_m, mid_m, new_m, args.out_dir / "08_three_way_mae_r2.png")
        # Also keep disk CAD bag metrics noted
        three = {
            "old_cad_crow_same_holdout": old_m,
            "network_bag4_same_holdout": mid_m,
            "network_hybrid_same_holdout": new_m,
            "cad_bag4_hgb_random_split_reference": cad_m,
        }
    else:
        three = None

    # Feature importance from first bag model
    imp = (
        pd.DataFrame({"feature": cols, "importance": bag_models[0].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    top = imp.head(18).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#c0392b")
    ax.set_xlabel("LightGBM split importance")
    ax.set_title("Hybrid model top features")
    _save(fig, args.out_dir / "07_new_feature_importance.png")

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
        "three_way": three,
        "figures": sorted(p.name for p in args.out_dir.glob("*.png")),
        "note": (
            "Old = previous CAD crow-flies origin + context LightGBM. "
            "New = network origins + OOF location priors + residual 4-seed bag + HGB. "
            "Same chronological holdout on travel_time_network.parquet."
        ),
    }
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (ROOT / "data" / "processed" / "models" / "upgrade_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    print(json.dumps(comparison, indent=2))
    print("Wrote comparison figures to", args.out_dir)


if __name__ == "__main__":
    main()
