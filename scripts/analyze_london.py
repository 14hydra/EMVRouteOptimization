#!/usr/bin/env python3
"""
Initial analysis of the London Fire Brigade data, and which street
characteristics have the biggest impact on fire engine driving time.

Inputs: data/processed/lfb_travel_training.csv (scripts/build_lfb_travel_training.py),
data/raw/london/lfb_incidents_planner.csv, data/raw/london/lfb_mobilisations_*.csv.

Writes data/figures/london_eda/*.png and summary.json:

  01  driving and turnout time distributions
  02  driving time vs road distance (with the Kolesar curve)
  03  driving time and turnout by hour of day
  04  first-engine attendance by borough vs the 6-minute standard
  05  delay reasons LFB recorded
  06  monthly median driving time, 2021-2026
  07  model comparison on held-out 2025 trips
  08  street feature impact: mean |SHAP| (seconds per trip)
  09  street feature impact: MAE increase when a feature group is removed
  10  street feature effects: seconds per km per +1 SD (pace regression, 95% bootstrap CI)
  11  how the top street features move predicted driving time (SHAP dependence)

Street features describe the shortest legal route, not the route actually driven.
Effects are associations, not causes.

Run:  .venv/bin/python scripts/analyze_london.py
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emvro.kolesar import fit_kolesar  # noqa: E402
from emvro.route_features import ROUTE_FEATURES  # noqa: E402
import train_travel_time_lfb as T  # noqa: E402

RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "london_eda"

# Reference palette (dataviz skill, light mode).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED, MUTED = "#2a78d6", "#eb6834", "#e34948", "#9a9893"

FEATURE_LABELS = {
    "share_major": "Share on motorway / trunk roads",
    "share_primary": "Share on A roads (primary)",
    "share_secondary": "Share on B roads (secondary)",
    "share_tertiary": "Share on tertiary roads",
    "share_residential": "Share on residential streets",
    "share_minor": "Share on minor / unclassified roads",
    "share_oneway": "Share on one-way streets",
    "share_20mph": "Share with 20 mph limit",
    "share_bus_lane": "Share with a bus lane",
    "share_bridge_tunnel": "Share on bridges / tunnels",
    "mean_lanes": "Average lanes",
    "signals_per_km": "Traffic signals per km",
    "calming_per_km": "Traffic calming per km (humps etc.)",
    "crossings_per_km": "Pedestrian crossings per km",
    "give_way_per_km": "Give-way / stop signs per km",
    "junctions_per_km": "Junctions per km",
    "turns_per_km": "Turns per km (> 45°)",
    "sharp_turns_per_km": "Sharp turns per km (> 120°)",
    "roundabouts_per_km": "Roundabouts per km",
}
GROUPS = {
    "Road class mix": [f for f in ROUTE_FEATURES if f.startswith("share_") and f not in
                       ("share_oneway", "share_20mph", "share_bus_lane", "share_bridge_tunnel")],
    "One-way streets": ["share_oneway"],
    "20 mph limits": ["share_20mph"],
    "Bus lanes": ["share_bus_lane"],
    "Bridges / tunnels": ["share_bridge_tunnel"],
    "Lanes": ["mean_lanes"],
    "Traffic signals": ["signals_per_km"],
    "Traffic calming": ["calming_per_km"],
    "Crossings / give-way": ["crossings_per_km", "give_way_per_km"],
    "Junctions / turns / roundabouts": ["junctions_per_km", "turns_per_km", "sharp_turns_per_km",
                                        "roundabouts_per_km"],
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "sans-serif", "legend.frameon": False,
        "legend.fontsize": 9,
    })


def save(fig, name: str, written: list):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)


def note(ax, text):
    ax.text(0, -0.16, text, transform=ax.transAxes, fontsize=8, color=INK2, va="top")


# --------------------------------------------------------------------------- #
# Exploratory charts
# --------------------------------------------------------------------------- #


def fig_distributions(d, w):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, color, title in [
        (axes[0], "drive_s", BLUE, "Driving time (station → incident)"),
        (axes[1], "turnout_s", ORANGE, "Turnout time (call → engine leaves)"),
    ]:
        v = d[col].dropna()
        v = v[(v > 0) & (v < 1200)] / 60
        ax.hist(v, bins=80, color=color, edgecolor=SURFACE, linewidth=0.3)
        med = v.median()
        ax.axvline(med, color=INK, lw=1, ls="--")
        ax.text(med, ax.get_ylim()[1] * 0.95, f"  median {med:.1f} min", color=INK, fontsize=9, va="top")
        ax.set_title(title)
        ax.set_xlabel("minutes")
        ax.set_ylabel("engine trips")
    save(fig, "01_time_distributions.png", w)


def fig_time_vs_distance(d, kol, w):
    s = d.sample(min(300_000, len(d)), random_state=0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    hb = ax.hexbin(s["road_km"], s["drive_s"] / 60, gridsize=70, extent=(0, 10, 0, 20), mincnt=1,
                   cmap="Blues", bins="log", linewidths=0)
    x = np.linspace(0.05, 10, 200)
    ax.plot(x, kol.predict(x) / 60, color=ORANGE, lw=2, label="Kolesar curve (fit on 2021–24)")
    b = s.assign(bin=pd.cut(s["road_km"], np.arange(0, 10.5, 0.5))).groupby("bin", observed=True)["drive_s"].median()
    ax.plot([i.mid for i in b.index], b.values / 60, "o", ms=5, color=INK, label="median per 0.5 km")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.set_xlabel("shortest legal road distance (km)")
    ax.set_ylabel("driving time (min)")
    ax.set_title("Driving time grows fast at first, then roughly linearly")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="trips (log scale)")
    save(fig, "02_time_vs_distance.png", w)


def fig_by_hour(d, w):
    g = d.groupby("hour_of_call").agg(pace=("pace_s_per_km", "median"), turnout=("turnout_s", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(g.index, g["pace"], color=BLUE, lw=2, marker="o", ms=4)
    axes[0].set_title("Driving pace by hour (seconds per road km)")
    axes[0].set_ylabel("median s / km (higher = slower)")
    axes[1].plot(g.index, g["turnout"], color=ORANGE, lw=2, marker="o", ms=4)
    axes[1].set_title("Turnout time by hour")
    axes[1].set_ylabel("median seconds")
    for ax in axes:
        ax.set_xlabel("hour of call")
        ax.set_xticks(range(0, 24, 3))
    save(fig, "03_by_hour.png", w)
    return g


def fig_by_borough(inc, w):
    inc = inc[inc["cal_year"].isin([2024, 2025])]
    g = (inc.groupby("borough")["travel_seconds"].agg(["mean", "size"])
         .query("size >= 500").sort_values("mean"))
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [RED if v > 360 else BLUE for v in g["mean"]]
    names = [string.capwords(b.lower()).replace(" And ", " and ").replace(" Upon ", " upon ").replace(" Of ", " of ")
             for b in g.index]
    ax.barh(names, g["mean"] / 60, color=colors, height=0.7)
    ax.axvline(6, color=INK, lw=1, ls="--")
    ax.text(6.05, -0.9, "LFB standard: 6 min average", fontsize=9, color=INK)
    ax.set_xlabel("average first-engine attendance (min), 2024–25")
    ax.set_title("First-engine attendance by borough")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "04_attendance_by_borough.png", w)
    return g


def fig_delays(w):
    parts = []
    for f in ["lfb_mobilisations_2021_2024.csv", "lfb_mobilisations_2025.csv"]:
        parts.append(pd.read_csv(RAW / f, usecols=["DelayCode_Description"], encoding="utf-8-sig"))
    m = pd.concat(parts)["DelayCode_Description"]
    n_all = len(m)
    held = m[m.notna() & ~m.isin(["Not held up", "NULL"])].value_counts()
    held = held[held / n_all >= 0.001].sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    street = {"Traffic, roadworks, etc", "Traffic calming measures"}
    ax.barh(held.index, held.values / n_all * 100, color=[ORANGE if k in street else BLUE for k in held.index],
            height=0.7)
    for i, v in enumerate(held.values / n_all * 100):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("% of all engine mobilisations, 2021–2026")
    ax.set_title("Why engines were held up (LFB's own delay codes)")
    ax.grid(axis="y", visible=False)
    note(ax, "Orange = street-related causes. Most mobilisations have no delay code recorded.")
    save(fig, "05_delay_reasons.png", w)
    return (held / n_all).round(4).to_dict()


def fig_monthly(d, w):
    m = d.assign(month=pd.to_datetime(d["date_of_call"]).dt.to_period("M")).groupby("month")["drive_s"].median()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(m.index.to_timestamp(), m.values / 60, color=BLUE, lw=2)
    ax.set_ylabel("median driving time (min)")
    ax.set_title("Monthly median driving time")
    save(fig, "06_monthly_drive_time.png", w)


# --------------------------------------------------------------------------- #
# Street feature impact
# --------------------------------------------------------------------------- #


def features(df, cats, street: list[str]):
    x = T.add_features(df, cats)
    for f in street:
        x[f] = df[f].to_numpy()
    return x


def fit_one(x, y, seed=42):
    m = T.make_regressor(seed)
    m.fit(x, y, categorical_feature=["borough_code"])
    return m


def street_impact(d, w, kol_road):
    tr = d[d.cal_year.between(2021, 2024)].reset_index(drop=True)
    te = d[d.cal_year == 2025].reset_index(drop=True)
    cats = sorted(tr["borough"].astype(str).str.upper().unique())
    ytr, yte = tr["drive_s"].to_numpy(), te["drive_s"].to_numpy()
    street = list(ROUTE_FEATURES)
    xtr_b, xte_b = features(tr, cats, []), features(te, cats, [])
    xtr, xte = features(tr, cats, street), features(te, cats, street)
    clip = lambda p: np.clip(p, *T.DRIVE_CLIP_S)  # noqa: E731
    mae = lambda p: float(np.abs(clip(p) - yte).mean())  # noqa: E731

    print("Training LightGBM without / with street features (4 seeds each)…")
    base = T.fit_bag(xtr_b, ytr)
    full = T.fit_bag(xtr, ytr)
    p_base, p_full = T.predict_bag(base, xte_b), T.predict_bag(full, xte)
    kol_crow = fit_kolesar(tr["crow_km"], ytr)
    road_kph = tr["road_km"].sum() / ytr.sum() * 3600
    models = {
        "Constant guess (median)": mae(np.full_like(yte, np.median(ytr), dtype=float)),
        "Straight line @ 32 km/h": mae(te["crow_km"] / 32 * 3600),
        f"Road distance @ {road_kph:.0f} km/h": mae(te["road_km"] / road_kph * 3600),
        "Kolesar (straight line)": mae(kol_crow.predict(te["crow_km"])),
        "Kolesar (road distance)": mae(kol_road.predict(te["road_km"])),
        "LightGBM (distance + time)": mae(p_base),
        "LightGBM + street features": mae(p_full),
    }
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(models)[::-1]
    vals = [models[k] for k in names]
    ax.barh(names, vals, color=[ORANGE if k.endswith("street features") else BLUE for k in names], height=0.65)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f} s", va="center", fontsize=9, color=INK)
    ax.set_xlabel("mean absolute error on held-out 2025 trips (seconds, lower is better)")
    ax.set_title("Predicting driving time: model comparison")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    save(fig, "07_model_comparison.png", w)

    # SHAP values (LightGBM's exact TreeSHAP via pred_contrib), averaged over the 4 seeds.
    print("SHAP values…")
    samp = xte.sample(40_000, random_state=1)
    contrib = np.mean([m.predict(samp, pred_contrib=True) for m in full], axis=0)[:, :-1]
    shap = pd.DataFrame(contrib, columns=xte.columns, index=samp.index)
    imp = shap[street].abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh([FEATURE_LABELS[f] for f in imp.index], imp.values, color=BLUE, height=0.7)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f" {v:.1f} s", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("mean |SHAP| (seconds the feature moves a trip's predicted driving time)")
    ax.set_title("Street features ranked by impact on predicted driving time")
    ax.grid(axis="y", visible=False)
    note(ax, f"For scale: road distance {shap['road_km'].abs().mean():.0f} s, "
             f"straight-line distance {shap['crow_km'].abs().mean():.0f} s, hour {shap['hour'].abs().mean():.0f} s.")
    save(fig, "08_street_shap.png", w)

    # Drop-one-group ablation (1 seed each; compared with the 1-seed full model).
    print("Ablation…")
    full1 = mae(fit_one(xtr, ytr).predict(xte))
    abl = {}
    for gname, cols in GROUPS.items():
        keep = [c for c in xtr.columns if c not in cols]
        abl[gname] = mae(fit_one(xtr[keep], ytr).predict(xte[keep])) - full1
    abl["All street features"] = mae(fit_one(xtr_b, ytr).predict(xte_b)) - full1
    a = pd.Series(abl).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(a.index, a.values, color=[ORANGE if k == "All street features" else BLUE for k in a.index], height=0.7)
    for i, v in enumerate(a.values):
        ax.text(max(v, 0), i, f" {v:+.2f} s", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("increase in MAE when the group is removed (seconds)")
    ax.set_title("What the model loses without each street feature group")
    ax.grid(axis="y", visible=False)
    note(ax, "Groups overlap (e.g. 20 mph streets are mostly residential), so a removed group can be partly "
             "covered by the others.")
    save(fig, "09_street_ablation.png", w)

    # Pace regression with bootstrap CIs.
    print("Pace regression + bootstrap…")
    effects = pace_regression(d[d.cal_year.between(2021, 2025)], street)
    fig_effects(effects, w)


    # SHAP dependence for the top 4 street features.
    top = imp.index[::-1][:4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, f in zip(axes, top):
        v = samp[f]
        bins = pd.qcut(v.rank(method="first"), 20, labels=False)
        g = pd.DataFrame({"x": v, "s": shap[f]}).groupby(bins).median()
        ax.scatter(v, shap[f], s=2, alpha=0.08, color=BLUE, linewidths=0)
        ax.plot(g["x"], g["s"], color=ORANGE, lw=2)
        ax.axhline(0, color=INK2, lw=0.8)
        lo, hi = np.nanpercentile(v, [1, 99])
        ax.set_xlim(lo, hi)
        ax.set_title(FEATURE_LABELS[f], fontsize=10)
    axes[0].set_ylabel("effect on predicted driving time (s)")
    fig.suptitle("How the top street features move predicted driving time", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    save(fig, "11_shap_dependence.png", w)

    return {
        "model_mae_2025_s": {k: round(v, 2) for k, v in models.items()},
        "shap_mean_abs_s": imp[::-1].round(2).to_dict(),
        "shap_scale_s": {c: round(float(shap[c].abs().mean()), 1) for c in ["road_km", "crow_km", "hour"]},
        "ablation_mae_increase_s": a[::-1].round(3).to_dict(),
        "pace_effects_s_per_km_per_sd": effects.round(2).to_dict(orient="index"),
    }


def fig_effects(effects: pd.DataFrame, w):
    e = effects.sort_values("coef")
    fig, ax = plt.subplots(figsize=(9, 7.2))
    y = np.arange(len(e))
    # Gray when the 95% CI crosses zero (no clear direction).
    color = [MUTED if lo <= 0 <= hi else RED if c > 0 else BLUE for c, lo, hi in zip(e["coef"], e["lo"], e["hi"])]
    ax.hlines(y, e["lo"], e["hi"], color=color, lw=2)
    ax.plot(e["coef"], y, "o", color=SURFACE, ms=9, zorder=3)
    ax.scatter(e["coef"], y, color=color, s=40, zorder=4)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [FEATURE_LABELS[f] for f in e.index])
    ax.set_xlabel("seconds per km for +1 standard deviation (95% bootstrap CI)")
    ax.set_title("Street features linked to slower (red) or faster (blue) driving")
    ax.grid(axis="y", visible=False)
    note(ax, "Linear model of pace (s/km) on trips ≥ 1 km, controlling for distance, hour and busy engines.\n"
             "Road classes are relative to residential streets. Gray = 95% CI crosses zero. Associations, not causes.")
    save(fig, "10_street_effects_per_km.png", w)


def pace_regression(d, street, n_boot=200, n_rows=300_000, seed=0):
    """OLS of pace (s/km) on standardized street features + controls; bootstrap CIs."""
    d = d[(d["road_km"] >= 1.0)].dropna(subset=street + ["drive_s"])
    d = d.sample(min(n_rows, len(d)), random_state=seed)
    pace = (d["drive_s"] / d["road_km"]).to_numpy()
    feats = [f for f in street if f != "share_residential"]  # reference road class (shares sum to 1)
    Z = (d[feats] - d[feats].mean()) / d[feats].std()
    ctrl = pd.concat(
        [np.log(d["road_km"]).rename("log_km"), d["busy_flag"].astype(float),
         pd.get_dummies(d["hour_of_call"], prefix="h", drop_first=True, dtype=float)], axis=1)
    X = np.column_stack([np.ones(len(d)), Z.to_numpy(), ctrl.to_numpy()])
    coef = np.linalg.lstsq(X, pace, rcond=None)[0][1 : 1 + len(feats)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boots.append(np.linalg.lstsq(X[i], pace[i], rcond=None)[0][1 : 1 + len(feats)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    sd = d[feats].std()
    return pd.DataFrame({"coef": coef, "lo": lo, "hi": hi, "sd_of_feature": sd.to_numpy()}, index=feats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    written: list[str] = []

    d = pd.read_csv(args.data)
    d["pace_s_per_km"] = d["drive_s"] / d["road_km"].where(d["road_km"] >= 1.0)
    inc = pd.read_csv(RAW / "lfb_incidents_planner.csv")
    kol_road = fit_kolesar(d.loc[d.cal_year.between(2021, 2024), "road_km"],
                           d.loc[d.cal_year.between(2021, 2024), "drive_s"])

    print("Exploratory charts…")
    fig_distributions(d, written)
    fig_time_vs_distance(d, kol_road, written)
    hour = fig_by_hour(d, written)
    bor = fig_by_borough(inc, written)
    delays = fig_delays(written)
    fig_monthly(d, written)
    impact = street_impact(d, written, kol_road)

    summary = {
        "n_trips": int(len(d)),
        "years": sorted(int(y) for y in d["cal_year"].unique()),
        "median_drive_s": float(d["drive_s"].median()),
        "median_turnout_s": float(d["turnout_s"].median()),
        "median_road_km": float(d["road_km"].median()),
        "hour_median_pace_s_per_km": hour["pace"].round(1).to_dict(),
        "hour_median_turnout_s": hour["turnout"].round(1).to_dict(),
        "borough_mean_attendance_s_2024_25": bor["mean"].round(1).to_dict(),
        "boroughs_over_6_min": [b for b, v in bor["mean"].items() if v > 360],
        "delay_reason_share": delays,
        **impact,
        "figures": written,
        "caveat": "Street features describe the shortest legal route, not the route driven; associations, not causes.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("model_mae_2025_s", "shap_mean_abs_s", "ablation_mae_increase_s")},
                     indent=2))
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
