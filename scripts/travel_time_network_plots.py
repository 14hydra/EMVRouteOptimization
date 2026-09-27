"""Figures for scripts/train_travel_time_network.py (test window only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEW = "#2a78d6"   # new model
PREV = "#9a988f"  # previous design (reference, recessive)
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab"]

FINAL = "7 + traffic congestion"
PREVIOUS = "1 Previous design"


def _style():
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )


def _title(fig, title, sub):
    h = fig.get_figheight()
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK, y=1 - 0.12 / h)
    fig.text(0.02, 1 - 0.46 / h, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def ablation(results: dict, out: Path, n_test: int):
    steps = list(results)
    mae = [results[s]["mae_s"] for s in steps]
    r2 = [results[s]["r2"] for s in steps]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [AXIS] + [BLUE_RAMP[min(i, len(BLUE_RAMP) - 1)] for i in range(len(steps) - 1)]
    y = np.arange(len(steps))[::-1]
    ax.barh(y, mae, color=colors, height=0.62, zorder=2)
    for yi, m, r, s in zip(y, mae, r2, steps):
        ax.text(m + 1.5, yi, f"{m:.1f} s   ·   R² {max(r, 0):.2f}", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(steps, fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_xlim(0, max(mae) * 1.28)
    ax.set_xlabel("Mean absolute error on future incidents (seconds, lower is better)")
    ax.spines["left"].set_visible(False)
    _title(fig, "What each fix buys", f"Same {n_test:,} test incidents (Jan–Mar 2025); every model trained only on 2024.")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.84, bottom=0.14)
    fig.savefig(out / "01_ablation_mae.png", dpi=170)
    plt.close(fig)


def pred_vs_actual(preds, y, results, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    lim = 900
    for ax, key, color in ((axes[0], PREVIOUS, PREV), (axes[1], FINAL, NEW)):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", [SURFACE, color])
        ax.hexbin(y, preds[key], gridsize=60, extent=(0, lim, 0, lim), cmap=cmap, mincnt=1, bins="log", linewidths=0)
        ax.plot([0, lim], [0, lim], color=INK2, lw=1)
        r = results[key]
        ax.set_title(f"{'Previous design' if key == PREVIOUS else 'New model'}", color=INK)
        ax.text(0.03, 0.95, f"MAE {r['mae_s']:.0f} s · R² {r['r2']:.2f}", transform=ax.transAxes, va="top", color=INK2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Actual travel time (s)")
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("Predicted (s)")
    _title(fig, "Predicted vs actual", "Darker = more incidents (log scale). Diagonal = perfect prediction. Clipped at 15 min (99% of incidents).")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.08)
    fig.savefig(out / "02_pred_vs_actual.png", dpi=170)
    plt.close(fig)


def error_cdf(preds, y, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for key, color, label in ((PREVIOUS, PREV, "Previous design"), (FINAL, NEW, "New model")):
        ae = np.sort(np.abs(preds[key] - y))
        ax.plot(ae, np.arange(1, len(ae) + 1) / len(ae), color=color, lw=2, label=label)
        for t in (60, 120):
            f = np.mean(ae <= t)
            ax.plot([t], [f], "o", color=color, ms=6, mec=SURFACE, mew=1.5, zorder=3)
            ax.text(t + 6, f - 0.045 if color == PREV else f + 0.02, f"{100 * f:.0f}%", color=INK2, fontsize=9)
    for t in (60, 120):
        ax.axvline(t, color=AXIS, lw=0.8, ls=":")
    ax.set_xlim(0, 480)
    ax.set_ylim(0, 1.01)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel("Absolute error (seconds)")
    ax.set_ylabel("Share of test incidents")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    _title(fig, "How close predictions land", "Share of incidents predicted within a given error; dots mark 1 and 2 minutes.")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(out / "03_error_cdf.png", dpi=170)
    plt.close(fig)


def grouped_mae(te, preds, col, order, fname, title, sub):
    df = pd.DataFrame({"g": te[col].values, "y": te["travel_seconds"].values,
                       "prev": np.abs(preds[PREVIOUS] - te["travel_seconds"].values),
                       "new": np.abs(preds[FINAL] - te["travel_seconds"].values)})
    g = df.groupby("g").agg(n=("y", "size"), prev=("prev", "mean"), new=("new", "mean"))
    g = g.reindex([o for o in order if o in g.index])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(g))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, g["prev"], w, color=PREV, label="Previous design", zorder=2)
    ax.bar(x + w / 2 + 0.01, g["new"], w, color=NEW, label="New model", zorder=2)
    for xi, (_, r) in zip(x, g.iterrows()):
        ax.text(xi + w / 2, r["new"] + 1.5, f"{r['new']:.0f}", ha="center", fontsize=9, color=INK2)
        ax.text(xi - w / 2, r["prev"] + 1.5, f"{r['prev']:.0f}", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{str(i).replace('_', ' ').title()}\nn={int(r.n):,}" for i, r in g.iterrows()], fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_ylabel("MAE (seconds)")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2, ncol=2)
    ax.set_ylim(0, g[["prev", "new"]].max().max() * 1.2)
    _title(fig, title, sub)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.17)
    fig.savefig(fname, dpi=170)
    plt.close(fig)


def calibration(preds, y, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for key, color, label in ((PREVIOUS, PREV, "Previous design"), (FINAL, NEW, "New model")):
        p = preds[key]
        q = pd.qcut(p, 20, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(g["p"], g["y"], "-o", color=color, lw=2, ms=5, mec=SURFACE, mew=1.2, label=label)
    lo, hi = 150, 700
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted travel time (s), 20 equal-count bins")
    ax.set_ylabel("Actual mean travel time (s)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    _title(fig, "Calibration", "On the diagonal = when the model says X seconds, trips take X on average. Wider spread = more useful.")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(out / "06_calibration.png", dpi=170)
    plt.close(fig)


def importance(imp: pd.Series, out: Path):
    top = imp.head(18)[::-1]
    share = 100 * top / imp.sum()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(np.arange(len(top)), share, color=NEW, height=0.62, zorder=2)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.set_xlabel("Share of total gain")
    ax.spines["left"].set_visible(False)
    _title(fig, "What the new model relies on", "LightGBM gain importance (improvement contributed by each feature's splits).")
    fig.subplots_adjust(left=0.3, right=0.97, top=0.88, bottom=0.09)
    fig.savefig(out / "07_feature_importance.png", dpi=170)
    plt.close(fig)


def make_all(results, preds, te, imp, out: Path):
    _style()
    y = te["travel_seconds"].to_numpy()
    ablation(results, out, len(te))
    pred_vs_actual(preds, y, results, out)
    error_cdf(preds, y, out)
    grouped_mae(te, preds, "borough", ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"],
                out / "04_mae_by_borough.png", "Error by borough", "Test window, Jan–Mar 2025.")
    grouped_mae(te, preds, "dest_source", ["alarm_box", "intersection_geocode", "zip_centroid"],
                out / "05_mae_by_destination.png", "Error by how the destination was located",
                "Intersection-geocoded incidents used to be ZIP centroids in the previous design.")
    calibration(preds, y, out)
    importance(imp, out)


def learning_curve(lc: pd.DataFrame, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for key, color, label in ((PREVIOUS, PREV, "Previous design"), (FINAL, NEW, "New model")):
        g = lc[lc["model"] == key].sort_values("n_train")
        ax.plot(g["n_train"], g["mae_s"], "-o", color=color, lw=2, ms=6, mec=SURFACE, mew=1.5, label=label)
        last = g.iloc[-1]
        ax.text(last["n_train"] * 1.08, last["mae_s"], f"{last['mae_s']:.0f} s", va="center", color=INK2, fontsize=9)
        first = g.iloc[0]
        ax.text(first["n_train"] / 1.08, first["mae_s"], f"{first['mae_s']:.0f} s", ha="right", va="center", color=INK2, fontsize=9)
    ax.axvline(2692, color=AXIS, lw=0.8, ls=":")
    ax.text(2692 * 1.07, lc["mae_s"].min(), "old training set\n(2,692 incidents)", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    ax.set_xlabel("Training incidents (log scale)")
    ax.set_ylabel("Test MAE (seconds)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlim(1500, lc["n_train"].max() * 1.6)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
    _title(fig, "More data is the biggest single win", "Same Jan–Mar 2025 test incidents; models trained on random subsamples of 2024.")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.84, bottom=0.13)
    fig.savefig(out / "08_learning_curve.png", dpi=170)
    plt.close(fig)
