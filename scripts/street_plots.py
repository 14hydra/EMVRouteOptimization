"""Figures for scripts/analyze_street_characteristics.py.

Styling follows the dataviz palette: blue↔red diverging with a neutral gray
midpoint for signed quantities, categorical blue/orange for the two model
specs, recessive grid, ink-colored text (never series-colored).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from emvro.street_analysis import STREET_FEATURES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#184f95", "#2a78d6", "#9ec5f4", NEUTRAL, "#f2a9a4", "#e34948", "#a62b2b"],
)

SPEC_STYLE = {
    "All routes": (BLUE, "o"),
    "All routes + borough FE": (ORANGE, "s"),
}
GROUP_ORDER = ["Road geometry", "Priority lanes", "Road class", "Friction", "Route shape"]


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
        }
    )


def _luma(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# --------------------------------------------------------------------------
# 1. Correlation heatmap
# --------------------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame, out: Path, summary: dict) -> pd.DataFrame:
    order = list(STREET_FEATURES) + ["log_crow_km", "log_travel_s"]
    labels = {k: v[0] for k, v in STREET_FEATURES.items()}
    labels["log_crow_km"] = "Straight-line distance (log)"
    labels["log_travel_s"] = "EMS travel time (log)"
    groups = {k: v[1] for k, v in STREET_FEATURES.items()}
    groups["log_crow_km"] = "Route shape"
    groups["log_travel_s"] = "Outcome"

    corr = df[order].corr(method="spearman")
    corr.to_csv(out / "correlation_matrix.csv")

    # Staircase: rows drop the first variable, columns drop the last (no diagonal)
    rows, cols = order[1:], order[:-1]
    m = len(rows)
    fig, ax = plt.subplots(figsize=(11.5, 9.0))
    mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    data = np.ma.array(corr.loc[rows, cols].to_numpy(), mask=mask)
    im = ax.imshow(data, cmap=DIVERGING, vmin=-1, vmax=1)

    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)

    for i in range(m):
        for j in range(i + 1):
            v = corr.loc[rows[i], cols[j]]
            txt = INK if _luma(DIVERGING((v + 1) / 2)) > 0.45 else "#ffffff"
            ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=8.6, color=txt)

    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels([labels[c] for c in cols], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([labels[c] for c in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Emphasise the outcome row and call out its headline
    ax.axhline(m - 1.5, color=INK2, linewidth=1.0)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    max_rho = float(corr.loc["log_travel_s", list(STREET_FEATURES)].abs().max())
    ax.text(m - 0.6, 1.0,
            f"Bottom row: every street characteristic has |ρ| ≤ {max_rho:.2f}\nwith observed EMS travel time. None tracks it\non its own; the strong structure is among the\ncharacteristics themselves (e.g. width ↔ lanes).",
            ha="right", va="center", fontsize=10, color=INK2, linespacing=1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.72)
    cbar.set_label("Spearman rank correlation", color=INK2)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["−1", "−0.5", "0", "+0.5", "+1"])

    fig.suptitle("Which street characteristics move together, and with EMS travel time",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.985)
    fig.text(0.06, 0.945,
             f"Spearman correlations across {summary['incidents']:,} EMS incidents on {summary['unique_routes']} distinct "
             "routes (NYC, 1–9 Jun 2024). Bottom row: correlation with travel time.",
             ha="left", fontsize=10, color=INK2)
    fig.text(0.06, 0.012,
             "Street attributes: NYC Centerline + DOT bus lanes + OpenStreetMap, length-weighted along the inferred "
             "route. Route starts are inferred (unit GPS is not public).",
             ha="left", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.9, bottom=0.2)
    fig.savefig(out / "01_correlation_heatmap.png", dpi=170)
    plt.close(fig)
    return corr


# --------------------------------------------------------------------------
# 2. Regression coefficient plot
# --------------------------------------------------------------------------
def _fmt_sd(feature: str, sd: float) -> str:
    if feature.endswith("_share"):
        return f"{100 * sd:.0f} pts"
    if feature == "street_width_ft":
        return f"{sd:.1f} ft"
    if feature == "posted_speed_mph":
        return f"{sd:.1f} mph"
    if feature == "block_length_m":
        return f"{sd:.0f} m"
    return f"{sd:.2f}"


def coefficient_plot(coefs: pd.DataFrame, scale: pd.DataFrame, out: Path, summary: dict) -> None:
    specs = [s for s in SPEC_STYLE if s in set(coefs["spec"])]
    feats = [f for g in GROUP_ORDER for f in STREET_FEATURES if STREET_FEATURES[f][1] == g]

    # y positions with a gap + header per group
    ypos, y, group_rows = {}, 0.0, []
    last = None
    for f in feats:
        g = STREET_FEATURES[f][1]
        if g != last:
            y += 0.9
            group_rows.append((g, y))
            y += 0.75
            last = g
        ypos[f] = y
        y += 1.0
    height = y + 1.3

    fig, ax = plt.subplots(figsize=(11, 0.34 * height + 2.3))
    ax.set_ylim(height - 0.6, -0.4)
    off = {s: d for s, d in zip(specs, np.linspace(-0.16, 0.16, len(specs)))}

    xs_lo = coefs["pct_lo"].min()
    xs_hi = coefs["pct_hi"].max()
    lim = max(abs(xs_lo), abs(xs_hi)) * 1.08
    ax.set_xlim(-lim, lim * 1.55)  # room on the right for the "1 SD =" column

    ax.axvline(0, color=INK2, linewidth=1.0, zorder=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for s in specs:
        color, marker = SPEC_STYLE[s]
        sub = coefs[coefs["spec"] == s].set_index("feature")
        for f in feats:
            r = sub.loc[f]
            yy = ypos[f] + off[s]
            ax.plot([r["pct_lo"], r["pct_hi"]], [yy, yy], color=color, linewidth=1.8, solid_capstyle="round", zorder=2)
            ax.scatter(
                [r["pct_change"]], [yy], s=52, marker=marker, zorder=3,
                facecolor=color if r["significant"] else SURFACE, edgecolor=color, linewidth=1.8,
            )

    ax.set_yticks([ypos[f] for f in feats])
    ax.set_yticklabels([STREET_FEATURES[f][0] for f in feats], fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    for g, gy in group_rows:
        ax.text(-lim, gy - 0.05, g.upper(), fontsize=8.5, fontweight="bold", color=MUTED, va="center", ha="left")
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    # "1 SD =" column
    xcol = lim * 1.03
    ax.text(xcol, -0.15, "1 SD =", fontsize=8.5, fontweight="bold", color=MUTED, ha="left", va="center")
    sc = scale["sd"]
    for f in feats:
        ax.text(xcol, ypos[f], _fmt_sd(f, float(sc[f])), fontsize=9, color=INK2, ha="left", va="center")

    ax.set_xticks([t for t in ax.get_xticks() if -lim <= t <= lim])
    ax.set_xlabel("Change in EMS travel time per +1 SD of the characteristic, holding the others fixed (%)",
                  fontsize=10, color=INK2, labelpad=10)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%".replace("-", "−") if v else "0")

    # Legend: spec identity + filled/hollow meaning
    from matplotlib.lines import Line2D

    handles = []
    for s in specs:
        c, m = SPEC_STYLE[s]
        handles.append(Line2D([0], [0], color=c, marker=m, markersize=7, linewidth=1.8, markerfacecolor=c, label=s))
    handles.append(Line2D([0], [0], color=MUTED, marker="o", markersize=7, linewidth=0,
                          markerfacecolor=SURFACE, markeredgewidth=1.6, label="Hollow = 95% CI includes 0"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=len(handles), frameon=False,
              fontsize=9, labelcolor=INK2, handletextpad=0.5, columnspacing=1.6)

    fig.suptitle("Impact of street characteristics on EMS travel time", x=0.06, ha="left", fontsize=15,
                 fontweight="bold", color=INK, y=0.985)
    fig.text(0.06, 0.935,
             f"OLS on log(travel seconds), {summary['incidents']:,} incidents / {summary['unique_routes']} routes; "
             "95% CIs clustered by route. Controls: distance, rush/night/weekend,\nprecipitation, severity, start type"
             f". R² = {summary['r2_all']:.2f}.",
             ha="left", va="top", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.885, bottom=0.15)
    fig.savefig(out / "02_regression_coefficients.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Partial-residual panels
# --------------------------------------------------------------------------
def partial_residual_panels(data: pd.DataFrame, res, coefs: pd.DataFrame, out: Path, n_extra: int = 4) -> list[str]:
    main = coefs[coefs["spec"] == "All routes"].copy()
    pinned = ["street_width_ft", "bus_lane_share"]
    rest = main[~main["feature"].isin(pinned)].assign(absb=lambda d: d["coef_log"].abs())
    # Rank remaining features by |t|-like signal: |coef| / CI half-width
    rest["snr"] = rest["coef_log"].abs() / ((rest["pct_hi"] - rest["pct_lo"]).abs() / 200 + 1e-9)
    chosen = pinned + rest.sort_values("snr", ascending=False)["feature"].head(n_extra).tolist()

    ncol = 3
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.9 * nrow), squeeze=False)
    for ax, f in zip(axes.ravel(), chosen):
        zc = f"z_{f}"
        b = float(res.params[zc])
        z = data[zc].to_numpy()
        zbar = z.mean()  # incident-weighted, so the line passes through the "typical" incident
        partial = res.resid.to_numpy() + b * (z - zbar)  # log-time net of all other terms
        x = data[f].to_numpy()
        sd = float(data[f].std(ddof=0))
        xbar = float(x.mean())

        # One point per route (size = incidents), mean partial residual
        g = (
            pd.DataFrame({"od": data["od_id"], "x": x, "y": partial})
            .groupby("od")
            .agg(x=("x", "first"), y=("y", "mean"), n=("y", "size"))
        )
        pct = lambda logv: 100 * (np.exp(logv) - 1)  # noqa: E731
        ax.scatter(g["x"], pct(g["y"]), s=np.clip(g["n"] / 6, 8, 70), color=BLUE, alpha=0.35, edgecolor="none", zorder=2)

        # Binned means (quantile bins; zero-heavy features collapse to fewer bins)
        try:
            bins = pd.qcut(g["x"], 8, duplicates="drop")
            bm = g.groupby(bins, observed=True).agg(x=("x", "mean"), y=("y", "mean"))
            ax.plot(bm["x"], pct(bm["y"]), color=INK, linewidth=1.8, marker="o", markersize=5,
                    markerfacecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        except ValueError:
            pass

        xs = np.linspace(g["x"].min(), g["x"].max(), 50)
        ax.plot(xs, pct(b * (xs - xbar) / sd), color=ORANGE, linewidth=2.2, zorder=3)

        lo, hi = np.percentile(pct(g["y"]), [3, 97])
        pad = 0.15 * (hi - lo)
        ymin, ymax = min(lo - pad, -20), max(hi + pad, 20)
        off = int(((pct(g["y"]) < ymin) | (pct(g["y"]) > ymax)).sum())
        ax.set_ylim(ymin, ymax)
        if off:
            ax.text(0.02, 0.96, f"{off} routes off-scale", transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")

        sig = bool(main.loc[main["feature"] == f, "significant"].iloc[0])
        est = float(main.loc[main["feature"] == f, "pct_change"].iloc[0])
        ax.set_title(f"{STREET_FEATURES[f][0]}", loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.text(0.98, 0.04, f"{est:+.1f}% per SD".replace("-", "−") + ("" if sig else " (n.s.)"), transform=ax.transAxes,
                ha="right", fontsize=9, color=INK2)
        ax.axhline(0, color=AXIS, linewidth=0.9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%".replace("-", "−") if v else "0")
        if f.endswith("_share"):
            ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.tick_params(length=0)
    for ax in axes.ravel()[len(chosen):]:
        ax.set_visible(False)

    fig.suptitle("Partial effect on EMS travel time, net of all other characteristics and controls",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.995)
    fig.text(0.04, 0.945,
             "Dots: individual routes (size = incidents). Black: binned means. Orange: regression line.",
             fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "03_partial_effects.png", dpi=170)
    plt.close(fig)
    return chosen


def make_all(df, data, res, coefs, scale, out: Path, summary: dict) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    correlation_heatmap(df, out, summary)
    coefficient_plot(coefs, scale, out, summary)
    partial_residual_panels(data, res, coefs, out)
