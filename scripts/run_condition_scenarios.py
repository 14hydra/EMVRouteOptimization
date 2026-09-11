#!/usr/bin/env python3
"""
Run route models across traffic × weather conditions (ASI eval variables).

Shows that EMV routers using special right-of-way (primary corridors / bus
lanes) pull further ahead of civilian GPS as congestion and weather worsen.

Conditions use:
  - hour-based congestion priors (TomTom-style live flow can replace later)
  - Open-Meteo weather factors (cached NYC hourly + harsh-hour picks)
  - bus-lane / primary ROW discounts on EMV edge costs

Example:
  PYTHONPATH=src python scripts/run_condition_scenarios.py
  PYTHONPATH=src python scripts/run_condition_scenarios.py --with-open-meteo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

sns.set_theme(style="whitegrid", context="notebook")

from emvro.routing import (  # noqa: E402
    control_civilian_time,
    nearest_node,
    prepare_routing_graph,
    solve_composite_drl,
    solve_gbdt_route,
    solve_mipsstw_mcs,
)
from emvro.routing.conditions import (  # noqa: E402
    CONDITION_PRESETS,
    get_conditions,
    pick_harsh_open_meteo_hours,
)
from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model  # noqa: E402
from emvro.routing.graph import dijkstra_route  # noqa: E402

# Midtown → Civic Center (same demo OD as run_route_models)
DEFAULT_ORIGIN = (40.7580, -73.9855)
DEFAULT_DEST = (40.7115, -74.0060)

MODEL_COLORS = {
    "control_civilian_time": "#7f8c8d",
    "emv_dijkstra": "#2980b9",
    "mipsstw_mcs": "#c0392b",
    "composite_drl": "#8e44ad",
    "gbdt_router": "#1e8449",
}
MODEL_LABELS = {
    "control_civilian_time": "Civilian GPS",
    "emv_dijkstra": "EMV Dijkstra",
    "mipsstw_mcs": "MIPSSTW+MCS",
    "composite_drl": "Composite DRL",
    "gbdt_router": "GBDT router",
}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_one(
    graph_path: Path,
    conditions,
    *,
    origin_ll,
    dest_ll,
    mcs_iters: int,
    drl_episodes: int,
    train_gbdt: bool,
) -> list[dict]:
    G = prepare_routing_graph(graph_path, conditions=conditions)
    origin = nearest_node(G, origin_ll[1], origin_ll[0])
    dest = nearest_node(G, dest_ll[1], dest_ll[0])

    results = {
        "control_civilian_time": control_civilian_time(G, origin, dest),
        "emv_dijkstra": dijkstra_route(G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"),
        "mipsstw_mcs": solve_mipsstw_mcs(
            G, origin, dest, n_iterations=mcs_iters, n_nests=8, seed=42
        ),
        "composite_drl": solve_composite_drl(
            G, origin, dest, episodes=drl_episodes, seed=42
        ),
    }
    if train_gbdt:
        edge_df = build_edge_training_frame(G, hours=[conditions.hour], max_edges=3500)
        bundle = train_gbdt_edge_model(edge_df)
        results["gbdt_router"] = solve_gbdt_route(
            G, origin, dest, bundle=bundle, hour=conditions.hour
        )

    civ = results["control_civilian_time"]
    civ_min = civ.travel_seconds / 60.0 if civ.ok else np.nan
    rows = []
    meta = conditions.to_meta()
    for name, r in results.items():
        mins = r.travel_seconds / 60.0 if r.ok else np.nan
        saved = (civ_min - mins) if (civ_min == civ_min and mins == mins) else np.nan
        pct = 100.0 * saved / civ_min if civ_min and civ_min == civ_min and civ_min > 0 else np.nan
        rows.append(
            {
                "condition_id": conditions.id,
                "condition_label": conditions.label,
                "hour": conditions.hour,
                "congestion": conditions.congestion,
                "wx_precip_mm": conditions.wx_precip_mm,
                "wx_snow_mm": conditions.wx_snow_mm,
                "weather_factor_civilian": meta["weather_factor_civilian"],
                "weather_factor_emv": meta["weather_factor_emv"],
                "row_bonus": meta["row_bonus"],
                "source": conditions.source,
                "model": name,
                "travel_minutes": round(float(mins), 3) if mins == mins else None,
                "distance_km": round(r.distance_m / 1000.0, 3) if r.ok else None,
                "n_edges": r.n_edges if r.ok else 0,
                "minutes_saved_vs_civilian": round(float(saved), 3) if saved == saved else None,
                "pct_faster_vs_civilian": round(float(pct), 2) if pct == pct else None,
            }
        )
    return rows


def plot_minutes_by_condition(df: pd.DataFrame, out: Path):
    keep = [
        "control_civilian_time",
        "emv_dijkstra",
        "mipsstw_mcs",
        "composite_drl",
        "gbdt_router",
    ]
    d = df[df["model"].isin(keep)].copy()
    order = [c for c in CONDITION_PRESETS if c in set(d["condition_id"])]
    extra = [c for c in d["condition_id"].unique() if c not in order]
    cond_order = order + extra
    label_map = (
        d.drop_duplicates("condition_id").set_index("condition_id")["condition_label"].to_dict()
    )

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    x = np.arange(len(cond_order))
    models = [m for m in keep if m in set(d["model"])]
    width = 0.15
    for i, m in enumerate(models):
        vals = []
        for cid in cond_order:
            hit = d[(d["condition_id"] == cid) & (d["model"] == m)]
            vals.append(float(hit["travel_minutes"].iloc[0]) if len(hit) else np.nan)
        ax.bar(
            x + (i - len(models) / 2) * width + width / 2,
            vals,
            width=width,
            color=MODEL_COLORS.get(m, "#333"),
            label=MODEL_LABELS.get(m, m),
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(c, c) for c in cond_order], rotation=15, ha="right")
    ax.set_ylabel("Travel time (minutes)")
    ax.set_title("EMV right-of-way advantage under traffic & weather")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    _save(fig, out)


def plot_pct_saved(df: pd.DataFrame, out: Path):
    emv_models = ["emv_dijkstra", "mipsstw_mcs", "composite_drl", "gbdt_router"]
    d = df[df["model"].isin(emv_models)].copy()
    order = [c for c in CONDITION_PRESETS if c in set(d["condition_id"])]
    extra = [c for c in d["condition_id"].unique() if c not in order]
    cond_order = order + extra
    label_map = (
        d.drop_duplicates("condition_id").set_index("condition_id")["condition_label"].to_dict()
    )

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    for m in emv_models:
        if m not in set(d["model"]):
            continue
        ys = []
        for cid in cond_order:
            hit = d[(d["condition_id"] == cid) & (d["model"] == m)]
            ys.append(float(hit["pct_faster_vs_civilian"].iloc[0]) if len(hit) else np.nan)
        ax.plot(
            range(len(cond_order)),
            ys,
            marker="o",
            lw=2.2,
            color=MODEL_COLORS.get(m, "#333"),
            label=MODEL_LABELS.get(m, m),
        )
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([label_map.get(c, c) for c in cond_order], rotation=15, ha="right")
    ax.set_ylabel("% faster than civilian GPS")
    ax.set_title("ROW payoff grows as congestion / weather worsen")
    ax.legend(loc="best", fontsize=9)
    _save(fig, out)


def plot_dashboard(df: pd.DataFrame, out: Path):
    """Compact ASI slide panel: civilian vs best EMV + condition table."""
    civ = df[df["model"] == "control_civilian_time"][
        ["condition_id", "condition_label", "travel_minutes", "congestion", "weather_factor_civilian"]
    ].rename(columns={"travel_minutes": "civilian_min"})
    emv = (
        df[df["model"] != "control_civilian_time"]
        .sort_values("travel_minutes")
        .groupby("condition_id", as_index=False)
        .first()[
            [
                "condition_id",
                "model",
                "travel_minutes",
                "minutes_saved_vs_civilian",
                "pct_faster_vs_civilian",
            ]
        ]
        .rename(
            columns={
                "travel_minutes": "best_emv_min",
                "model": "best_emv_model",
            }
        )
    )
    panel = civ.merge(emv, on="condition_id")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    labels = panel["condition_label"].tolist()
    x = np.arange(len(labels))
    w = 0.36
    axes[0].bar(x - w / 2, panel["civilian_min"], width=w, color="#7f8c8d", label="Civilian GPS")
    axes[0].bar(x + w / 2, panel["best_emv_min"], width=w, color="#8e44ad", label="Best EMV model")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("Minutes")
    axes[0].set_title("Civilian vs best EMV route")
    axes[0].legend(loc="upper left")

    axes[1].bar(labels, panel["pct_faster_vs_civilian"], color="#1e8449", edgecolor="white")
    axes[1].set_ylabel("% time saved vs civilian")
    axes[1].set_title("Special ROW payoff by condition")
    axes[1].tick_params(axis="x", rotation=18)
    for i, v in enumerate(panel["pct_faster_vs_civilian"]):
        if v == v:
            axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    _save(fig, out)
    return panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument("--origin-lat", type=float, default=DEFAULT_ORIGIN[0])
    p.add_argument("--origin-lon", type=float, default=DEFAULT_ORIGIN[1])
    p.add_argument("--dest-lat", type=float, default=DEFAULT_DEST[0])
    p.add_argument("--dest-lon", type=float, default=DEFAULT_DEST[1])
    p.add_argument(
        "--conditions",
        nargs="*",
        default=list(CONDITION_PRESETS.keys()),
        help="Preset condition ids (default: all)",
    )
    p.add_argument(
        "--with-open-meteo",
        action="store_true",
        help="Also add harshest hours from cached Open-Meteo NYC weather",
    )
    p.add_argument("--mcs-iters", type=int, default=10)
    p.add_argument("--drl-episodes", type=int, default=40)
    p.add_argument("--skip-gbdt", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "figures" / "route_models" / "conditions",
    )
    args = p.parse_args()

    if not args.graph.exists():
        raise SystemExit(f"Missing graph {args.graph}; run scripts/build_osm_graph.py first")

    scenarios = [get_conditions(cid) for cid in args.conditions]
    if args.with_open_meteo:
        wx_path = ROOT / "data" / "processed" / "weather_hourly_nyc.csv"
        scenarios.extend(pick_harsh_open_meteo_hours(wx_path, n=2))

    print(f"Running {len(scenarios)} condition scenarios…")
    rows: list[dict] = []
    for cond in scenarios:
        print(f"  · {cond.label} (cong={cond.congestion:.2f}, wx_civ={cond.weather_factor_civilian():.2f})")
        rows.extend(
            run_one(
                args.graph,
                cond,
                origin_ll=(args.origin_lat, args.origin_lon),
                dest_ll=(args.dest_lat, args.dest_lon),
                mcs_iters=args.mcs_iters,
                drl_episodes=args.drl_episodes,
                train_gbdt=not args.skip_gbdt,
            )
        )

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "condition_scenario_times.csv"
    df.to_csv(csv_path, index=False)

    plot_minutes_by_condition(df, args.out_dir / "01_minutes_by_condition.png")
    plot_pct_saved(df, args.out_dir / "02_pct_faster_vs_civilian.png")
    panel = plot_dashboard(df, args.out_dir / "00_row_advantage_dashboard.png")

    summary = {
        "od": {
            "origin": [args.origin_lat, args.origin_lon],
            "dest": [args.dest_lat, args.dest_lon],
        },
        "n_conditions": int(df["condition_id"].nunique()),
        "conditions": [c.to_meta() for c in scenarios],
        "mean_pct_faster_emv_models": float(
            df[df["model"] != "control_civilian_time"]["pct_faster_vs_civilian"].mean()
        ),
        "pct_faster_by_condition_best_emv": panel.set_index("condition_label")[
            "pct_faster_vs_civilian"
        ].round(2).to_dict(),
        "note": (
            "Civilian costs take full congestion×weather; EMV costs use reduced weather "
            "penalty + congestion-scaled ROW bonus on primary/bus edges (ASI eval variables)."
        ),
        "outputs": {
            "csv": str(csv_path),
            "figures": str(args.out_dir),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()
