#!/usr/bin/env python3
"""
Plan optimal fire-station locations (London LFB primary): redesign, replace
or expand.

Response = predicted drive (kolesar / crow scorer) + station turnout,
judged against LFB first-engine KPIs (mean <= 6 min, > 90 % within 10 min).
Demand comes from --demand-years, held-out scoring from --eval-years.

Examples:
  PYTHONPATH=src python scripts/plan_firehouse_locations.py --demo --mode expand --add-stations 2
  PYTHONPATH=src python scripts/plan_firehouse_locations.py \\
      --city london --mode redesign
  PYTHONPATH=src python scripts/plan_firehouse_locations.py \\
      --city london --mode replace --replace 10
  PYTHONPATH=src python scripts/plan_firehouse_locations.py \\
      --city london --mode expand --add-stations 3 --use-busy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.cities import (  # noqa: E402
    custom_city,
    get_city,
    list_cities,
    load_firehouses,
    load_incidents,
    with_overrides,
)
from emvro.firehouse_location import demo_inputs, run_firehouse_plan  # noqa: E402


def _write_map(result, out: Path, threshold_min: float) -> Path:
    import folium
    from branca.element import Element
    from folium.plugins import HeatMap

    cells = result.cells
    selected = result.selected
    closed = result.closed
    opened = result.opened
    candidates = result.candidates
    center = [float(cells["lat"].mean()), float(cells["lon"].mean())]
    # Never use tiles.openstreetmap.org — OSM's volunteer CDN 403s Folium apps
    # (osm.wiki/Blocked). Match patrol/visualize_data: Esri default + Carto fallbacks.
    m = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri &copy; OpenStreetMap contributors",
        name="Esri streets",
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri imagery",
        name="Esri imagery",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Carto light",
        show=False,
    ).add_to(m)

    # Demand heat
    fg_heat = folium.FeatureGroup(name="Incident demand (heat)", show=True)
    heat = [
        [float(r.lat), float(r.lon), float(r.weight)]
        for r in cells.itertuples()
        if np.isfinite(r.lat) and np.isfinite(r.lon)
    ]
    if heat:
        HeatMap(heat, radius=14, blur=18, max_zoom=12, min_opacity=0.35).add_to(fg_heat)
    fg_heat.add_to(m)

    # Response improvement surface (green = faster under plan)
    fg_delta = folium.FeatureGroup(name="Δ response (plan − current)", show=True)
    if "response_s_delta" in cells.columns:
        for _, row in cells.iterrows():
            d = float(row["response_s_delta"])
            # green faster, red slower
            if d <= -15:
                color = "#1a7f37"
            elif d < 0:
                color = "#6bbf6b"
            elif d < 15:
                color = "#c8c8c8"
            else:
                color = "#c0392b"
            folium.CircleMarker(
                [row["lat"], row["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                weight=0,
                popup=f"Δ={d:.0f}s · plan={row['response_s']/60:.1f} min · was={row.get('response_s_current', float('nan'))/60:.1f} min",
            ).add_to(fg_delta)
    fg_delta.add_to(m)

    # Kept existing (subtle)
    kept = selected[selected["is_existing"] == True] if "is_existing" in selected.columns else selected.iloc[0:0]
    fg_kept = folium.FeatureGroup(name="Kept firehouses", show=False)
    for _, row in kept.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=4,
            color="#1f4e79",
            fill=True,
            fill_opacity=0.6,
            weight=1,
            popup=f"KEPT: {row['site_name']}",
        ).add_to(fg_kept)
    fg_kept.add_to(m)

    if len(closed):
        fg_c = folium.FeatureGroup(name="Closed firehouses", show=True)
        for _, row in closed.iterrows():
            folium.Marker(
                [row["lat"], row["lon"]],
                popup=f"CLOSED: {row['site_name']}",
                icon=folium.Icon(color="black", icon="remove", prefix="glyphicon"),
            ).add_to(fg_c)
        fg_c.add_to(m)

    if len(opened):
        fg_o = folium.FeatureGroup(name="Opened sites", show=True)
        for _, row in opened.iterrows():
            folium.Marker(
                [row["lat"], row["lon"]],
                popup=f"OPENED: {row['site_name']}",
                icon=folium.Icon(color="green", icon="plus", prefix="glyphicon"),
            ).add_to(fg_o)
        fg_o.add_to(m)

    # New non-existing selected (redesign)
    new_sel = selected[selected["is_existing"] == False] if "is_existing" in selected.columns else selected.iloc[0:0]
    if len(new_sel) and not len(opened):
        fg_new = folium.FeatureGroup(name="New planned sites", show=True)
        for _, row in new_sel.iterrows():
            folium.Marker(
                [row["lat"], row["lon"]],
                popup=f"NEW: {row['site_name']}",
                icon=folium.Icon(color="red", icon="home", prefix="glyphicon"),
            ).add_to(fg_new)
        fg_new.add_to(m)

    s = result.summary
    plan = s["plan"]
    cur = s["baselines"]["current_firehouses"]
    delta = s.get("delta_vs_current_s")
    delta_txt = f"{delta:.0f} s" if delta is not None else "—"
    panel = f"""
    <div style="position:fixed;top:12px;left:12px;z-index:9999;background:rgba(255,255,255,0.96);
         padding:12px 14px;border:1px solid #888;border-radius:10px;font:13px/1.45 system-ui,sans-serif;
         width:380px;max-width:92vw;box-shadow:0 4px 14px rgba(0,0,0,.18);">
      <div style="font-weight:700;">Firehouse location plan</div>
      <div style="color:#555;font-size:12px;margin:4px 0 8px;">
        {s['city_id']} · mode=<b>{s['mode']}</b> · {s['objective']}
      </div>
      <table style="width:100%;font-size:12px;border-collapse:collapse;">
        <tr><td></td><th style="text-align:right;">E[T] min</th><th style="text-align:right;">≤{threshold_min} min</th></tr>
        <tr><td>Current</td><td style="text-align:right;">{cur['expected_response_s']/60:.2f}</td>
            <td style="text-align:right;">{100*cur['coverage_share_within_threshold']:.1f}%</td></tr>
        <tr><td><b>Plan</b></td><td style="text-align:right;"><b>{plan['expected_response_s']/60:.2f}</b></td>
            <td style="text-align:right;"><b>{100*plan['coverage_share_within_threshold']:.1f}%</b></td></tr>
      </table>
      <div style="font-size:11px;color:#555;margin-top:8px;">
        selected={s['n_selected']} · closed={s['n_closed']} · opened={s['n_opened']}<br>
        Δ vs current: <b>{delta_txt}</b> (negative = faster)<br>
        Map: heat = demand · green dots = cells that got faster
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(panel))
    folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def _write_figures(result, out_dir: Path, threshold_min: float) -> list[Path]:
    """Static comparison charts for the firehouse plan."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    s = result.summary
    plan = s["plan"]
    baselines = s["baselines"]
    cells = result.cells
    selected = result.selected
    closed = result.closed
    opened = result.opened

    # --- 01 metrics bars ---
    labels = ["Current"]
    et = [baselines["current_firehouses"]["expected_response_s"] / 60]
    cov = [100 * baselines["current_firehouses"]["coverage_share_within_threshold"]]
    colors = ["#7f8c8d"]
    for key in ("random_replace", "random_expand"):
        if key in baselines:
            labels.append(key.replace("_", " ").capitalize())
            et.append(baselines[key]["expected_response_s"] / 60)
            cov.append(100 * baselines[key]["coverage_share_within_threshold"])
            colors.append("#95a5a6")
    labels.append("Plan")
    et.append(plan["expected_response_s"] / 60)
    cov.append(100 * plan["coverage_share_within_threshold"])
    colors.append("#c0392b")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    bars = ax.bar(labels, et, color=colors)
    ax.set_ylabel("Expected response (min)")
    ax.set_title(f"{s['city_id']} · {s['mode']} · E[T]")
    for b, v in zip(bars, et):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax = axes[1]
    bars = ax.bar(labels, cov, color=colors)
    ax.set_ylabel(f"% demand ≤ {threshold_min:.0f} min")
    ax.set_title("Coverage")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, min(v + 1.5, 102), f"{v:.1f}%", ha="center", fontsize=9)
    fig.suptitle("Firehouse plan vs baselines", fontsize=13, fontweight="600", y=1.02)
    fig.tight_layout()
    path = out_dir / "01_metrics_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- 02 response CDF ---
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    w = cells["weight"].to_numpy(dtype=float)
    w = w / max(w.sum(), 1e-9)

    def _ecdf(vals, weights):
        order = np.argsort(vals)
        v = vals[order] / 60.0
        cw = np.cumsum(weights[order])
        return v, cw

    if "response_s_current" in cells.columns:
        v, c = _ecdf(cells["response_s_current"].to_numpy(dtype=float), w)
        ax.plot(v, c, color="#7f8c8d", lw=2.2, label="Current houses")
    v, c = _ecdf(cells["response_s"].to_numpy(dtype=float), w)
    ax.plot(v, c, color="#c0392b", lw=2.2, label="Plan")
    for mark, col in ((6, "#2980b9"), (10, "#8e44ad")):
        ax.axvline(mark, color=col, ls=":", lw=1.2, label=f"LFB {mark} min")
    if threshold_min not in (6, 10):
        ax.axvline(threshold_min, color="#222", ls="--", lw=1, label=f"{threshold_min:g} min threshold")
    ax.set_xlabel("Response time (min)")
    ax.set_ylabel("Demand-weighted CDF")
    ax.set_xlim(0, max(14, threshold_min * 1.5))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("How fast demand is reached")
    fig.tight_layout()
    path = out_dir / "02_response_cdf.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- 03 before/after map scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)
    for ax, col, title, cmap in (
        (axes[0], "response_s_current" if "response_s_current" in cells.columns else "response_s", "Current network", "YlOrRd"),
        (axes[1], "response_s", "Planned network", "YlOrRd"),
    ):
        vals = cells[col].to_numpy(dtype=float) / 60.0
        sc = ax.scatter(
            cells["lon"],
            cells["lat"],
            c=vals,
            s=12 + 40 * (cells["weight"] / cells["weight"].max()),
            cmap=cmap,
            vmin=0,
            vmax=max(8, np.nanpercentile(vals, 95)),
            alpha=0.85,
            edgecolors="none",
        )
        # overlay sites
        if col.startswith("response_s_current") or col == "response_s_current":
            exist = result.candidates[result.candidates["is_existing"]]
            ax.scatter(exist["lon"], exist["lat"], s=18, c="#1f4e79", marker="s", label="Existing", zorder=5)
        else:
            ax.scatter(selected["lon"], selected["lat"], s=22, c="#1f4e79", marker="s", label="Planned", zorder=5)
            if len(closed):
                ax.scatter(closed["lon"], closed["lat"], s=50, c="black", marker="x", label="Closed", zorder=6)
            if len(opened):
                ax.scatter(opened["lon"], opened["lat"], s=55, c="#27ae60", marker="^", label="Opened", zorder=6)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="min")
    axes[0].set_ylabel("Latitude")
    fig.suptitle(f"Response surface · {s['city_id']} · {s['mode']}", fontsize=13, fontweight="600")
    fig.tight_layout()
    path = out_dir / "03_response_surface_map.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- 04 close/open detail (replace) or top demand sites ---
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.scatter(cells["lon"], cells["lat"], c="#dde3ea", s=8, alpha=0.7, edgecolors="none", label="Demand cells")
    exist = result.candidates[result.candidates["is_existing"]]
    ax.scatter(exist["lon"], exist["lat"], s=16, c="#94a3b8", marker="s", alpha=0.7, label="Existing houses")
    if len(closed):
        ax.scatter(closed["lon"], closed["lat"], s=90, c="#111", marker="X", label=f"Closed ({len(closed)})", zorder=5)
        for _, r in closed.iterrows():
            ax.annotate(str(r["site_name"])[:22], (r["lon"], r["lat"]), fontsize=7, color="#111", xytext=(4, 4), textcoords="offset points")
    if len(opened):
        ax.scatter(opened["lon"], opened["lat"], s=90, c="#16a34a", marker="^", label=f"Opened ({len(opened)})", zorder=5)
        for _, r in opened.iterrows():
            ax.annotate(str(r["site_name"])[:22], (r["lon"], r["lat"]), fontsize=7, color="#14532d", xytext=(4, -10), textcoords="offset points")
    # draw lines closed→nearest opened as crude relocation hint
    if len(closed) and len(opened):
        o_lon = opened["lon"].to_numpy()
        o_lat = opened["lat"].to_numpy()
        for _, r in closed.iterrows():
            d2 = (o_lon - r["lon"]) ** 2 + (o_lat - r["lat"]) ** 2
            j = int(np.argmin(d2))
            ax.plot([r["lon"], o_lon[j]], [r["lat"], o_lat[j]], color="#64748b", lw=1.0, alpha=0.7, zorder=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Relocations · {s['city_id']} · closed={s['n_closed']} opened={s['n_opened']}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "04_relocations.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- 05 assigned demand by site (top 20) ---
    top = selected.sort_values("assigned_share", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    colors = ["#16a34a" if (not bool(x)) else "#1f4e79" for x in top.get("is_existing", pd.Series([True] * len(top)))]
    ax.barh(top["site_name"].astype(str).str.slice(0, 36), top["assigned_share"] * 100, color=colors)
    ax.set_xlabel("% of citywide demand assigned")
    ax.set_title("Top sites by assigned demand (navy=existing · green=new)")
    fig.tight_layout()
    path = out_dir / "05_site_demand_share.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", type=str, default="london", help=f"City id ({', '.join(list_cities())}) or custom")
    p.add_argument("--mode", choices=["redesign", "replace", "expand"], default="redesign")
    p.add_argument("--n-houses", type=int, default=None, help="Fleet size for redesign (default=current count)")
    p.add_argument("--replace", type=int, default=10, help="How many houses to relocate in replace mode")
    p.add_argument("--add-stations", type=int, default=1, help="New stations to open in expand mode")
    p.add_argument("--cell-km", type=float, default=0.9)
    p.add_argument("--demand-candidates", type=int, default=80)
    p.add_argument("--objective", choices=["response_time", "coverage"], default="response_time")
    p.add_argument("--threshold-min", type=float, default=10.0, help="Cover threshold (LFB: 10)")
    p.add_argument("--scorer", choices=["kolesar", "crow"], default="kolesar", help="Drive-time scorer")
    p.add_argument("--demand-years", type=str, default="2024", help="Comma list of cal_years for demand")
    p.add_argument("--eval-years", type=str, default="2025", help="Comma list of cal_years for held-out scoring")
    p.add_argument("--use-busy", action="store_true", help="Blend next-nearest engine by station busy rate")
    p.add_argument("--firehouses", type=Path, default=None)
    p.add_argument("--incidents", type=Path, default=None)
    p.add_argument("--max-incidents", type=int, default=200_000)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "firehouse_plan")
    p.add_argument("--demo", action="store_true", help="Synthetic city (no data files needed)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def _years(txt):
        return [int(x) for x in str(txt).replace(" ", "").split(",") if x]

    demand_years, eval_years = _years(args.demand_years), _years(args.eval_years)
    scorer = args.scorer

    if args.demo:
        city_id = "demo"
        houses, incidents = demo_inputs(seed=args.seed)
        if args.n_houses is None:
            args.n_houses = max(4, len(houses) // 2) if args.mode == "redesign" else None
        print(f"Demo city: {len(houses)} houses, {len(incidents)} incidents")
    else:
        if args.city.lower() in {"custom"}:
            if not args.firehouses or not args.incidents:
                p.error("custom city requires --firehouses and --incidents")
            spec = custom_city(
                firehouses=args.firehouses,
                incidents=args.incidents,
            )
        else:
            try:
                spec = get_city(args.city)
            except KeyError as e:
                p.error(str(e))
            spec = with_overrides(
                spec,
                firehouses=args.firehouses,
                incidents=args.incidents,
            )
        try:
            spec.require_paths(need_graph=False)
        except FileNotFoundError as e:
            p.error(str(e))
        city_id = spec.id
        houses = load_firehouses(spec)
        # LFB planner CSV: load demand + eval years whole; run_firehouse_plan splits
        # them and caps each split at --max-incidents.
        is_lfb = spec.meta.get("standards") == "lfb"
        incidents = load_incidents(
            spec,
            max_rows=None if is_lfb else args.max_incidents,
            years=sorted(set(demand_years + eval_years)) if is_lfb else None,
        )
        if args.n_houses is None and args.mode == "redesign":
            args.n_houses = len(houses)
        print(f"City {spec.name}: {len(houses)} firehouses, {len(incidents):,} demand rows")

    result = run_firehouse_plan(
        city_id=city_id,
        houses=houses,
        incidents=incidents,
        mode=args.mode,
        n_houses=args.n_houses,
        replace_r=args.replace,
        add_stations=args.add_stations,
        cell_km=args.cell_km,
        demand_candidates=args.demand_candidates,
        objective=args.objective,
        threshold_min=args.threshold_min,
        max_incident_rows=args.max_incidents,
        seed=args.seed,
        scorer=scorer,
        demand_years=demand_years,
        eval_years=eval_years,
        use_busy=args.use_busy,
    )

    # Write artifacts
    result.selected.to_csv(args.out_dir / "planned_firehouses.csv", index=False)
    result.cells.to_csv(args.out_dir / "demand_cells.csv", index=False)
    result.candidates.to_csv(args.out_dir / "candidates.csv", index=False)
    if len(result.closed):
        result.closed.to_csv(args.out_dir / "closed_firehouses.csv", index=False)
    if len(result.opened):
        result.opened.to_csv(args.out_dir / "opened_firehouses.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")

    try:
        figs = _write_figures(result, args.out_dir, args.threshold_min)
        for f in figs:
            print("Figure ->", f)
    except Exception as exc:  # noqa: BLE001
        print("Figures skipped:", exc)

    try:
        map_path = _write_map(result, args.out_dir / "firehouse_plan_map.html", args.threshold_min)
        print("Map ->", map_path)
    except Exception as exc:  # noqa: BLE001
        print("Map skipped:", exc)

    print(json.dumps(result.summary, indent=2))
    print("Wrote artifacts to", args.out_dir)


if __name__ == "__main__":
    main()
