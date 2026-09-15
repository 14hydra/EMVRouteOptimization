#!/usr/bin/env python3
"""
Find OD cases where EMV ROW (contraflow / busway) beats civilian GPS, then map them.

A case counts as a ROW win when:
  1. EMV Dijkstra is faster than civilian congested time by a meaningful margin
  2. The EMV path uses ≥1 EMV-only corridor edge (Google-untakeable)
  3. Removing those privileges (no contraflow / busway) loses most of that advantage
     → the win is attributable to special right-of-way, not just a generic speedup

Example:
  PYTHONPATH=src python scripts/find_row_wins.py
  PYTHONPATH=src python scripts/find_row_wins.py --limit 400 --top 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

from emvro.routing import (  # noqa: E402
    apply_routing_conditions,
    conditions_at,
    control_civilian_time,
    nearest_node,
    prepare_routing_graph,
)
from emvro.routing.emv_corridors import (  # noqa: E402
    count_corridor_edges,
    path_corridor_segments,
)
from emvro.routing.graph import dijkstra_route  # noqa: E402


def _strip_emv_privileges(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Copy with EMV-only edges removed and costs recomputed without corridors."""
    H = G.copy()
    drop = [
        (u, v, k)
        for u, v, k, d in H.edges(keys=True, data=True)
        if d.get("civilian_forbidden") or d.get("emv_corridor")
    ]
    H.remove_edges_from(drop)
    # Recompute costs as if no ROW privileges remain on remaining edges
    for _, _, _, data in H.edges(keys=True, data=True):
        data["civilian_forbidden"] = False
        data["emv_corridor"] = False
        data.pop("emv_corridor_kind", None)
    # Keep current congestion/weather by re-deriving from stored civilian/emv is messy;
    # instead scale from travel_time using the same hour meta on G.
    cond_meta = G.graph.get("routing_conditions") or {}
    hour = int(G.graph.get("routing_hour") or cond_meta.get("hour") or 17)
    weather = "clear"
    cid = str(cond_meta.get("id") or "")
    if cid.startswith("rain"):
        weather = "rain"
    elif cid.startswith("snow"):
        weather = "snow"
    apply_routing_conditions(H, conditions=conditions_at(hour, weather))
    # Force no forbidden edges in the stripped graph
    for _, _, _, data in H.edges(keys=True, data=True):
        if data.get("civilian_forbidden"):
            # Shouldn't happen after strip; belt-and-suspenders
            data["civilian_forbidden"] = False
            base = float(data.get("travel_time") or 0.0)
            cong = float(H.graph.get("routing_congestion") or 1.2)
            data["civilian_s"] = base * cong
            data["emv_s"] = data["civilian_s"] * 0.9
            data["weight_emv"] = data["emv_s"]
    return H


def _eval_od(G, G_no_row, origin, dest) -> dict | None:
    civ = control_civilian_time(G, origin, dest)
    emv = dijkstra_route(G, origin, dest, weight="weight_emv", model_name="emv_row")
    if not civ.ok or not emv.ok:
        return None
    n_corr = count_corridor_edges(G, emv.node_path)
    if n_corr < 1:
        return None

    emv_plain = dijkstra_route(
        G_no_row, origin, dest, weight="weight_emv", model_name="emv_no_row"
    )
    if not emv_plain.ok:
        return None

    civ_s = float(civ.travel_seconds)
    emv_s = float(emv.travel_seconds)
    plain_s = float(emv_plain.travel_seconds)
    if civ_s <= 0 or emv_s <= 0:
        return None

    saved_vs_civ = civ_s - emv_s
    pct_vs_civ = 100.0 * saved_vs_civ / civ_s
    # Advantage explained by ROW ≈ how much worse without corridor privileges
    row_attrib_s = plain_s - emv_s
    row_attrib_pct = 100.0 * row_attrib_s / civ_s if civ_s else 0.0

    return {
        "civ_s": civ_s,
        "emv_s": emv_s,
        "emv_no_row_s": plain_s,
        "saved_vs_civ_s": saved_vs_civ,
        "pct_vs_civ": pct_vs_civ,
        "row_attrib_s": row_attrib_s,
        "row_attrib_pct": row_attrib_pct,
        "emv_only_edges": n_corr,
        "civ_path": list(civ.node_path),
        "emv_path": list(emv.node_path),
        "emv_no_row_path": list(emv_plain.node_path),
        "civ_km": civ.distance_m / 1000.0,
        "emv_km": emv.distance_m / 1000.0,
    }


def sample_od_candidates(od_path: Path, *, limit: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(od_path, low_memory=False)
    need = ["start_lat", "start_lon", "dest_lat", "dest_lon"]
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).copy()
    if "crow_flies_km" in df.columns:
        df["crow_flies_km"] = pd.to_numeric(df["crow_flies_km"], errors="coerce")
        # Prefer trips long enough for corridor shortcuts to matter
        mid = df[(df["crow_flies_km"] >= 0.9) & (df["crow_flies_km"] <= 10.0)]
        if len(mid) >= max(50, limit // 4):
            df = mid
    boro_col = "borough_norm" if "borough_norm" in df.columns else (
        "borough" if "borough" in df.columns else None
    )
    if boro_col:
        b = df[boro_col].astype(str).str.upper()
        nyc_names = {"MANHATTAN", "BROOKLYN", "BRONX", "QUEENS", "STATEN ISLAND"}
        prefer = b.isin(nyc_names)
        # Only apply NYC borough preference when those labels actually exist
        if prefer.any():
            man = df[b == "MANHATTAN"]
            if len(man) >= limit // 2:
                rest = df[(b != "MANHATTAN") & prefer]
                rng = np.random.default_rng(seed)
                n_man = min(len(man), max(limit // 2, limit - len(rest)))
                take_man = man.iloc[rng.choice(len(man), size=n_man, replace=False)]
                n_rest = min(len(rest), limit - len(take_man))
                take_rest = (
                    rest.iloc[rng.choice(len(rest), size=n_rest, replace=False)]
                    if n_rest > 0
                    else rest.iloc[0:0]
                )
                df = pd.concat([take_man, take_rest], ignore_index=True)
            else:
                df = df[prefer | b.isna()].copy()
    rng = np.random.default_rng(seed)
    if len(df) > limit:
        idx = rng.choice(len(df), size=limit, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def build_row_wins_map(
    G,
    wins: list[dict],
    out: Path,
    *,
    condition_label: str,
) -> Path | None:
    try:
        import folium
        from branca.element import Element
    except ImportError:
        print("folium not installed; skipping map")
        return None

    def ll(node):
        d = G.nodes[node]
        return [float(d.get("y", d.get("lat"))), float(d.get("x", d.get("lon")))]

    def path_ll(path):
        return [ll(n) for n in path]

    m = folium.Map(location=[40.74, -73.97], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri streets",
    ).add_to(m)

    civ_fg = folium.FeatureGroup(name="Civilian GPS route", show=True)
    emv_fg = folium.FeatureGroup(name="EMV Dijkstra (with right-of-way)", show=True)
    gold_fg = folium.FeatureGroup(name="Roads civilians/Google Maps cannot use", show=True)
    markers_fg = folium.FeatureGroup(name="Example trips", show=True)
    for fg in (civ_fg, emv_fg, gold_fg, markers_fg):
        fg.add_to(m)

    bounds = []
    for i, w in enumerate(wins):
        o_ll = ll(w["origin"])
        d_ll = ll(w["dest"])
        civ_coords = path_ll(w["civ_path"])
        emv_coords = path_ll(w["emv_path"])
        civ_coords[0] = list(o_ll)
        civ_coords[-1] = list(d_ll)
        emv_coords[0] = list(o_ll)
        emv_coords[-1] = list(d_ll)
        bounds.extend(civ_coords)
        bounds.extend(emv_coords)

        saved_m = w["saved_vs_civ_s"] / 60.0
        row_m = w["row_attrib_s"] / 60.0
        title = f"Example {i+1}: {w.get('label') or w.get('trip_id')}"

        folium.CircleMarker(
            o_ll,
            radius=7,
            color="#1f4e79",
            fill=True,
            fill_color="#3498db",
            popup=f"<b>{title}</b><br>Start",
            tooltip=f"Start {i+1}",
        ).add_to(markers_fg)
        folium.CircleMarker(
            d_ll,
            radius=7,
            color="#b35900",
            fill=True,
            fill_color="#e67e22",
            popup=f"<b>{title}</b><br>Destination",
            tooltip=f"Destination {i+1}",
        ).add_to(markers_fg)
        folium.Marker(
            [(o_ll[0] + d_ll[0]) / 2, (o_ll[1] + d_ll[1]) / 2],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font:11px/1.2 sans-serif;background:rgba(255,255,255,.94);'
                    f'padding:2px 6px;border:1px solid #888;border-radius:4px;white-space:nowrap;">'
                    f"Ex. {i+1}: {saved_m:.1f} min faster "
                    f"(right-of-way {row_m:.1f} min)</div>"
                )
            ),
        ).add_to(markers_fg)

        popup = (
            f"<b>{title}</b><br>Conditions: {condition_label}<br>"
            f"Civilian GPS: {w['civ_s']/60:.2f} min → "
            f"EMV Dijkstra (with right-of-way): {w['emv_s']/60:.2f} min<br>"
            f"Faster by: <b>{saved_m:.2f} min ({w['pct_vs_civ']:.1f}%)</b><br>"
            f"Of which from EMV-only roads: <b>{row_m:.2f} min</b><br>"
            f"EMV-only road segments used: {w['emv_only_edges']}<br>"
            f"EMV Dijkstra without those privileges: {w['emv_no_row_s']/60:.2f} min"
        )
        folium.PolyLine(
            civ_coords,
            color="#7f8c8d",
            weight=4,
            opacity=0.55,
            dash_array="10 8",
            popup=popup,
            tooltip=f"Example {i+1} civilian GPS ({w['civ_s']/60:.1f} min)",
        ).add_to(civ_fg)
        folium.PolyLine(
            emv_coords,
            color="#8e44ad",
            weight=6,
            opacity=0.95,
            popup=popup,
            tooltip=(
                f"Example {i+1} EMV Dijkstra "
                f"({w['emv_s']/60:.1f} min, {saved_m:.1f} min faster)"
            ),
        ).add_to(emv_fg)

        for seg in path_corridor_segments(G, w["emv_path"]):
            kind = seg.get("kind") or "emv_only"
            kind_label = {
                "contraflow": "Wrong-way lane / contraflow (EMV only)",
                "busway": "Bus lane / busway (EMV only)",
                "restricted": "Restricted road (EMV only)",
            }.get(kind, f"EMV-only ({kind})")
            folium.PolyLine(
                seg["coords"],
                color="#f1c40f",
                weight=9,
                opacity=0.9,
                popup=f"<b>{kind_label}</b><br>{title}",
                tooltip=f"Example {i+1}: {kind_label}",
            ).add_to(gold_fg)

    if bounds:
        m.fit_bounds(bounds, padding=(30, 30))

    legend = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:rgba(255,255,255,0.97);
                padding:12px 14px;border:1px solid #999;border-radius:8px;font:13px/1.45 sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,.15);max-width:360px;">
      <div style="font-weight:700;margin-bottom:6px;">When EMV right-of-way beats civilian GPS</div>
      <div>Real trip examples where <b>EMV Dijkstra</b> is faster <b>because</b> it uses roads
           civilians and Google Maps cannot (wrong-way / bus lanes).
           Conditions: {condition_label}.</div>
      <hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">
      <div><span style="color:#7f8c8d;">╌ ╌</span> Civilian GPS route</div>
      <div><span style="color:#8e44ad;font-weight:700;">━━</span> EMV Dijkstra (with right-of-way)</div>
      <div><span style="color:#f1c40f;font-weight:700;">━━</span> Road civilians/Google Maps cannot use</div>
      <div style="margin-top:6px;"><span style="color:#3498db;">●</span> Start &nbsp;
           <span style="color:#e67e22;">●</span> Destination</div>
      <div style="margin-top:6px;font-size:12px;color:#444;">
        “Right-of-way X min” = minutes saved from those EMV-only roads
        (EMV Dijkstra without privileges − with privileges).
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument(
        "--od",
        type=Path,
        default=ROOT / "data" / "processed" / "od_pairs_inferred_starts.csv",
    )
    p.add_argument("--limit", type=int, default=350, help="OD candidates to evaluate")
    p.add_argument("--top", type=int, default=8, help="Wins to keep on the map")
    p.add_argument("--hour", type=int, default=17)
    p.add_argument("--weather", choices=["clear", "rain", "snow"], default="rain")
    p.add_argument("--min-pct", type=float, default=4.0, help="Min %% faster vs civilian")
    p.add_argument("--min-saved-s", type=float, default=25.0, help="Min seconds saved vs civilian")
    p.add_argument(
        "--min-row-attrib-s",
        type=float,
        default=15.0,
        help="Min seconds attributable to EMV-only corridors",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "figures" / "route_models",
    )
    args = p.parse_args()

    if not args.graph.exists():
        raise SystemExit(f"Missing graph {args.graph}")
    if not args.od.exists():
        raise SystemExit(f"Missing OD table {args.od}")

    cond = conditions_at(args.hour, args.weather)
    print(f"Loading graph under {cond.label}…")
    G = prepare_routing_graph(args.graph, conditions=cond)
    print("Building no-ROW comparison graph…")
    G_no = _strip_emv_privileges(G)

    cands = sample_od_candidates(args.od, limit=args.limit, seed=args.seed)
    print(f"Evaluating {len(cands)} OD candidates…")

    wins: list[dict] = []
    scanned = 0
    for rec in cands.itertuples(index=False):
        scanned += 1
        try:
            origin = nearest_node(G, float(rec.start_lon), float(rec.start_lat))
            dest = nearest_node(G, float(rec.dest_lon), float(rec.dest_lat))
        except Exception:  # noqa: BLE001
            continue
        if origin == dest:
            continue
        ev = _eval_od(G, G_no, origin, dest)
        if ev is None:
            continue
        if ev["pct_vs_civ"] < args.min_pct or ev["saved_vs_civ_s"] < args.min_saved_s:
            continue
        if ev["row_attrib_s"] < args.min_row_attrib_s:
            continue
        # Path must actually differ from the no-ROW EMV path
        if ev["emv_path"] == ev["emv_no_row_path"]:
            continue

        label = None
        boro = getattr(rec, "borough_norm", None) or getattr(rec, "borough", None)
        z = getattr(rec, "zipcode", None)
        if boro or z is not None:
            boro_s = str(boro).strip().title() if boro else "?"
            try:
                zip_s = str(int(float(z))) if z is not None and str(z).strip() else "?"
            except (TypeError, ValueError):
                zip_s = str(z).rstrip(".0") if z is not None else "?"
            label = f"{boro_s} → ZIP {zip_s}"
        wins.append(
            {
                **ev,
                "origin": origin,
                "dest": dest,
                "trip_id": str(getattr(rec, "incident_id", scanned)),
                "label": label,
                "start_lat": float(rec.start_lat),
                "start_lon": float(rec.start_lon),
                "dest_lat": float(rec.dest_lat),
                "dest_lon": float(rec.dest_lon),
                "borough": boro,
                "zipcode": z,
            }
        )
        if scanned % 50 == 0:
            print(f"  scanned {scanned}/{len(cands)} · wins so far {len(wins)}")

    wins.sort(key=lambda w: (w["row_attrib_s"], w["saved_vs_civ_s"]), reverse=True)

    # Deduplicate near-identical ODs (same snapped graph nodes)
    seen_od: set[tuple] = set()
    unique: list[dict] = []
    for w in wins:
        key = (w["origin"], w["dest"])
        if key in seen_od:
            continue
        seen_od.add(key)
        unique.append(w)

    # Diversify boroughs in the mapped set
    top: list[dict] = []
    boro_counts: dict[str, int] = {}
    for w in unique:
        b = str(w.get("borough") or "UNK").upper()
        if boro_counts.get(b, 0) >= max(2, args.top // 3) and len(top) < args.top:
            continue
        top.append(w)
        boro_counts[b] = boro_counts.get(b, 0) + 1
        if len(top) >= args.top:
            break
    if len(top) < args.top:
        for w in unique:
            if w in top:
                continue
            top.append(w)
            if len(top) >= args.top:
                break

    print(f"Found {len(wins)} ROW wins ({len(unique)} unique ODs); mapping top {len(top)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for i, w in enumerate(top):
        rows_out.append(
            {
                "rank": i + 1,
                "trip_id": w["trip_id"],
                "label": w["label"],
                "borough": w.get("borough"),
                "zipcode": w.get("zipcode"),
                "civ_min": round(w["civ_s"] / 60.0, 2),
                "emv_min": round(w["emv_s"] / 60.0, 2),
                "emv_no_row_min": round(w["emv_no_row_s"] / 60.0, 2),
                "saved_vs_civ_min": round(w["saved_vs_civ_s"] / 60.0, 2),
                "pct_vs_civ": round(w["pct_vs_civ"], 2),
                "row_attrib_min": round(w["row_attrib_s"] / 60.0, 2),
                "emv_only_edges": w["emv_only_edges"],
                "start_lat": w["start_lat"],
                "start_lon": w["start_lon"],
                "dest_lat": w["dest_lat"],
                "dest_lon": w["dest_lon"],
                "hour": args.hour,
                "weather": args.weather,
            }
        )
    csv_path = args.out_dir / "row_wins.csv"
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)

    map_path = None
    if top:
        map_path = build_row_wins_map(
            G,
            top,
            args.out_dir / "row_wins_map.html",
            condition_label=cond.label,
        )

    summary = {
        "condition": cond.to_meta(),
        "candidates_scanned": scanned,
        "wins_found": len(wins),
        "top_mapped": len(top),
        "thresholds": {
            "min_pct": args.min_pct,
            "min_saved_s": args.min_saved_s,
            "min_row_attrib_s": args.min_row_attrib_s,
        },
        "top": rows_out,
        "csv": str(csv_path),
        "map": str(map_path) if map_path else None,
    }
    (args.out_dir / "row_wins_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("wins_found", "top_mapped", "map", "csv")}, indent=2))
    if rows_out:
        print(pd.DataFrame(rows_out)[
            ["rank", "label", "civ_min", "emv_min", "saved_vs_civ_min", "row_attrib_min", "emv_only_edges"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
