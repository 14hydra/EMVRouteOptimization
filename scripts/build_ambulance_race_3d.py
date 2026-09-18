#!/usr/bin/env python3
"""
Build a Google Maps photorealistic-3D multi-ambulance race HTML.

Racers (all timed by **our** EMV travel-time model):
  • Google Maps alternatives (up to 3) — geometry from Google, scored by us
  • MIPSSTW + MCS
  • Composite DRL
  • GBDT edge-cost router

Google's own ETA is reference-only; the race clock is ours.

Usage:
  PYTHONPATH=src python scripts/build_ambulance_race_3d.py
  open data/figures/ambulance_race_3d.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEMPLATE = Path(__file__).with_name("ambulance_race_3d_template.html")

omp = "/opt/homebrew/opt/libomp/lib"
if Path(omp).exists():
    os.environ["DYLD_LIBRARY_PATH"] = omp + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

SCENARIO = {
    "id": "midtown_to_civic",
    "label": "Midtown → Civic Center",
    "origin": {"lat": 40.7580, "lon": -73.9855},
    "dest": {"lat": 40.7115, "lon": -74.0060},
    "hour": 17,
}

GMAPS_COLORS = ("#d0d5db", "#9aa7b5", "#6b7a8a")
MODEL_META = {
    "mipsstw_mcs": {"label": "MIPSSTW + MCS", "color": "#c084fc"},
    "composite_drl": {"label": "Composite DRL", "color": "#34d399"},
    "gbdt_router": {"label": "GBDT router", "color": "#f59e0b"},
}


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _path_ll(G, path: list) -> list[list[float]]:
    out = []
    for n in path:
        node = G.nodes[n]
        out.append(
            [float(node.get("y", node.get("lat"))), float(node.get("x", node.get("lon")))]
        )
    return out


def _node_ll(G, n) -> tuple[float, float]:
    node = G.nodes[n]
    return float(node.get("y", node.get("lat"))), float(node.get("x", node.get("lon")))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _path_length_m(G, path: list) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            return 1e18
        edge = min(
            edata.values(),
            key=lambda d: float(d.get("length") or d.get("length_m") or 1e18),
        )
        total += float(edge.get("length") or edge.get("length_m") or 0.0)
    return total


def _ensure_civilian_weights(G) -> None:
    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight_civilian"] = float(
            data.get("civilian_s") or data.get("travel_time") or 1.0
        )


def _snap_polyline_to_nodes(G, coords: list[list[float]], origin, dest) -> list:
    """Map-match a lat/lon polyline onto the drive graph with local hops only."""
    import networkx as nx

    from emvro.routing.graph import nearest_node

    _ensure_civilian_weights(G)

    step = max(1, len(coords) // 160)
    samples = [coords[0], *coords[1:-1:step], coords[-1]]

    nodes: list = [origin]
    for lat, lon in samples[1:]:
        n = nearest_node(G, float(lon), float(lat))
        if n == nodes[-1]:
            continue
        prev = nodes[-1]
        plat, plon = _node_ll(G, prev)
        nlat, nlon = _node_ll(G, n)
        crow = _haversine_m(plat, plon, nlat, nlon)

        if G.has_edge(prev, n):
            nodes.append(n)
            continue

        try:
            sp = nx.shortest_path(G, prev, n, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        sp_len = _path_length_m(G, sp)
        if sp_len <= max(180.0, 2.5 * crow) and len(sp) <= 8:
            nodes.extend(sp[1:])

    if nodes[-1] != dest:
        try:
            sp = nx.shortest_path(G, nodes[-1], dest, weight="length")
            sp_len = _path_length_m(G, sp)
            crow = _haversine_m(*_node_ll(G, nodes[-1]), *_node_ll(G, dest))
            if sp_len <= max(400.0, 3.0 * crow):
                nodes.extend(sp[1:])
            else:
                nodes = nx.shortest_path(G, origin, dest, weight="weight_civilian")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            nodes = nx.shortest_path(G, origin, dest, weight="weight_civilian")

    deduped = [nodes[0]]
    for n in nodes[1:]:
        if n != deduped[-1]:
            deduped.append(n)

    o_ll = _node_ll(G, origin)
    d_ll = _node_ll(G, dest)
    od_crow = _haversine_m(*o_ll, *d_ll)
    matched_len = _path_length_m(G, deduped)
    if matched_len > max(15_000.0, 4.0 * od_crow) or len(deduped) < 2:
        print(
            f"  map-match rejected (len={matched_len/1000:.1f} km); "
            "falling back to civilian Dijkstra on our graph"
        )
        deduped = nx.shortest_path(G, origin, dest, weight="weight_civilian")
    return deduped


def _corridor_ll(G, path: list) -> list[list[list[float]]]:
    from emvro.routing.emv_corridors import path_corridor_segments

    return [seg["coords"] for seg in path_corridor_segments(G, path)]


def _timed_track(G, nodes: list, n: int = 240) -> tuple[list[list[float]], float]:
    """Build [lat, lon, cum_seconds] using our emv_s edge costs."""
    if len(nodes) < 2:
        ll = _path_ll(G, nodes)
        return ([[ll[0][0], ll[0][1], 0.0]] if ll else []), 0.0

    samples: list[list[float]] = []
    cum = 0.0
    for i, n_id in enumerate(nodes):
        lat, lon = _node_ll(G, n_id)
        if i == 0:
            samples.append([lat, lon, 0.0])
            continue
        u, v = nodes[i - 1], n_id
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            samples.append([lat, lon, cum])
            continue
        edge = min(
            edata.values(),
            key=lambda d: float(d.get("emv_s") or d.get("travel_time") or 1e18),
        )
        dt = float(edge.get("emv_s") or edge.get("travel_time") or 0.0)
        cum += max(0.0, dt)
        samples.append([lat, lon, cum])

    total_t = cum if cum > 0 else 1.0
    out: list[list[float]] = []
    for k in range(n):
        t = (k / (n - 1)) * total_t
        j = 1
        while j < len(samples) and samples[j][2] < t:
            j += 1
        j = min(j, len(samples) - 1)
        t0, t1 = samples[j - 1][2], samples[j][2]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        a, b = samples[j - 1], samples[j]
        out.append([a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), t])
    return out, total_t


def _score_path(
    G,
    nodes: list,
    *,
    origin,
    dest,
    o_ll: list[float],
    d_ll: list[float],
    racer_id: str,
    label: str,
    color: str,
    group: str,
    google_api_minutes: float | None = None,
) -> dict:
    from emvro.routing.graph import path_stats

    nodes = [origin, *nodes[1:-1], dest] if len(nodes) >= 2 else [origin, dest]
    secs, dist, n_edges = path_stats(G, nodes)
    if not (secs == secs and secs > 0):
        raise RuntimeError(f"Our model returned no time for {racer_id}")
    track, _ = _timed_track(G, nodes, 240)
    total = float(secs)
    if track:
        track[0] = [o_ll[0], o_ll[1], 0.0]
        track[-1] = [d_ll[0], d_ll[1], total]
    out = {
        "id": racer_id,
        "group": group,
        "label": label,
        "color": color,
        "minutes": round(total / 60.0, 2),
        "coords": track,
        "distance_km": round(float(dist) / 1000.0, 3),
        "n_edges": int(n_edges),
        "corridor_segs": _corridor_ll(G, nodes) if group == "emv" else [],
    }
    if google_api_minutes is not None:
        out["google_api_minutes"] = round(float(google_api_minutes), 2)
    return out


def solve_routes(graph_path: Path) -> dict:
    from emvro.routing import (
        control_google_maps,
        nearest_node,
        prepare_routing_graph,
        solve_composite_drl,
        solve_gbdt_route,
        solve_mipsstw_mcs,
    )
    from emvro.routing.gbdt_router import build_edge_training_frame, train_gbdt_edge_model
    from emvro.routing.graph import dijkstra_route

    hour = SCENARIO["hour"]
    o, d = SCENARIO["origin"], SCENARIO["dest"]
    print(f"Preparing graph @ hour {hour}…")
    G = prepare_routing_graph(graph_path, hour=hour)
    origin = nearest_node(G, o["lon"], o["lat"])
    dest = nearest_node(G, d["lon"], d["lat"])
    o_ll = _path_ll(G, [origin])[0]
    d_ll = _path_ll(G, [dest])[0]

    print("Routing Google Maps alternatives (path only)…")
    gmaps = control_google_maps(
        o["lat"],
        o["lon"],
        d["lat"],
        d["lon"],
        cache_path=ROOT / "data" / "processed" / "gmaps_cache.json",
        alternatives=True,
    )
    alts = list((gmaps.meta or {}).get("alternatives") or [])
    if not alts and (gmaps.meta or {}).get("polyline_latlons"):
        alts = [
            {
                "polyline_latlons": gmaps.meta["polyline_latlons"],
                "duration_in_traffic_s": gmaps.travel_seconds,
                "duration_s": (gmaps.meta or {}).get("duration_s"),
                "distance_m": gmaps.distance_m,
            }
        ]

    google_alts: list[dict] = []
    seen: set[str] = set()
    for alt in alts:
        poly = list(alt.get("polyline_latlons") or [])
        if len(poly) < 2:
            continue
        mid = poly[len(poly) // 2]
        key = (
            f"{round(poly[0][0], 4)},{round(poly[0][1], 4)}|"
            f"{round(mid[0], 4)},{round(mid[1], 4)}|"
            f"{round(poly[-1][0], 4)},{round(poly[-1][1], 4)}"
        )
        if key in seen:
            continue
        seen.add(key)
        google_alts.append(alt)
        if len(google_alts) >= 3:
            break
    if not google_alts:
        raise RuntimeError("Google Maps returned no usable alternative polylines")
    print(f"  Google returned {len(google_alts)} distinct route(s)")

    print("Training GBDT edge model…")
    model_path = ROOT / "data" / "processed" / "models" / "gbdt_edge_costs.joblib"
    edge_df = build_edge_training_frame(G, hours=[hour], max_edges=6000)
    gbdt_bundle = train_gbdt_edge_model(edge_df, out_path=model_path)

    emv_dij = dijkstra_route(
        G, origin, dest, weight="weight_emv", model_name="emv_dijkstra"
    )
    deadline = float(emv_dij.travel_seconds) * 1.05 if emv_dij.ok else None

    print("Routing three EMV models…")
    model_results = {
        "mipsstw_mcs": solve_mipsstw_mcs(
            G,
            origin,
            dest,
            deadline_s=deadline,
            n_iterations=12,
            n_nests=10,
            seed=42,
        ),
        "composite_drl": solve_composite_drl(G, origin, dest, episodes=80, seed=42),
        "gbdt_router": solve_gbdt_route(
            G, origin, dest, bundle=gbdt_bundle, hour=hour
        ),
    }
    for mid, res in model_results.items():
        if not res.ok or not res.node_path:
            raise RuntimeError(f"{mid} failed: {res.meta}")

    print("Scoring all racers with our EMV travel-time model…")
    racers: list[dict] = []

    for i, alt in enumerate(google_alts):
        poly = list(alt.get("polyline_latlons") or [])
        nodes = _snap_polyline_to_nodes(G, poly, origin, dest)
        api_s = alt.get("duration_in_traffic_s") or alt.get("duration_s")
        api_min = float(api_s) / 60.0 if api_s else None
        racers.append(
            _score_path(
                G,
                nodes,
                origin=origin,
                dest=dest,
                o_ll=o_ll,
                d_ll=d_ll,
                racer_id=f"gmaps_{i + 1}",
                label=f"Google Maps #{i + 1}",
                color=GMAPS_COLORS[i % len(GMAPS_COLORS)],
                group="gmaps",
                google_api_minutes=api_min,
            )
        )

    for mid, res in model_results.items():
        meta = MODEL_META[mid]
        racers.append(
            _score_path(
                G,
                list(res.node_path),
                origin=origin,
                dest=dest,
                o_ll=o_ll,
                d_ll=d_ll,
                racer_id=mid,
                label=meta["label"],
                color=meta["color"],
                group="emv",
            )
        )

    gmaps_racers = [r for r in racers if r["group"] == "gmaps"]
    emv_racers = [r for r in racers if r["group"] == "emv"]
    best_g = min(gmaps_racers, key=lambda r: r["minutes"])
    best_e = min(emv_racers, key=lambda r: r["minutes"])
    pct = (
        100.0 * (1.0 - best_e["minutes"] / best_g["minutes"])
        if best_g["minutes"] > 0
        else 0.0
    )

    payload = {
        "scenario": SCENARIO,
        "timing_model": "emvro_emv_edge_costs",
        "racers": racers,
        "best_gmaps_id": best_g["id"],
        "best_emv_id": best_e["id"],
        "pct_faster": round(pct, 1),
        "gmaps": {
            "label": best_g["label"],
            "minutes": best_g["minutes"],
            "coords": best_g["coords"],
            "google_api_minutes": best_g.get("google_api_minutes"),
        },
        "emv": {
            "label": best_e["label"],
            "minutes": best_e["minutes"],
            "coords": best_e["coords"],
            "corridor_segs": best_e.get("corridor_segs") or [],
        },
    }
    print("  Same model clock:")
    for r in racers:
        extra = ""
        if r.get("google_api_minutes") is not None:
            extra = f" (API {r['google_api_minutes']:.2f})"
        print(f"    {r['label']:18s}  {r['minutes']:6.2f} min{extra}")
    print(
        f"  Best EMV ({best_e['label']}) is {pct:.1f}% faster than "
        f"best Google path ({best_g['label']}) on our clock"
    )
    return payload


def normalize_payload(raw: dict) -> dict:
    sc = raw.get("scenario") or SCENARIO
    if raw.get("racers"):
        racers = raw["racers"]
    else:
        g = raw["gmaps"]
        e = raw["emv"]
        racers = [
            {
                "id": "gmaps_1",
                "group": "gmaps",
                "label": g.get("label", "Google Maps"),
                "color": GMAPS_COLORS[0],
                "minutes": float(g["minutes"]),
                "coords": g["coords"],
                "corridor_segs": [],
            },
            {
                "id": "mipsstw_mcs",
                "group": "emv",
                "label": e.get("label", "Best EMV"),
                "color": MODEL_META["mipsstw_mcs"]["color"],
                "minutes": float(e["minutes"]),
                "coords": e["coords"],
                "corridor_segs": e.get("corridor_segs") or [],
            },
        ]
    gmaps_racers = [r for r in racers if r.get("group") == "gmaps"] or racers[:1]
    emv_racers = [r for r in racers if r.get("group") == "emv"] or racers[1:]
    best_g = min(gmaps_racers, key=lambda r: float(r["minutes"]))
    best_e = min(emv_racers, key=lambda r: float(r["minutes"]))
    return {
        "scenario": {
            "id": sc.get("id", SCENARIO["id"]),
            "label": sc.get("label", SCENARIO["label"]),
            "origin": sc.get("origin", SCENARIO["origin"]),
            "dest": sc.get("dest", SCENARIO["dest"]),
            "hour": sc.get("hour", SCENARIO["hour"]),
        },
        "timing_model": raw.get("timing_model", "emvro_emv_edge_costs"),
        "racers": racers,
        "best_gmaps_id": raw.get("best_gmaps_id", best_g["id"]),
        "best_emv_id": raw.get("best_emv_id", best_e["id"]),
        "pct_faster": float(raw.get("pct_faster") or 0),
        "gmaps": {
            "label": best_g["label"],
            "minutes": float(best_g["minutes"]),
            "coords": best_g["coords"],
        },
        "emv": {
            "label": best_e["label"],
            "minutes": float(best_e["minutes"]),
            "coords": best_e["coords"],
            "corridor_segs": best_e.get("corridor_segs") or [],
        },
    }


def write_html(payload: dict, out: Path, api_key: str) -> Path:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    html = (
        tpl.replace("__DATA__", json.dumps(payload))
        .replace("__LABEL__", payload["scenario"]["label"])
        .replace("__PCT__", f"{payload['pct_faster']:.0f}")
        .replace("__GMAPS_KEY__", api_key)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive.graphml")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "figures" / "ambulance_race_3d.html")
    p.add_argument(
        "--from-json",
        type=Path,
        nargs="?",
        const=ROOT / "data" / "figures" / "ambulance_race_3d.json",
        default=None,
        help="Rebuild HTML from existing route JSON (skip routing).",
    )
    args = p.parse_args()
    _load_dotenv()
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GOOGLE_MAPS_API_KEY missing in env / .env")
    if not TEMPLATE.exists():
        raise SystemExit(f"Missing template: {TEMPLATE}")

    if args.from_json:
        raw = json.loads(Path(args.from_json).read_text())
        payload = normalize_payload(raw)
        print(f"Loaded routes from {args.from_json}")
    else:
        payload = solve_routes(args.graph)
        meta = args.out.with_suffix(".json")
        meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Meta ", meta)

    path = write_html(payload, args.out, api_key)
    print("Wrote", path)


if __name__ == "__main__":
    main()
