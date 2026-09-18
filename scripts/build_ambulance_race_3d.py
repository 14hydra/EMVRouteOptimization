#!/usr/bin/env python3
"""
Build a Google Maps photorealistic-3D dual-ambulance race HTML.

Both ambulances are timed by **our** EMV travel-time model:
  • Grey  — follows the Google Maps *path* (geometry from Google)
  • Purple — follows our best EMV path (MIPSSTW + MCS)

Google's own ETA is stored only as a reference; the race clock is ours.

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


def _resample(coords: list[list[float]], n: int = 240) -> list[list[float]]:
    if len(coords) < 2:
        return coords

    def dist(a, b):
        dlat = (a[0] - b[0]) * 111_320
        dlon = (a[1] - b[1]) * 111_320 * math.cos(math.radians((a[0] + b[0]) / 2))
        return math.hypot(dlat, dlon)

    seg = [0.0]
    for i in range(1, len(coords)):
        seg.append(seg[-1] + dist(coords[i - 1], coords[i]))
    total = seg[-1] or 1.0
    out = []
    for k in range(n):
        t = (k / (n - 1)) * total
        j = 1
        while j < len(seg) and seg[j] < t:
            j += 1
        j = min(j, len(coords) - 1)
        t0, t1 = seg[j - 1], seg[j]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        a, b = coords[j - 1], coords[j]
        out.append([a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])])
    return out


def _timed_track(G, nodes: list, n: int = 240) -> tuple[list[list[float]], float]:
    """
    Build a time-parameterized track [lat, lon, cum_seconds] using our emv_s
    edge costs. Animation should advance by model time so shared edges move
    both ambulances at the same speed.
    """
    if len(nodes) < 2:
        ll = _path_ll(G, nodes)
        return [[ll[0][0], ll[0][1], 0.0]] if ll else [], 0.0

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
            # Missing edge — keep geometry, no time advance
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
    # Dense uniform-in-time resample for smooth playback
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
        out.append(
            [
                a[0] + u * (b[0] - a[0]),
                a[1] + u * (b[1] - a[1]),
                t,
            ]
        )
    return out, total_t


def _edge_data(G, u, v) -> dict:
    raw = G.get_edge_data(u, v) or {}
    if not raw:
        return {}
    first = next(iter(raw.values()), None)
    if isinstance(first, dict):
        for attrs in raw.values():
            if isinstance(attrs, dict) and (
                attrs.get("emv_corridor") or attrs.get("civilian_forbidden")
            ):
                return attrs
        return first
    return raw if isinstance(raw, dict) else {}


def _corridor_ll(G, path: list) -> list[list[list[float]]]:
    segs = []
    for u, v in zip(path[:-1], path[1:]):
        data = _edge_data(G, u, v)
        if not (data.get("emv_corridor") or data.get("civilian_forbidden")):
            continue
        nu, nv = G.nodes[u], G.nodes[v]
        segs.append(
            [
                [float(nu.get("y", nu.get("lat"))), float(nu.get("x", nu.get("lon")))],
                [float(nv.get("y", nv.get("lat"))), float(nv.get("x", nv.get("lon")))],
            ]
        )
    return segs


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
        edge = min(edata.values(), key=lambda d: float(d.get("length") or d.get("length_m") or 1e18))
        total += float(edge.get("length") or edge.get("length_m") or 0.0)
    return total


def _ensure_civilian_weights(G) -> None:
    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight_civilian"] = float(
            data.get("civilian_s") or data.get("travel_time") or 1.0
        )


def _snap_polyline_to_nodes(G, coords: list[list[float]], origin, dest) -> list:
    """
    Map-match a lat/lon polyline onto the drive graph.

    Walk the Google corridor and only accept hops that stay local (direct edge
    or a short connecting path). This avoids zigzag artifacts that explode
    length when nearest-node snaps bounce around.
    """
    import networkx as nx

    from emvro.routing import nearest_node

    _ensure_civilian_weights(G)

    # Dense samples along the Google polyline (~every 40–60 m of input spacing)
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

        # Allow a short legal connector, but reject long detours / U-turns
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
                # Fall back: civilian Dijkstra for the whole OD if match failed
                nodes = nx.shortest_path(G, origin, dest, weight="weight_civilian")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            nodes = nx.shortest_path(G, origin, dest, weight="weight_civilian")

    deduped = [nodes[0]]
    for n in nodes[1:]:
        if n != deduped[-1]:
            deduped.append(n)

    # Sanity: if matched path is absurdly long vs crow-flies OD, use civilian SP
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


def solve_routes(graph_path: Path) -> dict:
    from emvro.routing import (
        control_google_maps,
        nearest_node,
        prepare_routing_graph,
        solve_mipsstw_mcs,
    )
    from emvro.routing.graph import path_stats

    hour = SCENARIO["hour"]
    o, d = SCENARIO["origin"], SCENARIO["dest"]
    print(f"Preparing graph @ hour {hour}…")
    G = prepare_routing_graph(graph_path, hour=hour)
    origin = nearest_node(G, o["lon"], o["lat"])
    dest = nearest_node(G, d["lon"], d["lat"])

    print("Routing Google Maps (path only)…")
    gmaps = control_google_maps(
        o["lat"],
        o["lon"],
        d["lat"],
        d["lon"],
        cache_path=ROOT / "data" / "processed" / "gmaps_cache.json",
    )
    print("Routing best EMV (MIPSSTW + MCS)…")
    emv = solve_mipsstw_mcs(
        G, origin, dest, n_iterations=12, n_nests=10, seed=42
    )

    g_poly = list((gmaps.meta or {}).get("polyline_latlons") or [])
    if len(g_poly) < 2:
        raise RuntimeError("Google Maps returned no polyline")
    if not emv.ok or not emv.node_path:
        raise RuntimeError(f"EMV route failed: {emv.meta}")

    # Same origin/dest pads for both ambulances
    o_ll = _path_ll(G, [origin])[0]
    d_ll = _path_ll(G, [dest])[0]

    # Snap Google geometry onto our graph, then score BOTH paths with our EMV model
    print("Scoring both paths with our EMV travel-time model…")
    g_nodes = _snap_polyline_to_nodes(G, g_poly, origin, dest)
    if len(g_nodes) < 2:
        raise RuntimeError("Failed to map-match Google polyline onto the drive graph")

    g_secs, g_dist, g_edges = path_stats(G, g_nodes)
    e_secs, e_dist, e_edges = path_stats(G, emv.node_path)
    if not (g_secs == g_secs and g_secs > 0):
        raise RuntimeError("Our model returned no time for the Google path")
    if not (e_secs == e_secs and e_secs > 0):
        raise RuntimeError("Our model returned no time for the EMV path")

    # Force identical start/end pads, then time-parameterize with our emv_s costs
    g_nodes = [origin, *g_nodes[1:-1], dest] if len(g_nodes) >= 2 else [origin, dest]
    e_nodes = (
        [origin, *emv.node_path[1:-1], dest]
        if len(emv.node_path) >= 2
        else [origin, dest]
    )
    g_track, g_total = _timed_track(G, g_nodes, 240)
    e_track, e_total = _timed_track(G, e_nodes, 240)
    # Prefer path_stats totals (canonical); stamp onto final samples
    g_total = float(g_secs)
    e_total = float(e_secs)
    if g_track:
        g_track[0] = [o_ll[0], o_ll[1], 0.0]
        g_track[-1] = [d_ll[0], d_ll[1], g_total]
    if e_track:
        e_track[0] = [o_ll[0], o_ll[1], 0.0]
        e_track[-1] = [d_ll[0], d_ll[1], e_total]

    g_min = g_total / 60.0
    e_min = e_total / 60.0
    pct = 100.0 * (1.0 - e_min / g_min) if g_min > 0 else 0.0
    gmaps_api_min = float(gmaps.travel_seconds) / 60.0

    payload = {
        "scenario": SCENARIO,
        "timing_model": "emvro_emv_edge_costs",
        "gmaps": {
            "label": "Google Maps path",
            "minutes": round(g_min, 2),
            "coords": g_track,  # [lat, lon, cum_s] — same model clock
            "distance_km": round(float(g_dist) / 1000.0, 3),
            "n_edges": int(g_edges),
            # Reference only — not used for the race clock
            "google_api_minutes": round(gmaps_api_min, 2),
        },
        "emv": {
            "label": "Best EMV path",
            "minutes": round(e_min, 2),
            "coords": e_track,  # [lat, lon, cum_s] — same model clock
            "distance_km": round(float(e_dist) / 1000.0, 3),
            "n_edges": int(e_edges),
            "corridor_segs": _corridor_ll(G, emv.node_path),
        },
        "pct_faster": round(pct, 1),
    }
    print(
        f"  Same model clock · Google path {g_min:.2f} min "
        f"(API said {gmaps_api_min:.2f}) · EMV path {e_min:.2f} min · "
        f"{pct:.1f}% faster · corridor={len(payload['emv']['corridor_segs'])}"
    )
    return payload


def normalize_payload(raw: dict) -> dict:
    sc = raw.get("scenario") or SCENARIO
    g = raw["gmaps"]
    e = raw["emv"]
    return {
        "scenario": {
            "id": sc.get("id", SCENARIO["id"]),
            "label": sc.get("label", SCENARIO["label"]),
            "origin": sc.get("origin", SCENARIO["origin"]),
            "dest": sc.get("dest", SCENARIO["dest"]),
            "hour": sc.get("hour", SCENARIO["hour"]),
        },
        "gmaps": {
            "label": g.get("label", "Google Maps"),
            "minutes": float(g["minutes"]),
            "coords": g["coords"],
        },
        "emv": {
            "label": e.get("label", "Best EMV"),
            "minutes": float(e["minutes"]),
            "coords": e["coords"],
            "corridor_segs": e.get("corridor_segs") or [],
        },
        "pct_faster": float(raw.get("pct_faster") or 0),
    }


def write_html(payload: dict, out: Path, api_key: str) -> Path:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    html = (
        tpl.replace("__DATA__", json.dumps(payload))
        .replace("__LABEL__", payload["scenario"]["label"])
        .replace("__GMAPS_MIN__", f"{payload['gmaps']['minutes']:.2f}")
        .replace("__EMV_MIN__", f"{payload['emv']['minutes']:.2f}")
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
