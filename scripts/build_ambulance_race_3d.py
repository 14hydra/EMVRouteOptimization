#!/usr/bin/env python3
"""
Build a Google Maps photorealistic-3D dual-ambulance race HTML.

One ambulance follows Google Maps (civilian). The other follows the best EMV
model (MIPSSTW + MCS) on Midtown → Civic Center, where EMV beats GMaps.

Usage:
  PYTHONPATH=src python scripts/build_ambulance_race_3d.py --from-json
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


def solve_routes(graph_path: Path) -> dict:
    from emvro.routing import (
        control_google_maps,
        nearest_node,
        prepare_routing_graph,
        solve_mipsstw_mcs,
    )

    hour = SCENARIO["hour"]
    o, d = SCENARIO["origin"], SCENARIO["dest"]
    print(f"Preparing graph @ hour {hour}…")
    G = prepare_routing_graph(graph_path, hour=hour)
    origin = nearest_node(G, o["lon"], o["lat"])
    dest = nearest_node(G, d["lon"], d["lat"])

    print("Routing Google Maps…")
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

    g_coords = list((gmaps.meta or {}).get("polyline_latlons") or [])
    if len(g_coords) < 2:
        raise RuntimeError("Google Maps returned no polyline")
    if not emv.ok or not emv.node_path:
        raise RuntimeError(f"EMV route failed: {emv.meta}")

    e_coords = _path_ll(G, emv.node_path)
    o_ll = _path_ll(G, [origin])[0]
    d_ll = _path_ll(G, [dest])[0]
    g_coords = [o_ll] + g_coords[1:-1] + [d_ll]
    e_coords = [o_ll] + e_coords[1:-1] + [d_ll]

    g_min = float(gmaps.travel_seconds) / 60.0
    e_min = float(emv.travel_seconds) / 60.0
    pct = 100.0 * (1.0 - e_min / g_min) if g_min > 0 else 0.0

    payload = {
        "scenario": SCENARIO,
        "gmaps": {
            "label": "Google Maps",
            "minutes": round(g_min, 2),
            "coords": _resample(g_coords, 240),
        },
        "emv": {
            "label": "Best EMV · MIPSSTW + MCS",
            "minutes": round(e_min, 2),
            "coords": _resample(e_coords, 240),
            "corridor_segs": _corridor_ll(G, emv.node_path),
        },
        "pct_faster": round(pct, 1),
    }
    print(
        f"  GMaps {g_min:.2f} min · EMV {e_min:.2f} min · "
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
