#!/usr/bin/env python3
"""
Visualize EMVRouteOptimization datasets and inferred OD pairs.

Produces static PNGs under data/figures/ and an interactive Folium map
(data/figures/emv_nyc_map.html).

If GOOGLE_MAPS_API_KEY is set, also runs a small Google Maps control
classification sample and plots station_plausible vs on_road_required.
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

sns.set_theme(style="whitegrid", context="talk")


def _load(raw: Path, processed: Path, samples: Path):
    od = pd.read_csv(processed / "od_pairs_inferred_starts.csv", low_memory=False)
    for c in ("travel_seconds", "response_seconds", "crow_flies_km", "start_lat", "start_lon", "dest_lat", "dest_lon"):
        if c in od.columns:
            od[c] = pd.to_numeric(od[c], errors="coerce")
    stations = pd.read_csv(raw / "ems_stations.csv") if (raw / "ems_stations.csv").exists() else None
    hospitals = pd.read_csv(raw / "hospital_bays.csv") if (raw / "hospital_bays.csv").exists() else None
    firehouses = pd.read_csv(raw / "fdny_firehouses.csv") if (raw / "fdny_firehouses.csv").exists() else None
    csls = pd.read_csv(processed / "synthetic_csls.csv") if (processed / "synthetic_csls.csv").exists() else None
    gmaps = (
        pd.read_csv(processed / "origin_class_gmaps_control.csv")
        if (processed / "origin_class_gmaps_control.csv").exists()
        else None
    )
    return od, stations, hospitals, firehouses, csls, gmaps


def fig_layer_and_mode(od: pd.DataFrame, out: Path):
    usable = od.dropna(subset=["start_lat", "dest_lat"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layer_counts = usable["depot_layer"].fillna("missing").value_counts()
    axes[0].bar(layer_counts.index.astype(str), layer_counts.values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"][: len(layer_counts)])
    axes[0].set_title("Inferred start layer")
    axes[0].set_ylabel("Incidents")
    axes[0].tick_params(axis="x", rotation=20)

    if "start_mode" in usable.columns:
        mode_counts = usable["start_mode"].fillna("missing").value_counts()
        axes[1].bar(mode_counts.index.astype(str), mode_counts.values, color=["#9467bd", "#8c564b", "#e377c2"][: len(mode_counts)])
        axes[1].set_title("Start mode (hybrid rule)")
        axes[1].tick_params(axis="x", rotation=20)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out / "01_start_layers.png", dpi=160)
    plt.close(fig)


def fig_travel_time(od: pd.DataFrame, out: Path):
    df = od.dropna(subset=["travel_seconds"]).copy()
    df = df[(df["travel_seconds"] > 0) & (df["travel_seconds"] < 3600)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df, x="travel_seconds", bins=40, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Observed EMS travel time")
    axes[0].set_xlabel("Travel seconds")

    if "borough" in df.columns:
        order = df.groupby("borough")["travel_seconds"].median().sort_values().index
        sns.boxplot(data=df, x="borough", y="travel_seconds", order=order, ax=axes[1], color="#aec7e8")
        axes[1].set_title("Travel time by borough")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(out / "02_travel_times.png", dpi=160)
    plt.close(fig)


def fig_crowflies_vs_travel(od: pd.DataFrame, out: Path):
    df = od.dropna(subset=["travel_seconds", "crow_flies_km", "depot_layer"]).copy()
    df = df[(df["travel_seconds"] > 0) & (df["travel_seconds"] < 3600) & (df["crow_flies_km"] < 30)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for layer, g in df.groupby("depot_layer"):
        ax.scatter(g["crow_flies_km"], g["travel_seconds"], s=12, alpha=0.35, label=str(layer))
    ax.set_xlabel("Crow-flies start→dest (km)")
    ax.set_ylabel("Observed travel time (s)")
    ax.set_title("Distance vs observed travel time by inferred start layer")
    ax.legend(markerscale=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out / "03_distance_vs_travel.png", dpi=160)
    plt.close(fig)


def fig_spatial_scatter(od: pd.DataFrame, out: Path):
    df = od.dropna(subset=["start_lat", "start_lon", "dest_lat", "dest_lon"]).copy()
    sample = df.sample(n=min(800, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sample["dest_lon"], sample["dest_lat"], s=10, alpha=0.35, c="#ff7f0e", label="Incident ZIP centroid")
    ax.scatter(sample["start_lon"], sample["start_lat"], s=10, alpha=0.35, c="#1f77b4", label="Inferred start")
    # draw a few OD lines
    for _, r in sample.sample(n=min(60, len(sample)), random_state=1).iterrows():
        ax.plot([r["start_lon"], r["dest_lon"]], [r["start_lat"], r["dest_lat"]], color="gray", alpha=0.15, lw=0.8)
    ax.set_title("NYC inferred OD pairs (sample)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out / "04_od_spatial.png", dpi=160)
    plt.close(fig)


def fig_depot_inventory(stations, hospitals, firehouses, csls, out: Path):
    counts = {
        "EMS stations": 0 if stations is None else len(stations),
        "Hospital bays": 0 if hospitals is None else len(hospitals),
        "Firehouses": 0 if firehouses is None else len(firehouses),
        "Synthetic CSLs": 0 if csls is None else len(csls),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    ax.set_title("Depot / staging inventory")
    ax.set_ylabel("Locations")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + max(counts.values()) * 0.01, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out / "05_depot_inventory.png", dpi=160)
    plt.close(fig)


def fig_gmaps_control(gmaps: pd.DataFrame | None, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if gmaps is None or not len(gmaps):
        for ax in axes:
            ax.axis("off")
        axes[0].text(
            0.5,
            0.5,
            "Google Maps control not run yet.\n\nexport GOOGLE_MAPS_API_KEY=...\nPYTHONPATH=src python scripts/classify_origins_with_gmaps.py --limit 100\nthen re-run visualize_data.py",
            ha="center",
            va="center",
            fontsize=12,
            wrap=True,
        )
        fig.tight_layout()
        fig.savefig(out / "06_gmaps_control.png", dpi=160)
        plt.close(fig)
        return

    counts = gmaps["origin_class"].value_counts()
    axes[0].bar(counts.index.astype(str), counts.values, color=["#d62728", "#2ca02c", "#7f7f7f"][: len(counts)])
    axes[0].set_title("GMaps control: origin class")
    axes[0].tick_params(axis="x", rotation=15)

    plot_df = gmaps.dropna(subset=["actual_travel_s", "gmaps_station_s"]).copy()
    if len(plot_df):
        colors = plot_df["origin_class"].map(
            {"on_road_required": "#d62728", "station_plausible": "#2ca02c", "unknown": "#7f7f7f"}
        ).fillna("#7f7f7f")
        axes[1].scatter(plot_df["gmaps_station_s"], plot_df["actual_travel_s"], c=colors, s=28, alpha=0.7)
        lim = max(plot_df["gmaps_station_s"].max(), plot_df["actual_travel_s"].max())
        axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="actual = GMaps")
        axes[1].set_xlabel("Google Maps station ETA (s)")
        axes[1].set_ylabel("Observed EMS travel (s)")
        axes[1].set_title("Civilian GMaps vs observed travel")
        axes[1].legend()
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "06_gmaps_control.png", dpi=160)
    plt.close(fig)


def build_folium_map(od, stations, hospitals, csls, out: Path):
    import json

    import folium
    from branca.element import Element, MacroElement, Template
    from folium.plugins import MarkerCluster

    df = od.dropna(subset=["start_lat", "start_lon", "dest_lat", "dest_lon"])
    sample = df.sample(n=min(400, len(df)), random_state=42)
    # Avoid tiles.openstreetmap.org — returns 403 for Folium apps that violate their
    # tile usage policy. Esri + Carto CDN are usable without a project API key.
    m = folium.Map(location=[40.75, -73.97], zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri &copy; OpenStreetMap contributors",
        name="Esri streets",
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Carto light",
        control=True,
        show=False,
        subdomains="abcd",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Carto streets",
        control=True,
        show=False,
        subdomains="abcd",
    ).add_to(m)

    key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GMAPS_API_KEY")
    if key:
        folium.TileLayer(
            tiles=f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={key}",
            attr="Google",
            name="Google Maps",
            overlay=False,
            control=True,
        ).add_to(m)

    # OD pairs as GeoJSON + custom JS so hover draws start→end line
    od_features = []
    for i, (_, r) in enumerate(sample.iterrows()):
        layer = str(r.get("depot_layer") or "other")
        start_color = {
            "csl": "#9467bd",
            "ems_station": "#1f77b4",
            "hospital_bay": "#2ca02c",
        }.get(layer, "#7f7f7f")
        pair = {
            "id": i,
            "start": [float(r["start_lat"]), float(r["start_lon"])],
            "end": [float(r["dest_lat"]), float(r["dest_lon"])],
            "start_color": start_color,
            "end_color": "#ff7f0e",
            "popup_start": (
                f"Inferred EMV start<br>{layer}: {r.get('depot_name')}"
                f"<br>mode={r.get('start_mode')}"
            ),
            "popup_end": (
                f"Incident ZIP centroid<br>id={r.get('incident_id')}"
                f"<br>ZIP {r.get('zipcode')}"
            ),
        }
        od_features.append(pair)

    od_group_name = "OD sample (hover for route line)"
    # Placeholder layer so LayerControl lists it; points/lines added in JS
    folium.FeatureGroup(name=od_group_name, show=True).add_to(m)

    if stations is not None and len(stations):
        g = MarkerCluster(name="EMS stations (facility layer)").add_to(m)
        for _, r in stations.iterrows():
            if pd.isna(r.get("latitude")):
                continue
            folium.Marker(
                [float(r["latitude"]), float(r["longitude"])],
                icon=folium.Icon(color="blue", icon="plus-sign"),
                popup=f"EMS station<br>{r.get('facname')}",
            ).add_to(g)

    if hospitals is not None and len(hospitals):
        g = MarkerCluster(name="Hospital bays (facility layer)").add_to(m)
        for _, r in hospitals.iterrows():
            if pd.isna(r.get("latitude")):
                continue
            folium.Marker(
                [float(r["latitude"]), float(r["longitude"])],
                icon=folium.Icon(color="green", icon="plus-sign"),
                popup=f"Hospital bay<br>{r.get('facname')}",
            ).add_to(g)

    if csls is not None and len(csls):
        g = MarkerCluster(name="Synthetic CSLs (on-road staging)").add_to(m)
        lat_col = "latitude" if "latitude" in csls.columns else "start_lat"
        lon_col = "longitude" if "longitude" in csls.columns else "start_lon"
        name_col = "facname" if "facname" in csls.columns else "location"
        for _, r in csls.iterrows():
            if pd.isna(r.get(lat_col)):
                continue
            folium.CircleMarker(
                [float(r[lat_col]), float(r[lon_col])],
                radius=4,
                color="purple",
                fill=True,
                popup=f"Synthetic CSL<br>{r.get(name_col)}",
            ).add_to(g)

    legend_html = """
    <div id="emvro-legend" style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        border: 1px solid #999;
        border-radius: 8px;
        padding: 12px 14px;
        font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
        font-size: 13px;
        line-height: 1.45;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        max-width: 340px;
    ">
      <div style="font-weight:700; margin-bottom:8px;">EMV map key</div>
      <div style="margin-bottom:6px;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff7f0e;margin-right:8px;"></span>Incident ZIP centroid (destination)</div>
      <div style="margin-bottom:6px;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#1f77b4;margin-right:8px;"></span>Inferred start: EMS station</div>
      <div style="margin-bottom:6px;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2ca02c;margin-right:8px;"></span>Inferred start: hospital bay</div>
      <div style="margin-bottom:6px;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#9467bd;margin-right:8px;"></span>Inferred start / CSL: on-road staging</div>
      <div style="margin-bottom:6px;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#7f7f7f;margin-right:8px;"></span>Inferred start: other / fallback</div>
      <hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">
      <div style="margin-bottom:4px;"><b>Blue + marker</b> — EMS station facility</div>
      <div style="margin-bottom:4px;"><b>Green + marker</b> — Hospital bay facility</div>
      <div style="margin-bottom:4px;"><b>Purple dots (clustered)</b> — Synthetic CSL candidates</div>
      <hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">
      <div style="font-weight:600;margin-bottom:4px;">Numbered circles (clusters)</div>
      <div style="margin-bottom:4px;">
        <span style="display:inline-block;width:18px;height:18px;border-radius:50%;background:#53e073;color:#000;font-size:10px;text-align:center;line-height:18px;margin-right:6px;">3</span>
        Green = small cluster (few nearby points)
      </div>
      <div style="margin-bottom:4px;">
        <span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#f1d357;color:#000;font-size:10px;text-align:center;line-height:22px;margin-right:6px;">12</span>
        Yellow = medium cluster
      </div>
      <div style="margin-bottom:4px;">
        <span style="display:inline-block;width:26px;height:26px;border-radius:50%;background:#f18072;color:#000;font-size:10px;text-align:center;line-height:26px;margin-right:6px;">40</span>
        Red/orange = large cluster
      </div>
      <div style="margin-top:4px;color:#555;font-size:11px;">
        The number is how many markers are grouped there. Zoom in or click the cluster to expand.
      </div>
      <div style="margin-top:8px;color:#555;font-size:11px;">
        Hover a start or destination dot to draw the OD line. Toggle layers at top right.
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    # Custom Leaflet JS: OD markers + hover polylines (runs after map init)
    pairs_json = json.dumps(od_features)
    map_name = m.get_name()

    class ODHover(MacroElement):
        _template = Template(
            """
            {% macro script(this, kwargs) %}
            (function() {
              var pairs = """
            + pairs_json
            + """;
              var map = """
            + map_name
            + """;
              var odLayer = L.layerGroup().addTo(map);
              var activeLine = null;
              function clearLine() {
                if (activeLine) { odLayer.removeLayer(activeLine); activeLine = null; }
              }
              pairs.forEach(function(p) {
                var startLL = L.latLng(p.start[0], p.start[1]);
                var endLL = L.latLng(p.end[0], p.end[1]);
                var startMk = L.circleMarker(startLL, {
                  radius: 5, color: p.start_color, fillColor: p.start_color,
                  fillOpacity: 0.85, weight: 1
                }).bindPopup(p.popup_start);
                var endMk = L.circleMarker(endLL, {
                  radius: 5, color: p.end_color, fillColor: p.end_color,
                  fillOpacity: 0.85, weight: 1
                }).bindPopup(p.popup_end);
                function showLine() {
                  clearLine();
                  activeLine = L.polyline([startLL, endLL], {
                    color: '#222222', weight: 3, opacity: 0.9, dashArray: '6 6'
                  }).addTo(odLayer);
                }
                startMk.on('mouseover', showLine);
                endMk.on('mouseover', showLine);
                startMk.on('mouseout', clearLine);
                endMk.on('mouseout', clearLine);
                startMk.addTo(odLayer);
                endMk.addTo(odLayer);
              });
            })();
            {% endmacro %}
            """
        )

    m.add_child(ODHover())

    folium.LayerControl(collapsed=False).add_to(m)
    path = out / "emv_nyc_map.html"
    m.save(str(path))
    return path


def maybe_run_gmaps(limit: int, processed: Path) -> pd.DataFrame | None:
    key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GMAPS_API_KEY")
    if not key:
        return None
    if (processed / "origin_class_gmaps_control.csv").exists():
        return pd.read_csv(processed / "origin_class_gmaps_control.csv")
    # Run classifier in-process for a small sample
    from subprocess import run

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "classify_origins_with_gmaps.py"),
        "--limit",
        str(limit),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    run(cmd, cwd=str(ROOT), env=env, check=False)
    if (processed / "origin_class_gmaps_control.csv").exists():
        return pd.read_csv(processed / "origin_class_gmaps_control.csv")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "figures")
    p.add_argument("--gmaps-limit", type=int, default=80)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gmaps_df = maybe_run_gmaps(args.gmaps_limit, args.processed)
    od, stations, hospitals, firehouses, csls, gmaps_file = _load(args.raw, args.processed, args.samples)
    if gmaps_df is not None:
        gmaps_file = gmaps_df

    fig_layer_and_mode(od, args.out)
    fig_travel_time(od, args.out)
    fig_crowflies_vs_travel(od, args.out)
    fig_spatial_scatter(od, args.out)
    fig_depot_inventory(stations, hospitals, firehouses, csls, args.out)
    fig_gmaps_control(gmaps_file, args.out)
    map_path = build_folium_map(od, stations, hospitals, csls, args.out)

    summary = {
        "figures": sorted([p.name for p in args.out.glob("*.png")]),
        "map": str(map_path),
        "od_rows": int(len(od)),
        "gmaps_control_rows": 0 if gmaps_file is None else int(len(gmaps_file)),
        "google_maps_key_present": bool(os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GMAPS_API_KEY")),
    }
    (args.out / "viz_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Open map:", map_path)


if __name__ == "__main__":
    main()
