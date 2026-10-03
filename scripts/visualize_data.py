#!/usr/bin/env python3
"""
Visualize London Fire Brigade (LFB) incident density on an interactive map.

Produces data/figures/london_incident_density.html: a smoothed incident-density
surface (kernel density, incidents per km² per year) for every LFB incident in
data/raw/london/lfb_incidents_2024_onwards.csv over a real London basemap, with
a per-incident-type switch, the ten densest 500 m hotspots, a hover readout,
and a summary panel (incident mix, hour-of-day profile, busiest boroughs).

Incident coordinates come from Easting_rounded / Northing_rounded (British
National Grid, EPSG:27700, rounded to 50 m by LFB). They are populated for
every row, unlike Latitude / Longitude (~38%).

Why a precomputed raster instead of a Leaflet heat plugin: client-side heat
layers add up overlapping blurred points, so colour depends on zoom level and
saturates across inner London. Here the density is computed once on a Web
Mercator grid (so it lines up exactly with the basemap tiles) and the legend
carries real units.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PIXEL_GROUND_M = 25  # density raster resolution on the ground (≈ native at zoom 14)
KDE_SIGMA_M = 90  # Gaussian kernel bandwidth (ground metres); larger = blurrier
READOUT_BLOCK_PX = 20  # hover readout cell = 20 px = 500 m
LOG_DECADES = 2.0  # colour scale spans vmax/100 .. vmax (log)
HOTSPOT_CELL_M = 500  # hotspot cell size, ~a few city blocks
N_HOTSPOTS = 10
CMAP = "YlOrRd"  # sequential: density is a magnitude, one hue family low→high

GROUP_COLORS = {"False Alarm": "#7f7f7f", "Special Service": "#1f77b4", "Fire": "#d62728"}

USECOLS = [
    "DateOfCall",
    "HourOfCall",
    "IncidentGroup",
    "PropertyCategory",
    "ProperCase",
    "IncGeo_WardNameNew",
    "Postcode_district",
    "Easting_rounded",
    "Northing_rounded",
    "FirstPumpArriving_AttendanceTime",
]

# Esri tiles need no API key (Carto's basemaps.cartocdn.com now returns
# "API KEY REQUIRED" tiles; tiles.openstreetmap.org 403s Folium apps).
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"


def _load(path: Path) -> pd.DataFrame:
    from pyproj import Transformer

    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    df = df.dropna(subset=["Easting_rounded", "Northing_rounded"])
    df["attend_s"] = pd.to_numeric(df["FirstPumpArriving_AttendanceTime"], errors="coerce")
    e, n = df["Easting_rounded"].to_numpy(), df["Northing_rounded"].to_numpy()
    df["mx"], df["my"] = Transformer.from_crs("EPSG:27700", "EPSG:3857", always_xy=True).transform(e, n)
    return df


def _mode(s: pd.Series) -> str:
    s = s.dropna()
    if not len(s):
        return "?"
    # capwords, not str.title(): "EARL'S COURT" -> "Earl's Court", not "Earl'S Court".
    return string.capwords(str(s.mode().iloc[0]).strip()).replace(" And ", " and ").replace(" Of ", " of ")


def _fmt_mmss(s: float | None) -> str:
    if s is None or np.isnan(s):
        return "–"
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _nice(v: float) -> str:
    """Round a legend value to 2 significant figures."""
    r = float(f"{v:.2g}")
    return f"{r:,.0f}" if r >= 10 else f"{r:g}"


class Grid:
    """Web Mercator raster covering all incidents.

    Mercator metres are stretched by 1/cos(lat) relative to the ground, so the
    pixel size is set from PIXEL_GROUND_M at the data's mean latitude.
    """

    def __init__(self, df: pd.DataFrame):
        from pyproj import Transformer

        self.to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lat0 = np.radians(self.to_ll.transform(df["mx"].mean(), df["my"].mean())[1])
        self.k = 1 / np.cos(lat0)  # mercator metres per ground metre
        self.px = PIXEL_GROUND_M * self.k
        pad = 2000 * self.k
        self.x0, self.x1 = df["mx"].min() - pad, df["mx"].max() + pad
        self.y0, self.y1 = df["my"].min() - pad, df["my"].max() + pad
        self.nx = int(np.ceil((self.x1 - self.x0) / self.px))
        self.ny = int(np.ceil((self.y1 - self.y0) / self.px))
        self.x1, self.y1 = self.x0 + self.nx * self.px, self.y0 + self.ny * self.px
        self.pixel_km2 = (PIXEL_GROUND_M / 1000) ** 2

    def counts(self, sub: pd.DataFrame) -> np.ndarray:
        """Incident counts per pixel, row 0 = north (image orientation)."""
        h, _, _ = np.histogram2d(
            sub["my"], sub["mx"], bins=[self.ny, self.nx], range=[[self.y0, self.y1], [self.x0, self.x1]]
        )
        return h[::-1]

    def bounds_latlon(self) -> list[list[float]]:
        lon0, lat0 = self.to_ll.transform(self.x0, self.y0)
        lon1, lat1 = self.to_ll.transform(self.x1, self.y1)
        return [[lat0, lon0], [lat1, lon1]]


def _density_layer(grid: Grid, sub: pd.DataFrame, years: float) -> dict:
    """Gaussian KDE → RGBA PNG (data URI), legend ticks, and a hover readout grid."""
    from matplotlib import colormaps
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    counts = grid.counts(sub)
    dens = gaussian_filter(counts, sigma=KDE_SIGMA_M / PIXEL_GROUND_M) / grid.pixel_km2 / years

    vmax = float(np.percentile(dens[dens > 0], 99.9))
    vmin = vmax / 10**LOG_DECADES
    t = np.clip((np.log10(np.maximum(dens, 1e-12)) - np.log10(vmin)) / LOG_DECADES, 0, 1)
    rgba = colormaps[CMAP](t)
    # Fade the low end out so the basemap stays readable where little happens.
    rgba[..., 3] = np.where(dens < vmin, 0.0, 0.18 + 0.67 * t**0.8)
    buf = io.BytesIO()
    # 256-colour palette PNG: ~5x smaller than RGBA at this resolution, no visible banding.
    img = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").quantize(256, method=Image.Quantize.FASTOCTREE)
    img.save(buf, format="PNG", optimize=True)

    # Hover readout: raw incidents/year in READOUT_BLOCK_PX × READOUT_BLOCK_PX blocks.
    b = READOUT_BLOCK_PX
    ny, nx = counts.shape[0] // b * b, counts.shape[1] // b * b
    blocks = counts[:ny, :nx].reshape(ny // b, b, nx // b, b).sum(axis=(1, 3)) / years

    ticks = [vmin * 10 ** (LOG_DECADES * f) for f in (0, 0.25, 0.5, 0.75, 1)]
    return {
        "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "ticks": [_nice(v) for v in ticks],
        "readout": np.round(blocks, 1).tolist(),
        "n": int(len(sub)),
    }


def _hotspots(df: pd.DataFrame, n_days: int) -> list[dict]:
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    df = df.assign(
        _gx=(df["Easting_rounded"] // HOTSPOT_CELL_M).astype(int),
        _gy=(df["Northing_rounded"] // HOTSPOT_CELL_M).astype(int),
    )
    top = df.groupby(["_gx", "_gy"]).size().nlargest(N_HOTSPOTS)
    out = []
    for rank, ((x, y), n) in enumerate(top.items(), start=1):
        g = df[(df["_gx"] == x) & (df["_gy"] == y)]
        lon, lat = to_ll.transform((x + 0.5) * HOTSPOT_CELL_M, (y + 0.5) * HOTSPOT_CELL_M)
        mix = g["IncidentGroup"].value_counts(normalize=True)
        out.append(
            {
                "rank": rank,
                "lat": lat,
                "lon": lon,
                "n": int(n),
                "per_day": n / n_days,
                "ward": _mode(g["IncGeo_WardNameNew"]),
                "borough": _mode(g["ProperCase"]),
                "postcode": _mode(g["Postcode_district"]).upper(),
                "property": _mode(g["PropertyCategory"]),
                "mix": {k: float(mix.get(k, 0.0)) for k in GROUP_COLORS},
                "median_attend_s": float(g["attend_s"].median()) if g["attend_s"].notna().any() else None,
            }
        )
    return out


def _hour_svg(df: pd.DataFrame) -> str:
    """Tiny inline bar chart of incidents by hour of call."""
    h = df["HourOfCall"].value_counts().reindex(range(24), fill_value=0)
    w, ht, bw = 264, 54, 11
    peak = h.max()
    bars = []
    for hour, v in h.items():
        bh = 0 if peak == 0 else v / peak * (ht - 4)
        fill = "#bd0026" if v == peak else "#fd8d3c"
        bars.append(
            f'<rect x="{hour * bw}" y="{ht - bh:.1f}" width="{bw - 2}" height="{bh:.1f}" rx="1.5" fill="{fill}">'
            f"<title>{hour:02d}:00 – {v:,} incidents</title></rect>"
        )
    ticks = "".join(
        f'<text x="{t * bw + (bw - 2) / 2}" y="{ht + 11}" text-anchor="middle">{t:02d}</text>' for t in (0, 6, 12, 18, 23)
    )
    return (
        f'<svg viewBox="0 0 {w} {ht + 14}" width="100%" style="display:block;font-size:9px;fill:#666">'
        f"{''.join(bars)}{ticks}</svg>"
    )


def _panel_html(df: pd.DataFrame, n_days: int, hotspots: list[dict], layer_names: list[str]) -> str:
    from matplotlib import colormaps

    total = len(df)
    start_s = pd.Timestamp(df["DateOfCall"].min()).strftime("%b %Y")
    end_s = pd.Timestamp(df["DateOfCall"].max()).strftime("%b %Y")
    mix = df["IncidentGroup"].value_counts()
    mix_bar = "".join(
        f'<div title="{k}: {mix.get(k, 0):,}" style="width:{mix.get(k, 0) / total * 100:.2f}%;background:{c}"></div>'
        for k, c in GROUP_COLORS.items()
    )
    mix_rows = "".join(
        f'<div class="row"><span><i class="sw" style="background:{c}"></i>{k}</span>'
        f"<b>{mix.get(k, 0):,} <em>({mix.get(k, 0) / total:.0%})</em></b></div>"
        for k, c in GROUP_COLORS.items()
    )
    bor = (
        df.groupby("ProperCase")
        .agg(n=("ProperCase", "size"), attend=("attend_s", "median"))
        .sort_values("n", ascending=False)
        .head(8)
    )
    bmax = bor["n"].max()
    bor_rows = "".join(
        f'<tr><td>{name}</td><td class="bar"><div style="width:{r.n / bmax * 100:.0f}%"></div></td>'
        f"<td>{r.n / n_days:.0f}</td><td>{_fmt_mmss(r.attend)}</td></tr>"
        for name, r in bor.iterrows()
    )
    hot_rows = "".join(
        f'<tr class="hot" data-rank="{h["rank"]}"><td><span class="pin">{h["rank"]}</span></td>'
        f'<td>{h["ward"]}<br><em>{h["borough"]} · {h["postcode"]}</em></td><td>{h["per_day"]:.1f}</td></tr>'
        for h in hotspots[:5]
    )
    cmap = colormaps[CMAP]
    grad = ", ".join(
        f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{0.18 + 0.67 * t**0.8:.2f}) {t * 100:.0f}%"
        for t in np.linspace(0, 1, 9)
        for r, g, b, _ in [cmap(t)]
    )
    radios = "".join(
        f'<label><input type="radio" name="lfb-layer" value="{i}"{" checked" if i == 0 else ""}> {name}</label>'
        for i, name in enumerate(layer_names)
    )

    return f"""
<style>
  #lfb-panel {{
    position: fixed; top: 12px; left: 56px; z-index: 9999; width: 300px;
    max-height: calc(100vh - 24px); overflow-y: auto;
    background: rgba(255,255,255,0.97); border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.22); color: #222;
    font: 12.5px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #lfb-panel h1 {{ font-size: 16px; margin: 0 0 2px; }}
  #lfb-panel h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #666;
                   margin: 14px 0 6px; border-top: 1px solid #e6e6e6; padding-top: 10px; }}
  #lfb-panel h2 em {{ text-transform: none; letter-spacing: 0; }}
  #lfb-panel .sub {{ color: #666; font-size: 11.5px; }}
  #lfb-panel .kpis {{ display: flex; gap: 8px; margin-top: 10px; }}
  #lfb-panel .kpi {{ flex: 1; background: #f6f6f6; border-radius: 6px; padding: 6px 8px; }}
  #lfb-panel .kpi b {{ display: block; font-size: 15px; }}
  #lfb-panel .kpi span {{ font-size: 10.5px; color: #666; }}
  #lfb-panel .layers label {{ display: block; cursor: pointer; }}
  #lfb-panel .ramp {{ height: 12px; border-radius: 3px; margin-top: 8px;
     background: linear-gradient(to right, {grad}), repeating-conic-gradient(#ddd 0 25%, #fff 0 50%) 0 0 / 8px 8px; }}
  #lfb-panel .ramp-l {{ display: flex; justify-content: space-between; font-size: 10.5px; color: #444;
     font-variant-numeric: tabular-nums; margin-top: 2px; }}
  #lfb-panel .ramp-u {{ font-size: 10.5px; color: #666; text-align: center; }}
  #lfb-panel .mixbar {{ display: flex; height: 8px; border-radius: 3px; overflow: hidden; gap: 1px; margin-bottom: 6px; }}
  #lfb-panel .row {{ display: flex; justify-content: space-between; }}
  #lfb-panel em {{ font-style: normal; color: #888; font-weight: 400; font-size: 11px; }}
  #lfb-panel .sw {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }}
  #lfb-panel table {{ width: 100%; border-collapse: collapse; }}
  #lfb-panel th {{ font-weight: 600; font-size: 10.5px; color: #666; text-align: left; padding-bottom: 3px; }}
  #lfb-panel td {{ padding: 2px 0; vertical-align: middle; font-variant-numeric: tabular-nums; }}
  #lfb-panel th:last-child, #lfb-panel td:last-child,
  #lfb-panel .bor th:nth-last-child(2), #lfb-panel .bor td:nth-last-child(2) {{ text-align: right; padding-left: 6px; }}
  #lfb-panel td.bar {{ width: 70px; padding: 0 4px; }}
  #lfb-panel td.bar div {{ height: 7px; background: #f03b20; border-radius: 2px; }}
  #lfb-panel tr.hot {{ cursor: pointer; }}
  #lfb-panel tr.hot:hover {{ background: #fff3e6; }}
  #lfb-panel .pin, .lfb-pin {{ display: inline-flex; align-items: center; justify-content: center;
     width: 20px; height: 20px; border-radius: 50%; background: #222; color: #fff;
     font: 700 11px -apple-system, sans-serif; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.4); }}
  #lfb-panel .note {{ font-size: 10.5px; color: #777; margin-top: 10px; }}
  #lfb-toggle {{ float: right; border: none; background: none; font-size: 16px; cursor: pointer; color: #666; }}
  #lfb-panel.collapsed .body {{ display: none; }}
  #lfb-readout {{ position: fixed; bottom: 26px; right: 10px; z-index: 9999; background: rgba(255,255,255,.95);
     border-radius: 6px; padding: 5px 9px; box-shadow: 0 1px 6px rgba(0,0,0,.2); color: #222;
     font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; pointer-events: none; }}
  @media (max-width: 600px) {{
    #lfb-panel {{ left: 10px; right: 10px; width: auto; top: auto; bottom: 24px; max-height: 50vh; }}
    #lfb-readout {{ display: none; }}
  }}
</style>
<div id="lfb-panel">
  <button id="lfb-toggle" title="Collapse / expand">–</button>
  <h1>London fire incident density</h1>
  <div class="sub">London Fire Brigade incidents, {start_s} – {end_s}</div>
  <div class="body">
    <div class="kpis">
      <div class="kpi"><b>{total:,}</b><span>incidents</span></div>
      <div class="kpi"><b>{total / n_days:,.0f}</b><span>per day</span></div>
      <div class="kpi"><b>{_fmt_mmss(df["attend_s"].median())}</b><span>median 1st pump</span></div>
    </div>

    <h2>Show</h2>
    <div class="layers">{radios}</div>
    <div class="ramp"></div>
    <div class="ramp-l" id="lfb-ticks"></div>
    <div class="ramp-u">incidents per km² per year (log scale)</div>

    <h2>Incident mix</h2>
    <div class="mixbar">{mix_bar}</div>
    {mix_rows}

    <h2>Densest hotspots <em>({HOTSPOT_CELL_M} m squares, all types)</em></h2>
    <table><tr><th></th><th>Ward</th><th>/ day</th></tr>{hot_rows}</table>

    <h2>Busiest boroughs</h2>
    <table class="bor"><tr><th>Borough</th><th></th><th>/ day</th><th>1st pump</th></tr>{bor_rows}</table>

    <h2>Calls by hour of day</h2>
    {_hour_svg(df)}

    <div class="note">
      Density is a kernel estimate ({KDE_SIGMA_M} m Gaussian) of incident locations, which LFB
      rounds to 50 m. Colour scale is set per incident type, so compare patterns across types,
      not shades. Click a numbered pin for details. "1st pump" = median time from call to the
      first fire engine on scene.
    </div>
  </div>
</div>
<div id="lfb-readout">Hover the map for local incident counts</div>
"""


def build_map(df: pd.DataFrame, out: Path) -> tuple[Path, dict]:
    import folium
    from branca.element import Element, MacroElement, Template

    dates = pd.to_datetime(df["DateOfCall"])
    n_days = int((dates.max() - dates.min()).days) + 1
    years = n_days / 365.25

    grid = Grid(df)
    subsets = [("All incidents", df)] + [(k, df[df["IncidentGroup"] == k]) for k in GROUP_COLORS]
    layers = [dict(name=name, **_density_layer(grid, sub, years)) for name, sub in subsets]
    layer_names = [f"{l['name']} <em>({l['n']:,})</em>" for l in layers]

    m = folium.Map(location=[51.5, -0.2], zoom_start=11, tiles=None, control_scale=True, max_zoom=18)
    # max_native_zoom: Esri's Canvas (gray) tiles stop at z16 and return "Map data not yet
    # available" beyond it, so Leaflet upscales the z16 tile instead of requesting more.
    for svc, name, show, native in [
        ("Canvas/World_Light_Gray_Base", "Light gray", True, 16),
        ("World_Street_Map", "Streets", False, 18),
        ("World_Imagery", "Satellite", False, 18),
    ]:
        folium.TileLayer(
            tiles=ESRI.format(svc), attr=ESRI_ATTR, name=name, show=show, max_zoom=18, max_native_zoom=native
        ).add_to(m)
    # Place-name labels drawn above the density surface so boroughs stay legible.
    folium.map.CustomPane("labels", z_index=450).add_to(m)
    folium.TileLayer(
        tiles=ESRI.format("Canvas/World_Light_Gray_Reference"),
        attr=ESRI_ATTR, name="Place labels", overlay=True, control=True, show=True, pane="labels", max_zoom=18, max_native_zoom=16,
    ).add_to(m)

    hotspots = _hotspots(df, n_days)
    hot_group = folium.FeatureGroup(name=f"Top {N_HOTSPOTS} hotspots", show=True).add_to(m)
    for h in hotspots:
        mix = "".join(
            f'<div style="display:flex;justify-content:space-between;gap:12px">'
            f'<span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{c};margin-right:5px"></i>{k}</span>'
            f"<b>{h['mix'][k]:.0%}</b></div>"
            for k, c in GROUP_COLORS.items()
        )
        popup = (
            f'<div style="font:12px/1.45 -apple-system,sans-serif;min-width:200px">'
            f'<div style="font-weight:700;font-size:13px">#{h["rank"]} {h["ward"]}</div>'
            f'<div style="color:#666;margin-bottom:6px">{h["borough"]} · {h["postcode"]} · {HOTSPOT_CELL_M} m square</div>'
            f'<div><b>{h["n"]:,}</b> incidents (<b>{h["per_day"]:.1f}</b>/day)</div>'
            f'<div>Median 1st pump: <b>{_fmt_mmss(h["median_attend_s"])}</b></div>'
            f'<div style="margin-bottom:6px">Most common property: <b>{h["property"]}</b></div>{mix}</div>'
        )
        folium.Marker(
            [h["lat"], h["lon"]],
            icon=folium.DivIcon(
                html=f'<span class="lfb-pin">{h["rank"]}</span>', icon_size=(24, 24), icon_anchor=(12, 12)
            ),
            tooltip=f"#{h['rank']} {h['ward']} – {h['per_day']:.1f} incidents/day",
            popup=folium.Popup(popup, max_width=280),
        ).add_to(hot_group)

    m.get_root().html.add_child(Element(_panel_html(df, n_days, hotspots, layer_names)))

    payload = json.dumps(
        {
            "bounds": grid.bounds_latlon(),
            "merc": {"x0": grid.x0, "y1": grid.y1, "cell": grid.px * READOUT_BLOCK_PX},
            "cellM": PIXEL_GROUND_M * READOUT_BLOCK_PX,
            "layers": [{k: l[k] for k in ("name", "png", "ticks", "readout")} for l in layers],
            "hot": [{"rank": h["rank"], "ll": [h["lat"], h["lon"]]} for h in hotspots],
        }
    )
    map_name = m.get_name()

    class DensityJS(MacroElement):
        _template = Template(
            """
            {% macro script(this, kwargs) %}
            (function() {
              var map = """ + map_name + """, D = """ + payload + """;
              var current = 0, overlay = null;
              function show(i) {
                current = i;
                if (overlay) map.removeLayer(overlay);
                overlay = L.imageOverlay(D.layers[i].png, D.bounds, {opacity: 1, interactive: false}).addTo(map);
                overlay.bringToBack();
                if (typeof fade === 'function') fade();
                document.getElementById('lfb-ticks').innerHTML =
                  D.layers[i].ticks.map(function(t) { return '<span>' + t + '</span>'; }).join('');
              }
              show(0);
              // Fade the surface at street level so the streets underneath stay readable.
              function fade() {
                var z = map.getZoom();
                if (overlay) overlay.setOpacity(z >= 16 ? 0.5 : z >= 14 ? 0.7 : 1);
              }
              map.on('zoomend', fade);
              document.querySelectorAll('input[name=lfb-layer]').forEach(function(r) {
                r.onchange = function() { show(+r.value); };
              });

              var readout = document.getElementById('lfb-readout');
              map.on('mousemove', function(e) {
                var p = L.CRS.EPSG3857.project(e.latlng);
                var col = Math.floor((p.x - D.merc.x0) / D.merc.cell);
                var row = Math.floor((D.merc.y1 - p.y) / D.merc.cell);
                var g = D.layers[current].readout;
                var v = (g[row] || [])[col];
                readout.innerHTML = v === undefined ? 'Outside LFB area' :
                  '<b>' + (v >= 10 ? Math.round(v).toLocaleString() : v) + '</b> ' +
                  D.layers[current].name.toLowerCase().replace('all incidents', 'incidents') +
                  ' / year within this ~' + D.cellM + ' m square';
              });

              var panel = document.getElementById('lfb-panel');
              L.DomEvent.disableScrollPropagation(panel);
              L.DomEvent.disableClickPropagation(panel);
              var btn = document.getElementById('lfb-toggle');
              btn.onclick = function() {
                panel.classList.toggle('collapsed');
                btn.textContent = panel.classList.contains('collapsed') ? '+' : '–';
              };
              document.querySelectorAll('#lfb-panel tr.hot').forEach(function(tr) {
                tr.onclick = function() {
                  var h = D.hot.find(function(x) { return x.rank == tr.dataset.rank; });
                  if (h) map.flyTo(h.ll, 15);
                };
              });
            })();
            {% endmacro %}
            """
        )

    m.add_child(DensityJS())
    folium.LayerControl(collapsed=False).add_to(m)

    path = out / "london_incident_density.html"
    m.save(str(path))
    summary = {
        "map": str(path.relative_to(ROOT)),
        "incidents": int(len(df)),
        "date_range": [str(df["DateOfCall"].min()), str(df["DateOfCall"].max())],
        "incidents_per_day": round(len(df) / n_days, 1),
        "median_first_pump_s": float(df["attend_s"].median()),
        "kde_sigma_m": KDE_SIGMA_M,
        "pixel_m": PIXEL_GROUND_M,
        "legend_incidents_per_km2_yr": {l["name"]: l["ticks"] for l in layers},
        "hotspots": [
            {k: h[k] for k in ("rank", "ward", "borough", "postcode", "n", "lat", "lon")} for h in hotspots
        ],
    }
    return path, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--incidents", type=Path, default=ROOT / "data" / "raw" / "london" / "lfb_incidents_2024_onwards.csv"
    )
    p.add_argument("--out", type=Path, default=ROOT / "data" / "figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = _load(args.incidents)
    map_path, summary = build_map(df, args.out)
    (args.out / "viz_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Open map:", map_path)


if __name__ == "__main__":
    main()
