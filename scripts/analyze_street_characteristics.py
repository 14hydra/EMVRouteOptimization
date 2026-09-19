#!/usr/bin/env python3
"""
Impact of street characteristics on real EMS travel time.

Pipeline
  1. Download NYC Centerline (width, lanes, speed) + DOT bus lanes (cached).
  2. Match them onto the OSM drive graph edges (needs ``--rich-tags`` graph).
  3. Route every inferred start → destination pair and aggregate street
     characteristics (length-weighted) along the route.
  4. Regress log(EMS travel seconds) on standardized characteristics, and plot a
     correlation heatmap + coefficient plot + partial-residual panels.

Outputs land in data/figures/street_characteristics/.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.street_analysis import (  # noqa: E402
    BUS_LANES_ID,
    CENTERLINE_ID,
    CENTERLINE_SELECT,
    STREET_FEATURES,
    build_edge_table,
    fetch_geo_dataset,
    prepare_graph,
    route_table_for_od,
)

CONTROLS = ["log_crow_km", "is_rush", "is_night", "is_weekend", "wx_is_precip", "severity"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_edge_table(args) -> tuple[object, pd.DataFrame]:
    G = prepare_graph(args.graph)
    cache = args.processed / "street_edge_table.pkl"
    if cache.exists() and not args.refresh:
        return G, pickle.loads(cache.read_bytes())

    centerline = fetch_geo_dataset(
        CENTERLINE_ID, args.raw / "centerline.geojson", select=CENTERLINE_SELECT
    )
    bus = fetch_geo_dataset(BUS_LANES_ID, args.raw / "bus_lanes_local.geojson")
    table = build_edge_table(G, centerline, bus)
    args.processed.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(table))
    return G, table


def build_dataset(args, G, edge_table) -> pd.DataFrame:
    df = pd.read_csv(args.training, low_memory=False)
    df = df[(df["valid_travel"] == 1) & df["travel_seconds"].between(60, 3600)].copy()

    coord_cols = ["start_lat", "start_lon", "dest_lat", "dest_lon"]
    df["od_id"] = df[coord_cols].round(4).astype(str).agg("|".join, axis=1)
    od = df.drop_duplicates("od_id")[["od_id", *coord_cols, "crow_flies_km"]].rename(
        columns={"crow_flies_km": "crow_km"}
    )
    routes = route_table_for_od(G, edge_table, od)
    df = df.merge(routes, on="od_id", how="left")

    df["log_travel_s"] = np.log(df["travel_seconds"])
    df["log_crow_km"] = np.log(df["crow_flies_km"].clip(lower=0.05))
    df["severity"] = df["severity"].fillna(df["severity"].median())
    df["wx_is_precip"] = df["wx_is_precip"].fillna(0)
    return df


def filter_routes(df: pd.DataFrame, *, min_crow_km: float, min_lion_share: float) -> pd.DataFrame:
    feats = list(STREET_FEATURES)
    keep = (
        df["path_km"].notna()
        & (df["crow_flies_km"] >= min_crow_km)
        & (df["lion_match_share"] >= min_lion_share)
        & df[feats].notna().all(axis=1)
    )
    return df[keep].copy()


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def standardize(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score features on the *route* distribution (each route counted once)."""
    routes = df.drop_duplicates("od_id")
    mu, sd = routes[feats].mean(), routes[feats].std(ddof=0).replace(0, np.nan)
    z = (df[feats] - mu) / sd
    z.columns = [f"z_{c}" for c in feats]
    scale = pd.DataFrame({"mean": mu, "sd": sd})
    return z, scale


def fit_ols(df: pd.DataFrame, z: pd.DataFrame, *, borough_fe: bool):
    import statsmodels.formula.api as smf

    data = pd.concat([df.reset_index(drop=True), z.reset_index(drop=True)], axis=1)
    terms = list(z.columns) + CONTROLS
    if data["primary_origin_layer"].nunique() > 1:
        terms.append("C(primary_origin_layer)")
    if borough_fe:
        terms.append("C(borough)")
    model = smf.ols("log_travel_s ~ " + " + ".join(terms), data=data)
    groups = pd.factorize(data["od_id"])[0]
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups}), data


def coef_table(res, feats: list[str], spec: str) -> pd.DataFrame:
    rows = []
    ci = res.conf_int()
    for f in feats:
        k = f"z_{f}"
        b = float(res.params[k])
        lo, hi = float(ci.loc[k, 0]), float(ci.loc[k, 1])
        rows.append(
            {
                "spec": spec,
                "feature": f,
                "label": STREET_FEATURES[f][0],
                "group": STREET_FEATURES[f][1],
                "coef_log": b,
                "pct_change": 100 * (np.exp(b) - 1),
                "pct_lo": 100 * (np.exp(lo) - 1),
                "pct_hi": 100 * (np.exp(hi) - 1),
                "p_value": float(res.pvalues[k]),
                "significant": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)


def vif_table(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    routes = df.drop_duplicates("od_id")
    X = routes[feats + ["log_crow_km"]].copy()
    X = (X - X.mean()) / X.std(ddof=0)
    X.insert(0, "const", 1.0)
    vals = [variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])]
    return pd.DataFrame({"feature": feats + ["log_crow_km"], "vif": vals})


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "nyc_drive_rich.graphml")
    p.add_argument("--training", type=Path, default=ROOT / "data" / "processed" / "travel_time_training.csv")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "figures" / "street_characteristics")
    p.add_argument("--min-crow-km", type=float, default=0.25)
    p.add_argument("--min-lion-share", type=float, default=0.5)
    p.add_argument("--refresh", action="store_true", help="Rebuild the cached OSM↔Centerline edge table")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    if not args.graph.exists():
        raise SystemExit(
            f"Missing {args.graph}; run: PYTHONPATH=src python scripts/build_osm_graph.py "
            f"--rich-tags --out {args.graph}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    G, edge_table = load_edge_table(args)
    print(
        "edge table:",
        len(edge_table),
        "edges | LION matched %.1f%% | DOT bus %.2f%% | OSM bus %.2f%% | any bus %.2f%%"
        % (
            100 * edge_table["lion_matched"].mean(),
            100 * edge_table["bus_dot"].mean(),
            100 * edge_table["bus_osm"].mean(),
            100 * edge_table["bus_lane"].mean(),
        ),
    )

    raw = build_dataset(args, G, edge_table)
    raw.to_csv(args.processed / "street_route_dataset.csv", index=False)
    df = filter_routes(raw, min_crow_km=args.min_crow_km, min_lion_share=args.min_lion_share)
    feats = list(STREET_FEATURES)
    print(
        f"incidents: {len(raw)} -> {len(df)} after route filters | "
        f"unique routes {raw['od_id'].nunique()} -> {df['od_id'].nunique()}"
    )

    z, scale = standardize(df, feats)
    res_all, data_all = fit_ols(df, z, borough_fe=False)
    res_fe, _ = fit_ols(df, z, borough_fe=True)

    coefs = pd.concat(
        [
            coef_table(res_all, feats, "All routes"),
            coef_table(res_fe, feats, "All routes + borough FE"),
        ],
        ignore_index=True,
    )
    coefs.to_csv(args.out / "regression_coefficients.csv", index=False)
    vif = vif_table(df, feats)
    vif.to_csv(args.out / "vif.csv", index=False)
    scale.rename_axis("feature").reset_index().to_csv(args.out / "feature_scale.csv", index=False)

    routes = df.drop_duplicates("od_id")
    summary = {
        "target": "log(EMS incident travel seconds); coefficients reported as % change per +1 SD",
        "incidents": int(len(df)),
        "unique_routes": int(routes["od_id"].nunique()),
        "cluster_unit": "route (start→destination pair)",
        "filters": {"min_crow_km": args.min_crow_km, "min_lion_match_share": args.min_lion_share},
        "r2_all": float(res_all.rsquared),
        "r2_borough_fe": float(res_fe.rsquared),
        "median_route_lion_match": float(routes["lion_match_share"].median()),
        "max_vif": float(vif["vif"].max()),
        "edge_coverage_note": "graph bbox includes NJ/Westchester, which Centerline (NYC only) cannot match",
        "edge_coverage": {
            "lion_matched": float(edge_table["lion_matched"].mean()),
            "bus_dot": float(edge_table["bus_dot"].mean()),
            "bus_osm": float(edge_table["bus_osm"].mean()),
            "bus_any": float(edge_table["bus_lane"].mean()),
        },
        "routes_with_bus_lane": int((routes["bus_lane_share"] > 0).sum()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(json.dumps(summary, indent=2))
        print(coefs[coefs.spec == "All routes"][["label", "pct_change", "pct_lo", "pct_hi", "p_value"]])
        print(vif.sort_values("vif", ascending=False).head(8))

    if not args.no_plots:
        from street_plots import make_all  # noqa: WPS433

        make_all(df, data_all, res_all, coefs, scale, args.out, summary)
        print("figures ->", args.out)


if __name__ == "__main__":
    main()
