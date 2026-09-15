"""MODEL 3 — Gradient Boosted Decision Trees route scoring (LightGBM).

First draft: learn edge-level EMV travel costs from street + context features,
then run Dijkstra with predicted edge weights. This matches the slides' GBDT
routing model and reuses the same feature philosophy as the route-ETA model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd

from .graph import RouteResult, dijkstra_route, path_stats


EDGE_FEATURE_COLS = [
    "length_m",
    "speed_kph",
    "is_primary",
    "is_motorway",
    "is_residential",
    "has_bus_lane",
    "is_emv_corridor",
    "civilian_forbidden",
    "hour",
    "is_rush",
    "is_night",
    "cong_prior",
]


def _highway_flags(hw) -> tuple[int, int, int]:
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    hw = str(hw or "").lower()
    is_primary = int(hw in {"primary", "primary_link", "trunk", "trunk_link", "secondary", "secondary_link"})
    is_motorway = int(hw in {"motorway", "motorway_link"})
    is_residential = int(hw in {"residential", "living_street", "unclassified", "tertiary", "tertiary_link"})
    return is_primary, is_motorway, is_residential


def _cong_prior(hour: int) -> float:
    if hour in (7, 8, 9, 16, 17, 18, 19):
        return 1.45
    if hour >= 22 or hour < 6:
        return 1.05
    return 1.20


def edge_feature_row(data: dict, *, hour: int) -> dict[str, float]:
    length = float(data.get("length_m") or data.get("length") or 0.0)
    speed = float(data.get("speed_kph") or 30.0)
    hw = data.get("highway")
    is_primary, is_motorway, is_residential = _highway_flags(hw)
    return {
        "length_m": length,
        "speed_kph": speed,
        "is_primary": is_primary,
        "is_motorway": is_motorway,
        "is_residential": is_residential,
        "has_bus_lane": int(bool(data.get("busway") or data.get("lanes:bus"))),
        "is_emv_corridor": int(bool(data.get("emv_corridor") or data.get("emv_corridor_kind"))),
        "civilian_forbidden": int(bool(data.get("civilian_forbidden"))),
        "hour": float(hour),
        "is_rush": float(hour in (7, 8, 9, 16, 17, 18, 19)),
        "is_night": float(hour >= 22 or hour < 6),
        "cong_prior": float(_cong_prior(hour)),
    }


def build_edge_training_frame(
    G,
    *,
    hours: list[int] | None = None,
    max_edges: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """Edge labels from graph EMV seconds, with hour/congestion modulation."""
    rng = np.random.default_rng(seed)
    hours = hours or [0, 6, 8, 12, 17, 21]
    edges = list(G.edges(keys=True, data=True))
    if len(edges) > max_edges:
        idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in idx]

    rows = []
    for _, _, _, data in edges:
        base = float(data.get("emv_s") or data.get("travel_time") or 0.0)
        if base <= 0:
            continue
        # Corridor edges: EMVs are faster relative to the civilian clock
        corridor_bonus = 0.88 if data.get("emv_corridor") or data.get("civilian_forbidden") else 1.0
        for hour in hours:
            cong = _cong_prior(hour)
            y = base * cong * corridor_bonus
            feat = edge_feature_row(data, hour=hour)
            feat["travel_seconds"] = float(y)
            rows.append(feat)
    return pd.DataFrame(rows)


def train_gbdt_edge_model(
    df: pd.DataFrame,
    *,
    out_path: Path | str | None = None,
    seed: int = 42,
):
    X = df[EDGE_FEATURE_COLS]
    y = df["travel_seconds"].astype(float)
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=48,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X, y)
    bundle = {"model": model, "features": EDGE_FEATURE_COLS}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)
    return bundle


def apply_gbdt_edge_weights(G, bundle, *, hour: int = 12) -> None:
    """Write predicted EMV seconds onto each edge as weight_gbdt."""
    model = bundle["model"]
    feats = []
    keys = []
    for u, v, k, data in G.edges(keys=True, data=True):
        feats.append(edge_feature_row(data, hour=hour))
        keys.append((u, v, k))
    if not feats:
        return
    X = pd.DataFrame(feats)[EDGE_FEATURE_COLS]
    pred = model.predict(X)
    for (u, v, k), p in zip(keys, pred):
        G[u][v][k]["weight_gbdt"] = float(max(0.1, p))


def solve_gbdt_route(
    G,
    origin,
    dest,
    *,
    bundle: dict | None = None,
    hour: int = 12,
    model_path: Path | str | None = None,
) -> RouteResult:
    """Dijkstra on LightGBM-predicted edge EMV costs."""
    if bundle is None:
        if model_path and Path(model_path).exists():
            bundle = joblib.load(model_path)
        else:
            # Fit a quick on-the-fly model from this graph
            df = build_edge_training_frame(G, hours=[hour], max_edges=8000)
            bundle = train_gbdt_edge_model(df)
    apply_gbdt_edge_weights(G, bundle, hour=hour)
    result = dijkstra_route(G, origin, dest, weight="weight_gbdt", model_name="gbdt_router")
    result.meta["hour"] = hour
    result.meta["n_features"] = len(EDGE_FEATURE_COLS)
    return result
