"""Per-segment speed model for route-level travel time (scaffold).

Idea (planning doc; after Zhan et al.): learn *edge-level* speeds from only
*route-level* observations. Iteratively:

  1. Start with speeds from a prior (e.g. maxspeed * factor).
  2. Re-route every training OD with Dijkstra on current edge times.
  3. Regress observed route time on the chosen routes' edge features
     (route time = sum_i length_i / speed_i, so with slowness s_i = 1/speed_i
     = w . x_i the route time is LINEAR in w:  T = w . sum_i length_i x_i).
  4. Update edge speeds from w, go to 2 until the routes stop changing.

This module provides the building blocks: edge featurisation, a ridge-regularised
:class:`SegmentSpeedModel` (shrunk toward a maxspeed prior), a one-shot route
fit (:meth:`SegmentSpeedModel.fit_routes`, step 3 for fixed routes) and
:func:`route_time_seconds`. The outer Dijkstra loop is left as a TODO
(:func:`iterative_reroute_fit` shows the intended shape).

Features per edge: intercept, 3600/maxspeed (s/km at the limit), lanes, width (m),
road_type one-hots, hour (rush flag and sin/cos). Target: slowness in s/km.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROAD_TYPES = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "other")
MIN_SPEED_KPH, MAX_SPEED_KPH = 5.0, 120.0
DEFAULT_MAXSPEED_KPH = {
    "motorway": 112.0, "trunk": 96.0, "primary": 48.0, "secondary": 48.0,
    "tertiary": 48.0, "residential": 32.0, "other": 32.0,
}
MPH_TO_KPH = 1.609344


def _first(v: Any) -> Any:
    return v[0] if isinstance(v, (list, tuple)) and v else v


def parse_maxspeed_kph(v: Any, road_type: str = "other") -> float:
    v = _first(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return DEFAULT_MAXSPEED_KPH.get(road_type, 32.0)
    s = str(v).strip().lower()
    digits = "".join(ch for ch in s.split()[0] if ch.isdigit() or ch == ".") if s else ""
    if not digits:
        return DEFAULT_MAXSPEED_KPH.get(road_type, 32.0)
    val = float(digits)
    return val * MPH_TO_KPH if "mph" in s else val


def parse_road_type(highway: Any) -> str:
    h = str(_first(highway) or "other").lower().replace("_link", "")
    return h if h in ROAD_TYPES else "other"


def _num(v: Any, default: float) -> float:
    v = _first(v)
    try:
        f = float("".join(ch for ch in str(v).split()[0] if ch.isdigit() or ch == "."))
        return f
    except (ValueError, IndexError, TypeError):
        return default


def edge_attrs_to_record(data: Mapping[str, Any]) -> dict[str, Any]:
    """OSMnx edge-data dict -> {length_m, maxspeed_kph, lanes, width_m, road_type}."""
    rt = parse_road_type(data.get("highway"))
    return {
        "length_m": float(_first(data.get("length", 0.0)) or 0.0),
        "maxspeed_kph": parse_maxspeed_kph(data.get("maxspeed"), rt),
        "lanes": _num(data.get("lanes"), 1.0 if rt in ("residential", "other") else 2.0),
        "width_m": _num(data.get("width"), 3.5 * _num(data.get("lanes"), 1.0 if rt in ("residential", "other") else 2.0)),
        "road_type": rt,
    }


def feature_names() -> list[str]:
    return ["intercept", "limit_s_per_km", "lanes", "width_m", *[f"rt_{r}" for r in ROAD_TYPES],
            "rush", "hour_sin", "hour_cos"]


def edge_features(rec: Mapping[str, Any], hour: int) -> np.ndarray:
    rt = rec["road_type"]
    ms = max(float(rec["maxspeed_kph"]), 1.0)
    ang = 2 * np.pi * (hour % 24) / 24.0
    rush = 1.0 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0.0
    return np.array(
        [1.0, 3600.0 / ms, float(rec["lanes"]), float(rec["width_m"]),
         *[1.0 if rt == r else 0.0 for r in ROAD_TYPES], rush, np.sin(ang), np.cos(ang)]
    )


@dataclass
class SegmentSpeedModel:
    """Linear model of *slowness* (s/km) per segment, ridge-shrunk toward a prior.

    Prior: slowness = prior_factor * 3600 / maxspeed  (i.e. speed = maxspeed / prior_factor).
    """

    ridge: float = 1.0
    prior_factor: float = 1.4
    w: np.ndarray = field(default_factory=lambda: np.zeros(len(feature_names())))
    fitted: bool = False

    def __post_init__(self) -> None:
        if not self.fitted:
            self.w = self._prior_w()

    def _prior_w(self) -> np.ndarray:
        w = np.zeros(len(feature_names()))
        w[feature_names().index("limit_s_per_km")] = self.prior_factor
        return w

    def slowness_s_per_km(self, rec: Mapping[str, Any], hour: int) -> float:
        floor = 3600.0 / MAX_SPEED_KPH
        return float(np.clip(edge_features(rec, hour) @ self.w, floor, 3600.0 / MIN_SPEED_KPH))

    def predict_speed_kph(self, rec: Mapping[str, Any], hour: int) -> float:
        return 3600.0 / self.slowness_s_per_km(rec, hour)

    def fit_routes(
        self, route_features: np.ndarray, route_seconds: np.ndarray, *, weights: np.ndarray | None = None
    ) -> "SegmentSpeedModel":
        """Ridge fit toward the prior. ``route_features[r] = sum_i (length_km_i * x_i)``.

        Solves min_w ||W^{1/2}(X w - t)||^2 + ridge*||w - w0||^2 (X scaled by
        feature standard deviation inside the penalty is NOT done; keep it simple).
        """
        x = np.asarray(route_features, float)
        t = np.asarray(route_seconds, float)
        sw = np.ones(len(t)) if weights is None else np.asarray(weights, float)
        w0 = self._prior_w()
        lam = float(self.ridge) * max(len(t), 1) * 1e-3 * np.mean(np.sum(x**2, axis=1) + 1e-9) / x.shape[1]
        a = x.T @ (x * sw[:, None]) + lam * np.eye(x.shape[1])
        b = x.T @ (sw * t) + lam * w0
        self.w = np.linalg.solve(a, b)
        self.fitted = True
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"features": feature_names(), "w": self.w.tolist(), "ridge": self.ridge,
                "prior_factor": self.prior_factor, "fitted": self.fitted}


def _edge_rec(G, u: Any, v: Any) -> dict[str, Any]:
    """Pick the shortest parallel edge u->v and featurise it."""
    data = G.get_edge_data(u, v)
    if data is None:
        raise KeyError(f"No edge {u}->{v}")
    if "length" in data:  # simple Graph / DiGraph
        return edge_attrs_to_record(data)
    best = min(data.values(), key=lambda d: float(_first(d.get("length", 0.0)) or 0.0))
    return edge_attrs_to_record(best)


def route_features(G, path: Sequence[Any], hour: int) -> tuple[np.ndarray, float]:
    """(sum_i length_km_i * x_i, total_km) for a node path."""
    acc = np.zeros(len(feature_names()))
    km = 0.0
    for u, v in zip(path[:-1], path[1:]):
        rec = _edge_rec(G, u, v)
        lk = rec["length_m"] / 1000.0
        acc += lk * edge_features(rec, hour)
        km += lk
    return acc, km


def route_time_seconds(G, path: Sequence[Any], model: SegmentSpeedModel, hour: int) -> float:
    """Sum over edges of length / predicted speed (seconds)."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        rec = _edge_rec(G, u, v)
        total += (rec["length_m"] / 1000.0) / model.predict_speed_kph(rec, hour) * 3600.0
    return float(total)


def iterative_reroute_fit(
    G, od_pairs: Iterable[tuple[Any, Any, int, float]], model: SegmentSpeedModel | None = None,
    n_iter: int = 5,
) -> SegmentSpeedModel:
    """Iterative Dijkstra re-route training (Zhan et al. style).

    ``od_pairs``: (origin_node, dest_node, hour, observed_seconds). Each
    iteration: weight edges by current predicted time, route every OD with
    Dijkstra, then refit the ridge on the chosen routes. Stops early when the
    routes stop changing. Minimal, not optimised (per-hour reweighting is O(E)).
    """
    import networkx as nx

    model = model or SegmentSpeedModel()
    ods = list(od_pairs)
    prev: list[tuple] | None = None
    for _ in range(n_iter):
        paths, feats, ys = [], [], []
        by_hour: dict[int, Any] = {}
        for o, d, hour, sec in ods:
            if hour not in by_hour:
                H = G.copy()
                for u, v, k, data in H.edges(keys=True, data=True) if H.is_multigraph() else (
                    (u, v, None, data) for u, v, data in H.edges(data=True)
                ):
                    rec = edge_attrs_to_record(data)
                    data["_t"] = rec["length_m"] / 1000.0 / model.predict_speed_kph(rec, hour) * 3600.0
                by_hour[hour] = H
            try:
                p = nx.shortest_path(by_hour[hour], o, d, weight="_t")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            f, _km = route_features(G, p, hour)
            paths.append(tuple(p)); feats.append(f); ys.append(sec)
        if not paths:
            break
        model.fit_routes(np.array(feats), np.array(ys))
        if prev == paths:
            break
        prev = paths
    return model
