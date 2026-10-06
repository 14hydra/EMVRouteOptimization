"""Discrete facility location: demand grid, candidate sites, p-median / max-cover.

Historical incidents are binned into a demand grid::

    w_c      = incident weight of grid cell c
    t(s, c)  = response seconds from site s to cell c
    S        = the k sites chosen from the candidates

    objective="response_time"   minimize  E[T] = sum_c w_c * min_{s in S} t(s,c) / sum_c w_c
    objective="coverage"        maximize  C(T) = sum_c w_c * 1[min_{s in S} t(s,c) <= T] / sum_c w_c

``response_time`` is the p-median problem, ``coverage`` is maximal covering
location (Church & ReVelle, 1974); both are NP-hard, so ``select_posts`` uses a
greedy seed plus a bounded k-medoids style swap pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle distance in km (broadcasts like numpy)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_R_KM * 2.0 * np.arcsin(np.sqrt(a))


@dataclass
class TravelTimeSurrogate:
    """Crow-flies -> drive seconds: ``fixed + km * detour / speed``.

    ``detour_factor`` is the classic street-network circuity correction (~1.3 in
    dense grids); ``speed_kph`` is an *effective* door-to-scene speed including
    intersections, so it is well below posted limits.
    """

    speed_kph: float = 28.0
    detour_factor: float = 1.30
    fixed_seconds: float = 15.0  # acceleration + first-intersection overhead
    meta: dict[str, Any] = field(default_factory=dict)

    def seconds(self, lon1, lat1, lon2, lat2):
        km = haversine_km(lon1, lat1, lon2, lat2) * self.detour_factor
        return self.fixed_seconds + km / max(self.speed_kph, 1.0) * 3600.0

    def matrix(self, from_df: pd.DataFrame, to_df: pd.DataFrame) -> np.ndarray:
        """Seconds matrix [len(from_df), len(to_df)] using lon/lat columns."""
        flon = from_df["lon"].to_numpy(dtype=float)[:, None]
        flat = from_df["lat"].to_numpy(dtype=float)[:, None]
        tlon = to_df["lon"].to_numpy(dtype=float)[None, :]
        tlat = to_df["lat"].to_numpy(dtype=float)[None, :]
        return np.asarray(self.seconds(flon, flat, tlon, tlat), dtype=float)


def build_demand_grid(
    incidents: pd.DataFrame,
    *,
    lat_col: str = "dest_lat",
    lon_col: str = "dest_lon",
    cell_km: float = 0.75,
    weight_col: str | None = None,
    severity_col: str | None = None,
    min_incidents: int = 1,
) -> pd.DataFrame:
    """Bin historical incidents into an equal-area-ish demand grid.

    Returns one row per non-empty cell with the demand-weighted centroid (the
    point we actually route to) and the cell's share of citywide demand.

    ``severity_col`` (e.g. ``initial_severity_level_code``, 1 = most acute)
    up-weights acute calls: weight = 1 + (max_level - level) / max_level.
    """
    df = incidents[[c for c in {lat_col, lon_col, weight_col, severity_col} if c]].copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])
    if not len(df):
        raise ValueError("No incidents with usable coordinates for the demand grid")

    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        w = pd.Series(1.0, index=df.index)
    if severity_col and severity_col in df.columns:
        sev = pd.to_numeric(df[severity_col], errors="coerce")
        top = float(sev.max()) if sev.notna().any() else np.nan
        if top == top and top > 0:
            w = w * (1.0 + (top - sev.fillna(top)) / top)
    df["_w"] = w

    lat0 = float(df[lat_col].mean())
    dlat = cell_km / 110.574
    dlon = cell_km / (111.320 * max(math.cos(math.radians(lat0)), 0.1))
    df["_iy"] = np.floor(df[lat_col] / dlat).astype(int)
    df["_ix"] = np.floor(df[lon_col] / dlon).astype(int)

    # Demand-weighted centroid per cell = Σ(w·coord) / Σw (no groupby.apply).
    df["_wpos"] = df["_w"].clip(lower=1e-9)
    df["_wlat"] = df[lat_col] * df["_wpos"]
    df["_wlon"] = df[lon_col] * df["_wpos"]
    g = df.groupby(["_ix", "_iy"], sort=False)
    agg = g.agg(
        n_incidents=("_w", "size"),
        weight=("_w", "sum"),
        _wsum=("_wpos", "sum"),
        _wlat=("_wlat", "sum"),
        _wlon=("_wlon", "sum"),
    )
    agg["lat"] = agg["_wlat"] / agg["_wsum"]
    agg["lon"] = agg["_wlon"] / agg["_wsum"]
    cells = agg.drop(columns=["_wsum", "_wlat", "_wlon"]).reset_index()
    cells = cells[cells["n_incidents"] >= max(1, int(min_incidents))]
    cells = cells.sort_values("weight", ascending=False).reset_index(drop=True)
    cells.insert(0, "cell_id", np.arange(len(cells)))
    cells["share"] = cells["weight"] / cells["weight"].sum()
    cells["cell_km"] = cell_km
    cells["cell_dlat"] = dlat
    cells["cell_dlon"] = dlon
    return cells


def add_demand_cell_candidates(
    candidates: pd.DataFrame, cells: pd.DataFrame, *, top_n: int = 60
) -> pd.DataFrame:
    """Add the highest-demand cell centroids as extra on-road post candidates.

    Without these the optimizer can only pick from existing facilities, which
    biases posts toward legacy station real estate rather than where calls are.
    """
    if top_n <= 0 or not len(cells):
        return candidates
    top = cells.head(int(top_n))
    extra = pd.DataFrame(
        {
            "post_name": ["Demand cell " + str(int(c)) for c in top["cell_id"]],
            "layer": "demand_cell",
            "borough": None,
            "lon": top["lon"].to_numpy(dtype=float),
            "lat": top["lat"].to_numpy(dtype=float),
            "is_anchor": False,
        }
    )
    out = pd.concat([candidates.drop(columns=["candidate_id"]), extra], ignore_index=True)
    out.insert(0, "candidate_id", np.arange(len(out)))
    return out


@dataclass
class SelectionResult:
    objective: str
    indices: list[int]
    value: float
    assignment: np.ndarray  # cell -> position in `indices`
    response_s: np.ndarray  # cell -> travel seconds from its nearest post
    n_swaps: int = 0
    greedy_value: float = float("nan")


def objective_value(
    T: np.ndarray,
    indices: Sequence[int],
    weights: np.ndarray,
    *,
    objective: str,
    threshold_s: float,
) -> float:
    """Lower is better for both objectives (coverage is negated demand covered)."""
    best = T[list(indices), :].min(axis=0)
    total = max(weights.sum(), 1e-9)
    if objective == "coverage":
        return -float((weights * (best <= threshold_s)).sum() / total)
    return float((weights * best).sum() / total)


def select_posts(
    T: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    objective: str = "response_time",
    threshold_s: float = 480.0,
    must_include: Sequence[int] = (),
    allowed: Sequence[int] | None = None,
    max_swap_rounds: int = 8,
) -> SelectionResult:
    """Greedy seed + bounded swap improvement over candidate rows of ``T``.

    ``T`` is [n_candidates, n_cells] travel seconds, ``weights`` is demand per
    cell. ``allowed`` restricts selection to a candidate subset (used to pick
    home-base anchors from facility layers only).
    """
    if objective not in {"response_time", "coverage"}:
        raise ValueError(f"Unknown objective={objective!r}")
    n_cand = T.shape[0]
    k = int(min(max(1, k), n_cand))
    pool = list(range(n_cand)) if allowed is None else [int(i) for i in allowed]
    selected = [int(i) for i in must_include if int(i) in set(pool)]

    # ---- greedy seed -----------------------------------------------------
    if selected:
        best = T[selected, :].min(axis=0)
    else:
        best = np.full(T.shape[1], np.inf)
    while len(selected) < k:
        remaining = [j for j in pool if j not in selected]
        if not remaining:
            break
        cand = np.asarray(remaining, dtype=int)
        if objective == "coverage":
            covered = best <= threshold_s
            gain = ((T[cand, :] <= threshold_s) & ~covered[None, :]) @ weights
            pick = int(cand[int(np.argmax(gain))])
        else:
            cost = np.minimum(best[None, :], T[cand, :]) @ weights
            pick = int(cand[int(np.argmin(cost))])
        selected.append(pick)
        best = np.minimum(best, T[pick, :])

    greedy_value = objective_value(
        T, selected, weights, objective=objective, threshold_s=threshold_s
    )

    # ---- k-medoids style swaps ------------------------------------------
    locked = set(int(i) for i in must_include)
    current = objective_value(
        T, selected, weights, objective=objective, threshold_s=threshold_s
    )
    n_swaps = 0
    for _ in range(int(max_swap_rounds)):
        improved = False
        for pos, out_idx in enumerate(list(selected)):
            if out_idx in locked:
                continue
            base = [s for i, s in enumerate(selected) if i != pos]
            best_gain = 0.0
            best_in = None
            for in_idx in pool:
                if in_idx in selected:
                    continue
                val = objective_value(
                    T, base + [in_idx], weights, objective=objective, threshold_s=threshold_s
                )
                if val < current - 1e-9 and (current - val) > best_gain:
                    best_gain = current - val
                    best_in = in_idx
            if best_in is not None:
                selected[pos] = best_in
                current -= best_gain
                n_swaps += 1
                improved = True
        if not improved:
            break

    sub = T[selected, :]
    assignment = sub.argmin(axis=0)
    response_s = sub.min(axis=0)
    return SelectionResult(
        objective=objective,
        indices=[int(i) for i in selected],
        value=float(current),
        assignment=assignment,
        response_s=response_s,
        n_swaps=n_swaps,
        greedy_value=float(greedy_value),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cw[-1], cw, v))


def posture_metrics(
    response_s: np.ndarray,
    weights: np.ndarray,
    *,
    threshold_s: float,
    label: str = "",
    n_posts: int | None = None,
) -> dict[str, Any]:
    """Demand-weighted response-time / coverage summary for one posture."""
    total = max(float(np.sum(weights)), 1e-9)
    finite = np.isfinite(response_s)
    return {
        "label": label,
        "n_posts": n_posts,
        "expected_response_s": round(float((weights[finite] * response_s[finite]).sum() / total), 1),
        "median_response_s": round(_weighted_quantile(response_s[finite], weights[finite], 0.5), 1),
        "p90_response_s": round(_weighted_quantile(response_s[finite], weights[finite], 0.9), 1),
        "worst_cell_response_s": round(float(np.max(response_s[finite])), 1) if finite.any() else None,
        "coverage_share_within_threshold": round(
            float((weights[finite] * (response_s[finite] <= threshold_s)).sum() / total), 4
        ),
        "threshold_s": float(threshold_s),
        "demand_cells": int(len(response_s)),
    }
