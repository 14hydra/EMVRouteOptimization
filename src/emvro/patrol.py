"""Optimal EMV patrol posts + patrol loops (pre-positioning, not dispatch).

This module is the demand-side companion to the inferred starting-location work
(``depots.py`` / ``csl.py`` / ``docs/starting_locations.md``). Those modules answer
*"where did a unit probably start?"*; this one answers *"where should units wait or
circulate so the next incident is closer?"*.

Objective
---------
Historical EMS incidents are binned into a spatial demand grid::

    w_c      = historical incident weight of grid cell c (count, optionally
               severity-weighted)
    t(p, c)  = EMV travel seconds from patrol post p to cell c
    P        = the k posts we choose out of the candidate locations
               (inferred FDNY firehouses / synthetic CSLs /
                high-demand cell anchors)

    objective="response_time"   minimize  E[T] = Σ_c w_c · min_{p∈P} t(p,c) / Σ_c w_c
    objective="coverage"        maximize  C(T) = Σ_c w_c · 1[min_{p∈P} t(p,c) ≤ T] / Σ_c w_c

``response_time`` is the p-median problem, ``coverage`` is maximal covering
location; both are NP-hard, so we use a greedy seed plus a bounded k-medoids
style swap pass. That runs in seconds on a laptop for a few hundred candidates
and a few thousand demand cells, which is the right trade for an MVP.

Only *travel* time is modelled. Call-processing and chute (turnout) time are
excluded, so numbers are "wheels-rolling to on-scene", not NFPA total response.

Travel time is computed in two stages so the model still works without OSM:

1. **screening** — calibrated crow-flies surrogate (``TravelTimeSurrogate``),
   used for the candidate × cell matrix and the swap search.
2. **scoring / geometry** — OSM graph Dijkstra on the EMV edge costs from
   ``emvro.routing`` for the *selected* posts only. Optional: when the graph or
   osmnx is unavailable everything degrades to the surrogate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .depots import prepare_depots

EARTH_R_KM = 6371.0

# Candidate layers that can host a fleet home base / patrol anchor (a unit can
# actually be relieved, restocked and crew-changed there). Synthetic CSLs and
# demand-cell anchors are on-road posts only.
ANCHOR_LAYERS = ("fdny_firehouse",)

TimeFn = Callable[[float, float, float, float], float]


def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle distance in km (broadcasts like numpy)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_R_KM * 2.0 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Travel-time surrogate
# --------------------------------------------------------------------------- #


@dataclass
class TravelTimeSurrogate:
    """Crow-flies → EMV seconds: ``fixed + km · detour / speed``.

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

    @classmethod
    def calibrate(
        cls,
        od: pd.DataFrame | None,
        *,
        detour_factor: float = 1.30,
        default_speed_kph: float = 28.0,
        km_col: str = "crow_flies_km",
        seconds_col: str = "travel_seconds",
        min_km: float = 0.3,
        seconds_range: tuple[float, float] = (60.0, 1800.0),
        accept_kph: tuple[float, float] = (15.0, 45.0),
        enabled: bool = False,
    ) -> "TravelTimeSurrogate":
        """Effective EMV speed: prior by default, optionally fit from OD medians.

        Calibration is **opt-in** because the OD starts are inferred, not CAD unit
        GPS (``docs/starting_locations.md``): the nearest-depot/CSL rule puts starts
        a median ~0.3 km from a ZIP-centroid destination while observed travel time
        is ~8 min, so the implied speed (~6 kph) is an artifact of the inference,
        not a driving speed. When the fitted median lands outside ``accept_kph`` we
        keep the prior and record why.
        """
        meta: dict[str, Any] = {
            "speed_source": "prior",
            "detour_factor": detour_factor,
            "prior_speed_kph": default_speed_kph,
            "n_calibration_rows": 0,
        }
        speed = default_speed_kph
        if enabled and od is not None and km_col in od.columns and seconds_col in od.columns:
            km = pd.to_numeric(od[km_col], errors="coerce")
            sec = pd.to_numeric(od[seconds_col], errors="coerce")
            ok = km.notna() & sec.notna() & (km >= min_km)
            ok &= sec.between(*seconds_range)
            if int(ok.sum()) >= 50:
                med = float(np.median((km[ok] * detour_factor) / (sec[ok] / 3600.0)))
                meta["n_calibration_rows"] = int(ok.sum())
                meta["fitted_median_kph"] = round(med, 2)
                if med == med and accept_kph[0] <= med <= accept_kph[1]:
                    speed = float(med)
                    meta["speed_source"] = "median_observed_od"
                else:
                    meta["speed_source"] = "prior_fit_out_of_band"
                    meta["accept_kph"] = list(accept_kph)
        meta["speed_kph"] = round(float(speed), 2)
        return cls(speed_kph=speed, detour_factor=detour_factor, meta=meta)


# --------------------------------------------------------------------------- #
# Demand
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Candidate posts
# --------------------------------------------------------------------------- #


def build_candidate_posts(
    layers: Iterable[tuple[pd.DataFrame | None, str]],
    *,
    anchor_layers: Sequence[str] = ANCHOR_LAYERS,
) -> pd.DataFrame:
    """Normalize inferred depot / CSL tables into a candidate-post table.

    Reuses ``depots.prepare_depots`` so the same layer labels and names appear in
    OD pairs and in patrol output (auditable end to end).
    """
    frames = []
    for df, label in layers:
        if df is None or not len(df):
            continue
        norm = prepare_depots(df, source_label=label)
        frames.append(
            pd.DataFrame(
                {
                    "post_name": norm["depot_name"].fillna(label),
                    "layer": norm["depot_layer"],
                    "borough": norm["borough"],
                    "lon": norm["start_lon"],
                    "lat": norm["start_lat"],
                }
            )
        )
    if not frames:
        raise ValueError("No candidate layers provided")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["layer", "lon", "lat"])
    out["is_anchor"] = out["layer"].isin(list(anchor_layers))
    out = out.reset_index(drop=True)
    out.insert(0, "candidate_id", np.arange(len(out)))
    return out


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


def screen_candidates(
    candidates: pd.DataFrame,
    T: np.ndarray,
    weights: np.ndarray,
    *,
    max_candidates: int = 400,
    keep_anchors: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep the most useful candidates (individually lowest weighted mean time).

    Pure speed guard for the swap pass. Anchors are kept regardless so every
    borough still has a home base available.
    """
    if len(candidates) <= max_candidates:
        return candidates.reset_index(drop=True), T
    score = (T * weights[None, :]).sum(axis=1) / max(weights.sum(), 1e-9)
    order = np.argsort(score)
    keep = list(order[:max_candidates])
    if keep_anchors:
        anchors = np.flatnonzero(candidates["is_anchor"].to_numpy())
        keep = sorted(set(keep) | set(int(a) for a in anchors))
    keep = np.asarray(sorted(keep), dtype=int)
    return candidates.iloc[keep].reset_index(drop=True), T[keep, :]


# --------------------------------------------------------------------------- #
# Post selection (p-median / max-coverage)
# --------------------------------------------------------------------------- #


@dataclass
class SelectionResult:
    objective: str
    indices: list[int]
    value: float
    assignment: np.ndarray  # cell -> position in `indices`
    response_s: np.ndarray  # cell -> travel seconds from its nearest post
    n_swaps: int = 0
    greedy_value: float = float("nan")


def _objective_value(
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

    greedy_value = _objective_value(
        T, selected, weights, objective=objective, threshold_s=threshold_s
    )

    # ---- k-medoids style swaps ------------------------------------------
    locked = set(int(i) for i in must_include)
    current = _objective_value(
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
                val = _objective_value(
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


# --------------------------------------------------------------------------- #
# Units: home-base anchors, balanced post assignment, patrol loops
# --------------------------------------------------------------------------- #


@dataclass
class UnitPlan:
    unit_id: int
    anchor_candidate_id: int
    anchor_name: str
    anchor_layer: str
    anchor_lon: float
    anchor_lat: float
    post_order: list[int]  # post ids (or candidate ids) in loop order, anchor excluded
    demand_share: float = 0.0
    loop_seconds: float = float("nan")
    loop_km: float = float("nan")


def select_unit_anchors(
    candidates: pd.DataFrame,
    posts: pd.DataFrame,
    post_weights: np.ndarray,
    n_units: int,
    surrogate: TravelTimeSurrogate,
) -> list[int]:
    """Pick ``n_units`` home bases (facility layers only) close to the posts.

    Same p-median machinery as post selection, but demand = the patrol posts'
    assigned demand. Anchors are where a unit starts/ends its tour, which is
    exactly the "inferred start" the OD pipeline assumes.
    """
    anchor_pool = np.flatnonzero(candidates["is_anchor"].to_numpy())
    if not len(anchor_pool):
        anchor_pool = np.arange(len(candidates))
    A = surrogate.matrix(candidates, posts)
    res = select_posts(
        A,
        np.asarray(post_weights, dtype=float),
        n_units,
        objective="response_time",
        allowed=[int(i) for i in anchor_pool],
        max_swap_rounds=4,
    )
    return res.indices


def _order_loop(D: np.ndarray) -> list[int]:
    """Nearest-neighbour tour from node 0 (the anchor) + 2-opt, closed loop."""
    n = D.shape[0]
    if n <= 2:
        return list(range(n))
    order = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        order.append(nxt)
        unvisited.remove(nxt)

    def tour_cost(seq: list[int]) -> float:
        return float(sum(D[seq[i], seq[(i + 1) % len(seq)]] for i in range(len(seq))))

    best = tour_cost(order)
    for _ in range(60):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                c = tour_cost(cand)
                if c < best - 1e-9:
                    order, best = cand, c
                    improved = True
        if not improved:
            break
    return order


def build_unit_plans(
    candidates: pd.DataFrame,
    posts: pd.DataFrame,
    anchor_rows: Sequence[int],
    surrogate: TravelTimeSurrogate,
    *,
    balance: float = 1.35,
) -> list[UnitPlan]:
    """Assign posts to units (demand-balanced) and order each unit's patrol loop.

    Each unit gets a closed loop anchor → posts → anchor. A "post" is where the
    unit parks and waits; the loop is how it rotates between posts so coverage
    does not collapse onto one corner while the crew is out of position.
    """
    anchors = candidates.iloc[list(anchor_rows)].reset_index(drop=True)
    n_units = len(anchors)
    if not n_units:
        raise ValueError("No anchors selected")
    id_col = "post_id" if "post_id" in posts.columns else "candidate_id"

    W = posts["assigned_weight"].to_numpy(dtype=float)
    A = surrogate.matrix(anchors, posts)  # [n_units, n_posts]
    cap = (W.sum() / n_units) * float(balance) if W.sum() > 0 else np.inf
    load = np.zeros(n_units)
    members: list[list[int]] = [[] for _ in range(n_units)]

    for p in np.argsort(-W):  # heaviest demand first
        order = np.argsort(A[:, p])
        placed = False
        for u in order:
            if load[u] + W[p] <= cap or not members[u]:
                members[u].append(int(p))
                load[u] += W[p]
                placed = True
                break
        if not placed:
            u = int(order[0])
            members[u].append(int(p))
            load[u] += W[p]

    # Never ship an empty unit: pull the closest post off the largest unit.
    for u in range(n_units):
        if members[u]:
            continue
        donor = int(np.argmax([len(m) for m in members]))
        if len(members[donor]) <= 1:
            continue
        take = min(members[donor], key=lambda p: A[u, p])
        members[donor].remove(take)
        members[u].append(take)

    plans: list[UnitPlan] = []
    total_w = max(W.sum(), 1e-9)
    for u in range(n_units):
        idxs = members[u]
        a = anchors.iloc[u]
        pts = pd.concat(
            [
                pd.DataFrame({"lon": [float(a["lon"])], "lat": [float(a["lat"])]}),
                posts.iloc[idxs][["lon", "lat"]].reset_index(drop=True),
            ],
            ignore_index=True,
        )
        D = surrogate.matrix(pts, pts)
        order = _order_loop(D)
        seq = [idxs[i - 1] for i in order if i != 0]
        plans.append(
            UnitPlan(
                unit_id=u + 1,
                anchor_candidate_id=int(a["candidate_id"]),
                anchor_name=str(a["post_name"]),
                anchor_layer=str(a["layer"]),
                anchor_lon=float(a["lon"]),
                anchor_lat=float(a["lat"]),
                post_order=[int(posts.iloc[i][id_col]) for i in seq],
                demand_share=float(W[idxs].sum() / total_w),
            )
        )
    return plans


# --------------------------------------------------------------------------- #
# Optional OSM graph refinement
# --------------------------------------------------------------------------- #


def graph_nearest_nodes(G, lons: Sequence[float], lats: Sequence[float]) -> list[Any]:
    """Snap lon/lat pairs to graph nodes (vectorized when osmnx allows)."""
    from .street_features import _try_import_ox

    ox = _try_import_ox()
    lons = [float(x) for x in lons]
    lats = [float(y) for y in lats]
    try:
        return list(ox.nearest_nodes(G, lons, lats))
    except Exception:  # noqa: BLE001 - fall back to one call per point
        return [ox.nearest_nodes(G, lo, la) for lo, la in zip(lons, lats)]


def graph_time_matrix(
    G,
    source_nodes: Sequence[Any],
    target_nodes: Sequence[Any],
    *,
    weight: str = "weight_emv",
    cutoff_s: float | None = None,
    fallback: np.ndarray | None = None,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact EMV travel seconds source × target via cutoff-limited Dijkstra.

    Returns ``(matrix, resolved_mask)``. Entries the graph could not resolve
    (unreachable, or beyond ``cutoff_s``) keep their ``fallback`` (surrogate)
    value so downstream math never sees ``inf``; ``resolved_mask`` records which
    entries are true graph times.
    """
    import networkx as nx

    shape = (len(source_nodes), len(target_nodes))
    out = np.array(fallback, dtype=float) if fallback is not None else np.full(shape, np.inf)
    resolved = np.zeros(shape, dtype=bool)
    for i, src in enumerate(source_nodes):
        try:
            dist = nx.single_source_dijkstra_path_length(G, src, cutoff=cutoff_s, weight=weight)
        except Exception:  # noqa: BLE001
            continue
        for j, tgt in enumerate(target_nodes):
            d = dist.get(tgt)
            if d is not None:
                out[i, j] = float(d)
                resolved[i, j] = True
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    … {i + 1}/{len(source_nodes)} sources")
    return out, resolved


def graph_segment(G, node_a: Any, node_b: Any, *, weight: str = "weight_emv") -> dict[str, Any]:
    """Shortest EMV path between two graph nodes as latlons + seconds + metres."""
    from .routing.graph import dijkstra_route

    r = dijkstra_route(G, node_a, node_b, weight=weight, model_name="patrol_leg")
    if not r.ok:
        return {"ok": False, "latlons": [], "seconds": float("nan"), "meters": float("nan")}
    latlons = []
    for n in r.node_path:
        d = G.nodes[n]
        latlons.append([float(d.get("y", d.get("lat"))), float(d.get("x", d.get("lon")))])
    return {
        "ok": True,
        "latlons": latlons,
        "seconds": float(r.travel_seconds),
        "meters": float(r.distance_m),
        "n_edges": int(r.n_edges),
    }


# --------------------------------------------------------------------------- #
# Synthetic demo (keeps the MVP runnable with no NYC data at all)
# --------------------------------------------------------------------------- #


def synthetic_demo_data(
    *,
    n_incidents: int = 4000,
    center: tuple[float, float] = (-73.95, 40.75),
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fake incidents + stations + CSLs so the pipeline and map still build.

    Three demand hotspots plus a diffuse background, which is enough to show the
    optimizer moving posts off the stations and toward demand.
    """
    rng = np.random.default_rng(seed)
    lon0, lat0 = center
    hotspots = [
        (lon0 + 0.00, lat0 + 0.00, 0.020, 0.45),
        (lon0 - 0.06, lat0 - 0.04, 0.015, 0.30),
        (lon0 + 0.07, lat0 + 0.05, 0.018, 0.25),
    ]
    rows = []
    for hx, hy, sd, share in hotspots:
        n = int(n_incidents * share * 0.8)
        rows.append(
            pd.DataFrame(
                {
                    "dest_lon": rng.normal(hx, sd, n),
                    "dest_lat": rng.normal(hy, sd * 0.8, n),
                }
            )
        )
    n_bg = n_incidents - sum(len(r) for r in rows)
    rows.append(
        pd.DataFrame(
            {
                "dest_lon": rng.uniform(lon0 - 0.12, lon0 + 0.12, n_bg),
                "dest_lat": rng.uniform(lat0 - 0.09, lat0 + 0.09, n_bg),
            }
        )
    )
    incidents = pd.concat(rows, ignore_index=True)
    incidents["travel_seconds"] = rng.uniform(180, 600, len(incidents))
    incidents["crow_flies_km"] = rng.uniform(0.5, 4.0, len(incidents))
    incidents["borough"] = "DEMO"

    n_st = 10
    stations = pd.DataFrame(
        {
            "facname": [f"DEMO Station {i+1}" for i in range(n_st)],
            "factype": "EMS STATION",
            "latitude": rng.uniform(lat0 - 0.08, lat0 + 0.08, n_st),
            "longitude": rng.uniform(lon0 - 0.11, lon0 + 0.11, n_st),
            "borough": "DEMO",
        }
    )
    n_csl = 90
    csls = pd.DataFrame(
        {
            "facname": [f"DEMO CSL {i+1}" for i in range(n_csl)],
            "factype": "SYNTHETIC_CSL",
            "latitude": rng.uniform(lat0 - 0.09, lat0 + 0.09, n_csl),
            "longitude": rng.uniform(lon0 - 0.12, lon0 + 0.12, n_csl),
            "borough": "DEMO",
        }
    )
    return incidents, stations, csls
