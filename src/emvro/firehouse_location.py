"""Optimal firehouse facility location (redesign / replace).

Places or relocates firehouses so firetrucks reach historical demand as fast
as possible. Combines:

- graph theory: firetruck-weighted Dijkstra on the OSM drive graph
- travel-time model: affine calibration of graph seconds to CAD / hybrid scale
- discrete facility location: p-median / max-cover (reuse ``patrol.select_posts``)

Modes
-----
``redesign`` — choose ``k`` sites from the candidate set (full redesign).
``replace``  — keep ``n - r`` current houses; jointly swap out ``r`` existing
               for ``r`` new candidates (constant fleet size).

Travel-only: no land cost, zoning, or chute time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .patrol import (
    TravelTimeSurrogate,
    add_demand_cell_candidates,
    build_demand_grid,
    graph_nearest_nodes,
    graph_time_matrix,
    posture_metrics,
    select_posts,
    synthetic_demo_data,
)
from .routing.firetruck import WEIGHT_KEY as FIRETRUCK_WEIGHT, prepare_firetruck_graph


@dataclass
class AffineCalibrator:
    """Map graph seconds → CAD-like seconds: ``y ≈ a + b * x``."""

    a: float = 0.0
    b: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = self.a + self.b * np.asarray(x, dtype=float)
        return np.clip(out, 30.0, 1800.0)

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, **self.meta}


def fit_affine_calibrator(
    graph_s: np.ndarray,
    observed_s: np.ndarray,
    *,
    min_n: int = 40,
) -> AffineCalibrator:
    """Fit affine map from firetruck graph times to observed travel seconds."""
    g = np.asarray(graph_s, dtype=float).ravel()
    y = np.asarray(observed_s, dtype=float).ravel()
    ok = np.isfinite(g) & np.isfinite(y) & (g >= 45) & (g <= 1500) & (y >= 45) & (y <= 1200)
    meta: dict[str, Any] = {"n_pairs": int(ok.sum()), "method": "identity"}
    if int(ok.sum()) < min_n:
        meta["reason"] = "too_few_pairs"
        return AffineCalibrator(a=0.0, b=1.0, meta=meta)
    # Robust: median ratio for slope, median residual for intercept
    ratio = np.median(y[ok] / g[ok])
    if not np.isfinite(ratio) or ratio <= 0.2 or ratio > 3.0:
        meta["reason"] = "ratio_out_of_band"
        meta["raw_ratio"] = float(ratio) if np.isfinite(ratio) else None
        return AffineCalibrator(a=0.0, b=1.0, meta=meta)
    resid = y[ok] - ratio * g[ok]
    a = float(np.median(resid))
    # Keep intercept mild
    a = float(np.clip(a, -60.0, 120.0))
    meta.update({"method": "median_ratio_affine", "ratio": float(ratio), "intercept": a})
    return AffineCalibrator(a=a, b=float(ratio), meta=meta)


def calibrate_from_od_sample(
    G,
    houses: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    sample_n: int = 400,
    seed: int = 42,
    weight: str = FIRETRUCK_WEIGHT,
) -> AffineCalibrator:
    """Sample house→incident graph times vs observed CAD seconds for calibration."""
    if G is None or "travel_seconds" not in incidents.columns:
        return AffineCalibrator(meta={"method": "identity", "reason": "no_graph_or_labels"})
    df = incidents.dropna(subset=["dest_lat", "dest_lon", "travel_seconds"]).copy()
    if "start_lat" in df.columns and "start_lon" in df.columns:
        starts = df.dropna(subset=["start_lat", "start_lon"])
    else:
        starts = pd.DataFrame()
    if len(starts) < 30:
        # Pair random houses to random dests with observed y — weak but usable
        rng = np.random.default_rng(seed)
        if len(df) < 30 or len(houses) < 1:
            return AffineCalibrator(meta={"method": "identity", "reason": "insufficient_od"})
        take = min(sample_n, len(df))
        samp = df.sample(take, random_state=seed)
        hix = rng.integers(0, len(houses), size=take)
        src = houses.iloc[hix][["lon", "lat"]].reset_index(drop=True)
        tgt = samp[["dest_lon", "dest_lat"]].rename(columns={"dest_lon": "lon", "dest_lat": "lat"})
        y = samp["travel_seconds"].to_numpy(dtype=float)
    else:
        take = min(sample_n, len(starts))
        samp = starts.sample(take, random_state=seed)
        src = samp[["start_lon", "start_lat"]].rename(columns={"start_lon": "lon", "start_lat": "lat"})
        tgt = samp[["dest_lon", "dest_lat"]].rename(columns={"dest_lon": "lon", "dest_lat": "lat"})
        y = samp["travel_seconds"].to_numpy(dtype=float)

    try:
        src_nodes = graph_nearest_nodes(G, src["lon"].to_numpy(), src["lat"].to_numpy())
        tgt_nodes = graph_nearest_nodes(G, tgt["lon"].to_numpy(), tgt["lat"].to_numpy())
    except Exception as exc:  # noqa: BLE001
        return AffineCalibrator(meta={"method": "identity", "reason": f"snap_failed:{exc}"})

    # Per-row Dijkstra would be slow; use unique sources
    graph_s = np.full(len(src), np.nan)
    unique_src = {}
    for i, sn in enumerate(src_nodes):
        unique_src.setdefault(sn, []).append(i)
    import networkx as nx

    for sn, idxs in unique_src.items():
        try:
            dist = nx.single_source_dijkstra_path_length(G, sn, cutoff=1500, weight=weight)
        except Exception:  # noqa: BLE001
            continue
        for i in idxs:
            d = dist.get(tgt_nodes[i])
            if d is not None:
                graph_s[i] = float(d)
    return fit_affine_calibrator(graph_s, y)


def build_house_candidates(
    houses: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    demand_candidates: int = 80,
) -> pd.DataFrame:
    """Existing firehouses + top demand centroids as relocatable sites."""
    existing = pd.DataFrame(
        {
            "site_name": houses["facilityname"].astype(str),
            "layer": "existing_firehouse",
            "lon": houses["lon"].astype(float),
            "lat": houses["lat"].astype(float),
            "is_existing": True,
            "borough": houses["borough"] if "borough" in houses.columns else None,
        }
    )
    cand = existing.copy()
    cand.insert(0, "candidate_id", np.arange(len(cand)))
    # Reuse patrol helper shape: needs post_name/layer/lon/lat
    patrol_shaped = cand.rename(columns={"site_name": "post_name"})
    patrol_shaped["is_anchor"] = True
    enriched = add_demand_cell_candidates(patrol_shaped, cells, top_n=demand_candidates)
    # Normalize columns for this module
    out = pd.DataFrame(
        {
            "candidate_id": np.arange(len(enriched)),
            "site_name": enriched["post_name"].astype(str),
            "layer": enriched["layer"].astype(str),
            "lon": enriched["lon"].astype(float),
            "lat": enriched["lat"].astype(float),
        }
    )
    out["is_existing"] = out["layer"].eq("existing_firehouse")
    if "borough" in enriched.columns:
        out["borough"] = enriched["borough"]
    return out.reset_index(drop=True)


def build_travel_matrix(
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    G=None,
    surrogate: TravelTimeSurrogate | None = None,
    calibrator: AffineCalibrator | None = None,
    cutoff_s: float | None = 1200.0,
    max_graph_sources: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Candidate × demand travel seconds (surrogate → graph → calibrate)."""
    surrogate = surrogate or TravelTimeSurrogate()
    T = surrogate.matrix(candidates, cells)
    meta: dict[str, Any] = {
        "surrogate_speed_kph": surrogate.speed_kph,
        "graph_used": False,
        "graph_resolved_share": 0.0,
        "calibrator": (calibrator or AffineCalibrator()).to_dict(),
    }
    if G is None:
        if calibrator is not None:
            T = calibrator.transform(T)
        return T, meta

    src_df = candidates
    if max_graph_sources is not None and len(candidates) > max_graph_sources:
        # Prefer existing houses + top demand layers already in candidates; take all
        src_df = candidates

    try:
        src_nodes = graph_nearest_nodes(G, src_df["lon"].to_numpy(), src_df["lat"].to_numpy())
        tgt_nodes = graph_nearest_nodes(G, cells["lon"].to_numpy(), cells["lat"].to_numpy())
        Tg, resolved = graph_time_matrix(
            G,
            src_nodes,
            tgt_nodes,
            weight=FIRETRUCK_WEIGHT,
            cutoff_s=cutoff_s,
            fallback=T,
            progress_every=25,
        )
        meta["graph_used"] = True
        meta["graph_resolved_share"] = float(resolved.mean())
        meta["weight"] = FIRETRUCK_WEIGHT
        T = Tg
    except Exception as exc:  # noqa: BLE001
        meta["graph_error"] = str(exc)

    if calibrator is not None:
        T = calibrator.transform(T)
        meta["calibrator"] = calibrator.to_dict()
    return T, meta


def plan_redesign(
    T: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    objective: str = "response_time",
    threshold_s: float = 480.0,
    max_swap_rounds: int = 10,
):
    return select_posts(
        T,
        weights,
        k,
        objective=objective,
        threshold_s=threshold_s,
        max_swap_rounds=max_swap_rounds,
    )


def plan_replace(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    *,
    replace_r: int,
    new_pool: Sequence[int] | None = None,
    objective: str = "response_time",
    threshold_s: float = 480.0,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """Joint swaps: drop ``r`` existing houses, open ``r`` new candidates.

    Starts from the full existing set; each round swaps one existing site for
    the best unused new candidate. Total fleet size stays ``len(existing_idx)``.
    """
    from .patrol import _objective_value

    existing = [int(i) for i in existing_idx]
    n = len(existing)
    r = int(min(max(0, replace_r), n))
    if new_pool is None:
        new_pool = [i for i in range(T.shape[0]) if i not in set(existing)]
    else:
        new_pool = [int(i) for i in new_pool if int(i) not in set(existing)]

    selected = list(existing)
    if r == 0 or not new_pool:
        best = T[selected, :].min(axis=0)
        return {
            "indices": selected,
            "closed": [],
            "opened": [],
            "value": _objective_value(T, selected, weights, objective=objective, threshold_s=threshold_s),
            "response_s": best,
            "n_swaps": 0,
        }

    rounds = max_rounds if max_rounds is not None else r
    n_swaps = 0
    closed: list[int] = []
    opened: list[int] = []
    current = _objective_value(T, selected, weights, objective=objective, threshold_s=threshold_s)

    for _ in range(rounds):
        if len(closed) >= r:
            break
        best_gain = 0.0
        best_out = None
        best_in = None
        # Only swap out still-original existing houses not yet closed
        drop_candidates = [i for i in selected if i in set(existing) and i not in closed]
        add_candidates = [j for j in new_pool if j not in selected]
        if not drop_candidates or not add_candidates:
            break
        for out_idx in drop_candidates:
            base = [s for s in selected if s != out_idx]
            for in_idx in add_candidates:
                val = _objective_value(
                    T, base + [in_idx], weights, objective=objective, threshold_s=threshold_s
                )
                gain = current - val
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_out = out_idx
                    best_in = in_idx
        if best_out is None or best_in is None:
            break
        selected = [best_in if s == best_out else s for s in selected]
        closed.append(best_out)
        opened.append(best_in)
        current -= best_gain
        n_swaps += 1

    best = T[selected, :].min(axis=0)
    return {
        "indices": [int(i) for i in selected],
        "closed": [int(i) for i in closed],
        "opened": [int(i) for i in opened],
        "value": float(current),
        "response_s": best,
        "n_swaps": n_swaps,
        "objective": objective,
    }


def random_replace_baseline(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    *,
    replace_r: int,
    new_pool: Sequence[int],
    objective: str = "response_time",
    threshold_s: float = 480.0,
    seed: int = 0,
) -> dict[str, Any]:
    from .patrol import _objective_value

    rng = np.random.default_rng(seed)
    existing = [int(i) for i in existing_idx]
    pool = [int(i) for i in new_pool if int(i) not in set(existing)]
    r = min(int(replace_r), len(existing), len(pool))
    if r <= 0:
        selected = list(existing)
        closed, opened = [], []
    else:
        closed = [int(x) for x in rng.choice(existing, size=r, replace=False)]
        opened = [int(x) for x in rng.choice(pool, size=r, replace=False)]
        selected = [i for i in existing if i not in set(closed)] + opened
    best = T[selected, :].min(axis=0)
    return {
        "indices": selected,
        "closed": closed,
        "opened": opened,
        "value": _objective_value(T, selected, weights, objective=objective, threshold_s=threshold_s),
        "response_s": best,
        "label": "random_replace",
    }


@dataclass
class FirehousePlanResult:
    mode: str
    city_id: str
    candidates: pd.DataFrame
    cells: pd.DataFrame
    selected: pd.DataFrame
    closed: pd.DataFrame
    opened: pd.DataFrame
    metrics: dict[str, Any]
    baselines: dict[str, Any]
    travel_meta: dict[str, Any]
    summary: dict[str, Any]


def run_firehouse_plan(
    *,
    city_id: str,
    houses: pd.DataFrame,
    incidents: pd.DataFrame,
    mode: str = "redesign",
    n_houses: int | None = None,
    replace_r: int = 0,
    cell_km: float = 0.9,
    demand_candidates: int = 80,
    objective: str = "response_time",
    threshold_min: float = 8.0,
    graph_path: Path | str | None = None,
    hour: int = 17,
    use_graph: bool = True,
    max_incident_rows: int | None = 200_000,
    seed: int = 42,
) -> FirehousePlanResult:
    """End-to-end facility location for one city."""
    if mode not in {"redesign", "replace"}:
        raise ValueError(f"mode must be redesign|replace, got {mode!r}")

    if max_incident_rows is not None and len(incidents) > max_incident_rows:
        incidents = incidents.sample(max_incident_rows, random_state=seed).reset_index(drop=True)

    cells = build_demand_grid(incidents, lat_col="dest_lat", lon_col="dest_lon", cell_km=cell_km)
    candidates = build_house_candidates(houses, cells, demand_candidates=demand_candidates)
    weights = cells["weight"].to_numpy(dtype=float)
    threshold_s = float(threshold_min) * 60.0

    existing_idx = candidates.index[candidates["is_existing"]].tolist()
    if not existing_idx:
        # treat all current house rows as existing by position
        existing_idx = list(range(len(houses)))
        existing_idx = [i for i in existing_idx if i < len(candidates)]

    G = None
    calibrator = AffineCalibrator()
    if use_graph and graph_path is not None and Path(graph_path).exists():
        print(f"Loading firetruck graph from {graph_path} (hour={hour})…")
        G = prepare_firetruck_graph(graph_path, hour=hour)
        print("Calibrating graph times to observed CAD travel…")
        calibrator = calibrate_from_od_sample(G, houses, incidents, seed=seed)
        print("Calibrator:", calibrator.to_dict())

    surrogate = TravelTimeSurrogate()
    print(f"Building travel matrix ({len(candidates)} sites × {len(cells)} cells)…")
    T, travel_meta = build_travel_matrix(
        candidates,
        cells,
        G=G,
        surrogate=surrogate,
        calibrator=calibrator,
        cutoff_s=max(threshold_s * 2.5, 900.0),
    )

    # Current posture baseline
    cur_sel = existing_idx
    cur_resp = T[cur_sel, :].min(axis=0) if cur_sel else np.full(len(cells), np.nan)
    baselines = {
        "current_firehouses": posture_metrics(
            cur_resp, weights, threshold_s=threshold_s, label="current_firehouses", n_posts=len(cur_sel)
        )
    }

    closed_df = pd.DataFrame()
    opened_df = pd.DataFrame()

    if mode == "redesign":
        k = int(n_houses) if n_houses is not None else len(existing_idx)
        k = max(1, min(k, len(candidates)))
        print(f"Redesign: placing k={k} firehouses…")
        res = plan_redesign(T, weights, k, objective=objective, threshold_s=threshold_s)
        selected_idx = res.indices
        plan_resp = res.response_s
        plan_value = res.value
        # Compare which existing were kept vs new
        kept = [i for i in selected_idx if i in set(existing_idx)]
        opened_idx = [i for i in selected_idx if i not in set(existing_idx)]
        closed_idx = [i for i in existing_idx if i not in set(selected_idx)]
        closed_df = candidates.loc[closed_idx].copy() if closed_idx else pd.DataFrame()
        opened_df = candidates.loc[opened_idx].copy() if opened_idx else pd.DataFrame()
        extra = {"k": k, "n_swaps": res.n_swaps, "kept_existing": len(kept)}
    else:
        r = int(replace_r)
        if r <= 0:
            raise ValueError("--replace must be > 0 for mode=replace")
        new_pool = [i for i in range(len(candidates)) if i not in set(existing_idx)]
        print(f"Replace: relocating r={r} of {len(existing_idx)} firehouses…")
        res = plan_replace(
            T,
            weights,
            existing_idx,
            replace_r=r,
            new_pool=new_pool,
            objective=objective,
            threshold_s=threshold_s,
        )
        selected_idx = res["indices"]
        plan_resp = res["response_s"]
        plan_value = res["value"]
        closed_df = candidates.loc[res["closed"]].copy() if res["closed"] else pd.DataFrame()
        opened_df = candidates.loc[res["opened"]].copy() if res["opened"] else pd.DataFrame()
        rnd = random_replace_baseline(
            T,
            weights,
            existing_idx,
            replace_r=r,
            new_pool=new_pool,
            objective=objective,
            threshold_s=threshold_s,
            seed=seed,
        )
        baselines["random_replace"] = posture_metrics(
            rnd["response_s"],
            weights,
            threshold_s=threshold_s,
            label="random_replace",
            n_posts=len(rnd["indices"]),
        )
        extra = {"replace_r": r, "n_swaps": res["n_swaps"]}

    selected = candidates.loc[selected_idx].copy().reset_index(drop=True)
    selected["plan_rank"] = np.arange(len(selected))
    # Per-site assigned demand
    assign = T[selected_idx, :].argmin(axis=0)
    site_w = np.zeros(len(selected_idx))
    for j, a in enumerate(assign):
        site_w[a] += weights[j]
    selected["assigned_demand"] = site_w
    selected["assigned_share"] = site_w / max(float(weights.sum()), 1e-9)

    plan_metrics = posture_metrics(
        plan_resp,
        weights,
        threshold_s=threshold_s,
        label=f"plan_{mode}",
        n_posts=len(selected_idx),
    )
    plan_metrics["objective_value"] = float(plan_value)

    cells_out = cells.copy()
    cells_out["response_s"] = plan_resp
    cells_out["response_s_current"] = cur_resp
    cells_out["response_s_delta"] = plan_resp - cur_resp
    cells_out["assigned_site"] = [selected_idx[int(a)] for a in assign]
    cells_out["covered"] = plan_resp <= threshold_s
    cells_out["covered_current"] = cur_resp <= threshold_s

    cur_et = baselines["current_firehouses"]["expected_response_s"]
    plan_et = plan_metrics["expected_response_s"]
    summary = {
        "city_id": city_id,
        "mode": mode,
        "objective": objective,
        "n_existing_firehouses": len(existing_idx),
        "n_candidates": len(candidates),
        "n_demand_cells": len(cells),
        "n_incidents": int(len(incidents)),
        "n_selected": len(selected),
        "n_closed": int(len(closed_df)),
        "n_opened": int(len(opened_df)),
        "threshold_min": threshold_min,
        "plan": plan_metrics,
        "baselines": baselines,
        "delta_vs_current_s": round(float(plan_et - cur_et), 1) if cur_et == cur_et else None,
        "travel": travel_meta,
        **extra,
        "honesty": (
            "Travel-only facility location on discrete candidates (existing houses + "
            "demand centroids). No parcel, zoning, or staffing constraints. "
            "Graph times use firetruck edge weights; affine-calibrated to CAD travel when available."
        ),
    }

    return FirehousePlanResult(
        mode=mode,
        city_id=city_id,
        candidates=candidates,
        cells=cells_out,
        selected=selected,
        closed=closed_df.reset_index(drop=True) if len(closed_df) else closed_df,
        opened=opened_df.reset_index(drop=True) if len(opened_df) else opened_df,
        metrics=plan_metrics,
        baselines=baselines,
        travel_meta=travel_meta,
        summary=summary,
    )


def demo_inputs(*, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic houses + incidents for --demo."""
    incidents, stations, _csl = synthetic_demo_data(seed=seed)
    houses = pd.DataFrame(
        {
            "facilityname": stations["facname"].astype(str),
            "lat": pd.to_numeric(stations["latitude"], errors="coerce"),
            "lon": pd.to_numeric(stations["longitude"], errors="coerce"),
            "borough": stations.get("borough", "DEMO"),
        }
    ).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    houses.insert(0, "house_id", houses.index.astype(int))
    return houses, incidents.reset_index(drop=True)
