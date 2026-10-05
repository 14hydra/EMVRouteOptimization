"""Optimal firehouse facility location (London LFB primary).

Places or relocates fire stations so the first engine reaches historical
demand fast enough to meet the LFB standards. The scored quantity is

    attendance(station s -> cell j) = predicted_drive(s, j) + turnout(s)

and is judged against the LFB first-engine KPIs (mean <= 6 min, > 90 % within
10 min; ``emvro.lfb_standards.score_first_engine``). Travel-only scoring (no
turnout) was WRONG for LFB: public attendance is mobilise -> arrive, so drive
alone understates response and misses the 6/10-minute standards.

Building blocks:

- travel scorers (``scorer=``): ``kolesar`` (crow distance -> Kolesar
  piecewise model, refit on LFB trips), ``crow`` (flat-speed surrogate) and
  ``network`` (firetruck-weighted Dijkstra + affine calibration)
- turnout: per-station prior (median attendance residual vs. Kolesar drive,
  clipped to [30, 180] s) for existing houses, ``DEFAULT_TURNOUT_S`` for new sites
- discrete facility location: p-median / max-cover (``patrol.select_posts``)
- optional busy-engine blend (``use_busy``) and demand/eval year split

Modes
-----
``redesign`` - choose ``k`` sites from the candidate set (full redesign).
``replace``  - keep ``n - r`` current houses; jointly swap out ``r`` existing
               for ``r`` new candidates (constant fleet size).
``expand``   - keep ALL current houses; greedily open ``m`` new ones.

No land cost, zoning, staffing or borough-boundary constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .busy_engines import estimate_busy_rates, expected_first_arrival
from .kolesar import KolesarModel, fit_kolesar
from .lfb_standards import DEFAULT_TURNOUT_S, attendance_seconds, score_first_engine
from .patrol import (
    TravelTimeSurrogate,
    add_demand_cell_candidates,
    build_demand_grid,
    graph_nearest_nodes,
    graph_time_matrix,
    haversine_km,
    posture_metrics,
    select_posts,
    synthetic_demo_data,
)
from .routing.firetruck import WEIGHT_KEY as FIRETRUCK_WEIGHT, prepare_firetruck_graph

SCORERS = ("network", "kolesar", "crow")
MODES = ("redesign", "replace", "expand")


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
    scorer: str = "network",
    kolesar: KolesarModel | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Candidate × demand **drive** seconds (turnout is added by the caller).

    ``scorer``:
      * ``network`` — surrogate → graph Dijkstra → affine calibration (needs ``G``;
        falls back to the crow surrogate when no graph is given)
      * ``kolesar`` — crow-flies km → :class:`KolesarModel` (NYC prior if none given)
      * ``crow``    — flat-speed surrogate only (graph ignored)
    """
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {SCORERS}, got {scorer!r}")
    surrogate = surrogate or TravelTimeSurrogate()
    meta: dict[str, Any] = {
        "scorer": scorer,
        "surrogate_speed_kph": surrogate.speed_kph,
        "graph_used": False,
        "graph_resolved_share": 0.0,
        "calibrator": (calibrator or AffineCalibrator()).to_dict(),
    }
    if scorer == "kolesar":
        model = kolesar or KolesarModel.nyc_default()
        km = haversine_km(
            candidates["lon"].to_numpy(dtype=float)[:, None],
            candidates["lat"].to_numpy(dtype=float)[:, None],
            cells["lon"].to_numpy(dtype=float)[None, :],
            cells["lat"].to_numpy(dtype=float)[None, :],
        )
        meta["kolesar"] = model.to_dict()
        return np.maximum(model.predict(km), 15.0), meta

    T = surrogate.matrix(candidates, cells)
    if G is None or scorer == "crow":
        if scorer == "network":
            meta["scorer_effective"] = "crow"
            meta["note"] = "no graph supplied; used crow surrogate"
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


def plan_expand(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    add_m: int,
    *,
    new_pool: Sequence[int] | None = None,
    objective: str = "response_time",
    threshold_s: float = 480.0,
    max_swap_rounds: int = 4,
) -> dict[str, Any]:
    """Keep ALL existing houses; open ``add_m`` new candidates from ``new_pool``.

    Uses ``select_posts`` with ``must_include=existing`` (greedy seed + swaps among
    the new sites only; existing rows are locked).
    """
    existing = [int(i) for i in existing_idx]
    ex = set(existing)
    pool = [i for i in range(T.shape[0]) if i not in ex] if new_pool is None else [
        int(i) for i in new_pool if int(i) not in ex
    ]
    m = int(min(max(0, add_m), len(pool)))
    if m == 0:
        return {
            "indices": existing, "closed": [], "opened": [], "n_swaps": 0,
            "value": _expand_value(T, existing, weights, objective, threshold_s),
            "response_s": T[existing, :].min(axis=0), "objective": objective,
        }
    res = select_posts(
        T, weights, len(existing) + m,
        objective=objective, threshold_s=threshold_s,
        must_include=existing, allowed=existing + pool, max_swap_rounds=max_swap_rounds,
    )
    return {
        "indices": res.indices,
        "closed": [],
        "opened": [i for i in res.indices if i not in ex],
        "value": float(res.value),
        "response_s": res.response_s,
        "n_swaps": res.n_swaps,
        "objective": objective,
    }


def _expand_value(T, idx, weights, objective, threshold_s) -> float:
    from .patrol import _objective_value

    return _objective_value(T, idx, weights, objective=objective, threshold_s=threshold_s)


def random_expand_baseline(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    add_m: int,
    *,
    new_pool: Sequence[int],
    objective: str = "response_time",
    threshold_s: float = 480.0,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    existing = [int(i) for i in existing_idx]
    pool = [int(i) for i in new_pool if int(i) not in set(existing)]
    m = min(int(add_m), len(pool))
    opened = [int(x) for x in rng.choice(pool, size=m, replace=False)] if m > 0 else []
    selected = existing + opened
    return {
        "indices": selected,
        "opened": opened,
        "value": _expand_value(T, selected, weights, objective, threshold_s),
        "response_s": T[selected, :].min(axis=0),
        "label": "random_expand",
    }


# --------------------------------------------------------------------------- #
# Turnout / Kolesar fit / busy / year-split helpers
# --------------------------------------------------------------------------- #


def _norm_name(x: Any) -> str:
    return " ".join(str(x).strip().lower().split())


def _station_ground_pairs(
    incidents: pd.DataFrame, houses: pd.DataFrame
) -> pd.DataFrame | None:
    """Rows of (station, crow_km, observed_s) for non-busy calls (house = origin)."""
    need = {"station_ground", "travel_seconds", "dest_lat", "dest_lon"}
    if not need.issubset(incidents.columns):
        return None
    df = incidents.dropna(subset=list(need)).copy()
    if "deployed_from_station" in df.columns:
        same = df["station_ground"].map(_norm_name) == df["deployed_from_station"].map(_norm_name)
        df = df[same]
    df["_key"] = df["station_ground"].map(_norm_name)
    h = houses.assign(_key=houses["facilityname"].map(_norm_name))[["_key", "lat", "lon"]]
    h = h.drop_duplicates("_key")
    df = df.merge(h, on="_key", how="inner")
    if df.empty:
        return None
    df["crow_km"] = haversine_km(df["lon"], df["lat"], df["dest_lon"], df["dest_lat"])
    df["observed_s"] = df["travel_seconds"].astype(float)
    df = df[df["crow_km"].between(0.05, 40.0) & df["observed_s"].between(60, 1800)]
    return df[["_key", "crow_km", "observed_s"]].rename(columns={"_key": "station"}) if len(df) else None


def fit_scorer_kolesar(
    incidents: pd.DataFrame,
    houses: pd.DataFrame,
    *,
    min_pairs: int = 200,
    max_pairs: int = 50_000,
    turnout_s: float = DEFAULT_TURNOUT_S,
    seed: int = 0,
) -> tuple[KolesarModel, dict[str, Any], pd.DataFrame | None]:
    """Fit a **drive-only** Kolesar model from house→dest crow km.

    LFB ``travel_seconds`` is attendance (turnout + drive), so ``turnout_s`` is
    subtracted before fitting; the planner adds turnout back per station.
    Falls back to the NYC prior when too few usable pairs exist.
    """
    pairs = _station_ground_pairs(incidents, houses)
    if pairs is None or len(pairs) < min_pairs:
        return (
            KolesarModel.nyc_default(),
            {"source": "nyc_default", "n_pairs": 0 if pairs is None else int(len(pairs))},
            pairs,
        )
    fit_df = pairs.sample(max_pairs, random_state=seed) if len(pairs) > max_pairs else pairs
    drive = np.maximum(fit_df["observed_s"].to_numpy() - float(turnout_s), 30.0)
    model = fit_kolesar(fit_df["crow_km"].to_numpy(), drive)
    return model, {"source": "fit_house_to_dest_crow_km", "n_pairs": int(len(fit_df)),
                   "turnout_removed_s": float(turnout_s)}, pairs


def estimate_station_turnout(
    pairs: pd.DataFrame | None,
    model: KolesarModel,
    *,
    default_s: float = DEFAULT_TURNOUT_S,
    min_n: int = 20,
    clip: tuple[float, float] = (30.0, 180.0),
) -> dict[str, float]:
    """Per-station turnout: default + median(observed - default - Kolesar drive)."""
    if pairs is None or pairs.empty:
        return {}
    resid = pairs["observed_s"].to_numpy() - default_s - model.predict(pairs["crow_km"].to_numpy())
    df = pd.DataFrame({"station": pairs["station"].to_numpy(), "resid": resid})
    g = df.groupby("station")["resid"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    return {k: float(np.clip(default_s + v, *clip)) for k, v in g["median"].items()}


def resolve_turnout_vector(
    candidates: pd.DataFrame,
    turnout_s: float | Mapping[str, float] | None,
    estimated: Mapping[str, float] | None = None,
    *,
    default_s: float = DEFAULT_TURNOUT_S,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turnout per candidate row. Scalar → all rows; dict/estimates → existing by name."""
    n = len(candidates)
    if isinstance(turnout_s, (int, float)) and not isinstance(turnout_s, bool):
        return np.full(n, float(turnout_s)), {"source": "scalar", "default_s": float(turnout_s)}
    table = {_norm_name(k): float(v) for k, v in (estimated or {}).items()}
    source = "estimated_station_prior" if table else "default"
    if isinstance(turnout_s, Mapping):
        table.update({_norm_name(k): float(v) for k, v in turnout_s.items()})
        source = "mapping" + ("+estimated" if estimated else "")
    vec = np.full(n, float(default_s))
    existing = candidates["is_existing"].to_numpy(dtype=bool)
    keys = candidates["site_name"].map(_norm_name).to_numpy()
    hit = 0
    for i in range(n):
        if existing[i] and keys[i] in table:
            vec[i] = table[keys[i]]
            hit += 1
    vec = np.clip(vec, 0.0, 300.0)
    ex = vec[existing] if existing.any() else vec
    return vec, {
        "source": source, "default_s": float(default_s), "n_existing_with_prior": hit,
        "existing_min_s": float(ex.min()), "existing_median_s": float(np.median(ex)),
        "existing_max_s": float(ex.max()),
    }


def busy_prob_vector(
    candidates: pd.DataFrame, incidents: pd.DataFrame, *, min_n: int = 20
) -> np.ndarray | None:
    """Busy probability per candidate (new sites get the existing-house mean)."""
    if not {"station_ground", "deployed_from_station"}.issubset(incidents.columns):
        return None
    rates = estimate_busy_rates(incidents)["by_station"]
    rates = rates[rates["n_incidents"] >= min_n]
    if rates.empty:
        return None
    table = {_norm_name(k): float(v) for k, v in zip(rates["station_ground"], rates["busy_prob"])}
    keys = candidates["site_name"].map(_norm_name).to_numpy()
    existing = candidates["is_existing"].to_numpy(dtype=bool)
    p = np.array([table.get(k, np.nan) if e else np.nan for k, e in zip(keys, existing)])
    mean_p = float(np.nanmean(p)) if np.isfinite(p).any() else float(np.mean(list(table.values())))
    return np.where(np.isfinite(p), p, mean_p)


def posture_response(A: np.ndarray, idx: Sequence[int], busy_p: np.ndarray | None = None) -> np.ndarray:
    """Per-cell first-engine seconds; with ``busy_p`` blend nearest with next-nearest."""
    idx = np.asarray(list(idx), dtype=int)
    sub = A[idx, :]
    if busy_p is None or len(idx) < 2:
        return sub.min(axis=0)
    order = np.argsort(sub, axis=0)[:2]
    cols = np.arange(sub.shape[1])
    t1, t2 = sub[order[0], cols], sub[order[1], cols]
    return expected_first_arrival(t1, busy_p[idx[order[0]]], t2)


def snap_to_cells(points: pd.DataFrame, cells: pd.DataFrame, *, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Nearest demand cell (crow km) for each incident; returns (cell_idx, km)."""
    lat0 = float(cells["lat"].mean())
    kx = 111.32 * np.cos(np.radians(lat0))
    cx, cy = cells["lon"].to_numpy(float) * kx, cells["lat"].to_numpy(float) * 110.574
    px, py = points["dest_lon"].to_numpy(float) * kx, points["dest_lat"].to_numpy(float) * 110.574
    idx = np.empty(len(points), dtype=int)
    dist = np.empty(len(points))
    for s in range(0, len(points), chunk):
        d2 = (px[s:s + chunk, None] - cx[None, :]) ** 2 + (py[s:s + chunk, None] - cy[None, :]) ** 2
        j = d2.argmin(axis=1)
        idx[s:s + chunk] = j
        dist[s:s + chunk] = np.sqrt(d2[np.arange(len(j)), j])
    return idx, dist


def drop_far_incidents(
    incidents: pd.DataFrame, houses: pd.DataFrame, *, max_km: float = 60.0
) -> pd.DataFrame:
    """Drop geocode outliers farther than ``max_km`` from the median house location."""
    c_lat, c_lon = float(houses["lat"].median()), float(houses["lon"].median())
    km = haversine_km(incidents["dest_lon"], incidents["dest_lat"], c_lon, c_lat)
    return incidents[np.asarray(km) <= max_km]


def _split_years(
    incidents: pd.DataFrame,
    demand_years: Sequence[int] | None,
    eval_years: Sequence[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Demand / eval incident frames by ``cal_year`` (no-op without that column)."""
    if "cal_year" not in incidents.columns:
        return incidents, None
    yr = pd.to_numeric(incidents["cal_year"], errors="coerce")
    demand = incidents
    if demand_years:
        demand = incidents[yr.isin({int(y) for y in demand_years})]
        if demand.empty:
            raise ValueError(f"No incidents for demand_years={list(demand_years)}")
    ev = None
    if eval_years:
        ev = incidents[yr.isin({int(y) for y in eval_years})]
        if ev.empty:
            ev = None
    return demand.reset_index(drop=True), None if ev is None else ev.reset_index(drop=True)


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
    add_stations: int = 0,
    cell_km: float = 0.9,
    demand_candidates: int = 80,
    objective: str = "response_time",
    threshold_min: float = 10.0,
    graph_path: Path | str | None = None,
    hour: int = 17,
    use_graph: bool = True,
    max_incident_rows: int | None = 200_000,
    seed: int = 42,
    scorer: str = "network",
    turnout_s: float | Mapping[str, float] | None = None,
    demand_years: Sequence[int] | None = None,
    eval_years: Sequence[int] | None = None,
    use_busy: bool = False,
) -> FirehousePlanResult:
    """End-to-end facility location for one city.

    Response = drive (``scorer``) + turnout. ``turnout_s``: scalar (all sites), a
    ``{station_name: seconds}`` mapping, or ``None`` (estimate a per-station prior
    from LFB attendance residuals; new sites get ``DEFAULT_TURNOUT_S``).
    ``demand_years`` builds the demand grid; ``eval_years`` (if present in the data)
    re-scores current vs plan on held-out incidents snapped to demand cells.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be {'|'.join(MODES)}, got {mode!r}")
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be {'|'.join(SCORERS)}, got {scorer!r}")

    has_years = "cal_year" in incidents.columns
    n_raw = len(incidents)
    incidents = drop_far_incidents(incidents, houses)
    n_far = n_raw - len(incidents)
    incidents, eval_inc = _split_years(incidents, demand_years, eval_years)
    if max_incident_rows is not None:
        if len(incidents) > max_incident_rows:
            incidents = incidents.sample(max_incident_rows, random_state=seed).reset_index(drop=True)
        if eval_inc is not None and len(eval_inc) > max_incident_rows:
            eval_inc = eval_inc.sample(max_incident_rows, random_state=seed).reset_index(drop=True)

    cells = build_demand_grid(incidents, lat_col="dest_lat", lon_col="dest_lon", cell_km=cell_km)
    candidates = build_house_candidates(houses, cells, demand_candidates=demand_candidates)
    weights = cells["weight"].to_numpy(dtype=float)
    threshold_s = float(threshold_min) * 60.0

    existing_idx = candidates.index[candidates["is_existing"]].tolist()
    if not existing_idx:
        existing_idx = [i for i in range(len(houses)) if i < len(candidates)]

    # --- Kolesar drive model (scorer + turnout prior) -----------------------
    kmodel, kmeta, pairs = fit_scorer_kolesar(incidents, houses, seed=seed)
    print(f"Kolesar drive model: {kmeta['source']} (pairs={kmeta['n_pairs']})")

    # --- turnout per candidate ---------------------------------------------
    estimated = None
    if turnout_s is None and kmeta["source"] != "nyc_default":
        estimated = estimate_station_turnout(pairs, kmodel)
    turnout_vec, turnout_meta = resolve_turnout_vector(candidates, turnout_s, estimated)
    print("Turnout:", turnout_meta)

    # --- graph (network scorer only) ---------------------------------------
    G = None
    calibrator = AffineCalibrator()
    if scorer == "network" and use_graph and graph_path is not None and Path(graph_path).exists():
        print(f"Loading firetruck graph from {graph_path} (hour={hour})…")
        G = prepare_firetruck_graph(graph_path, hour=hour)
        print("Calibrating graph times to observed CAD travel…")
        calib_inc = incidents
        if "station_ground" in incidents.columns and "travel_seconds" in incidents.columns:
            # LFB attendance includes turnout; calibrate graph to drive-only seconds.
            calib_inc = incidents.assign(travel_seconds=incidents["travel_seconds"] - DEFAULT_TURNOUT_S)
        calibrator = calibrate_from_od_sample(G, houses, calib_inc, seed=seed)
        print("Calibrator:", calibrator.to_dict())

    print(f"Building {scorer} travel matrix ({len(candidates)} sites × {len(cells)} cells)…")
    T, travel_meta = build_travel_matrix(
        candidates,
        cells,
        G=G,
        surrogate=TravelTimeSurrogate(),
        calibrator=calibrator,
        cutoff_s=max(threshold_s * 2.5, 900.0),
        scorer=scorer,
        kolesar=kmodel,
    )
    travel_meta["kolesar_fit"] = kmeta
    travel_meta["turnout"] = turnout_meta
    # Attendance = drive + station turnout; everything below optimises/scores A.
    A = attendance_seconds(T, turnout_vec[:, None])

    busy_p = None
    if use_busy:
        busy_p = busy_prob_vector(candidates, incidents)
        travel_meta["busy"] = {
            "used": busy_p is not None,
            "mean_existing_busy_prob": None if busy_p is None
            else round(float(busy_p[candidates["is_existing"].to_numpy()].mean()), 4),
        }
        if busy_p is None:
            print("use_busy requested but station_ground/deployed_from_station missing; ignoring.")

    # --- current posture ----------------------------------------------------
    cur_sel = existing_idx
    cur_resp = posture_response(A, cur_sel, busy_p) if cur_sel else np.full(len(cells), np.nan)
    cur_metrics = posture_metrics(
        cur_resp, weights, threshold_s=threshold_s, label="current_firehouses", n_posts=len(cur_sel)
    )
    cur_metrics["lfb_first_engine"] = score_first_engine(cur_resp, weights)
    baselines = {"current_firehouses": cur_metrics}

    closed_df = pd.DataFrame()
    opened_df = pd.DataFrame()
    new_pool = [i for i in range(len(candidates)) if i not in set(existing_idx)]

    if mode == "redesign":
        k = int(n_houses) if n_houses is not None else len(existing_idx)
        k = max(1, min(k, len(candidates)))
        print(f"Redesign: placing k={k} firehouses…")
        res = plan_redesign(A, weights, k, objective=objective, threshold_s=threshold_s)
        selected_idx = res.indices
        plan_value = res.value
        kept = [i for i in selected_idx if i in set(existing_idx)]
        opened_idx = [i for i in selected_idx if i not in set(existing_idx)]
        closed_idx = [i for i in existing_idx if i not in set(selected_idx)]
        closed_df = candidates.loc[closed_idx].copy() if closed_idx else pd.DataFrame()
        opened_df = candidates.loc[opened_idx].copy() if opened_idx else pd.DataFrame()
        extra = {"k": k, "n_swaps": res.n_swaps, "kept_existing": len(kept)}
    elif mode == "replace":
        r = int(replace_r)
        if r <= 0:
            raise ValueError("--replace must be > 0 for mode=replace")
        print(f"Replace: relocating r={r} of {len(existing_idx)} firehouses…")
        res = plan_replace(
            A, weights, existing_idx,
            replace_r=r, new_pool=new_pool, objective=objective, threshold_s=threshold_s,
        )
        selected_idx = res["indices"]
        plan_value = res["value"]
        closed_df = candidates.loc[res["closed"]].copy() if res["closed"] else pd.DataFrame()
        opened_df = candidates.loc[res["opened"]].copy() if res["opened"] else pd.DataFrame()
        rnd = random_replace_baseline(
            A, weights, existing_idx,
            replace_r=r, new_pool=new_pool, objective=objective, threshold_s=threshold_s, seed=seed,
        )
        baselines["random_replace"] = _baseline_metrics(
            A, rnd["indices"], weights, threshold_s, busy_p, "random_replace"
        )
        extra = {"replace_r": r, "n_swaps": res["n_swaps"]}
    else:  # expand
        m = int(add_stations)
        if m <= 0:
            raise ValueError("--add-stations must be > 0 for mode=expand")
        print(f"Expand: keeping {len(existing_idx)} houses, opening m={m} new…")
        res = plan_expand(
            A, weights, existing_idx, m,
            new_pool=new_pool, objective=objective, threshold_s=threshold_s,
        )
        selected_idx = res["indices"]
        plan_value = res["value"]
        opened_df = candidates.loc[res["opened"]].copy() if res["opened"] else pd.DataFrame()
        rnd = random_expand_baseline(
            A, weights, existing_idx, m,
            new_pool=new_pool, objective=objective, threshold_s=threshold_s, seed=seed,
        )
        baselines["random_expand"] = _baseline_metrics(
            A, rnd["indices"], weights, threshold_s, busy_p, "random_expand"
        )
        extra = {"add_stations": m, "n_swaps": res["n_swaps"]}

    plan_resp = posture_response(A, selected_idx, busy_p)
    selected = candidates.loc[selected_idx].copy().reset_index(drop=True)
    selected["plan_rank"] = np.arange(len(selected))
    selected["turnout_s"] = turnout_vec[selected_idx]
    assign = A[selected_idx, :].argmin(axis=0)
    site_w = np.zeros(len(selected_idx))
    for j, a in enumerate(assign):
        site_w[a] += weights[j]
    selected["assigned_demand"] = site_w
    selected["assigned_share"] = site_w / max(float(weights.sum()), 1e-9)

    plan_metrics = posture_metrics(
        plan_resp, weights, threshold_s=threshold_s, label=f"plan_{mode}", n_posts=len(selected_idx)
    )
    plan_metrics["objective_value"] = float(plan_value)  # nearest-engine objective
    plan_metrics["lfb_first_engine"] = score_first_engine(plan_resp, weights)

    # --- held-out evaluation on eval_years incidents ------------------------
    eval_summary = None
    if eval_inc is not None and len(eval_inc):
        ok = eval_inc[["dest_lat", "dest_lon"]].notna().all(axis=1)
        ev = eval_inc[ok].reset_index(drop=True)
        cidx, ckm = snap_to_cells(ev, cells)
        cur_score = score_first_engine(cur_resp[cidx])
        plan_score = score_first_engine(plan_resp[cidx])
        cur_metrics["lfb_first_engine_eval"] = cur_score
        plan_metrics["lfb_first_engine_eval"] = plan_score
        eval_summary = {
            "years": [int(y) for y in (eval_years or [])],
            "n_incidents": int(len(ev)),
            "snap_median_km": round(float(np.median(ckm)), 3),
            "snap_p95_km": round(float(np.percentile(ckm, 95)), 3),
            "current": cur_score,
            "plan": plan_score,
        }
        if "travel_seconds" in ev.columns and ev["travel_seconds"].notna().any():
            eval_summary["observed_attendance"] = score_first_engine(ev["travel_seconds"].dropna())

    cells_out = cells.copy()
    cells_out["response_s"] = plan_resp
    cells_out["response_s_current"] = cur_resp
    cells_out["response_s_delta"] = plan_resp - cur_resp
    cells_out["assigned_site"] = [selected_idx[int(a)] for a in assign]
    cells_out["drive_s"] = T[np.asarray(selected_idx)[assign], np.arange(len(cells))]
    cells_out["covered"] = plan_resp <= threshold_s
    cells_out["covered_current"] = cur_resp <= threshold_s

    cur_et = cur_metrics["expected_response_s"]
    plan_et = plan_metrics["expected_response_s"]
    summary = {
        "city_id": city_id,
        "mode": mode,
        "objective": objective,
        "scorer": scorer,
        "demand_years": [int(y) for y in (demand_years or [])] if has_years else [],
        "eval_years": [int(y) for y in (eval_years or [])] if has_years else [],
        "use_busy": bool(busy_p is not None),
        "n_existing_firehouses": len(existing_idx),
        "n_candidates": len(candidates),
        "n_demand_cells": len(cells),
        "n_incidents": int(len(incidents)),
        "n_incidents_dropped_far": int(n_far),
        "n_selected": len(selected),
        "n_closed": int(len(closed_df)),
        "n_opened": int(len(opened_df)),
        "threshold_min": threshold_min,
        "plan": plan_metrics,
        "baselines": baselines,
        "eval": eval_summary,
        "delta_vs_current_s": round(float(plan_et - cur_et), 1) if cur_et == cur_et else None,
        "travel": travel_meta,
        **extra,
        "honesty": (
            "Response = predicted drive + station turnout (LFB attendance is mobilise->arrive; "
            "travel-only scoring was wrong). Judged on LFB first-engine KPIs: mean <= 6 min, "
            "> 90% within 10 min. Drive comes from the chosen scorer "
            f"({scorer}); new sites use the default turnout prior, existing houses a per-station "
            "estimate when available. Discrete candidates (existing houses + demand centroids); "
            "house sites are station-ground centroids, not surveyed. No land, zoning or staffing "
            "constraints. Busy engines are only blended into reported response when use_busy=True "
            "(optimisation itself uses nearest engine)."
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


def _baseline_metrics(A, idx, weights, threshold_s, busy_p, label) -> dict[str, Any]:
    resp = posture_response(A, idx, busy_p)
    out = posture_metrics(resp, weights, threshold_s=threshold_s, label=label, n_posts=len(idx))
    out["lfb_first_engine"] = score_first_engine(resp, weights)
    return out


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
