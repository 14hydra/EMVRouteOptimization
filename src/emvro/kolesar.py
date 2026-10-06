"""Kolesar et al. (1975) piecewise emergency-vehicle travel-time model.

Kolesar, Walker & Hausner (1975), "Determining the relation between fire engine
travel times and travel distances in New York City", Operations Research 23(4).

Travel time ``T`` (seconds) as a function of distance ``d`` (km)::

    T = a + b * sqrt(d)     for d <= breakpoint   (still accelerating)
    T = c + e * d           for d >  breakpoint   (cruising speed)

Distances are km. :func:`fit_kolesar` refits the curve on local trips;
``KolesarModel.london_default`` is that fit on London Fire Brigade trips and is
only a fallback when a caller has too few trips to fit its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Fit on 701,000 LFB engine trips 2021-2024 (scripts/train_travel_time_lfb.py):
# real station origin, straight-line km, recorded driving time (TravelTimeSeconds).
LONDON_DEFAULT_KM: dict[str, float] = {
    "a": 15.846,
    "b": 180.913,
    "c": 145.697,
    "e": 71.783,
    "breakpoint_km": 1.913,
}


@dataclass(frozen=True)
class KolesarModel:
    """Piecewise travel-time model; distances in km, times in seconds."""

    a: float
    b: float
    c: float
    e: float
    breakpoint_km: float

    def predict(self, distance_km: Any) -> np.ndarray:
        """Predict travel seconds for scalar/array distance in km."""
        d = np.maximum(np.asarray(distance_km, dtype=float), 0.0)
        short = self.a + self.b * np.sqrt(d)
        long_ = self.c + self.e * d
        return np.where(d <= self.breakpoint_km, short, long_)

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, params: Mapping[str, float]) -> "KolesarModel":
        return cls(
            a=float(params["a"]),
            b=float(params["b"]),
            c=float(params["c"]),
            e=float(params["e"]),
            breakpoint_km=float(params["breakpoint_km"]),
        )

    @classmethod
    def london_default(cls) -> "KolesarModel":
        """London fallback curve (see ``LONDON_DEFAULT_KM``)."""
        return cls.from_dict(LONDON_DEFAULT_KM)


def _lstsq(x: np.ndarray, y: np.ndarray, robust: bool, n_iter: int = 20) -> np.ndarray:
    """Fit y ~ X @ beta; optionally Huber-IRLS for outlier resistance."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    if not robust:
        return beta
    for _ in range(n_iter):
        r = y - x @ beta
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-9)
        u = np.abs(r) / (1.345 * scale)
        w = np.where(u <= 1.0, 1.0, 1.0 / u)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
    return beta


def fit_kolesar(
    distance_km: Any,
    travel_s: Any,
    *,
    robust: bool = True,
    n_breakpoints: int = 25,
    min_segment: int = 20,
) -> KolesarModel:
    """Refit (a, b, c, e, breakpoint) by grid search over the breakpoint.

    For each candidate breakpoint (quantiles of distance) the short segment is
    fit as ``T ~ 1, sqrt(d)`` and the long segment as ``T ~ 1, d`` (Huber IRLS
    if ``robust``). The breakpoint with the lowest absolute-error sum wins.
    Falls back to a single linear fit (breakpoint = 0) when data is too small.
    """
    d = np.asarray(distance_km, dtype=float)
    t = np.asarray(travel_s, dtype=float)
    ok = np.isfinite(d) & np.isfinite(t) & (d > 0) & (t > 0)
    d, t = d[ok], t[ok]
    if d.size < 2 * min_segment:
        if d.size < 2:
            raise ValueError("Need at least 2 valid (distance, time) pairs.")
        c, e = _lstsq(np.column_stack([np.ones_like(d), d]), t, robust)
        return KolesarModel(a=float(c), b=0.0, c=float(c), e=float(e), breakpoint_km=0.0)

    best: tuple[float, KolesarModel] | None = None
    for bp in np.unique(np.quantile(d, np.linspace(0.05, 0.6, n_breakpoints))):
        lo, hi = d <= bp, d > bp
        if lo.sum() < min_segment or hi.sum() < min_segment:
            continue
        a, b = _lstsq(np.column_stack([np.ones(lo.sum()), np.sqrt(d[lo])]), t[lo], robust)
        c, e = _lstsq(np.column_stack([np.ones(hi.sum()), d[hi]]), t[hi], robust)
        model = KolesarModel(float(a), float(b), float(c), float(e), float(bp))
        loss = float(np.abs(t - model.predict(d)).sum())
        if best is None or loss < best[0]:
            best = (loss, model)
    if best is None:
        raise ValueError("Could not find a valid breakpoint; check input data.")
    return best[1]


def fit_kolesar_from_frame(
    df: pd.DataFrame, dist_col: str, time_col: str, **kwargs: Any
) -> KolesarModel:
    """Convenience wrapper: fit from DataFrame columns (km, seconds)."""
    return fit_kolesar(df[dist_col].to_numpy(), df[time_col].to_numpy(), **kwargs)
