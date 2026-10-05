"""London Fire Brigade (LFB, 2022) attendance standards and scoring.

Standards (LFB London Safety Plan 2022):
  * First engine: average 6 min; 90th percentile within 10 min
    (share within 10 min should be > 0.90).
  * Second engine: average 8 min; 95th percentile within 12 min.

Data caveat: the public LFB incident CSV only provides
``FirstPumpArriving_AttendanceTime`` (mobilise -> arrive), i.e. turnout and
travel are not separable. A true turnout-vs-drive split requires LFB
mobilisation records when available; until then :data:`DEFAULT_TURNOUT_S` is
a fallback prior for new/hypothetical sites.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FIRST_ENGINE_AVG_S = 6 * 60
FIRST_ENGINE_P90_S = 10 * 60
SECOND_ENGINE_AVG_S = 8 * 60
SECOND_ENGINE_P95_S = 12 * 60

FIRST_ENGINE_TARGET_SHARE = 0.90
SECOND_ENGINE_TARGET_SHARE = 0.95

DEFAULT_TURNOUT_S = 60.0
MAX_TURNOUT_S = 300.0


def attendance_seconds(drive_s: Any, turnout_s: Any = DEFAULT_TURNOUT_S) -> np.ndarray:
    """Attendance = drive + turnout, with drive >= 0 and turnout in [0, 300] s."""
    drive = np.maximum(np.asarray(drive_s, dtype=float), 0.0)
    turnout = np.clip(np.asarray(turnout_s, dtype=float), 0.0, MAX_TURNOUT_S)
    return drive + turnout


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w) / w.sum()
    return float(x[min(np.searchsorted(cum, q), x.size - 1)])


def score_first_engine(response_s: Any, weights: Any = None) -> dict[str, float | bool]:
    """Score first-engine response times (seconds) against LFB 2022 standards.

    ``weights`` optionally weights each incident (e.g. by risk or volume).
    Returns mean_s, p90_s, share_within_6min, share_within_10min,
    meets_avg_6, meets_90pct_10 and overall ``passes``.
    """
    x = np.asarray(response_s, dtype=float).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != x.shape:
        raise ValueError("weights must match response_s in length.")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if x.size == 0:
        raise ValueError("No valid response times.")

    mean_s = float(np.average(x, weights=w))
    p90_s = _weighted_quantile(x, w, 0.90)
    within6 = float(w[x <= FIRST_ENGINE_AVG_S].sum() / w.sum())
    within10 = float(w[x <= FIRST_ENGINE_P90_S].sum() / w.sum())
    meets_avg = mean_s <= FIRST_ENGINE_AVG_S
    meets_p90 = within10 > FIRST_ENGINE_TARGET_SHARE
    return {
        "mean_s": mean_s,
        "p90_s": p90_s,
        "share_within_6min": within6,
        "share_within_10min": within10,
        "meets_avg_6": bool(meets_avg),
        "meets_90pct_10": bool(meets_p90),
        "passes": bool(meets_avg and meets_p90),
    }
