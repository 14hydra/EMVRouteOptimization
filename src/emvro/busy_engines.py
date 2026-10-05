"""Estimate engine "busy" rates from LFB incident data.

An engine's home station is considered busy when the first pump was deployed
from a different station than the incident's ground station
(``FirstPumpArriving_DeployedFromStation`` != ``station_ground``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def estimate_busy_rates(
    incidents_df: pd.DataFrame,
    ground_col: str = "station_ground",
    deployed_col: str = "deployed_from_station",
    hour_col: str = "hour",
) -> dict[str, pd.DataFrame]:
    """Busy rate per ground station (and per station-hour if ``hour`` exists).

    Rows with a null in either station column are dropped. Returns a dict with
    ``by_station`` (station_ground, n_incidents, n_busy, busy_prob) and, when
    the hour column is present, ``by_station_hour``.
    """
    df = incidents_df.dropna(subset=[ground_col, deployed_col]).copy()
    df["busy"] = (df[ground_col].astype(str).str.strip() != df[deployed_col].astype(str).str.strip())

    def _agg(keys: list[str]) -> pd.DataFrame:
        out = (
            df.groupby(keys, as_index=False)["busy"]
            .agg(n_incidents="size", n_busy="sum")
            .astype({"n_busy": int})
        )
        out["busy_prob"] = out["n_busy"] / out["n_incidents"]
        return out.rename(columns={ground_col: "station_ground"})

    result = {"by_station": _agg([ground_col])}
    if hour_col in df.columns:
        result["by_station_hour"] = _agg([ground_col, hour_col])
    return result


def expected_first_arrival(t_local: Any, busy_prob: Any, backup_times: Any) -> np.ndarray:
    """Soft expected first-arrival time: ``(1 - p) * t_local + p * t_backup``.

    ``t_local`` may be a matrix (stations x points) or a 1-D row; ``busy_prob``
    is a scalar or per-station vector (broadcast down rows); ``backup_times``
    is a scalar or array broadcastable to ``t_local``.
    """
    t = np.asarray(t_local, dtype=float)
    p = np.clip(np.asarray(busy_prob, dtype=float), 0.0, 1.0)
    if p.ndim == 1 and t.ndim == 2:
        p = p[:, None]
    b = np.asarray(backup_times, dtype=float)
    return (1.0 - p) * t + p * b
