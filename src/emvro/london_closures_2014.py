"""The ten London fire stations closed on 9 January 2014."""

from __future__ import annotations

import difflib
import re
from datetime import date
from typing import Iterable

import pandas as pd

closure_date = date(2014, 1, 9)

CLOSED_2014: list[str] = [
    "Belsize",
    "Bow",
    "Clerkenwell",
    "Downham",
    "Kingsland",
    "Knightsbridge",
    "Silvertown",
    "Southwark",
    "Westminster",
    "Woolwich",
]

_NOISE = re.compile(r"\b(fire|station|stn|lfb)\b")


def _norm(name: object) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", _NOISE.sub(" ", s)).strip()


def match_closed(
    names: Iterable[str], candidates: Iterable[str] = CLOSED_2014, cutoff: float = 0.85
) -> dict[str, str]:
    """Map each name to its matching closed station (exact/substring/fuzzy)."""
    cands = {_norm(c): c for c in candidates}
    out: dict[str, str] = {}
    for name in names:
        n = _norm(name)
        if not n:
            continue
        hit = cands.get(n) or next(
            (orig for key, orig in cands.items() if re.search(rf"\b{re.escape(key)}\b", n)), None
        )
        if hit is None:
            close = difflib.get_close_matches(n, list(cands), n=1, cutoff=cutoff)
            hit = cands[close[0]] if close else None
        if hit is not None:
            out[name] = hit
    return out


def flag_closed(df: pd.DataFrame, name_col: str = "facilityname", cutoff: float = 0.85) -> pd.Series:
    """Boolean Series: True where ``df[name_col]`` matches a 2014-closed station."""
    matches = match_closed(df[name_col].dropna().unique(), cutoff=cutoff)
    return df[name_col].isin(matches)


# APPROXIMATE (lat, lon) of the ten stations closed on 9 Jan 2014. Hand-entered
# from general knowledge of each station's neighbourhood (good to ~300-500 m,
# NOT surveyed). Fine for crow-fly planning-scale scoring; do not cite as
# authoritative site coordinates.
CLOSED_2014_APPROX_COORDS: dict[str, tuple[float, float]] = {
    "Belsize": (51.5486, -0.1650),
    "Bow": (51.5290, -0.0210),
    "Clerkenwell": (51.5265, -0.1085),
    "Downham": (51.4262, 0.0130),
    "Kingsland": (51.5459, -0.0759),
    "Knightsbridge": (51.5018, -0.1600),
    "Silvertown": (51.5030, 0.0300),
    "Southwark": (51.5025, -0.0985),
    "Westminster": (51.4975, -0.1325),
    "Woolwich": (51.4905, 0.0640),
}
