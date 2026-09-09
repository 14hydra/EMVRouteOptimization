"""Google Maps Directions as the civilian control travel-time model."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"


class GoogleMapsControl:
    """
    Civilian Google Maps driving times (control / baseline).

    Used to decide whether an observed EMS travel time is plausible from an
    official depot, or so fast that the unit must already have been on the road.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_path: Path | str | None = None,
        pause_s: float = 0.05,
        timeout_s: int = 30,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GMAPS_API_KEY")
        self.cache_path = Path(cache_path) if cache_path else None
        self.pause_s = pause_s
        self.timeout_s = timeout_s
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _key(self, origin: tuple[float, float], dest: tuple[float, float], mode: str) -> str:
        # Round to ~11m so nearby queries share cache entries.
        o = (round(origin[0], 4), round(origin[1], 4))
        d = (round(dest[0], 4), round(dest[1], 4))
        raw = f"{mode}|{o}|{d}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))

    def driving_seconds(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        *,
        departure_time: str | int | None = "now",
    ) -> dict[str, Any]:
        """
        Return civilian Google Maps driving duration seconds for origin→dest.

        Keys: ok, duration_s, duration_in_traffic_s, distance_m, status, error
        """
        if not self.api_key:
            return {
                "ok": False,
                "error": "missing_api_key",
                "message": "Set GOOGLE_MAPS_API_KEY to use the Google Maps control model.",
            }

        origin = (float(origin_lat), float(origin_lon))
        dest = (float(dest_lat), float(dest_lon))
        ck = self._key(origin, dest, "driving")
        if ck in self._cache:
            hit = dict(self._cache[ck])
            hit["cached"] = True
            return hit

        params: dict[str, Any] = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{dest[0]},{dest[1]}",
            "mode": "driving",
            "key": self.api_key,
        }
        # Traffic-aware ETA when supported (Directions API).
        if departure_time is not None:
            params["departure_time"] = departure_time

        try:
            r = requests.get(GMAPS_DIRECTIONS_URL, params=params, timeout=self.timeout_s)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "request_failed", "message": str(exc)}

        status = payload.get("status")
        if status != "OK" or not payload.get("routes"):
            result = {
                "ok": False,
                "error": "directions_status",
                "status": status,
                "message": payload.get("error_message") or status,
            }
            self._cache[ck] = result
            self._save_cache()
            return result

        leg = payload["routes"][0]["legs"][0]
        duration_s = int(leg["duration"]["value"])
        traffic = leg.get("duration_in_traffic", {}).get("value")
        distance_m = int(leg.get("distance", {}).get("value") or 0)
        result = {
            "ok": True,
            "duration_s": duration_s,
            "duration_in_traffic_s": int(traffic) if traffic is not None else None,
            "distance_m": distance_m,
            "status": status,
            "cached": False,
        }
        self._cache[ck] = {k: v for k, v in result.items() if k != "cached"}
        self._save_cache()
        if self.pause_s:
            time.sleep(self.pause_s)
        return result


def classify_origin_with_gmaps(
    actual_travel_s: float,
    gmaps_station_s: float | None,
    gmaps_csl_s: float | None = None,
    *,
    emv_speed_factor: float = 0.75,
    slack: float = 1.15,
) -> dict[str, Any]:
    """
    Decide whether the observed EMS travel time is plausible from an official depot.

    Logic (Google Maps = civilian control):
      expected_emv_from_station ≈ gmaps_station_s * emv_speed_factor
      If even that EMV-adjusted station trip is slower than actual * slack,
      the unit could not have left the station → on_road_required.
      Else station_plausible.

    Optional: if CSL GMaps time is available, also report which origin's
    EMV-adjusted ETA is closer to the observed travel time.
    """
    out: dict[str, Any] = {
        "actual_travel_s": actual_travel_s,
        "gmaps_station_s": gmaps_station_s,
        "gmaps_csl_s": gmaps_csl_s,
        "emv_speed_factor": emv_speed_factor,
        "slack": slack,
    }
    if actual_travel_s is None or actual_travel_s != actual_travel_s or actual_travel_s <= 0:
        out.update(origin_class="unknown", reason="missing_or_invalid_actual_travel_time")
        return out
    if gmaps_station_s is None or gmaps_station_s != gmaps_station_s or gmaps_station_s <= 0:
        out.update(origin_class="unknown", reason="missing_gmaps_station_time")
        return out

    expected_station = gmaps_station_s * emv_speed_factor
    out["expected_emv_station_s"] = expected_station

    # Too fast to have come from the station, even as an EMV.
    if expected_station > actual_travel_s * slack:
        out["origin_class"] = "on_road_required"
        out["reason"] = (
            "EMV-adjusted Google Maps time from nearest official depot exceeds "
            "observed travel time → station origin implausible"
        )
    else:
        out["origin_class"] = "station_plausible"
        out["reason"] = (
            "Observed travel time is compatible with an EMV leaving the nearest "
            "official depot under the Google Maps civilian control"
        )

    # Which candidate better matches observed time (control comparison).
    candidates = [("station", expected_station)]
    if gmaps_csl_s is not None and gmaps_csl_s == gmaps_csl_s and gmaps_csl_s > 0:
        expected_csl = gmaps_csl_s * emv_speed_factor
        out["expected_emv_csl_s"] = expected_csl
        candidates.append(("csl", expected_csl))

    best = min(candidates, key=lambda kv: abs(kv[1] - actual_travel_s))
    out["best_match_origin"] = best[0]
    out["best_match_abs_error_s"] = abs(best[1] - actual_travel_s)
    return out
