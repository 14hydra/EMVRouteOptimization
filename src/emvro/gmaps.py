"""Google Maps Routes API as the civilian control travel-time model."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

# New Routes API (legacy Directions is disabled on many new projects)
GMAPS_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
# Fallback legacy endpoint (only if still enabled)
GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

_ROOT = Path(__file__).resolve().parents[2]


def load_api_key_from_dotenv(*paths: Path | str) -> str | None:
    """Load GOOGLE_MAPS_API_KEY from a local .env without printing it."""
    candidates = list(paths) or [_ROOT / ".env", Path.cwd() / ".env"]
    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in {"GOOGLE_MAPS_API_KEY", "GMAPS_API_KEY"}:
                val = v.strip().strip('"').strip("'")
                if val:
                    os.environ.setdefault(k.strip(), val)
                    return val
    return None


def decode_polyline(encoded: str) -> list[list[float]]:
    """Decode a Google encoded polyline into [[lat, lon], ...]."""
    if not encoded:
        return []
    coords: list[list[float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)
    while index < length:
        for coord_name in ("lat", "lon"):
            shift = 0
            result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if coord_name == "lat":
                lat += delta
            else:
                lon += delta
        coords.append([lat / 1e5, lon / 1e5])
    return coords


def _parse_duration_s(value: Any) -> int | None:
    """Parse Routes API duration ('123s' or number) to integer seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s.endswith("s"):
        s = s[:-1]
    try:
        return int(float(s))
    except ValueError:
        return None


class GoogleMapsControl:
    """
    Civilian Google Maps driving times (control / baseline).

    Uses the **Routes API** (`computeRoutes`) with TRAFFIC_AWARE preference.
    Falls back to legacy Directions only if Routes is unavailable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_path: Path | str | None = None,
        pause_s: float = 0.05,
        timeout_s: int = 30,
        prefer_routes_api: bool = True,
    ):
        if not api_key:
            load_api_key_from_dotenv()
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_MAPS_API_KEY")
            or os.environ.get("GMAPS_API_KEY")
        )
        self.cache_path = Path(cache_path) if cache_path else None
        self.pause_s = pause_s
        self.timeout_s = timeout_s
        self.prefer_routes_api = prefer_routes_api
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _key(
        self,
        origin: tuple[float, float],
        dest: tuple[float, float],
        mode: str,
        departure_time: str | int | None,
    ) -> str:
        o = (round(origin[0], 4), round(origin[1], 4))
        d = (round(dest[0], 4), round(dest[1], 4))
        dep = "none" if departure_time is None else str(departure_time)
        raw = f"{mode}|{o}|{d}|{dep}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))

    def _routes_api(
        self,
        origin: tuple[float, float],
        dest: tuple[float, float],
        *,
        include_polyline: bool,
    ) -> dict[str, Any]:
        field_mask = "routes.duration,routes.staticDuration,routes.distanceMeters"
        if include_polyline:
            field_mask += ",routes.polyline.encodedPolyline"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key or "",
            "X-Goog-FieldMask": field_mask,
        }
        body = {
            "origin": {
                "location": {"latLng": {"latitude": origin[0], "longitude": origin[1]}}
            },
            "destination": {
                "location": {"latLng": {"latitude": dest[0], "longitude": dest[1]}}
            },
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "computeAlternativeRoutes": False,
            "languageCode": "en-US",
            "units": "METRIC",
        }
        try:
            r = requests.post(
                GMAPS_ROUTES_URL, headers=headers, json=body, timeout=self.timeout_s
            )
            payload = r.json() if r.content else {}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "request_failed", "message": str(exc)}

        if r.status_code >= 400:
            err = payload.get("error") or {}
            return {
                "ok": False,
                "error": "routes_api_http",
                "status": r.status_code,
                "message": err.get("message") or payload.get("message") or r.text[:300],
            }
        routes = payload.get("routes") or []
        if not routes:
            return {
                "ok": False,
                "error": "routes_empty",
                "message": (payload.get("error") or {}).get("message") or "no routes",
            }
        route0 = routes[0]
        traffic_s = _parse_duration_s(route0.get("duration"))
        static_s = _parse_duration_s(route0.get("staticDuration"))
        duration_s = static_s or traffic_s
        distance_m = int(route0.get("distanceMeters") or 0)
        result: dict[str, Any] = {
            "ok": True,
            "duration_s": duration_s,
            "duration_in_traffic_s": traffic_s,
            "distance_m": distance_m,
            "status": "OK",
            "api": "routes_v2",
            "cached": False,
        }
        if include_polyline:
            enc = ((route0.get("polyline") or {}).get("encodedPolyline")) or ""
            result["polyline_latlons"] = decode_polyline(enc)
        return result

    def _directions_api(
        self,
        origin: tuple[float, float],
        dest: tuple[float, float],
        *,
        departure_time: str | int | None,
        include_polyline: bool,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{dest[0]},{dest[1]}",
            "mode": "driving",
            "key": self.api_key,
        }
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
            return {
                "ok": False,
                "error": "directions_status",
                "status": status,
                "message": payload.get("error_message") or status,
            }
        route0 = payload["routes"][0]
        leg = route0["legs"][0]
        duration_s = int(leg["duration"]["value"])
        traffic = leg.get("duration_in_traffic", {}).get("value")
        distance_m = int(leg.get("distance", {}).get("value") or 0)
        result: dict[str, Any] = {
            "ok": True,
            "duration_s": duration_s,
            "duration_in_traffic_s": int(traffic) if traffic is not None else None,
            "distance_m": distance_m,
            "status": status,
            "api": "directions_legacy",
            "cached": False,
        }
        if include_polyline:
            enc = (route0.get("overview_polyline") or {}).get("points") or ""
            result["polyline_latlons"] = decode_polyline(enc)
        return result

    def driving_seconds(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        *,
        departure_time: str | int | None = "now",
        include_polyline: bool = False,
    ) -> dict[str, Any]:
        """
        Return civilian Google Maps driving duration seconds for origin→dest.

        Keys: ok, duration_s, duration_in_traffic_s, distance_m, status, error,
        optional polyline_latlons
        """
        if not self.api_key:
            return {
                "ok": False,
                "error": "missing_api_key",
                "message": "Set GOOGLE_MAPS_API_KEY to use the Google Maps control model.",
            }

        origin = (float(origin_lat), float(origin_lon))
        dest = (float(dest_lat), float(dest_lon))
        mode_tag = "routes_poly" if include_polyline else "routes"
        ck = self._key(origin, dest, mode_tag, departure_time)
        if ck in self._cache:
            hit = dict(self._cache[ck])
            hit["cached"] = True
            return hit

        if self.prefer_routes_api:
            result = self._routes_api(origin, dest, include_polyline=include_polyline)
            if not result.get("ok"):
                legacy = self._directions_api(
                    origin,
                    dest,
                    departure_time=departure_time,
                    include_polyline=include_polyline,
                )
                if legacy.get("ok"):
                    result = legacy
        else:
            result = self._directions_api(
                origin,
                dest,
                departure_time=departure_time,
                include_polyline=include_polyline,
            )

        self._cache[ck] = {k: v for k, v in result.items() if k != "cached"}
        self._save_cache()
        if self.pause_s:
            time.sleep(self.pause_s)
        return result

    def control_travel_seconds(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        *,
        departure_time: str | int | None = "now",
        prefer_traffic: bool = True,
    ) -> float | None:
        """Civilian control seconds (traffic-aware when available)."""
        r = self.driving_seconds(
            origin_lat, origin_lon, dest_lat, dest_lon, departure_time=departure_time
        )
        if not r.get("ok"):
            return None
        if prefer_traffic and r.get("duration_in_traffic_s"):
            return float(r["duration_in_traffic_s"])
        return float(r["duration_s"]) if r.get("duration_s") else None


def classify_origin_with_gmaps(
    actual_travel_s: float,
    gmaps_station_s: float | None,
    gmaps_csl_s: float | None = None,
    *,
    emv_speed_factor: float = 0.75,
    slack: float = 1.15,
) -> dict[str, Any]:
    """Classify depot vs on-road origin using Google Maps civilian control."""
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

    candidates = [("station", expected_station)]
    if gmaps_csl_s is not None and gmaps_csl_s == gmaps_csl_s and gmaps_csl_s > 0:
        expected_csl = gmaps_csl_s * emv_speed_factor
        out["expected_emv_csl_s"] = expected_csl
        candidates.append(("csl", expected_csl))

    best = min(candidates, key=lambda kv: abs(kv[1] - actual_travel_s))
    out["best_match_origin"] = best[0]
    out["best_match_abs_error_s"] = abs(best[1] - actual_travel_s)
    return out
