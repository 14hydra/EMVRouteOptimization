"""City registry for multi-city EMV planning.

NYC is fully wired from repo data paths. Other cities need explicit
``--incidents`` / ``--firehouses`` / ``--graph`` overrides (or matching files
under ``data/``); ``get_city`` raises a clear error otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CitySpec:
    """Resolved paths + metadata for one planning city."""

    id: str
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    firehouses_path: Path | None = None
    incidents_path: Path | None = None
    graph_path: Path | None = None
    hybrid_model_path: Path | None = None
    default_n_houses: int | None = None
    center: tuple[float, float] = (0.0, 0.0)  # lon, lat
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def require_paths(self, *, need_graph: bool = True) -> None:
        missing = []
        if self.firehouses_path is None or not Path(self.firehouses_path).exists():
            missing.append("firehouses")
        if self.incidents_path is None or not Path(self.incidents_path).exists():
            missing.append("incidents")
        if need_graph and (self.graph_path is None or not Path(self.graph_path).exists()):
            missing.append("graph")
        if missing:
            raise FileNotFoundError(
                f"City {self.id!r} is missing data for: {', '.join(missing)}. "
                f"Pass --firehouses / --incidents / --graph, or add files under data/."
            )


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _nyc_spec(root: Path = ROOT) -> CitySpec:
    raw = root / "data" / "raw"
    big = raw / "big"
    processed = root / "data" / "processed"
    rich = raw / "nyc_drive_rich.graphml"
    plain = raw / "nyc_drive.graphml"
    firehouses = _first_existing(big / "fdny_firehouses.csv", raw / "fdny_firehouses.csv")
    incidents = _first_existing(
        processed / "travel_time_network.parquet",
        processed / "od_pairs_inferred_starts.csv",
    )
    graph = _first_existing(rich, plain)
    hybrid = _first_existing(processed / "models" / "travel_time_hybrid.joblib")
    return CitySpec(
        id="nyc",
        name="New York City",
        bbox=(-74.26, 40.49, -73.70, 40.92),
        firehouses_path=firehouses,
        incidents_path=incidents,
        graph_path=graph,
        hybrid_model_path=hybrid,
        default_n_houses=None,  # filled from firehouse count at load time
        center=(-73.95, 40.75),
        notes="FDNY firehouses + network/OD demand; firetruck EMV graph",
        meta={"demand_lat": "dest_lat", "demand_lon": "dest_lon"},
    )


def _sf_stub(root: Path = ROOT) -> CitySpec:
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    return CitySpec(
        id="sf",
        name="San Francisco",
        bbox=(-122.52, 37.70, -122.35, 37.83),
        firehouses_path=_first_existing(raw / "sf_firehouses.csv", processed / "sf_firehouses.csv"),
        incidents_path=_first_existing(processed / "od_pairs_sf_ems.csv"),
        graph_path=_first_existing(raw / "sf_drive.graphml"),
        hybrid_model_path=None,
        default_n_houses=None,
        center=(-122.42, 37.77),
        notes="Stub: provide firehouses CSV (lat/lon) to plan; SF EMS OD is medical, not FDNY",
        meta={"demand_lat": "dest_lat", "demand_lon": "dest_lon"},
    )


CITY_BUILDERS = {
    "nyc": _nyc_spec,
    "new_york": _nyc_spec,
    "new_york_city": _nyc_spec,
    "sf": _sf_stub,
    "san_francisco": _sf_stub,
}


def list_cities() -> list[str]:
    return sorted({("nyc" if k.startswith("new") else k) for k in ("nyc", "sf")})


def get_city(city_id: str, *, root: Path = ROOT) -> CitySpec:
    key = str(city_id).strip().lower().replace(" ", "_").replace("-", "_")
    builder = CITY_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"Unknown city {city_id!r}. Known: {list_cities()}. "
            "For a custom city pass --city custom with --firehouses/--incidents/--graph."
        )
    return builder(root)


def custom_city(
    *,
    city_id: str = "custom",
    name: str | None = None,
    firehouses: Path,
    incidents: Path,
    graph: Path | None = None,
    hybrid_model: Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
) -> CitySpec:
    fh = Path(firehouses)
    inc = Path(incidents)
    return CitySpec(
        id=city_id,
        name=name or city_id,
        bbox=bbox or (-180.0, -90.0, 180.0, 90.0),
        firehouses_path=fh,
        incidents_path=inc,
        graph_path=Path(graph) if graph else None,
        hybrid_model_path=Path(hybrid_model) if hybrid_model else None,
        center=center or (0.0, 0.0),
        notes="Custom city paths from CLI",
    )


def load_firehouses(spec: CitySpec) -> pd.DataFrame:
    """Normalize firehouse table to facilityname, lat, lon [, borough]."""
    if spec.firehouses_path is None:
        raise FileNotFoundError(f"No firehouses path for city {spec.id}")
    path = Path(spec.firehouses_path)
    df = pd.read_csv(path)
    # NYC Open Data schema
    if "latitude" in df.columns and "longitude" in df.columns:
        out = pd.DataFrame(
            {
                "facilityname": df.get("facilityname", df.get("name", "firehouse")).astype(str),
                "lat": pd.to_numeric(df["latitude"], errors="coerce"),
                "lon": pd.to_numeric(df["longitude"], errors="coerce"),
                "borough": df["borough"].astype(str) if "borough" in df.columns else None,
            }
        )
    elif "lat" in df.columns and "lon" in df.columns:
        out = pd.DataFrame(
            {
                "facilityname": df.get("facilityname", df.get("name", "firehouse")).astype(str),
                "lat": pd.to_numeric(df["lat"], errors="coerce"),
                "lon": pd.to_numeric(df["lon"], errors="coerce"),
                "borough": df["borough"].astype(str) if "borough" in df.columns else None,
            }
        )
    else:
        raise ValueError(f"Firehouses file needs lat/lon or latitude/longitude: {path}")
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    out.insert(0, "house_id", out.index.astype(int))
    return out


def load_incidents(spec: CitySpec, *, max_rows: int | None = 250_000) -> pd.DataFrame:
    """Load demand points with dest_lat / dest_lon."""
    if spec.incidents_path is None:
        raise FileNotFoundError(f"No incidents path for city {spec.id}")
    path = Path(spec.incidents_path)
    lat_c = spec.meta.get("demand_lat", "dest_lat")
    lon_c = spec.meta.get("demand_lon", "dest_lon")

    if path.suffix.lower() == ".parquet":
        cols = None
        try:
            import pyarrow.parquet as pq

            schema_names = set(pq.read_schema(path).names)
            want = [c for c in (lat_c, lon_c, "travel_seconds", "start_lat", "start_lon", "borough") if c in schema_names]
            # fall back aliases
            for a, b in (("dest_lat", "latitude"), ("dest_lon", "longitude")):
                if a not in schema_names and b in schema_names:
                    want.append(b)
            df = pd.read_parquet(path, columns=list(dict.fromkeys(want)) or None)
        except Exception:  # noqa: BLE001
            df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, nrows=max_rows)

    if lat_c not in df.columns and "latitude" in df.columns:
        lat_c = "latitude"
    if lon_c not in df.columns and "longitude" in df.columns:
        lon_c = "longitude"
    if lat_c not in df.columns or lon_c not in df.columns:
        raise ValueError(f"Incidents need {lat_c}/{lon_c} (or lat/lon): {path}")

    out = pd.DataFrame(
        {
            "dest_lat": pd.to_numeric(df[lat_c], errors="coerce"),
            "dest_lon": pd.to_numeric(df[lon_c], errors="coerce"),
        }
    )
    for c in ("travel_seconds", "start_lat", "start_lon", "borough"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c != "borough" else df[c]
    out = out.dropna(subset=["dest_lat", "dest_lon"])
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(max_rows, random_state=42)
    return out.reset_index(drop=True)


def with_overrides(
    spec: CitySpec,
    *,
    firehouses: Path | None = None,
    incidents: Path | None = None,
    graph: Path | None = None,
    hybrid_model: Path | None = None,
) -> CitySpec:
    kw = {}
    if firehouses is not None:
        kw["firehouses_path"] = Path(firehouses)
    if incidents is not None:
        kw["incidents_path"] = Path(incidents)
    if graph is not None:
        kw["graph_path"] = Path(graph)
    if hybrid_model is not None:
        kw["hybrid_model_path"] = Path(hybrid_model)
    return replace(spec, **kw) if kw else spec
