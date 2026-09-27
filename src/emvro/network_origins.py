"""Street-network origin features for FDNY travel-time modelling.

The CAD label (``incident_travel_tm_seconds_qy``) runs from the first unit's
assignment to the *first unit's* arrival — whichever engine, ladder or squad
gets there first, not necessarily the first-due engine. Public data has no unit
GPS, so instead of committing to one guessed start we describe the plausible
origins on the drive network:

- first-due **engine** house and first-due **ladder** house (FDNY boundaries)
- the nearest 1st / 2nd / 3rd firehouses by free-flow network time
- how many houses can reach the box within 3 and 5 minutes

All shortest paths come from one multi-source Dijkstra over the OSM drive
graph (SciPy, C speed), so this scales to hundreds of thousands of incidents.
None of these features use the travel-time label.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .depots import _haversine_km

ENGINE_RE = re.compile(r"\bEngine\s+(\d+)\b", re.I)
LADDER_RE = re.compile(r"\bLadder\s+(\d+)\b", re.I)

# Cap on Dijkstra search (seconds of free-flow driving). Farther houses are
# irrelevant for first arrival and treated as unreachable.
SEARCH_LIMIT_S = 1500.0

NETWORK_FEATURE_COLUMNS = [
    "net_s_engine",
    "net_km_engine",
    "net_s_ladder",
    "net_km_ladder",
    "net_s_first_due_min",
    "net_s_nearest1",
    "net_s_nearest2",
    "net_s_nearest3",
    "net_km_nearest1",
    "net_s_engine_minus_nearest",
    "houses_within_180s",
    "houses_within_300s",
    "engine_is_nearest",
    "crow_km_engine",
    "crow_km_nearest",
    "engine_circuity",
    "dest_snap_m",
]


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
class DriveGraph:
    """CSR view of the OSM drive graph with free-flow time and length weights."""

    def __init__(self, node_ids, lat, lon, time_csr, length_csr):
        self.node_ids = np.asarray(node_ids)
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self.time = time_csr
        self.length = length_csr
        self._tree = None

    @classmethod
    def from_graphml(cls, path: Path | str, cache: Path | str | None = None) -> "DriveGraph":
        cache = Path(cache) if cache else None
        if cache is not None and cache.exists():
            z = np.load(cache, allow_pickle=False)
            n = len(z["node_ids"])
            t = sparse.csr_matrix((z["t_data"], z["t_indices"], z["t_indptr"]), shape=(n, n))
            l_ = sparse.csr_matrix((z["l_data"], z["l_indices"], z["l_indptr"]), shape=(n, n))
            return cls(z["node_ids"], z["lat"], z["lon"], t, l_)

        import osmnx as ox

        G = ox.load_graphml(Path(path))
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        nodes = list(G.nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        lat = np.array([G.nodes[n]["y"] for n in nodes], dtype=float)
        lon = np.array([G.nodes[n]["x"] for n in nodes], dtype=float)
        rows, cols, tt, ll = [], [], [], []
        for u, v, d in G.edges(data=True):
            rows.append(idx[u])
            cols.append(idx[v])
            tt.append(max(float(d.get("travel_time") or 0.0), 0.01))
            ll.append(max(float(d.get("length") or 0.0), 0.01))
        n = len(nodes)
        # Parallel edges: keep the minimum (coo→csr would sum duplicates).
        e = pd.DataFrame({"r": rows, "c": cols, "t": tt, "l": ll})
        et = e.groupby(["r", "c"], sort=False)["t"].min().reset_index()
        el = e.groupby(["r", "c"], sort=False)["l"].min().reset_index()
        t = sparse.csr_matrix((et["t"], (et["r"], et["c"])), shape=(n, n))
        l_ = sparse.csr_matrix((el["l"], (el["r"], el["c"])), shape=(n, n))
        g = cls(np.array(nodes, dtype=np.int64), lat, lon, t, l_)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                node_ids=g.node_ids, lat=lat, lon=lon,
                t_data=t.data, t_indices=t.indices, t_indptr=t.indptr,
                l_data=l_.data, l_indices=l_.indices, l_indptr=l_.indptr,
            )
        return g

    def nearest(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Nearest graph node index + snap distance (m) for each point."""
        from scipy.spatial import cKDTree

        # Local equirectangular projection (m) is plenty for NYC-scale snapping.
        k = np.cos(np.radians(40.7))
        if self._tree is None:
            self._tree = cKDTree(np.c_[self.lon * k * 111_320, self.lat * 111_320])
        q = np.c_[np.asarray(lon, float) * k * 111_320, np.asarray(lat, float) * 111_320]
        d, i = self._tree.query(q)
        return i, d


# --------------------------------------------------------------------------
# Firehouses and first-due companies
# --------------------------------------------------------------------------
def firehouse_table(firehouses: pd.DataFrame) -> pd.DataFrame:
    """One row per firehouse with the engine / ladder numbers it houses."""
    fh = firehouses.copy()
    fh["lat"] = pd.to_numeric(fh["latitude"], errors="coerce")
    fh["lon"] = pd.to_numeric(fh["longitude"], errors="coerce")
    fh = fh.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    name = fh["facilityname"].astype(str)
    fh["engines"] = name.map(lambda s: [int(x) for x in ENGINE_RE.findall(s)])
    fh["ladders"] = name.map(lambda s: [int(x) for x in LADDER_RE.findall(s)])
    fh["house_idx"] = np.arange(len(fh))
    return fh[["house_idx", "facilityname", "lat", "lon", "engines", "ladders"]]


def company_to_house(houses: pd.DataFrame, kind: str) -> dict[int, int]:
    col = "engines" if kind == "E" else "ladders"
    out: dict[int, int] = {}
    for h, nums in zip(houses["house_idx"], houses[col]):
        for n in nums:
            out.setdefault(int(n), int(h))
    return out


def assign_first_due_companies(
    points: pd.DataFrame,
    companies_geojson: Path | str,
    *,
    nearest_max_ft: float = 1500.0,
) -> pd.DataFrame:
    """First-due engine and ladder number for each (lat, lon) point.

    ``points`` needs ``dest_lat`` / ``dest_lon``; returns ``first_due_engine``
    and ``first_due_ladder`` aligned to its index.
    """
    import geopandas as gpd

    cos = gpd.read_file(companies_geojson).to_crs(2263)
    cos["num"] = pd.to_numeric(cos["fire_co_num"], errors="coerce")
    pts = gpd.GeoDataFrame(
        index=points.index,
        geometry=gpd.points_from_xy(points["dest_lon"], points["dest_lat"]),
        crs=4326,
    ).to_crs(2263)

    out = pd.DataFrame(index=points.index)
    for kind, col in (("E", "first_due_engine"), ("L", "first_due_ladder")):
        poly = cos[cos["fire_co_type"].astype(str).str.upper() == kind][["num", "geometry"]]
        j = gpd.sjoin(pts, poly, how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        num = j["num"].reindex(points.index)
        miss = num.isna()
        if miss.any():
            near = gpd.sjoin_nearest(pts.loc[miss], poly, how="left", max_distance=nearest_max_ft)
            near = near[~near.index.duplicated(keep="first")]
            num.loc[near.index] = near["num"]
        out[col] = num
    return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def house_to_dest_matrices(
    g: DriveGraph,
    house_nodes: np.ndarray,
    dest_nodes: np.ndarray,
    *,
    limit_s: float = SEARCH_LIMIT_S,
) -> tuple[np.ndarray, np.ndarray]:
    """(H × D) free-flow seconds and km from each house node to each dest node.

    The km matrix is the length of the shortest-*distance* path, which is a
    close proxy for the length of the fastest path.
    """
    t = dijkstra(g.time, directed=True, indices=house_nodes, limit=limit_s)
    km = dijkstra(g.length, directed=True, indices=house_nodes, limit=limit_s * 40.0) / 1000.0
    return t[:, dest_nodes].astype(np.float32), km[:, dest_nodes].astype(np.float32)


def network_origin_features(
    dests: pd.DataFrame,
    houses: pd.DataFrame,
    g: DriveGraph,
    engine_house: dict[int, int],
    ladder_house: dict[int, int],
) -> pd.DataFrame:
    """Network features for unique destinations.

    ``dests`` needs dest_lat, dest_lon, first_due_engine, first_due_ladder.
    Returns a frame aligned to ``dests.index``.
    """
    d_node, d_snap = g.nearest(dests["dest_lat"], dests["dest_lon"])
    h_node, _ = g.nearest(houses["lat"], houses["lon"])
    u_nodes, inv = np.unique(d_node, return_inverse=True)
    T, K = house_to_dest_matrices(g, h_node, u_nodes)
    T = T[:, inv]  # H × len(dests)
    K = K[:, inv]
    T[~np.isfinite(T)] = np.nan
    K[~np.isfinite(K)] = np.nan
    n = len(dests)
    cols = np.arange(n)

    def pick(company_col: str, mapping: dict[int, int]):
        comp = pd.to_numeric(dests[company_col], errors="coerce").to_numpy()
        h = np.array([mapping.get(int(c), -1) if c == c else -1 for c in comp])
        ok = h >= 0
        s = np.full(n, np.nan, dtype=np.float32)
        k = np.full(n, np.nan, dtype=np.float32)
        s[ok] = T[h[ok], cols[ok]]
        k[ok] = K[h[ok], cols[ok]]
        return h, s, k

    h_eng, s_eng, k_eng = pick("first_due_engine", engine_house)
    _, s_lad, k_lad = pick("first_due_ladder", ladder_house)

    Tf = np.where(np.isnan(T), np.inf, T)
    order = np.argsort(Tf, axis=0)[:3]
    srt = np.take_along_axis(Tf, order, axis=0)
    srt[~np.isfinite(srt)] = np.nan
    nearest_house = order[0]
    k_near = K[nearest_house, cols]

    dlat = dests["dest_lat"].to_numpy(float)
    dlon = dests["dest_lon"].to_numpy(float)
    hlat = houses["lat"].to_numpy(float)
    hlon = houses["lon"].to_numpy(float)
    crow_eng = np.full(n, np.nan)
    ok = h_eng >= 0
    crow_eng[ok] = _haversine_km(hlon[h_eng[ok]], hlat[h_eng[ok]], dlon[ok], dlat[ok])
    crow_near = _haversine_km(hlon[nearest_house], hlat[nearest_house], dlon, dlat)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(
            {
                "net_s_engine": s_eng,
                "net_km_engine": k_eng,
                "net_s_ladder": s_lad,
                "net_km_ladder": k_lad,
                "net_s_first_due_min": np.fmin(s_eng, s_lad),
                "net_s_nearest1": srt[0],
                "net_s_nearest2": srt[1],
                "net_s_nearest3": srt[2],
                "net_km_nearest1": k_near,
                "net_s_engine_minus_nearest": s_eng - srt[0],
                "houses_within_180s": (Tf <= 180).sum(axis=0),
                "houses_within_300s": (Tf <= 300).sum(axis=0),
                "engine_is_nearest": np.where(ok, (h_eng == nearest_house).astype(float), np.nan),
                "crow_km_engine": crow_eng,
                "crow_km_nearest": crow_near,
                "engine_circuity": np.where(crow_eng > 0.05, k_eng / crow_eng, np.nan),
                "dest_snap_m": d_snap,
            },
            index=dests.index,
        )
    return out
