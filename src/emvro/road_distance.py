"""Shortest road distance on London's directed drive network (one-way streets respected).

The graph is OpenStreetMap's drivable network for Greater London (Geofabrik extract,
filtered with the same rules as osmnx ``network_type="drive"``).
osmnx stores a one-way street as a single directed edge, so Dijkstra can only
travel it in its legal direction; two-way streets get an edge each way.

Routing assumption (until a route model exists): engines take the shortest-
distance route. Emergency exemptions (e.g. driving against a one-way street or
through a bus gate) are not modelled.

Usage::

    G = load_or_build_graph()
    router = RoadRouter(G)
    km = router.road_km(start_lat, start_lon, dest_lat, dest_lon)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LONDON_RAW = ROOT / "data" / "raw" / "london"
GRAPH_PATH = LONDON_RAW / "london_drive.graphml"
# Geofabrik's Greater London extract (one dated file; more reliable than the
# public Overpass API for an area this size).
PBF_URL = "https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf"
PBF_PATH = LONDON_RAW / "greater-london-latest.osm.pbf"

# Same rules as osmnx network_type="drive" (osmnx._overpass._get_network_filter).
# Overpass "!~" is an unanchored regex match, so re.search reproduces it.
DRIVE_EXCLUDE = {
    "area": "yes",
    "access": "private",
    "highway": (
        "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|"
        "path|pedestrian|planned|platform|proposed|raceway|razed|rest_area|service|services|steps|track"
    ),
    "motor_vehicle": "no",
    "motorcar": "no",
    "service": "alley|driveway|emergency_access|parking|parking_aisle|private",
}


def _is_drivable(tags) -> bool:
    import re

    if "highway" not in tags:
        return False
    return not any(k in tags and re.search(pat, tags[k]) for k, pat in DRIVE_EXCLUDE.items())


def _pbf_to_drive_xml(pbf: Path, xml: Path) -> None:
    """Write only drivable ways (all their tags, incl. oneway) and their nodes to OSM XML."""
    import osmium

    ways, node_ids = [], set()
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        if _is_drivable(w.tags):
            ways.append(w.replace(tags=dict(w.tags), nodes=[n.ref for n in w.nodes]))
            node_ids.update(n.ref for n in w.nodes)
    if xml.exists():
        xml.unlink()
    writer = osmium.SimpleWriter(str(xml))
    for n in osmium.FileProcessor(str(pbf), osmium.osm.NODE):
        if n.id in node_ids:
            writer.add_node(n)
    for w in ways:
        writer.add_way(w)
    writer.close()


def load_or_build_graph(path: Path = GRAPH_PATH):
    """Directed OSM drive graph, restricted to its largest strongly connected component.

    osmnx builds the graph from the filtered XML with its normal one-way rules:
    oneway=yes/true/1 and roundabouts get one forward edge, oneway=-1/reverse one
    backward edge, everything else an edge each way. Strong connectivity then
    guarantees every node can reach every other along legal streets.
    """
    import osmnx as ox

    if path.exists():
        return ox.load_graphml(path)
    if not PBF_PATH.exists():
        import urllib.request

        PBF_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PBF_URL, PBF_PATH)
    xml = PBF_PATH.with_name("london_drive.osm")
    _pbf_to_drive_xml(PBF_PATH, xml)
    G = ox.graph_from_xml(xml, bidirectional=False, simplify=True, retain_all=False)
    G = ox.truncate.largest_component(G, strongly=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    xml.unlink()
    return G


class RoadRouter:
    """Shortest-distance routing on a directed graph via one Dijkstra per origin."""

    def __init__(self, G):
        from scipy.sparse import csr_matrix

        self.G = G
        self.nodes = np.array(list(G.nodes))
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.node_lat = np.array([G.nodes[n]["y"] for n in self.nodes])
        self.node_lon = np.array([G.nodes[n]["x"] for n in self.nodes])
        # Directed edges u -> v only (one-way streets have no v -> u edge).
        # Parallel edges between the same pair: keep the shortest.
        best: dict[tuple[int, int], float] = {}
        for u, v, d in G.edges(data=True):
            key = (self.index[u], self.index[v])
            length = float(d.get("length", 0.0))
            if key not in best or length < best[key]:
                best[key] = length
        rows, cols = np.array(list(best)).T
        n = len(self.nodes)
        self.csr = csr_matrix((np.fromiter(best.values(), float), (rows, cols)), shape=(n, n))

    def snap(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Nearest graph node for each point, and the straight-line metres to it."""
        from sklearn.neighbors import BallTree

        if not hasattr(self, "_tree"):
            self._tree = BallTree(np.radians(np.c_[self.node_lat, self.node_lon]), metric="haversine")
        dist, idx = self._tree.query(np.radians(np.c_[lat, lon]), k=1)
        return idx[:, 0], dist[:, 0] * 6_371_008.8

    def road_km(self, start_lat, start_lon, dest_lat, dest_lon) -> np.ndarray:
        """Shortest legal road distance (km) from each start to its destination.

        Includes the straight-line hop from each point to its snapped graph node.
        """
        from scipy.sparse.csgraph import dijkstra

        s_idx, s_snap = self.snap(np.asarray(start_lat), np.asarray(start_lon))
        d_idx, d_snap = self.snap(np.asarray(dest_lat), np.asarray(dest_lon))
        out = np.full(len(s_idx), np.nan)
        origins = np.unique(s_idx)
        # One single-source Dijkstra per distinct origin (a station), reused for all its trips.
        for chunk in np.array_split(origins, max(1, len(origins) // 16)):
            dist = dijkstra(self.csr, directed=True, indices=chunk)
            for row, o in enumerate(chunk):
                mask = s_idx == o
                out[mask] = dist[row, d_idx[mask]]
        return (out + s_snap + d_snap) / 1000.0
