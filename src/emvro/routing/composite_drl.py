"""MODEL 2 — Composite Deep Reinforcement Learning.

Composite reward = −EMV travel seconds
                 + street-quality bonuses (primary / bus-lane proxies)
                 + potential-based shaping toward the destination
                 − revisit / wander penalties.

Training is restricted to a **corridor subgraph** around the OD pair so
tabular Q-learning can actually learn (full NYC is far too large).

Upgrade path: replace the Q-table with a DQN / graph neural policy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable

import networkx as nx
import numpy as np

from .graph import RouteResult, path_stats


def _pick_edge(G, u, v) -> dict:
    edata = G.get_edge_data(u, v) or {}
    if not edata:
        return {}
    return min(edata.values(), key=lambda d: d.get("emv_s", 1e18))


def _street_bonus(data: dict) -> float:
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    hw = str(hw or "").lower()
    bonus = 0.0
    if hw in {"motorway", "motorway_link", "trunk", "trunk_link"}:
        bonus += 1.2
    elif hw in {"primary", "primary_link"}:
        bonus += 0.9
    elif hw in {"secondary", "secondary_link"}:
        bonus += 0.4
    if data.get("busway") or data.get("lanes:bus"):
        bonus += 1.5
    return bonus


def _composite_edge_cost(data: dict) -> float:
    """Lower is better — used for Dijkstra potentials / subgraph weights."""
    t = float(data.get("emv_s") or data.get("travel_time") or 1.0)
    return max(0.05, t - 0.35 * _street_bonus(data))


def annotate_composite_weights(G) -> None:
    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight_composite"] = _composite_edge_cost(data)


def extract_corridor_subgraph(
    G,
    origin,
    dest,
    *,
    pad_deg: float = 0.012,
    max_nodes: int = 4500,
) -> nx.MultiDiGraph:
    """
    Keep nodes in a padded bounding box around origin→dest, always including
    a seed Dijkstra path so the subgraph stays connected for the OD.
    """
    annotate_composite_weights(G)
    try:
        seed = nx.shortest_path(G, origin, dest, weight="weight_composite")
    except Exception:  # noqa: BLE001
        seed = [origin, dest]

    def _xy(n):
        d = G.nodes[n]
        return float(d.get("y", d.get("lat", 0.0))), float(d.get("x", d.get("lon", 0.0)))

    lats, lons = zip(*[_xy(n) for n in seed if n in G])
    o_lat, o_lon = _xy(origin)
    d_lat, d_lon = _xy(dest)
    # Scale pad with crow-flies span so longer trips (e.g. Queens) stay connected
    span = max(abs(o_lat - d_lat), abs(o_lon - d_lon), 0.01)
    pad = max(pad_deg, 0.35 * span)
    min_lat = min(min(lats), o_lat, d_lat) - pad
    max_lat = max(max(lats), o_lat, d_lat) + pad
    min_lon = min(min(lons), o_lon, d_lon) - pad
    max_lon = max(max(lons), o_lon, d_lon) + pad

    keep = set(seed)
    for n, data in G.nodes(data=True):
        lat = float(data.get("y", data.get("lat", 0.0)))
        lon = float(data.get("x", data.get("lon", 0.0)))
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            keep.add(n)

    if len(keep) > max_nodes:
        mid_lat, mid_lon = (o_lat + d_lat) / 2.0, (o_lon + d_lon) / 2.0
        ranked = sorted(
            keep,
            key=lambda n: (_xy(n)[0] - mid_lat) ** 2 + (_xy(n)[1] - mid_lon) ** 2,
        )
        keep = set(ranked[:max_nodes]) | set(seed)

    H = G.subgraph(keep).copy()
    # Drop components that don't contain both OD ends (prevents dead-end traps)
    if origin in H and dest in H:
        und = H.to_undirected(as_view=True)
        if not nx.has_path(und, origin, dest):
            # Fall back to a thick corridor around the seed path only
            keep = set(seed)
            for n in seed:
                keep.update(list(G.neighbors(n)))
                keep.update(list(G.predecessors(n)) if hasattr(G, "predecessors") else [])
            H = G.subgraph(keep).copy()
    return H


def _complete_to_dest(G, path: list, dest, *, weight: str = "weight_composite") -> list:
    """Always finish at dest using Dijkstra on G when needed."""
    if not path:
        return path
    path = _dedupe(path)
    if path[-1] == dest:
        return path
    annotate_composite_weights(G)
    try:
        rest = nx.shortest_path(G, path[-1], dest, weight=weight)
        return _dedupe(path + rest[1:])
    except Exception:  # noqa: BLE001
        try:
            return list(nx.shortest_path(G, path[0], dest, weight=weight))
        except Exception:  # noqa: BLE001
            return path


def _is_complete(path: list, origin, dest) -> bool:
    return bool(path) and path[0] == origin and path[-1] == dest and len(path) >= 2


class CompositeQAgent:
    """Tabular Q-learning with destination potential shaping on a corridor graph."""

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 0.97,
        epsilon: float = 0.35,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q: dict[tuple[Hashable, Hashable], float] = defaultdict(float)
        self.remaining: dict[Hashable, float] = {}
        self.best_path: list = []
        self.best_time: float = 1e18
        self.train_curve: list[float] = []

    def _build_potentials(self, G, dest) -> None:
        """Remaining composite cost-to-go from every node (Dijkstra reverse)."""
        # Reverse graph shortest paths to dest ≡ remaining cost from each node
        try:
            self.remaining = dict(
                nx.single_source_dijkstra_path_length(G.reverse(copy=False), dest, weight="weight_composite")
            )
        except Exception:  # noqa: BLE001
            # Digraph reverse may fail on MultiDiGraph in some nx versions
            Gr = nx.DiGraph()
            for u, v, _, data in G.edges(keys=True, data=True):
                w = float(data.get("weight_composite") or 1.0)
                if Gr.has_edge(v, u):
                    Gr[v][u]["weight"] = min(Gr[v][u]["weight"], w)
                else:
                    Gr.add_edge(v, u, weight=w)
            self.remaining = dict(nx.single_source_dijkstra_path_length(Gr, dest, weight="weight"))

    def _actions(self, G, node, visited: set, dest) -> list:
        nbrs = list(G.neighbors(node))
        # Prefer unvisited; always allow dest
        fresh = [a for a in nbrs if a not in visited or a == dest]
        return fresh or nbrs

    def _progress_actions(self, actions, node, dest) -> list:
        """Neighbors that reduce remaining potential (make geometric/time progress)."""
        cur = self.remaining.get(node, 1e18)
        better = [a for a in actions if self.remaining.get(a, 1e18) < cur - 1e-6 or a == dest]
        return better or actions

    def _choose(self, G, node, dest, actions: list) -> Hashable:
        progress = self._progress_actions(actions, node, dest)
        # ε-greedy over progress set (biased exploration)
        if self.rng.random() < self.epsilon:
            # Softmax over −remaining among progress actions
            scores = np.array([-self.remaining.get(a, 1e18) for a in progress], dtype=float)
            scores = scores - scores.max()
            probs = np.exp(scores / 30.0)
            probs = probs / probs.sum()
            return progress[int(self.rng.choice(len(progress), p=probs))]
        return max(actions, key=lambda a: self.Q[(node, a)] - 0.01 * self.remaining.get(a, 0.0))

    def learn_episode(self, G, origin, dest, *, max_steps: int = 300) -> list:
        path = [origin]
        node = origin
        visited = {origin}
        for _ in range(max_steps):
            if node == dest:
                break
            actions = self._actions(G, node, visited, dest)
            if not actions:
                break
            nxt = self._choose(G, node, dest, actions)
            edge = _pick_edge(G, node, nxt)
            t = float(edge.get("emv_s") or edge.get("travel_time") or 1.0)
            reward = -t + _street_bonus(edge)

            # Potential-based shaping: F = γ Φ(s') − Φ(s)
            phi_s = -self.remaining.get(node, 0.0)
            phi_sp = -self.remaining.get(nxt, 0.0)
            reward += self.gamma * phi_sp - phi_s

            if nxt in visited and nxt != dest:
                reward -= 8.0
            if nxt == dest:
                reward += 80.0

            next_actions = list(G.neighbors(nxt)) if nxt in G else []
            max_next = max((self.Q[(nxt, a)] for a in next_actions), default=0.0)
            old = self.Q[(node, nxt)]
            self.Q[(node, nxt)] = old + self.alpha * (reward + self.gamma * max_next - old)

            path.append(nxt)
            visited.add(nxt)
            node = nxt

        # If we failed to reach dest, splice Dijkstra finish for a usable trajectory
        if path[-1] != dest:
            path = _complete_to_dest(G, path, dest)
        return _dedupe(path)

    def train(self, G, origin, dest, *, episodes: int = 80, max_steps: int = 300) -> list[float]:
        annotate_composite_weights(G)
        self._build_potentials(G, dest)
        self.train_curve = []
        for ep in range(episodes):
            self.epsilon = max(0.05, self.epsilon0 * (1.0 - ep / max(1, episodes - 1)))
            path = self.learn_episode(G, origin, dest, max_steps=max_steps)
            if not _is_complete(path, origin, dest):
                self.train_curve.append(np.nan)
                continue
            t, _, _ = path_stats(G, path)
            t = float(t) if t == t else 1e18
            self.train_curve.append(t)
            if t < self.best_time:
                self.best_time = t
                self.best_path = list(path)
        return self.train_curve

    def greedy_route(self, G, origin, dest, *, max_steps: int = 400) -> list:
        candidates = []
        if _is_complete(self.best_path, origin, dest):
            candidates.append(list(self.best_path))
        decoded = self._decode_greedy(G, origin, dest, max_steps=max_steps)
        decoded = _complete_to_dest(G, decoded, dest)
        if _is_complete(decoded, origin, dest):
            candidates.append(decoded)
        if not candidates:
            return _complete_to_dest(G, [origin], dest)
        # Pick lowest EMV time among complete paths
        best, best_t = candidates[0], 1e18
        for p in candidates:
            t, _, _ = path_stats(G, p)
            if t == t and t < best_t:
                best, best_t = p, t
        return best

    def _decode_greedy(self, G, origin, dest, *, max_steps: int = 400) -> list:
        path = [origin]
        node = origin
        visited = {origin}
        for _ in range(max_steps):
            if node == dest:
                break
            actions = self._actions(G, node, visited, dest)
            if not actions:
                break
            nxt = max(
                actions,
                key=lambda a: (
                    1000.0 if a == dest else 0.0,
                    self.Q[(node, a)],
                    -self.remaining.get(a, 1e18),
                ),
            )
            if nxt in visited and nxt != dest:
                path = _complete_to_dest(G, path, dest)
                break
            path.append(nxt)
            visited.add(nxt)
            node = nxt
        return _complete_to_dest(G, path, dest)


def _dedupe(path: list) -> list:
    if not path:
        return path
    out = [path[0]]
    for n in path[1:]:
        if n != out[-1]:
            out.append(n)
    return out


def solve_composite_drl(
    G,
    origin,
    dest,
    *,
    episodes: int = 80,
    seed: int = 42,
    pad_deg: float = 0.012,
) -> RouteResult:
    """
    Train composite Q-learning on an OD corridor subgraph, return best path
    (evaluated on the original graph's EMV times).
    """
    H = extract_corridor_subgraph(G, origin, dest, pad_deg=pad_deg)
    if origin not in H or dest not in H:
        # Fallback: composite Dijkstra on full graph
        annotate_composite_weights(G)
        try:
            path = nx.shortest_path(G, origin, dest, weight="weight_composite")
        except Exception:  # noqa: BLE001
            return RouteResult("composite_drl", [], np.nan, np.nan, 0, meta={"error": "no_path"})
        travel, dist, n_edges = path_stats(G, path)
        return RouteResult(
            "composite_drl",
            path,
            float(travel),
            float(dist),
            n_edges,
            meta={"fallback": "composite_dijkstra", "episodes": 0},
        )

    agent = CompositeQAgent(seed=seed)
    history = agent.train(H, origin, dest, episodes=episodes)
    # Decode on corridor, then always complete/validate on the FULL graph
    path = agent.greedy_route(H, origin, dest)
    path = _complete_to_dest(G, path, dest)
    used = "q_policy"

    if not _is_complete(path, origin, dest):
        annotate_composite_weights(G)
        try:
            path = list(nx.shortest_path(G, origin, dest, weight="weight_composite"))
            used = "full_graph_dijkstra_fallback"
        except Exception:  # noqa: BLE001
            return RouteResult(
                "composite_drl", [], np.nan, np.nan, 0, meta={"error": "incomplete_path"}
            )

    travel, dist, n_edges = path_stats(G, path)

    # Prefer the best complete training path if it still ends at dest on G
    if _is_complete(agent.best_path, origin, dest):
        best = _complete_to_dest(G, agent.best_path, dest)
        if _is_complete(best, origin, dest):
            t_b, d_b, n_b = path_stats(G, best)
            if t_b == t_b and (travel != travel or t_b <= travel):
                path, travel, dist, n_edges = best, t_b, d_b, n_b
                used = "train_best"

    # Cap: never worse than composite Dijkstra on the full graph
    try:
        annotate_composite_weights(G)
        dij = nx.shortest_path(G, origin, dest, weight="weight_composite")
        t_dij, d_dij, n_dij = path_stats(G, dij)
        if t_dij == t_dij and (travel != travel or t_dij < travel - 1e-6):
            path, travel, dist, n_edges = dij, t_dij, d_dij, n_dij
            used = "composite_dijkstra_cap"
    except Exception:  # noqa: BLE001
        pass

    # Final hard guarantee for the map / callers
    if not _is_complete(path, origin, dest):
        path = _complete_to_dest(G, [origin], dest)
        travel, dist, n_edges = path_stats(G, path)
        used = "forced_complete"

    return RouteResult(
        "composite_drl",
        list(path),
        float(travel) if travel == travel else np.nan,
        float(dist) if dist == dist else np.nan,
        n_edges,
        meta={
            "episodes": episodes,
            "subgraph_nodes": int(H.number_of_nodes()),
            "subgraph_edges": int(H.number_of_edges()),
            "train_best_s": float(np.nanmin(history)) if history else None,
            "train_curve": history,
            "q_entries": len(agent.Q),
            "decode": used,
            "reaches_dest": bool(_is_complete(path, origin, dest)),
        },
    )
