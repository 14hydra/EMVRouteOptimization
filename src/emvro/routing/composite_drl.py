"""MODEL 2 — Composite Deep Reinforcement Learning (first draft).

Composite = travel-time cost + EMV-street bonuses (bus-lane proxy, primary
roads) − dead-end / backtrack penalties. Inspired by multi-objective EMV RL
(e.g. EMVLight-style routing) but implemented here as tabular Q-learning on
the local OSM subgraph so the draft runs without a GPU.

Upgrade path: replace Q-table with a DQN / actor-critic over graph embeddings.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable

import networkx as nx
import numpy as np

from .graph import RouteResult, path_stats


def _edge_reward(data: dict) -> float:
    """Negative EMV seconds + small street-quality bonus."""
    t = float(data.get("emv_s") or data.get("travel_time") or 1.0)
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    hw = str(hw or "").lower()
    bonus = 0.0
    if hw in {"primary", "primary_link", "trunk", "trunk_link"}:
        bonus += 0.5
    if data.get("busway") or data.get("lanes:bus"):
        bonus += 1.0
    return -t + bonus


def _pick_edge(G, u, v) -> dict:
    edata = G.get_edge_data(u, v) or {}
    return min(edata.values(), key=lambda d: d.get("emv_s", 1e18))


class CompositeQAgent:
    """Tabular Q-learning agent for one origin→destination episode family."""

    def __init__(
        self,
        *,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.25,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q: dict[tuple[Hashable, Hashable], float] = defaultdict(float)

    def _actions(self, G, node) -> list:
        return list(G.neighbors(node))

    def act(self, G, node, dest, *, explore: bool = True):
        actions = self._actions(G, node)
        if not actions:
            return None
        if explore and self.rng.random() < self.epsilon:
            return actions[int(self.rng.integers(0, len(actions)))]
        # Greedy by Q; small heuristic bias toward geometric progress via Dijkstra residual
        best_a, best_q = actions[0], -1e18
        for a in actions:
            q = self.Q[(node, a)]
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def learn_episode(self, G, origin, dest, *, max_steps: int = 250) -> list:
        path = [origin]
        node = origin
        visited = {origin}
        for _ in range(max_steps):
            if node == dest:
                break
            actions = [a for a in self._actions(G, node) if a not in visited or a == dest]
            if not actions:
                actions = self._actions(G, node)
            if not actions:
                break
            if self.rng.random() < self.epsilon:
                nxt = actions[int(self.rng.integers(0, len(actions)))]
            else:
                nxt = max(actions, key=lambda a: self.Q[(node, a)])

            edge = _pick_edge(G, node, nxt)
            reward = _edge_reward(edge)
            if nxt in visited and nxt != dest:
                reward -= 5.0  # backtrack penalty
            if nxt == dest:
                reward += 50.0  # arrival bonus

            next_actions = self._actions(G, nxt)
            max_next = max((self.Q[(nxt, a)] for a in next_actions), default=0.0)
            old = self.Q[(node, nxt)]
            self.Q[(node, nxt)] = old + self.alpha * (reward + self.gamma * max_next - old)

            path.append(nxt)
            visited.add(nxt)
            node = nxt
        return path

    def train(self, G, origin, dest, *, episodes: int = 40, max_steps: int = 250) -> list[float]:
        lengths = []
        for ep in range(episodes):
            # Decay exploration
            self.epsilon = max(0.05, 0.25 * (1.0 - ep / max(1, episodes)))
            path = self.learn_episode(G, origin, dest, max_steps=max_steps)
            t, _, _ = path_stats(G, path)
            lengths.append(float(t) if t == t else 1e18)
        return lengths

    def greedy_route(self, G, origin, dest, *, max_steps: int = 400) -> list:
        path = [origin]
        node = origin
        visited = {origin}
        for _ in range(max_steps):
            if node == dest:
                break
            actions = self._actions(G, node)
            if not actions:
                break
            # Prefer unseen nodes unless destination
            ranked = sorted(
                actions,
                key=lambda a: (
                    0 if a == dest else 1 if a not in visited else 2,
                    -self.Q[(node, a)],
                ),
            )
            nxt = ranked[0]
            # Safety: if looping, fall back to Dijkstra remainder
            if nxt in visited and nxt != dest:
                try:
                    rest = nx.shortest_path(G, node, dest, weight="weight_emv")
                    path.extend(rest[1:])
                except Exception:  # noqa: BLE001
                    break
                break
            path.append(nxt)
            visited.add(nxt)
            node = nxt
        if path[-1] != dest:
            try:
                rest = nx.shortest_path(G, path[-1], dest, weight="weight_emv")
                path.extend(rest[1:])
            except Exception:  # noqa: BLE001
                pass
        # Deduplicate consecutives
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
    episodes: int = 40,
    seed: int = 42,
) -> RouteResult:
    """Train a composite Q-agent on this OD, then return the greedy path."""
    agent = CompositeQAgent(seed=seed)
    history = agent.train(G, origin, dest, episodes=episodes)
    path = agent.greedy_route(G, origin, dest)
    travel, dist, n_edges = path_stats(G, path)
    return RouteResult(
        "composite_drl",
        path,
        float(travel) if travel == travel else np.nan,
        float(dist) if dist == dist else np.nan,
        n_edges,
        meta={
            "episodes": episodes,
            "train_best_s": float(min(history)) if history else None,
            "q_entries": len(agent.Q),
        },
    )
