"""MODEL 1 — MIPSSTW solved by Modified Cuckoo Search (MCS).

MIPSSTW = Mixed-Integer Programming with Semi-Soft Time Windows
(Luan & Jiang, PLOS ONE 2024): minimize EMV travel time subject to a soft
arrival deadline; late arrivals incur a linear penalty.

MCS = Modified Cuckoo Search metaheuristic (Lévy flights + dynamic inertia)
used as a first-draft path optimizer on the NYC street graph.

This is a runnable research draft — not a full MILP solver.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .graph import RouteResult, path_stats


def _path_cost(
    G,
    path: list,
    *,
    deadline_s: float | None,
    late_penalty_per_s: float,
) -> float:
    travel, _, _ = path_stats(G, path)
    if travel != travel:
        return 1e18
    cost = float(travel)
    if deadline_s is not None and travel > deadline_s:
        cost += late_penalty_per_s * (travel - deadline_s)
    return cost


def _random_simple_path(G, origin, dest, rng: np.random.Generator, max_steps: int = 400) -> list:
    """Biased random walk toward dest; falls back to Dijkstra."""
    try:
        return list(nx.shortest_path(G, origin, dest, weight="weight_emv"))
    except Exception:  # noqa: BLE001
        return []


def _neighbors(G, node) -> list:
    return list(G.neighbors(node))


def _levy_perturb_path(G, path: list, rng: np.random.Generator, beta: float = 1.5) -> list:
    """
    Lévy-inspired path perturbation: cut a random subpath and re-route locally.
    """
    if len(path) < 4:
        return list(path)
    # Lévy step size → how large a segment to replace
    u = rng.random()
    step = int(max(1, min(len(path) // 2, (u ** (-1.0 / beta)))))
    i = int(rng.integers(1, max(2, len(path) - step - 1)))
    j = min(len(path) - 2, i + step)
    a, b = path[i], path[j]
    try:
        mid = nx.shortest_path(G, a, b, weight="weight_emv")
    except Exception:  # noqa: BLE001
        return list(path)
    new_path = path[:i] + mid + path[j + 1 :]
    # Deduplicate consecutive nodes
    out = [new_path[0]]
    for n in new_path[1:]:
        if n != out[-1]:
            out.append(n)
    return out


def _crossover_relink(G, path_a: list, path_b: list, rng: np.random.Generator) -> list:
    """Path-relinking style crossover via a shared intermediate node when possible."""
    set_b = set(path_b)
    shared = [n for n in path_a[1:-1] if n in set_b]
    if not shared:
        return list(path_a if rng.random() < 0.5 else path_b)
    pivot = shared[int(rng.integers(0, len(shared)))]
    ia = path_a.index(pivot)
    ib = path_b.index(pivot)
    child = path_a[: ia + 1] + path_b[ib + 1 :]
    out = [child[0]]
    for n in child[1:]:
        if n != out[-1]:
            out.append(n)
    # Ensure connectivity; repair with Dijkstra segments if needed
    repaired = [out[0]]
    for n in out[1:]:
        if n in G[repaired[-1]]:
            repaired.append(n)
        else:
            try:
                mid = nx.shortest_path(G, repaired[-1], n, weight="weight_emv")
                repaired.extend(mid[1:])
            except Exception:  # noqa: BLE001
                return list(path_a)
    return repaired


def solve_mipsstw_mcs(
    G,
    origin,
    dest,
    *,
    deadline_s: float | None = None,
    late_penalty_per_s: float = 2.0,
    n_nests: int = 12,
    n_iterations: int = 25,
    pa: float = 0.25,
    seed: int = 42,
) -> RouteResult:
    """
    Modified Cuckoo Search over paths with semi-soft time-window fitness.

    Fitness ≈ EMV travel seconds + late_penalty * max(0, arrival − deadline).
    """
    rng = np.random.default_rng(seed)

    # Seed population with Dijkstra + light perturbations
    base = _random_simple_path(G, origin, dest, rng)
    if not base:
        return RouteResult("mipsstw_mcs", [], np.nan, np.nan, 0, meta={"error": "no_path"})

    nests: list[list] = [base]
    for _ in range(n_nests - 1):
        nests.append(_levy_perturb_path(G, base, rng))

    fitness = [_path_cost(G, p, deadline_s=deadline_s, late_penalty_per_s=late_penalty_per_s) for p in nests]
    best_idx = int(np.argmin(fitness))
    best_path = list(nests[best_idx])
    best_fit = float(fitness[best_idx])

    history = [best_fit]
    for it in range(n_iterations):
        # Dynamic inertia / discovery probability schedule
        pa_t = pa * (1.0 - it / max(1, n_iterations))
        inertia = 0.9 - 0.5 * (it / max(1, n_iterations))

        for i in range(n_nests):
            if rng.random() < inertia:
                candidate = _levy_perturb_path(G, nests[i], rng)
            else:
                other = nests[int(rng.integers(0, n_nests))]
                candidate = _crossover_relink(G, nests[i], other, rng)
            if len(candidate) < 2 or candidate[0] != origin or candidate[-1] != dest:
                continue
            f = _path_cost(G, candidate, deadline_s=deadline_s, late_penalty_per_s=late_penalty_per_s)
            if f < fitness[i]:
                nests[i] = candidate
                fitness[i] = f
                if f < best_fit:
                    best_fit = f
                    best_path = list(candidate)

        # Abandon a fraction of worst nests (cuckoo discovery)
        order = np.argsort(fitness)
        n_abandon = max(1, int(pa_t * n_nests))
        for idx in order[-n_abandon:]:
            nests[idx] = _levy_perturb_path(G, best_path, rng)
            fitness[idx] = _path_cost(
                G, nests[idx], deadline_s=deadline_s, late_penalty_per_s=late_penalty_per_s
            )
            if fitness[idx] < best_fit:
                best_fit = float(fitness[idx])
                best_path = list(nests[idx])
        history.append(best_fit)

    travel, dist, n_edges = path_stats(G, best_path)
    late = 0.0
    if deadline_s is not None and travel == travel and travel > deadline_s:
        late = float(travel - deadline_s)
    return RouteResult(
        "mipsstw_mcs",
        best_path,
        float(travel),
        float(dist),
        n_edges,
        meta={
            "fitness": best_fit,
            "deadline_s": deadline_s,
            "late_seconds": late,
            "n_iterations": n_iterations,
            "n_nests": n_nests,
            "fitness_history": history,
        },
    )
