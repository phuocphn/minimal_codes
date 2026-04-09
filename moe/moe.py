import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

@dataclass(order=True)
class Node:
    """A search node tracking multiple cost dimensions."""
    f_scores: tuple          # heuristic-augmented scores for priority
    costs: tuple             # actual accumulated costs (one per objective)
    state: Any = field(compare=False)
    path: list  = field(default_factory=list, compare=False)


def dominates(costs_a: tuple, costs_b: tuple) -> bool:
    """Return True if costs_a Pareto-dominates costs_b.
    
    A dominates B when A is no worse in every objective
    and strictly better in at least one.
    """
    return (
        all(a <= b for a, b in zip(costs_a, costs_b)) and
        any(a <  b for a, b in zip(costs_a, costs_b))
    )


def prune_dominated(pareto_front: list[tuple]) -> list[tuple]:
    """Remove dominated cost vectors from the Pareto front."""
    pruned = []
    for candidate in pareto_front:
        if not any(dominates(other, candidate) for other in pareto_front if other != candidate):
            pruned.append(candidate)
    return pruned


def multi_objective_astar(
    start,
    goal,
    neighbors: Callable,          # neighbors(state) → [(next_state, edge_costs_tuple), ...]
    heuristics: list[Callable],   # one heuristic per objective: h(state, goal) → float
    num_objectives: int = 2,
) -> list[dict]:
    """
    Multi-Objective A* (MOA*) search.

    Returns a list of Pareto-optimal solutions, each as:
        {"path": [...states...], "costs": (cost1, cost2, ...)}

    Parameters
    ----------
    start       : hashable start state
    goal        : hashable goal state
    neighbors   : callable(state) → iterable of (next_state, costs_tuple)
    heuristics  : list of admissible heuristic functions, one per objective
    num_objectives : number of cost dimensions

    Notes
    -----
    The open set uses the *sum* of f-scores as tie-breaker priority.
    Pareto dominance pruning prevents expanding cost vectors that are
    already dominated by a known solution.
    """
    zero_costs = tuple(0.0 for _ in range(num_objectives))

    # g_costs[state] = list of non-dominated cost tuples seen so far
    g_costs: dict[Any, list[tuple]] = {start: [zero_costs]}

    # Pareto-optimal complete paths found so far
    pareto_solutions: list[dict] = []
    pareto_cost_front: list[tuple] = []

    def h_vector(state) -> tuple:
        return tuple(h(state, goal) for h in heuristics)

    def f_vector(costs, state) -> tuple:
        hv = h_vector(state)
        return tuple(c + h for c, h in zip(costs, hv))

    start_f = f_vector(zero_costs, start)
    open_set: list[Node] = []
    heapq.heappush(open_set, Node(
        f_scores=start_f,
        costs=zero_costs,
        state=start,
        path=[start],
    ))

    while open_set:
        node = heapq.heappop(open_set)
        state, costs, path = node.state, node.costs, node.path

        # ── Pruning: skip if this cost vector is now dominated ──────────
        if pareto_cost_front and any(dominates(front_c, costs) for front_c in pareto_cost_front):
            continue

        # ── Check if any known g-cost dominates this node's costs ───────
        known = g_costs.get(state, [])
        if any(dominates(k, costs) for k in known):
            continue

        # ── Goal check ──────────────────────────────────────────────────
        if state == goal:
            # Add to Pareto front if not dominated
            if not any(dominates(fc, costs) for fc in pareto_cost_front):
                pareto_cost_front = [c for c in pareto_cost_front if not dominates(costs, c)]
                pareto_cost_front.append(costs)
                pareto_solutions = [s for s in pareto_solutions if not dominates(costs, s["costs"])]
                pareto_solutions.append({"path": path, "costs": costs})
            continue

        # ── Expand neighbors ────────────────────────────────────────────
        for next_state, edge_costs in neighbors(state):
            new_costs = tuple(c + e for c, e in zip(costs, edge_costs))

            # Skip if dominated by Pareto front or known g-costs
            if pareto_cost_front and any(dominates(fc, new_costs) for fc in pareto_cost_front):
                continue

            next_known = g_costs.get(next_state, [])
            if any(dominates(k, new_costs) for k in next_known):
                continue

            # Update non-dominated g-costs for next_state
            next_known = [k for k in next_known if not dominates(new_costs, k)]
            next_known.append(new_costs)
            g_costs[next_state] = next_known

            new_f = f_vector(new_costs, next_state)
            heapq.heappush(open_set, Node(
                f_scores=new_f,
                costs=new_costs,
                state=next_state,
                path=path + [next_state],
            ))

    return pareto_solutions


import math

GRID = [
    # 0=open, 1=wall, 2=risky zone
    [0, 0, 0, 0, 2, 0],
    [0, 1, 1, 0, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
]
ROWS, COLS = len(GRID), len(GRID[0])

def grid_neighbors(state):
    r, c = state
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] != 1:
            distance_cost = 1.0
            risk_cost     = 5.0 if GRID[nr][nc] == 2 else 0.0
            yield (nr, nc), (distance_cost, risk_cost)

def euclidean(state, goal):
    return math.hypot(state[0] - goal[0], state[1] - goal[1])

def risk_heuristic(state, goal):
    return 0.0  # admissible: underestimates risk (unknown future cells)

solutions = multi_objective_astar(
    start=(0, 0),
    goal=(4, 5),
    neighbors=grid_neighbors,
    heuristics=[euclidean, risk_heuristic],
    num_objectives=2,
)

print(f"Found {len(solutions)} Pareto-optimal paths:\n")
for s in sorted(solutions, key=lambda x: x["costs"]):
    dist, risk = s["costs"]
    print(f"  distance={dist:.0f}  risk={risk:.0f}  path={s['path']}")

