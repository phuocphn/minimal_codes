**Key design decisions:**

The algorithm tracks a Pareto front rather than a single best path. A solution is kept only if no other solution is strictly better across all objectives simultaneously. The dominates() check prunes both the open set and goal solutions — any node whose cost vector is already beaten in every dimension gets skipped.

You plug in one heuristic per objective. Each must be admissible (never overestimates) for that objective to guarantee the returned solutions are truly Pareto-optimal. Using 0.0 as a fallback is always safe.

The neighbors() function returns a cost tuple per edge — add as many objectives as you need (distance, risk, energy, time, monetary cost, etc.) by widening the tuple and adding a matching heuristic.

