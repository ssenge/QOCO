from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def gap_pct(cost: float, opt_cost: float) -> float:
    if not np.isfinite(cost) or not np.isfinite(opt_cost) or opt_cost <= 0:
        return float("inf")
    return float((cost - opt_cost) / opt_cost * 100.0)


def mean_gap_pct(costs: Sequence[float], opt_costs: Sequence[float], feasible: Sequence[bool]) -> float:
    gaps = []
    for c, o, f in zip(costs, opt_costs, feasible):
        if f:
            g = gap_pct(c, o)
            if np.isfinite(g):
                gaps.append(g)
    return float(np.mean(gaps)) if gaps else float("inf")


def feasibility_rate(feasible: Sequence[bool]) -> float:
    if not feasible:
        return 0.0
    return float(np.mean(np.array(feasible, dtype=np.float32)))
