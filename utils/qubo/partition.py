from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np

from qoco.core.qubo import QUBO, QuboPartition
from qoco.core.solution import InfoSolution, OptimizerRun, ProblemSummary, Status
from qoco.core.optimizer import Optimizer
from qoco.optimizers.decomposition.multistage import StageTask
from qoco.optimizers.decomposition.sequential import SequentialSolver


def evaluate_qubo(qubo: QUBO, x: np.ndarray) -> float:
    Q = np.asarray(qubo.Q, dtype=float)
    xx = np.asarray(x, dtype=int).reshape(-1)
    return float(xx @ Q @ xx + float(qubo.offset))


def _names_by_index(var_map: Mapping[str, int], n: int) -> List[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


@dataclass(frozen=True)
class SequentialQuboPartitionSolver:
    """Solve a QUBO partition by solving each sub-QUBO independently, then concatenating.

    This is intentionally simple: it ignores cross-block couplings, so it is a heuristic.
    """

    sub_solver: Optimizer[QUBO, QUBO, InfoSolution, OptimizerRun, ProblemSummary]

    def solve(self, *, qubo: QUBO, partition: QuboPartition) -> InfoSolution:
        n = int(qubo.n_vars)
        if n == 0:
            return InfoSolution(status=Status.OPTIMAL, objective=float(qubo.offset), var_values={}, var_arrays={"x": np.zeros((0,), dtype=int)})

        tasks = []
        for k, sub in enumerate(partition.sub_qubos):
            tasks.append(StageTask(tag=int(k), problem=sub, solver=self.sub_solver))

        def _combine(results):
            x = np.zeros((n,), dtype=int)
            for res, idx in zip(results, partition.global_indices):
                sol = res.solution
                if "x" not in sol.var_arrays:
                    raise RuntimeError("sub-solver must provide Solution.var_arrays['x']")
                xs = np.asarray(sol.var_arrays["x"], dtype=int).reshape(-1)
                if xs.shape[0] != int(idx.size):
                    raise RuntimeError("sub-solver returned x with wrong length")
                x[idx] = xs
            obj = evaluate_qubo(qubo, x)
            names = _names_by_index(qubo.var_map, n)
            var_values = {names[i]: int(x[i]) for i in range(n)}
            return InfoSolution(
                status=Status.FEASIBLE,
                objective=float(obj),
                var_values=var_values,
                var_arrays={"x": x},
                var_array_index={"x": list(names)},
                info={"n_blocks": int(len(partition.blocks)), "n_couplings": int(len(partition.couplings)), "note": "heuristic: couplings ignored"},
            )

        return SequentialSolver(tasks=tasks, combine=_combine).run()

