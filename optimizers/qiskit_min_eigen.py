from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

import numpy as np

from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import Solution, Status


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


def _result_to_solution(*, qubo: QUBO, x: np.ndarray, fval: float, info: dict[str, Any]) -> Solution:
    n = int(qubo.n_vars)
    xx = np.asarray(x, dtype=float).reshape(-1)
    if xx.shape[0] != n:
        raise ValueError("qiskit result x has wrong length")
    xi = np.asarray(np.rint(xx), dtype=int)
    names = _names_by_index(dict(qubo.var_map), n)
    var_values = {names[i]: int(xi[i]) for i in range(n)}
    return Solution(
        status=Status.FEASIBLE,
        objective=float(fval),
        var_values=var_values,
        var_arrays={"x": xi},
        var_array_index={"x": list(names)},
        info=info,
    )


@dataclass
class QiskitMinimumEigensolverOptimizer(Generic[P], Optimizer[P, QUBO, Solution]):
    """Generic wrapper for Qiskit Optimization's `MinimumEigenOptimizer` on QUBOs."""

    converter: Converter[P, QUBO]
    qubo_to_qp: Any = field(default_factory=QuboToQuadraticProgramConverter)
    minimum_eigensolver: Any = None

    def _optimize(self, qubo: QUBO) -> Solution:
        if self.minimum_eigensolver is None:
            raise ValueError("minimum_eigensolver must be provided")

        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        qp = self.qubo_to_qp.convert(qubo)
        opt = MinimumEigenOptimizer(self.minimum_eigensolver)
        res = opt.solve(qp)
        return _result_to_solution(
            qubo=qubo,
            x=np.asarray(res.x, dtype=float),
            fval=float(res.fval),
            info={"solver": "qiskit.MinimumEigenOptimizer", "minimum_eigensolver": type(self.minimum_eigensolver).__name__},
        )

