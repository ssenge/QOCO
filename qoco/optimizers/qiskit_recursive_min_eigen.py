from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

import numpy as np
from qiskit_optimization.algorithms import IntermediateResult, MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer

from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.qiskit_min_eigen import _result_to_solution
from qoco.optimizers.qiskit_legacy_qaoa import QiskitLegacyQAOAOptimizer


@dataclass
class QiskitRecursiveMinimumEigenOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """RecursiveMinimumEigenOptimizer wrapper (Qiskit Optimization).

    This is generic over the underlying minimum eigensolver; by default we reuse
    `QiskitLegacyQAOAOptimizer.make_qaoa()` to construct QAOA as the base eigensolver.
    """

    name: str = "QiskitRecursiveMinimumEigen"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_qp: Converter[QUBO, Any] = field(default_factory=QuboToQuadraticProgramConverter)

    base_qaoa: QiskitLegacyQAOAOptimizer[Any] = field(default_factory=QiskitLegacyQAOAOptimizer)
    min_num_vars: int = 1
    penalty: float | None = None
    history: IntermediateResult = IntermediateResult.LAST_ITERATION

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        qp = self.qubo_to_qp.convert(qubo)
        qaoa = self.base_qaoa.make_qaoa(qubo=qubo, n=qubo.n_vars)

        min_eigen_opt = MinimumEigenOptimizer(qaoa)
        opt = RecursiveMinimumEigenOptimizer(
            optimizer=min_eigen_opt,
            min_num_vars=int(self.min_num_vars),
            penalty=self.penalty,
            history=self.history,
        )
        res = opt.solve(qp)
        solution = _result_to_solution(qubo=qubo, x=np.asarray(res.x, dtype=float), fval=float(res.fval))
        return solution, OptimizerRun(name=self.name)

