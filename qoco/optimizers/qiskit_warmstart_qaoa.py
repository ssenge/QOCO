from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

import numpy as np
from qiskit_optimization.algorithms import SlsqpOptimizer, WarmStartQAOAOptimizer

from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.qiskit_min_eigen import _result_to_solution
from qoco.optimizers.qiskit_legacy_qaoa import QiskitLegacyQAOAOptimizer


@dataclass
class QiskitWarmStartQAOAOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Warm-start QAOA wrapper (Qiskit Optimization).

    This is QAOA-specific in Qiskit (`WarmStartQAOAOptimizer`).
    We reuse `QiskitLegacyQAOAOptimizer.make_qaoa()` to construct the underlying QAOA instance.
    """

    name: str = "QiskitWarmStartQAOA"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_qp: Converter[QUBO, Any] = field(default_factory=QuboToQuadraticProgramConverter)

    base_qaoa: QiskitLegacyQAOAOptimizer[Any] = field(default_factory=QiskitLegacyQAOAOptimizer)
    pre_solver: Any = field(default_factory=SlsqpOptimizer)
    relax_for_pre_solver: bool = True

    epsilon: float = 0.25
    num_initial_solutions: int = 1
    penalty: float | None = None

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        qp = self.qubo_to_qp.convert(qubo)
        qaoa = self.base_qaoa.make_qaoa(qubo=qubo, n=qubo.n_vars)

        opt = WarmStartQAOAOptimizer(
            pre_solver=self.pre_solver,
            relax_for_pre_solver=bool(self.relax_for_pre_solver),
            qaoa=qaoa,
            epsilon=float(self.epsilon),
            num_initial_solutions=int(self.num_initial_solutions),
            penalty=self.penalty,
        )
        res = opt.solve(qp)
        solution = _result_to_solution(qubo=qubo, x=np.asarray(res.x, dtype=float), fval=float(res.fval))
        return solution, OptimizerRun(name=self.name)

