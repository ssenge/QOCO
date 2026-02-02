from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import InfoSolution, OptimizerRun, ProblemSummary
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.qiskit_min_eigen import QiskitMinimumEigensolverOptimizer


@dataclass
class QiskitNumPyMinimumEigensolverOptimizer(Generic[P], Optimizer[P, QUBO, InfoSolution, OptimizerRun, ProblemSummary]):
    """Exact eigensolver baseline (only for very small QUBOs)."""

    name: str = "QiskitNumPyMES"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)

    def _optimize(self, qubo: QUBO) -> tuple[InfoSolution, OptimizerRun]:
        base = QiskitMinimumEigensolverOptimizer(converter=self.converter, minimum_eigensolver=NumPyMinimumEigensolver())
        return base._optimize(qubo)

