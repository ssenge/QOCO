from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import Solution
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.qiskit_min_eigen import QiskitMinimumEigensolverOptimizer


@dataclass
class QiskitNumPyMinimumEigensolverOptimizer(Generic[P], Optimizer[P, QUBO, Solution]):
    """Exact eigensolver baseline (only for very small QUBOs)."""

    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)

    def _optimize(self, qubo: QUBO) -> Solution:
        base = QiskitMinimumEigensolverOptimizer(converter=self.converter, minimum_eigensolver=NumPyMinimumEigensolver())
        return base._optimize(qubo)

