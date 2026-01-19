from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import Solution
from qoco.optimizers.qiskit_min_eigen import QiskitMinimumEigensolverOptimizer


@dataclass
class QiskitSamplingVQEOptimizer(Generic[P], Optimizer[P, QUBO, Solution]):
    """Qiskit SamplingVQE on QUBOs (via MinimumEigenOptimizer)."""

    converter: Converter[P, QUBO]

    reps: int = 1
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=100))
    sampler: Any = field(default_factory=SamplerV2)
    pass_manager: Any = field(default_factory=lambda: generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"]))
    seed: Optional[int] = 0

    def _optimize(self, qubo: QUBO) -> Solution:
        n = int(qubo.n_vars)
        if self.seed is not None:
            np.random.seed(int(self.seed))
        ansatz = efficient_su2(num_qubits=n, reps=int(self.reps))
        ansatz = self.pass_manager.run(ansatz)
        svqe = SamplingVQE(sampler=self.sampler, ansatz=ansatz, optimizer=self.classical_optimizer)
        base = QiskitMinimumEigensolverOptimizer(converter=self.converter, minimum_eigensolver=svqe)
        return base._optimize(qubo)

