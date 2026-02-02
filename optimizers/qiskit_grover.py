from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

import numpy as np

from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_optimization.algorithms import GroverOptimizer

from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import InfoSolution, OptimizerRun, ProblemSummary, Status
from qoco.converters.identity import IdentityConverter


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


@dataclass
class QiskitGroverOptimizer(Generic[P], Optimizer[P, QUBO, InfoSolution, OptimizerRun, ProblemSummary]):
    """Qiskit GroverOptimizer on QUBO (tiny problems only)."""

    name: str = "QiskitGrover"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_qp: Any = field(default_factory=QuboToQuadraticProgramConverter)

    num_value_qubits: int = 8
    num_iterations: int = 3
    sampler: Any = field(default_factory=SamplerV2)
    pass_manager: Any = field(default_factory=lambda: generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"]))

    def _optimize(self, qubo: QUBO) -> tuple[InfoSolution, OptimizerRun]:
        qp = self.qubo_to_qp.convert(qubo)
        grover = GroverOptimizer(
            num_value_qubits=int(self.num_value_qubits),
            num_iterations=int(self.num_iterations),
            sampler=self.sampler,
            pass_manager=self.pass_manager,
        )
        res = grover.solve(qp)
        x = np.asarray(res.x, dtype=float).reshape(-1)
        xi = np.asarray(np.rint(x), dtype=int)
        n = int(qubo.n_vars)
        if xi.shape[0] != n:
            raise ValueError("grover result x has wrong length")

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(xi[i]) for i in range(n)}
        solution = InfoSolution(
            status=Status.FEASIBLE,
            objective=float(res.fval),
            var_values=var_values,
            var_arrays={"x": xi},
            var_array_index={"x": list(names)},
            info={"solver": "qiskit.GroverOptimizer", "num_value_qubits": int(self.num_value_qubits), "num_iterations": int(self.num_iterations)},
        )
        return solution, OptimizerRun(name=self.name)

