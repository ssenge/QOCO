from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional

import numpy as np
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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


def _bitstring_to_x(bitstring: str, n: int) -> np.ndarray:
    s = bitstring.strip()
    if len(s) != int(n):
        raise ValueError("bitstring length mismatch")
    return np.array([1 if ch == "1" else 0 for ch in s], dtype=int)


@dataclass
class QiskitQAOAOptimizer(Generic[P], Optimizer[P, QUBO, InfoSolution, OptimizerRun, ProblemSummary]):
    """Solve QUBOs using Qiskit's QAOA (simulated via Aer SamplerV2 by default)."""

    name: str = "QiskitQAOA"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_qp: Converter[QUBO, Any] = field(default_factory=QuboToQuadraticProgramConverter)

    reps: int = 1
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=50))
    sampler: Any = field(default_factory=SamplerV2)
    transpiler: Any = field(default_factory=lambda: generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"]))
    seed: Optional[int] = 0

    def _optimize(self, qubo: QUBO) -> tuple[InfoSolution, OptimizerRun]:
        qp = self.qubo_to_qp.convert(qubo)
        op, offset = qp.to_ising()

        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=self.classical_optimizer,
            reps=int(self.reps),
            transpiler=self.transpiler,
        )
        if self.seed is not None:
            # Qiskit algorithms use numpy global RNG internally; seed is best-effort.
            np.random.seed(int(self.seed))

        res = qaoa.compute_minimum_eigenvalue(op)
        if not hasattr(res, "best_measurement"):
            raise RuntimeError("QAOA result missing best_measurement")
        bm = res.best_measurement
        bitstring = str(bm["bitstring"])
        value = float(np.real(bm["value"]))

        n = int(qubo.n_vars)
        x = _bitstring_to_x(bitstring, n)
        obj = float(value + float(offset))

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        solution = InfoSolution(
            status=Status.FEASIBLE,
            objective=float(obj),
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
            info={"solver": "qiskit.QAOA", "reps": int(self.reps)},
        )
        return solution, OptimizerRun(name=self.name)

