from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_optimization import QuadraticProgram

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


@dataclass
class QuboToQuadraticProgramConverter(Converter[QUBO, QuadraticProgram]):
    """Convert `QUBO` into a Qiskit `QuadraticProgram` (binary vars, quadratic objective)."""

    tol: float = 0.0

    def convert(self, problem: QUBO) -> QuadraticProgram:
        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])

        qp = QuadraticProgram(name="qubo")
        for i in range(n):
            qp.binary_var(name=f"x{i}")

        linear = [0.0] * n
        quadratic: dict[tuple[int, int], float] = {}

        for i in range(n):
            c = float(Q[i, i])
            if abs(c) > float(self.tol):
                linear[i] = c

        for i in range(n):
            for j in range(i + 1, n):
                c = float(Q[i, j]) + float(Q[j, i])
                if abs(c) > float(self.tol):
                    quadratic[(i, j)] = c

        qp.minimize(constant=float(problem.offset), linear=linear, quadratic=quadratic)
        return qp

