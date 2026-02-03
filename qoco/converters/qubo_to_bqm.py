from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import dimod

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


@dataclass
class QuboToBQMConverter(Converter[QUBO, dimod.BinaryQuadraticModel]):
    """Convert our `QUBO` to a dimod `BinaryQuadraticModel` (BINARY vartype).

    Our convention:
    - Diagonal Q[i,i] is linear term
    - Off-diagonal terms are stored in upper triangle Q[i,j] for i<j
    """

    tol: float = 0.0  # treat |coef| <= tol as zero

    def convert(self, problem: QUBO) -> dimod.BinaryQuadraticModel:
        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])
        linear: dict[int, float] = {}
        quadratic: dict[tuple[int, int], float] = {}

        for i in range(n):
            c = float(Q[i, i])
            if abs(c) > float(self.tol):
                linear[int(i)] = c

        for i in range(n):
            for j in range(i + 1, n):
                # be robust if caller ever filled both triangles
                c = float(Q[i, j]) + float(Q[j, i])
                if abs(c) > float(self.tol):
                    quadratic[(int(i), int(j))] = c

        return dimod.BinaryQuadraticModel(linear, quadratic, float(problem.offset), vartype=dimod.BINARY)

