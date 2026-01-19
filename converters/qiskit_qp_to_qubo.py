from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


@dataclass
class QiskitQuadraticProgramToQuboProgramConverter(Converter[QuadraticProgram, QuadraticProgram]):
    """Convert a Qiskit `QuadraticProgram` to an unconstrained QUBO `QuadraticProgram`."""

    penalty: float | None = None

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        conv = QuadraticProgramToQubo(penalty=self.penalty)
        return conv.convert(problem)


@dataclass
class QiskitQuadraticProgramToQuboMatrixConverter(Converter[QuadraticProgram, QUBO]):
    """Convert a Qiskit QUBO `QuadraticProgram` into our `QUBO` matrix type.

    Assumes:
    - all variables are binary
    - objective is quadratic (no constraints)
    """

    def convert(self, problem: QuadraticProgram) -> QUBO:
        n = int(problem.get_num_vars())
        Q = np.zeros((n, n), dtype=np.float64)

        obj = problem.objective
        offset = float(obj.constant)

        for idx, coef in obj.linear.to_dict().items():
            i = int(idx)
            Q[i, i] += float(coef)

        for (i, j), coef in obj.quadratic.to_dict().items():
            ii = int(i)
            jj = int(j)
            if ii == jj:
                Q[ii, ii] += float(coef)
            else:
                row, col = (ii, jj) if ii < jj else (jj, ii)
                Q[row, col] += float(coef)

        var_map = {problem.variables[i].name: i for i in range(n)}
        return QUBO(Q=Q, offset=float(offset), var_map=var_map)

