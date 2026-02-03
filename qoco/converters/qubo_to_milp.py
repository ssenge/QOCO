from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


@dataclass
class QuboToMILPConverter(Converter[QUBO, pyo.ConcreteModel]):
    """Convert a QUBO into an equivalent MILP by linearizing x_i * x_j terms.

    Minimizes: x^T Q x + offset, with x binary.

    For each i<j with nonzero coupling coefficient, introduces z[i,j] = x[i]*x[j].
    """

    tol: float = 0.0  # treat |coef| <= tol as zero

    def convert(self, problem: QUBO) -> pyo.ConcreteModel:
        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])

        m = pyo.ConcreteModel(name="QUBO_MILP")
        if n == 0:
            m.obj = pyo.Objective(expr=float(problem.offset), sense=pyo.minimize)
            return m

        m.I = pyo.RangeSet(0, n - 1)
        m.x = pyo.Var(m.I, domain=pyo.Binary)

        # z variables for i<j where coupling exists
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                coef = float(Q[i, j]) + float(Q[j, i])
                if abs(coef) > float(self.tol):
                    pairs.append((i, j))

        m.P = pyo.Set(initialize=pairs, dimen=2)
        m.z = pyo.Var(m.P, domain=pyo.Binary)

        # Linearization constraints:
        # z <= x_i, z <= x_j, z >= x_i + x_j - 1
        def z_le_i(mm, i, j):
            return mm.z[i, j] <= mm.x[i]

        def z_le_j(mm, i, j):
            return mm.z[i, j] <= mm.x[j]

        def z_ge(mm, i, j):
            return mm.z[i, j] >= mm.x[i] + mm.x[j] - 1

        m.z_le_i = pyo.Constraint(m.P, rule=z_le_i)
        m.z_le_j = pyo.Constraint(m.P, rule=z_le_j)
        m.z_ge = pyo.Constraint(m.P, rule=z_ge)

        # Objective: diag terms + coupling terms + offset
        expr = float(problem.offset)
        for i in range(n):
            expr = expr + float(Q[i, i]) * m.x[i]
        for (i, j) in pairs:
            coef = float(Q[i, j]) + float(Q[j, i])
            expr = expr + float(coef) * m.z[i, j]

        m.obj = pyo.Objective(expr=expr, sense=pyo.minimize)
        return m

