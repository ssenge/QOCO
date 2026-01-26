from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.core.problem import Problem


@dataclass
class TSP(Problem):
    name: str
    dist: List[List[float]]

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        n = len(self.dist)
        if n < 2:
            errors.append("n must be >= 2")
        if any(len(row) != n for row in self.dist):
            errors.append("dist must be square")
        return len(errors) == 0, errors

    def summary(self) -> str:
        valid, errors = self.validate()
        status = "✓ Valid" if valid else f"✗ {len(errors)} errors"
        return f"TSP('{self.name}') n={len(self.dist)}; {status}"

    class MILPConverter(Converter["TSP", pyo.ConcreteModel]):
        def convert(self, problem: "TSP") -> pyo.ConcreteModel:
            dist = problem.dist
            n = len(dist)
            model = pyo.ConcreteModel(name=getattr(problem, "name", None) or "TSP")
            model.N = pyo.RangeSet(0, n - 1)
            model.x = pyo.Var(model.N, model.N, domain=pyo.Binary)
            model.u = pyo.Var(model.N, domain=pyo.NonNegativeReals, bounds=(0, n - 1))

            def no_self_rule(m, i):
                return m.x[i, i] == 0

            model.no_self = pyo.Constraint(model.N, rule=no_self_rule)

            def out_rule(m, i):
                return sum(m.x[i, j] for j in m.N) == 1

            def in_rule(m, j):
                return sum(m.x[i, j] for i in m.N) == 1

            model.one_out = pyo.Constraint(model.N, rule=out_rule)
            model.one_in = pyo.Constraint(model.N, rule=in_rule)

            def mtz_rule(m, i, j):
                if i == 0 or j == 0 or i == j:
                    return pyo.Constraint.Skip
                return m.u[i] - m.u[j] + (n - 1) * m.x[i, j] <= n - 2

            model.mtz = pyo.Constraint(model.N, model.N, rule=mtz_rule)
            model.u[0].fix(0)

            model.obj = pyo.Objective(
                expr=sum(dist[i][j] * model.x[i, j] for i in range(n) for j in range(n)),
                sense=pyo.minimize,
            )
            return model
