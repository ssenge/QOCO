from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.core.problem import Problem


@dataclass
class BinPacking(Problem):
    name: str
    weights: list[float]
    capacity: float
    m: int  # bin upper bound 

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.capacity <= 0:
            errors.append("capacity must be > 0")
        if self.m <= 0:
            errors.append("m (bin upper bound) must be > 0")
        if not self.weights:
            errors.append("no items (n=0)")
        if any(w <= 0 for w in self.weights):
            errors.append("all weights must be > 0")
        if any(w > self.capacity for w in self.weights):
            errors.append("some weights exceed capacity (instance infeasible)")
        return len(errors) == 0, errors

    def summary(self) -> str:
        valid, errors = self.validate()
        status = "✓ Valid" if valid else f"✗ {len(errors)} errors"
        return (
            f"BinPacking('{self.name}')\n"
            f"  n={len(self.weights)} items\n"
            f"  m={self.m} bins (upper bound)\n"
            f"  capacity={self.capacity:.3g}\n"
            f"  Status: {status}"
        )

    class MILPConverter(Converter["BinPacking", pyo.ConcreteModel]):
        def convert(self, problem: "BinPacking") -> pyo.ConcreteModel:
            n = len(problem.weights)
            m = problem.m
            w = problem.weights
            cap = problem.capacity

            model = pyo.ConcreteModel(name=getattr(problem, "name", None) or "BinPacking")
            model.B = pyo.RangeSet(0, m - 1)
            model.I = pyo.RangeSet(0, n - 1)

            model.x = pyo.Var(model.B, model.I, domain=pyo.Binary)
            model.y = pyo.Var(model.B, domain=pyo.Binary)

            def cover_rule(mdl, i):
                return sum(mdl.x[b, i] for b in mdl.B) == 1

            model.cover = pyo.Constraint(model.I, rule=cover_rule)

            def cap_rule(mdl, b):
                return sum(w[i] * mdl.x[b, i] for i in mdl.I) <= cap * mdl.y[b]

            model.capacity = pyo.Constraint(model.B, rule=cap_rule)
            model.obj = pyo.Objective(expr=sum(model.y[b] for b in model.B), sense=pyo.minimize)
            return model


@dataclass
class Knapsack(BinPacking):
    m: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.m = 1


    class MILPConverter(BinPacking.MILPConverter):
        pass

