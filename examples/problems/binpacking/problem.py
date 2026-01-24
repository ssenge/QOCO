from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.core.problem import Problem


@dataclass(frozen=True)
class BinPackingMatrices:
    n: int
    m: int
    weights: np.ndarray
    capacity: float
    name: Optional[str] = None
    id: Optional[str] = None
    optimal_bins: Optional[int] = None
    optimal_time: Optional[float] = None


@dataclass
class BinPacking(Problem):
    name: str
    weights: np.ndarray
    capacity: float
    m: int

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.capacity <= 0:
            errors.append("capacity must be > 0")
        if self.m <= 0:
            errors.append("m (bin upper bound) must be > 0")
        if self.weights.ndim != 1:
            errors.append("weights must be a 1D array")
        if self.weights.size == 0:
            errors.append("no items (n=0)")
        if np.any(self.weights <= 0):
            errors.append("all weights must be > 0")
        if np.any(self.weights > self.capacity):
            errors.append("some weights exceed capacity (instance infeasible)")
        return len(errors) == 0, errors

    def summary(self) -> str:
        valid, errors = self.validate()
        status = "✓ Valid" if valid else f"✗ {len(errors)} errors"
        return (
            f"BinPacking('{self.name}')\n"
            f"  n={int(self.weights.size)} items\n"
            f"  m={int(self.m)} bins (upper bound)\n"
            f"  capacity={float(self.capacity):.3g}\n"
            f"  Status: {status}"
        )

    class MILPConverter(Converter[Any, pyo.ConcreteModel]):
        def convert(self, problem: Any) -> pyo.ConcreteModel:
            n = int(problem.n) if hasattr(problem, "n") else int(np.asarray(problem.weights).shape[0])
            m = int(problem.m)
            w = np.asarray(problem.weights, dtype=float)
            cap = float(problem.capacity)

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


def load_test_instances_json(path: Path, *, limit: int) -> List[BinPackingMatrices]:
    data = json.loads(path.read_text())
    out: List[BinPackingMatrices] = []
    for item in data[: int(limit)]:
        weights = np.array(item["weights"], dtype=np.float32)
        cap = float(item["capacity"])
        m = int(item.get("m", len(weights)))
        out.append(
            BinPackingMatrices(
                n=int(len(weights)),
                m=m,
                weights=weights,
                capacity=cap,
                name=item.get("name"),
                id=item.get("id"),
                optimal_bins=int(item["optimal_bins"]) if "optimal_bins" in item else None,
                optimal_time=float(item["optimal_time"]) if "optimal_time" in item else None,
            )
        )
    return out
