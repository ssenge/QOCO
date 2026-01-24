from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Sequence
from abc import ABC, abstractmethod

import pyomo.environ as pyo

from qoco.core.solution import Solution

RowKey = Hashable


@dataclass
class Column:
    id: str
    cost: float
    coverage: Dict[RowKey, float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PricingResult:
    column: Column | None
    reduced_cost: float


class PricingAdapter(ABC):
    def partitions(self) -> Sequence[Any] | None:
        return None

    @abstractmethod
    def build(self, duals: Dict[RowKey, float], partition: Any | None = None) -> Any: ...

    @abstractmethod
    def extract(self, problem: Any, solution: Solution) -> PricingResult: ...


@dataclass
class SetPartitionMaster:
    rows: Sequence[RowKey]
    columns: list[Column]
    lp_relaxation: bool = True
    exact_count: int | None = None

    model: pyo.ConcreteModel = field(init=False)
    _row_constraints: Dict[RowKey, pyo.Constraint] = field(init=False, default_factory=dict)
    _count_constraint: pyo.Constraint | None = field(init=False, default=None)
    _row_coeffs: Dict[RowKey, list[float]] = field(init=False, default_factory=dict)
    _costs: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.model = pyo.ConcreteModel()
        self.model.rows = pyo.Set(initialize=list(self.rows))
        domain = pyo.UnitInterval if self.lp_relaxation else pyo.Binary
        self.model.z = pyo.VarList(domain=domain)
        self.model.row_cons = pyo.ConstraintList()
        if self.lp_relaxation:
            self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        self._row_coeffs = {row: [] for row in self.rows}
        self._costs = []

        for col in self.columns:
            self._add_column_internal(col, rebuild=False)
        self._rebuild_constraints()
        self._update_objective()

    def add_column(self, column: Column) -> None:
        self.columns.append(column)
        self._add_column_internal(column, rebuild=True)

    def _add_column_internal(self, column: Column, *, rebuild: bool) -> None:
        self.model.z.add()
        self._costs.append(column.cost)
        for row in self.rows:
            self._row_coeffs[row].append(column.coverage.get(row, 0.0))
        if rebuild:
            self._rebuild_constraints()
            self._update_objective()

    def _rebuild_constraints(self) -> None:
        for con in self._row_constraints.values():
            con.deactivate()
        self._row_constraints = {}

        for row in self.rows:
            expr = sum(self._row_coeffs[row][j - 1] * self.model.z[j] for j in self.model.z)
            con = self.model.row_cons.add(expr == 1)
            self._row_constraints[row] = con
        if self.exact_count is not None:
            if self._count_constraint is not None:
                self._count_constraint.deactivate()
            expr = sum(self.model.z[j] for j in self.model.z)
            self._count_constraint = self.model.row_cons.add(expr == self.exact_count)

    def _update_objective(self) -> None:
        expr = sum(self._costs[j - 1] * self.model.z[j] for j in self.model.z)
        if hasattr(self.model, "obj"):
            self.model.obj.set_value(expr)
        else:
            self.model.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

    def get_duals(self) -> Dict[RowKey, float]:
        duals: Dict[RowKey, float] = {}
        for row, con in self._row_constraints.items():
            dv = self.model.dual.get(con, None)
            duals[row] = dv if dv is not None else 0.0
        return duals

    def coverage_for_indices(self, indices: Sequence[int]) -> Dict[RowKey, float]:
        coverage: Dict[RowKey, float] = {}
        for row in self.rows:
            total = 0.0
            for idx in indices:
                if 0 < idx <= len(self._row_coeffs[row]):
                    total += self._row_coeffs[row][idx - 1]
            coverage[row] = total
        return coverage

    def build_integer_model(self) -> pyo.ConcreteModel:
        return SetPartitionMaster(
            rows=self.rows,
            columns=list(self.columns),
            lp_relaxation=False,
            exact_count=self.exact_count,
        ).model


@dataclass
class ColumnGenerationDecomposition:
    master: SetPartitionMaster
    pricing: PricingAdapter
