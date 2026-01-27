from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import pyomo.environ as pyo

from qoco.converters.decomposition.column_generation import ColumnGenerationDecomposition, SetPartitionMaster
from qoco.core.optimizer import Optimizer
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.multistage import MultiStageSolver
from qoco.optimizers.decomposition.plans.column_generation import ClassicColumnGenPlan, ColumnGenState


class FinalStepStrategy(Protocol):
    def run(self, state: ColumnGenState) -> Solution: ...


@dataclass(frozen=True)
class NoFinalStepStrategy:
    def run(self, state: ColumnGenState) -> Solution:
        return ClassicColumnGenPlan().to_solution(state)


@dataclass(frozen=True)
class IntegerMasterStrategy:
    optimizer: Optimizer[pyo.ConcreteModel, object, Solution]

    def run(self, state: ColumnGenState) -> Solution:
        model = state.master.build_integer_model()
        sol = self.optimizer.optimize(model, log=False)
        sol.info = dict(sol.info)
        sol.info["colgen_iters"] = int(state.it)
        sol.info["colgen_columns"] = int(len(state.master.columns))
        return sol


@dataclass(frozen=True)
class RoundingStrategy:
    tol: float = 1e-6

    def run(self, state: ColumnGenState) -> Solution:
        master = state.master
        model = master.model
        chosen = [j for j in model.z if pyo.value(model.z[j]) >= 0.5]
        coverage = master.coverage_for_indices(chosen)
        feasible = all(abs(coverage[row] - 1) <= self.tol for row in master.rows)
        objective = sum(master.columns[idx - 1].cost for idx in chosen) if chosen else math.inf
        status = Status.FEASIBLE if feasible else Status.UNKNOWN
        var_values = {f"z[{idx}]": 1.0 for idx in chosen}
        info = {"colgen_iters": int(state.it), "colgen_columns": int(len(master.columns))}
        return Solution(status=status, objective=objective, var_values=var_values, info=info)


@dataclass(frozen=True)
class BranchAndPriceStrategy:
    master_optimizer: Optimizer[pyo.ConcreteModel, object, Solution]
    pricing_optimizer: Optimizer[Any, object, Solution]

    def run(self, state: ColumnGenState) -> Solution:
        raise NotImplementedError("Branch-and-price strategy not implemented yet.")


@dataclass
class ColGenOptimizer(Optimizer[ColumnGenerationDecomposition, ColumnGenerationDecomposition, Solution]):
    master_solver: Optimizer[pyo.ConcreteModel, object, Solution] | None = None
    pricing_solver: Optimizer[Any, object, Solution] | None = None
    final_step_strategy: FinalStepStrategy | None = None
    tol: float = 1e-6
    max_steps: int = 10_000
    time_limit_s: float | None = None

    def __post_init__(self) -> None:
        if self.master_solver is None:
            raise ValueError("master_solver is required")
        if self.pricing_solver is None:
            raise ValueError("pricing_solver is required")

    def _optimize(self, decomp: ColumnGenerationDecomposition) -> Solution:
        state = ColumnGenState(
            master=decomp.master,
            pricing=decomp.pricing,
            master_solver=self.master_solver,
            pricing_solver=self.pricing_solver,
            tol=self.tol,
        )
        plan = ClassicColumnGenPlan()
        sol = MultiStageSolver(
            plan=plan,
            state=state,
            max_steps=int(self.max_steps),
            time_limit_s=self.time_limit_s,
        ).run()
        if self.final_step_strategy is None:
            return sol
        return self.final_step_strategy.run(state)
