from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import pyomo.environ as pyo

from qoco.converters.decomposition.column_generation import PricingResult, SetPartitionMaster, PricingAdapter
from qoco.core.optimizer import Optimizer
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.multistage import StagePlan, StageResult, StageTask


@dataclass
class ColumnGenState:
    master: SetPartitionMaster
    pricing: PricingAdapter
    master_solver: Optimizer[pyo.ConcreteModel, object, Solution]
    pricing_solver: Optimizer[Any, object, Solution]
    tol: float = 1e-6

    phase: str = "master"  # master | pricing
    it: int = 0
    done: bool = False

    last_master_solution: Solution | None = None
    last_pricing_solution: Solution | None = None
    last_pricing_result: PricingResult | None = None
    last_duals: dict[Any, float] | None = None
    pricing_problem: Any | None = None
    partition_keys: list[Any] | None = None
    partition_pos: int = 0


@dataclass(frozen=True)
class ClassicColumnGenPlan(StagePlan[ColumnGenState, Any, str]):
    def next_task(self, state: ColumnGenState) -> Optional[StageTask[Any, str]]:
        if state.done:
            return None
        if state.phase == "master":
            return StageTask(tag="master", problem=state.master.model, solver=state.master_solver)
        if state.phase == "pricing":
            if state.last_duals is None:
                raise RuntimeError("column_generation: missing duals from master")
            if state.pricing_problem is None:
                partition = None
                if state.partition_keys is None:
                    parts = state.pricing.partitions()
                    state.partition_keys = list(parts) if parts is not None else None
                if state.partition_keys:
                    partition = state.partition_keys[state.partition_pos % len(state.partition_keys)]
                state.pricing_problem = state.pricing.build(state.last_duals, partition)
            return StageTask(tag="pricing", problem=state.pricing_problem, solver=state.pricing_solver)
        raise RuntimeError(f"column_generation: unknown phase {state.phase}")

    def apply(self, state: ColumnGenState, result: StageResult[Any, str]) -> ColumnGenState:
        if result.task.tag == "master":
            if result.solution.status not in (Status.OPTIMAL, Status.FEASIBLE):
                raise RuntimeError(f"column_generation: master solve failed: {result.solution.status}")
            state.last_master_solution = result.solution
            state.last_duals = state.master.get_duals()
            state.phase = "pricing"
            state.pricing_problem = None
            return state

        if result.task.tag == "pricing":
            if result.solution.status not in (Status.OPTIMAL, Status.FEASIBLE):
                raise RuntimeError(f"column_generation: pricing solve failed: {result.solution.status}")
            state.last_pricing_solution = result.solution
            if state.pricing_problem is None:
                raise RuntimeError("column_generation: missing pricing problem")
            pricing_result = state.pricing.extract(state.pricing_problem, result.solution)
            state.last_pricing_result = pricing_result
            if pricing_result.column is None or pricing_result.reduced_cost >= -state.tol:
                state.done = True
                return state
            state.master.add_column(pricing_result.column)
            state.it += 1
            state.phase = "master"
            if state.partition_keys:
                state.partition_pos = (state.partition_pos + 1) % len(state.partition_keys)
            return state

        raise RuntimeError("column_generation: unexpected stage tag")

    def converged(self, state: ColumnGenState) -> bool:
        return bool(state.done)

    def to_solution(self, state: ColumnGenState) -> Solution:
        if state.last_master_solution is None:
            return Solution(status=Status.UNKNOWN, objective=math.inf, var_values={}, info={"iters": int(state.it)})
        out = state.last_master_solution
        out.info = dict(out.info)
        out.info.setdefault("colgen_iters", int(state.it))
        out.info.setdefault("colgen_columns", int(len(state.master.columns)))
        return out
