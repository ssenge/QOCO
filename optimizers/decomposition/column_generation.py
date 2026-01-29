from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Protocol

import pyomo.environ as pyo

from qoco.converters.decomposition.column_generation import ColumnGenerationDecomposition, SetPartitionMaster
from qoco.core.optimizer import Optimizer
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.plans.column_generation import ColumnGenState


@dataclass(frozen=True)
class ColGenIterLog:
    it: int
    num_columns: int
    master_obj: float | None
    pricing_partition: Any | None
    pricing_reduced_cost: float | None
    added_column: tuple[Any, ...] | None
    duals: dict[Any, float] | None

    def __str__(self) -> str:
        return (
            f"ColGenIterLog(it={self.it}, columns={self.num_columns}, "
            f"master_obj={self.master_obj}, partition={self.pricing_partition}, "
            f"reduced_cost={self.pricing_reduced_cost}, added={self.added_column}, "
            f"duals={self.duals})"
        )


@dataclass(frozen=True)
class ColGenRunLog:
    iterations: list[ColGenIterLog]

    def __str__(self) -> str:
        return "ColGenRunLog\n" + "\n".join(str(it) for it in self.iterations)


class FinalStepStrategy(Protocol):
    def run(self, state: ColumnGenState) -> Solution: ...


@dataclass(frozen=True)
class NoFinalStepStrategy:
    def run(self, state: ColumnGenState) -> Solution:
        if state.last_master_solution is None:
            return Solution(status=Status.UNKNOWN, objective=math.inf, var_values={}, info={"iters": int(state.it)})
        out = state.last_master_solution
        out.info = dict(out.info)
        out.info.setdefault("colgen_iters", int(state.it))
        out.info.setdefault("colgen_columns", int(len(state.master.columns)))
        return out


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
    final_step_strategy: FinalStepStrategy | None = None
    tol: float = 1e-6
    max_steps: int = 10_000
    time_limit_s: float | None = None

    def __post_init__(self) -> None:
        if self.master_solver is None:
            raise ValueError("master_solver is required")

    def _optimize(self, decomp: ColumnGenerationDecomposition) -> Solution:
        state = ColumnGenState(master=decomp.master, pricing=decomp.pricing, tol=self.tol)
        t0 = time.perf_counter()
        partition_keys = list(state.pricing.partitions() or [])
        partition_pos = 0
        state.partition_keys = partition_keys
        log = ColGenRunLog(
            iterations=[
                ColGenIterLog(
                    it=0,
                    num_columns=int(len(state.master.columns)),
                    master_obj=None,
                    pricing_partition=None,
                    pricing_reduced_cost=None,
                    added_column=None,
                    duals=None,
                )
            ]
        )

        for _ in range(int(self.max_steps)):
            if self.time_limit_s is not None and (time.perf_counter() - t0) >= float(self.time_limit_s):
                break
            master_solution = self.master_solver.optimize(state.master.model, log=False)
            if master_solution.status not in (Status.OPTIMAL, Status.FEASIBLE):
                raise RuntimeError(f"column_generation: master solve failed: {master_solution.status}")
            state.last_master_solution = master_solution
            state.last_duals = state.master.get_duals()

            partition = None
            if partition_keys:
                partition = partition_keys[partition_pos % len(partition_keys)]
            pricing_result = state.pricing.price(state.last_duals, partition)
            state.last_pricing_result = pricing_result
            added_column = None
            if pricing_result.column is not None:
                added_column = tuple(sorted(pricing_result.column.coverage.keys(), key=str))
            log.iterations.append(
                ColGenIterLog(
                    it=int(state.it) + 1,
                    num_columns=int(len(state.master.columns)),
                    master_obj=float(pyo.value(state.master.model.obj)),
                    pricing_partition=partition,
                    pricing_reduced_cost=float(pricing_result.reduced_cost),
                    added_column=added_column,
                    duals=dict(state.last_duals),
                )
            )

            if pricing_result.column is None or pricing_result.reduced_cost >= -state.tol:
                break
            state.master.add_column(pricing_result.column)
            state.it += 1
            if partition_keys:
                partition_pos = (partition_pos + 1) % len(partition_keys)

        if self.final_step_strategy is None:
            solution = NoFinalStepStrategy().run(state)
        else:
            solution = self.final_step_strategy.run(state)
        solution.info = dict(solution.info)
        solution.info["colgen_log"] = log
        return solution
