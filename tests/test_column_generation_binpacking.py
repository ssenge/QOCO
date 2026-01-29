from __future__ import annotations

from dataclasses import dataclass
import pyomo.environ as pyo

from qoco.converters.decomposition.column_generation import (
    Column,
    ColumnGenerationDecomposition,
    PricingResult,
    PricingStrategy,
    SetPartitionMaster,
    set_pricing_objective,
)
from qoco.converters.identity import IdentityConverter
from qoco.examples.problems.binpacking.problem import BinPacking, Knapsack
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.column_generation import ColGenOptimizer, IntegerMasterStrategy
from qoco.optimizers.highs import HiGHSOptimizer


@dataclass
class BinPackingPricing(PricingStrategy):
    sizes: list[int]
    capacity: int
    solver: HiGHSOptimizer

    def price(self, duals: dict[int, float], partition=None) -> PricingResult:
        knapsack = Knapsack(
            name="colgen_pricing_knapsack",
            weights=[s for s in self.sizes],
            capacity=self.capacity,
        )
        model = Knapsack.MILPConverter().convert(knapsack)
        model.cover.deactivate()
        set_pricing_objective(model, 1 - sum(duals[i] * model.x[0, i] for i in model.I))
        solution = self.solver.optimize(model, log=False)
        reduced_cost = pyo.value(model.obj_reduced.expr)
        chosen = [idx[1] if isinstance(idx, tuple) else idx for idx in solution.selected_indices("x")]
        if not chosen:
            return PricingResult(column=None, reduced_cost=reduced_cost)
        col = Column(
            id=f"col_{'_'.join(str(i) for i in chosen)}",
            cost=1.0,
            coverage={i: 1.0 for i in chosen},
        )
        return PricingResult(column=col, reduced_cost=reduced_cost)

    def seed(self, duals: dict[int, float]) -> list[Column]:
        return [
            Column(id=f"bin_{i}", cost=1.0, coverage={i: 1.0})
            for i in range(len(self.sizes))
        ]


def test_column_generation_binpacking() -> None:
    sizes = [4, 4, 3, 2, 2]
    capacity = 8
    num_bins = 3
    problem = BinPacking(
        name="binpacking_colgen_test",
        weights=[float(s) for s in sizes],
        capacity=capacity,
        m=num_bins,
    )
    rows = list(range(len(sizes)))

    pricing_solver = HiGHSOptimizer()
    pricing = BinPackingPricing(sizes=sizes, capacity=capacity, solver=pricing_solver)
    initial_columns = pricing.seed({s: 0.0 for s in rows})
    master = SetPartitionMaster(
        rows=rows,
        columns=list(initial_columns),
        lp_relaxation=True,
    )
    decomp = ColumnGenerationDecomposition(master=master, pricing=pricing)

    master_solver = HiGHSOptimizer(capture_duals=True)
    final_solver = pricing_solver
    final_strategy = IntegerMasterStrategy(optimizer=final_solver)

    optimizer = ColGenOptimizer(
        converter=IdentityConverter(),
        master_solver=master_solver,
        final_step_strategy=final_strategy,
        max_steps=100,
    )
    solution = optimizer.optimize(decomp, log=False)

    oracle_converter = BinPacking.MILPConverter()
    oracle_model = oracle_converter.convert(problem)
    oracle_solver = HiGHSOptimizer()
    oracle_solution = oracle_solver.optimize(oracle_model, log=False)

    assert solution.status in (Status.OPTIMAL, Status.FEASIBLE)
    assert oracle_solution.status == Status.OPTIMAL
    assert solution.objective == oracle_solution.objective
    assert int(solution.info.get("colgen_iters", 0)) >= 1
    assert int(solution.info.get("colgen_columns", 0)) > len(initial_columns)
