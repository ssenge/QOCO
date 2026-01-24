from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pyomo.environ as pyo

from qoco.converters.decomposition.column_generation import (
    Column,
    ColumnGenerationDecomposition,
    PricingAdapter,
    PricingResult,
    SetPartitionMaster,
)
from qoco.converters.identity import IdentityConverter
from qoco.examples.problems.binpacking.problem import BinPacking
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.column_generation import ColumnGenSolver, IntegerMasterStrategy
from qoco.optimizers.highs import HiGHSOptimizer


@dataclass
class BinPackingPricing(PricingAdapter):
    sizes: list[int]
    capacity: int

    def build(self, duals: dict[int, float], partition=None) -> pyo.ConcreteModel:
        # Pricing for BinPacking degenerates to a 0-1 knapsack for one bin.
        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(0, len(self.sizes) - 1)
        model.x = pyo.Var(model.I, domain=pyo.Binary)
        model.cap = pyo.Constraint(
            expr=sum(self.sizes[i] * model.x[i] for i in model.I) <= self.capacity
        )
        model.obj = pyo.Objective(
            expr=1 - sum(duals.get(i, 0.0) * model.x[i] for i in model.I),
            sense=pyo.minimize,
        )
        return model

    def extract(self, problem: pyo.ConcreteModel, solution: Solution) -> PricingResult:
        reduced_cost = solution.objective
        chosen = solution.selected_indices("x")
        if not chosen:
            return PricingResult(column=None, reduced_cost=reduced_cost)
        col = Column(
            id=f"col_{'_'.join(str(i) for i in chosen)}",
            cost=1.0,
            coverage={i: 1.0 for i in chosen},
        )
        return PricingResult(column=col, reduced_cost=reduced_cost)


def test_column_generation_binpacking() -> None:
    sizes = [4, 4, 3, 2, 2]
    capacity = 8
    num_bins = 3
    problem = BinPacking(
        name="binpacking_colgen_test",
        weights=np.array(sizes, dtype=float),
        capacity=capacity,
        m=num_bins,
    )
    rows = list(range(len(sizes)))

    initial_columns = [
        Column(id="bin_0_1", cost=1.0, coverage={0: 1.0, 1: 1.0}),
        Column(id="bin_2_3", cost=1.0, coverage={2: 1.0, 3: 1.0}),
        Column(id="bin_4", cost=1.0, coverage={4: 1.0}),
    ]
    master = SetPartitionMaster(
        rows=rows,
        columns=list(initial_columns),
        lp_relaxation=True,
        exact_count=num_bins,
    )
    pricing = BinPackingPricing(sizes=sizes, capacity=capacity)
    decomp = ColumnGenerationDecomposition(master=master, pricing=pricing)

    master_solver = HiGHSOptimizer(converter=IdentityConverter(), verbose=False, capture_duals=True)
    pricing_solver = HiGHSOptimizer(converter=IdentityConverter(), verbose=False)
    final_solver = HiGHSOptimizer(converter=IdentityConverter(), verbose=False)
    final_strategy = IntegerMasterStrategy(optimizer=final_solver)

    solver = ColumnGenSolver(
        decomp=decomp,
        master_solver=master_solver,
        pricing_solver=pricing_solver,
        final_step_strategy=final_strategy,
        tol=1e-6,
        max_steps=100,
    )
    solution = solver.run()

    oracle_converter = BinPacking.MILPConverter()
    oracle_model = oracle_converter.convert(problem)
    oracle_model.count = pyo.Constraint(expr=sum(oracle_model.y[b] for b in oracle_model.B) == num_bins)
    oracle_solver = HiGHSOptimizer(converter=IdentityConverter(), verbose=False)
    oracle_solution = oracle_solver.optimize(oracle_model, log=False)

    assert solution.status in (Status.OPTIMAL, Status.FEASIBLE)
    assert oracle_solution.status == Status.OPTIMAL
    assert solution.objective == num_bins
    assert oracle_solution.objective == num_bins
    assert int(solution.info.get("colgen_iters", 0)) >= 1
    assert int(solution.info.get("colgen_columns", 0)) > len(initial_columns)
