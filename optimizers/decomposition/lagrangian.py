from __future__ import annotations

from dataclasses import dataclass
import pyomo.environ as pyo

from qoco.converters.decomposition.lagrangian import LagrangianSplit
from qoco.core.optimizer import Optimizer
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution
from qoco.optimizers.decomposition.multistage import MultiStageSolver
from qoco.optimizers.decomposition.plans.lagrangian import ClassicLagrangianPlan, LagrangianState


@dataclass(frozen=True)
class ClassicLagrangianSolver:
    split: LagrangianSplit
    sub_solver: Optimizer[pyo.ConcreteModel, object, Solution, OptimizerRun, ProblemSummary]
    step_size: float = 0.1
    max_iters: int = 25
    time_limit_s: float | None = None

    def run(self) -> Solution:
        state = LagrangianState(split=self.split, sub_solver=self.sub_solver, step_size=float(self.step_size), max_iters=int(self.max_iters))
        return MultiStageSolver(plan=ClassicLagrangianPlan(), state=state, max_steps=10_000, time_limit_s=self.time_limit_s).run()

