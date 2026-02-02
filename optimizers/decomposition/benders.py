from __future__ import annotations

from dataclasses import dataclass
import pyomo.environ as pyo

from qoco.converters.decomposition.benders import BendersDecomposition
from qoco.core.optimizer import Optimizer
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution
from qoco.optimizers.decomposition.multistage import MultiStageSolver
from qoco.optimizers.decomposition.plans.benders import BendersState, ClassicBendersPlan


@dataclass(frozen=True)
class ClassicBendersSolver:
    decomp: BendersDecomposition
    master_solver: Optimizer[pyo.ConcreteModel, object, Solution, OptimizerRun, ProblemSummary]
    sub_solver: Optimizer[pyo.ConcreteModel, object, Solution, OptimizerRun, ProblemSummary]
    tol: float = 1e-6
    max_steps: int = 10_000
    time_limit_s: float | None = None

    def run(self) -> Solution:
        state = BendersState(decomp=self.decomp, master_solver=self.master_solver, sub_solver=self.sub_solver, tol=float(self.tol))
        return MultiStageSolver(plan=ClassicBendersPlan(), state=state, max_steps=int(self.max_steps), time_limit_s=self.time_limit_s).run()

