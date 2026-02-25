"""
KOAOptimizer — qoco Optimizer wrapping the discrete Kepler Optimization Algorithm.

Converts an arbitrary Problem into a DiscreteProblem (via a problem-specific Converter),
runs DiscreteKOA, and returns a standard qoco Solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generic, Optional

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.koa.types import DiscreteKOAProblem

from qoco.optimizers.koa.engine.discrete_koa import DiscreteConfig, DiscreteKOA
from qoco.optimizers.koa.engine.base import PMHResult
from qoco.optimizers.koa.engine.components import (
    TrajectorySelector, RandomTrajectorySelector,
    MassComputer, RankBasedMassComputer,
    GravityScheduler, ExponentialDecayScheduler,
)
from qoco.optimizers.koa.engine.reference_selectors import ReferenceSelector, RandomSelector


@dataclass
class KOAOptimizer(Generic[P], Optimizer[P, DiscreteKOAProblem, Solution, OptimizerRun, ProblemSummary]):
    """
    Discrete KOA optimizer for the qoco framework.

    The Converter[P, DiscreteKOAProblem] is problem-specific and must produce a
    DiscreteKOAProblem containing the DiscreteProblem hooks, objective function,
    and variable-name mapping.
    """
    name: str = "DiscreteKOA"
    converter: Converter[P, DiscreteKOAProblem] = field(default_factory=IdentityConverter)

    population_size: int = 30
    max_evaluations: int = 10_000
    max_iterations: int = 2000
    seed: Optional[int] = None

    reference_selector: ReferenceSelector = field(default_factory=RandomSelector)
    trajectory_selector: TrajectorySelector = field(default_factory=RandomTrajectorySelector)
    mass_computer: MassComputer = field(default_factory=RankBasedMassComputer)
    gravity_scheduler: GravityScheduler = field(default_factory=ExponentialDecayScheduler)

    def _optimize(self, prob: DiscreteKOAProblem) -> tuple[Solution, OptimizerRun]:
        config = DiscreteConfig(
            dimension=prob.problem.dimension,
            population_size=self.population_size,
            max_evaluations=self.max_evaluations,
            max_iterations=self.max_iterations,
            seed=self.seed,
        )

        koa = DiscreteKOA(
            config=config,
            objective_fn=prob.objective_fn,
            problem=prob.problem,
            reference_selector=self.reference_selector,
            trajectory_selector=self.trajectory_selector,
            mass_computer=self.mass_computer,
            gravity_scheduler=self.gravity_scheduler,
        )

        ts_start = datetime.now(timezone.utc)
        result: PMHResult = koa.optimize()
        ts_end = datetime.now(timezone.utc)

        var_values = prob.decode_to_var_values(result.best_position)

        solution = Solution(
            status=Status.FEASIBLE,
            objective=result.best_fitness,
            var_values=var_values,
        )

        run = OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=ts_start,
            optimizer_timestamp_end=ts_end,
            metadata={
                "evaluations": result.evaluations,
                "wall_time": result.wall_time,
                "history_length": len(result.history),
                "population_size": self.population_size,
                "max_evaluations": self.max_evaluations,
            },
        )

        return solution, run
