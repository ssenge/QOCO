"""
HybridKOAOptimizer — qoco Optimizer wrapping the hybrid (classical + VQC) discrete KOA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generic, Optional

import numpy as np

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.koa.types import DiscreteKOAProblem

from qoco.optimizers.koa.engine.discrete_koa import DiscreteConfig
from qoco.optimizers.koa.engine.hybrid_discrete_koa import HybridDiscreteKOA, VQCPlanetConfig
from qoco.optimizers.koa.engine.base import PMHResult
from qoco.optimizers.koa.engine.components import (
    TrajectorySelector, RandomTrajectorySelector,
    MassComputer, RankBasedMassComputer,
    GravityScheduler, ExponentialDecayScheduler,
)
from qoco.optimizers.koa.engine.reference_selectors import ReferenceSelector, RandomSelector


@dataclass
class HybridKOAOptimizer(Generic[P], Optimizer[P, DiscreteKOAProblem, Solution, OptimizerRun, ProblemSummary]):
    """
    Hybrid (classical + VQC) discrete KOA optimizer for qoco.

    Requires a `bitstring_decoder` that maps a measured bitstring (0/1 array)
    to a feasible integer assignment vector compatible with the DiscreteProblem.
    """
    name: str = "HybridDiscreteKOA"
    converter: Converter[P, DiscreteKOAProblem] = field(default_factory=IdentityConverter)

    population_size: int = 30
    max_evaluations: int = 10_000
    max_iterations: int = 2000
    seed: Optional[int] = None

    n_quantum: int = 3
    vqc_n_qubits: int = 8
    vqc_n_layers: int = 2
    vqc_shots: int = 1

    bitstring_decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None

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

        vqc_config = VQCPlanetConfig(
            n_qubits=self.vqc_n_qubits,
            n_layers=self.vqc_n_layers,
            shots=self.vqc_shots,
        )

        if self.bitstring_decoder is None:
            raise ValueError("bitstring_decoder must be provided for HybridKOAOptimizer")

        koa = HybridDiscreteKOA(
            config=config,
            objective_fn=prob.objective_fn,
            problem=prob.problem,
            n_quantum=self.n_quantum,
            vqc_config=vqc_config,
            bitstring_decoder=self.bitstring_decoder,
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
                "n_quantum": self.n_quantum,
                "vqc_n_qubits": self.vqc_n_qubits,
                "vqc_n_layers": self.vqc_n_layers,
            },
        )

        return solution, run
