"""
DiscreteKOA — Natively discrete Kepler Optimization Algorithm.

Translates KOA physics (gravitational attraction, orbital trajectories, escape velocity)
into discrete move operators that work directly on assignment/permutation/integer vectors.

No continuous positions, no transfer functions, no binarization.
Solutions are integer arrays; moves are slot-level reassignments guided by KOA-derived
probabilities (force, orbital velocity, escape controller).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator

from qoco.optimizers.koa.engine.base import BasePMH2, PMHConfig, PMHState, PMHUtils
from qoco.optimizers.koa.engine.components import (
    TrajectoryType,
    TrajectorySelector, RandomTrajectorySelector,
    MassComputer, RankBasedMassComputer,
    GravityScheduler, ExponentialDecayScheduler,
)
from qoco.optimizers.koa.engine.reference_selectors import ReferenceSelector, RandomSelector


@dataclass
class DiscreteConfig(PMHConfig):
    """Configuration for discrete optimization."""
    pass


@dataclass
class DiscreteKOAState(PMHState):
    """Iteration state for the discrete KOA."""
    Sun_idx: int = 0
    Sun_Score: float = 0.0

    R: np.ndarray = field(default_factory=lambda: np.array([]))
    Rnorm: np.ndarray = field(default_factory=lambda: np.array([]))

    MS: np.ndarray = field(default_factory=lambda: np.array([]))
    m: np.ndarray = field(default_factory=lambda: np.array([]))
    M: float = 0.0

    Fg: np.ndarray = field(default_factory=lambda: np.array([]))
    Fg_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    a2: float = 0.0

    max_iter: int = 1000


@dataclass
class DiscreteProblem(ABC):
    """
    Abstract interface for a discrete optimization problem.

    Individual = 1-D int array of shape (dimension,).
    Each slot j holds an integer value from a problem-specific feasible set.
    """
    dimension: int

    @abstractmethod
    def init_solution(self, rng: Generator) -> np.ndarray:
        """Create a random feasible individual (int array of length dimension)."""
        raise NotImplementedError

    @abstractmethod
    def random_slot_value(self, slot: int, rng: Generator) -> int:
        """Return a uniformly random feasible value for the given slot."""
        raise NotImplementedError

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between two individuals. Default: Hamming."""
        return float(np.sum(a != b))

    def repair(self, x: np.ndarray) -> np.ndarray:
        """Optional repair operator for constraint satisfaction. Default: identity."""
        return x


@dataclass
class DiscreteKOA(BasePMH2[np.ndarray, DiscreteKOAState]):
    """
    Natively discrete Kepler Optimization Algorithm.

    Individual = np.ndarray of int, shape (dimension,).

    KOA concepts -> discrete analogs:
      Position         -> integer assignment vector
      Euclidean dist   -> Hamming distance (via problem.distance)
      Force Fg[i]      -> probability of changing each slot
      Trajectory type  -> determines *source* of new slot values
      Escape           -> large random perturbation
      Near-Sun         -> crossover toward best + small mutation
      Far-from-Sun     -> crossover from references + mutation
    """
    config: DiscreteConfig
    problem: DiscreteProblem = field(repr=False)

    Tc: int = 3
    eps: float = 2.2e-16

    reference_selector: ReferenceSelector = field(default_factory=RandomSelector)
    trajectory_selector: TrajectorySelector = field(default_factory=RandomTrajectorySelector)
    mass_computer: MassComputer = field(default_factory=RankBasedMassComputer)
    gravity_scheduler: GravityScheduler = field(default_factory=ExponentialDecayScheduler)

    orbital: np.ndarray = field(init=False)
    T_orbital: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        N = self.config.population_size
        self.orbital = self.rng.random(N)
        self.T_orbital = np.abs(self.rng.standard_normal(N))

    def init_population(self) -> list[np.ndarray]:
        return [self.problem.init_solution(self.rng) for _ in range(self.config.population_size)]

    def constrain(self, x: np.ndarray) -> np.ndarray:
        return self.problem.repair(x)

    def _find_sun(self) -> Tuple[int, float]:
        idx = int(np.argmin(self.fit))
        return idx, float(self.fit[idx])

    def _compute_distances(self, sun_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sun = self.pop[sun_idx]
        R = np.array([self.problem.distance(self.pop[i], sun) for i in range(len(self.pop))])
        Rnorm = PMHUtils.normalize(R)
        return R, Rnorm

    def _compute_gravitational_force(self, M: float, MSnorm: np.ndarray,
                                     Mnorm: np.ndarray, Rnorm: np.ndarray) -> np.ndarray:
        N = self.config.population_size
        return self.orbital[:N] * M * (MSnorm * Mnorm) / (Rnorm ** 2 + self.eps) + self.rng.random(N)

    def _compute_escape_controller(self) -> float:
        cycle = max(1, self.config.max_evaluations / self.Tc)
        return -1.0 - ((self.evals % cycle) / cycle)

    def prepare_state_begin(self, t: int, state: Optional[DiscreteKOAState]) -> DiscreteKOAState:
        sun_idx, sun_score = self._find_sun()
        R, Rnorm = self._compute_distances(sun_idx)

        MS, m = self.mass_computer.compute(self.fit, sun_score, self.rng, state)
        MSnorm = PMHUtils.normalize(MS)
        Mnorm = PMHUtils.normalize(m)

        M = self.gravity_scheduler.get_M(self.evals, self.config.max_evaluations, self.fit, state)
        Fg = self._compute_gravitational_force(M, MSnorm, Mnorm, Rnorm)

        Fg_max = Fg.max()
        Fg_norm = Fg / Fg_max if Fg_max > self.eps else np.zeros_like(Fg)

        a2 = self._compute_escape_controller()

        return DiscreteKOAState(
            t=t,
            Sun_idx=sun_idx, Sun_Score=sun_score,
            R=R, Rnorm=Rnorm,
            MS=MS, m=m, M=M,
            Fg=Fg, Fg_norm=Fg_norm, a2=a2,
            max_iter=self.config.max_evaluations // max(1, self.config.population_size),
        )

    def propose(self, i: int, s: DiscreteKOAState) -> np.ndarray:
        D = self.config.dimension
        xi = self.pop[i]

        planet_a, planet_b = self.reference_selector.select(
            i, np.array(self.pop), self.fit, self.rng, s,
        )

        trajectory = self.trajectory_selector.select(
            i, s.Rnorm[i], np.array(self.pop), self.fit, self.rng, s,
        )

        if trajectory == TrajectoryType.ESCAPE:
            return self._escape_move(xi, planet_a, planet_b, s, D)
        elif trajectory == TrajectoryType.NEAR_SUN:
            return self._near_sun_move(i, xi, planet_a, planet_b, s, D)
        else:
            return self._far_from_sun_move(i, xi, planet_a, planet_b, s, D)

    def _escape_move(self, xi: np.ndarray, planet_a: int, planet_b: int,
                     s: DiscreteKOAState, D: int) -> np.ndarray:
        """Large perturbation: randomize a fraction of slots, blend from references."""
        p_change = 0.3 + 0.5 * abs(s.a2 + 1)
        result = xi.copy()
        mask = self.rng.random(D) < p_change
        n_change = int(mask.sum())
        if n_change == 0:
            mask[self.rng.integers(D)] = True

        xm_sources = [self.pop[planet_b], self.pop[s.Sun_idx], xi]
        for j in np.where(mask)[0]:
            source = xm_sources[self.rng.integers(3)]
            if self.rng.random() < 0.3:
                result[j] = self.problem.random_slot_value(int(j), self.rng)
            else:
                result[j] = source[j]
        return result

    def _near_sun_move(self, i: int, xi: np.ndarray, planet_a: int, planet_b: int,
                       s: DiscreteKOAState, D: int) -> np.ndarray:
        """Crossover toward Sun, blended with reference, plus mutation."""
        fg = s.Fg_norm[i]
        rnorm_i = s.Rnorm[i]

        p_sun = 0.1 + 0.5 * fg
        p_ref = 0.05 + 0.2 * (1.0 - rnorm_i)
        p_mut = 0.02 + 0.08 * (1.0 - fg)

        result = xi.copy()
        sun = self.pop[s.Sun_idx]
        ref_b = self.pop[planet_b]

        for j in range(D):
            r = self.rng.random()
            if r < p_sun:
                result[j] = sun[j]
            elif r < p_sun + p_ref:
                result[j] = ref_b[j]
            elif r < p_sun + p_ref + p_mut:
                result[j] = self.problem.random_slot_value(j, self.rng)

        if np.array_equal(result, xi):
            slot = self.rng.integers(D)
            result[slot] = sun[slot]

        return result

    def _far_from_sun_move(self, i: int, xi: np.ndarray, planet_a: int, planet_b: int,
                           s: DiscreteKOAState, D: int) -> np.ndarray:
        """Crossover from references with exploration bias."""
        fg = s.Fg_norm[i]
        rnorm_i = s.Rnorm[i]

        p_ref_a = 0.1 + 0.3 * fg
        p_ref_b = 0.05 + 0.15 * fg
        p_mut = 0.05 + 0.15 * rnorm_i

        result = xi.copy()
        ref_a = self.pop[planet_a]
        ref_b = self.pop[planet_b]

        for j in range(D):
            r = self.rng.random()
            if r < p_ref_a:
                result[j] = ref_a[j]
            elif r < p_ref_a + p_ref_b:
                result[j] = ref_b[j]
            elif r < p_ref_a + p_ref_b + p_mut:
                result[j] = self.problem.random_slot_value(j, self.rng)

        p_pull = 0.05 + 0.1 * fg
        sun = self.pop[s.Sun_idx]
        for j in range(D):
            if self.rng.random() < p_pull and result[j] != sun[j]:
                result[j] = sun[j]

        if np.array_equal(result, xi):
            slot = self.rng.integers(D)
            result[slot] = self.problem.random_slot_value(slot, self.rng)

        return result
