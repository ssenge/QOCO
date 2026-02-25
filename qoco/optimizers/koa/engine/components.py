"""
Pluggable KOA components — trajectory selection, mass computation, gravity scheduling.

These components are representation-agnostic: they operate on scalar fitness values,
normalized distances, and eval counts, so they work for both continuous and discrete KOA.

Ported from KOA/src/algos/new/koa_components.py (subset relevant for DiscreteKOA).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from enum import Enum

import numpy as np
from numpy.random import Generator


# ══════════════════════════════════════════════════════════════════════════════
#                         TRAJECTORY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryType(Enum):
    ESCAPE = "escape"
    NEAR_SUN = "near_sun"
    FAR_FROM_SUN = "far_from_sun"


class TrajectorySelector(ABC):
    @abstractmethod
    def select(self, i: int, Rnorm_i: float, pop: np.ndarray,
               fit: np.ndarray, rng: Generator, state: Any) -> TrajectoryType:
        raise NotImplementedError


@dataclass
class RandomTrajectorySelector(TrajectorySelector):
    """Original KOA: random escape vs distance-based."""
    def select(self, i: int, Rnorm_i: float, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> TrajectoryType:
        if rng.random() < rng.random():
            return TrajectoryType.ESCAPE
        return TrajectoryType.NEAR_SUN if Rnorm_i < 0.5 else TrajectoryType.FAR_FROM_SUN


@dataclass
class AdaptiveTrajectorySelector(TrajectorySelector):
    """Adaptive: more escape early, more exploitation late."""
    early_escape_prob: float = 0.4
    late_escape_prob: float = 0.1

    def select(self, i: int, Rnorm_i: float, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> TrajectoryType:
        progress = getattr(state, 't', 0) / max(1, getattr(state, 'max_iter', 100))
        progress = min(1.0, progress)
        escape_prob = (1.0 - progress) * self.early_escape_prob + progress * self.late_escape_prob
        if rng.random() < escape_prob:
            return TrajectoryType.ESCAPE
        return TrajectoryType.NEAR_SUN if Rnorm_i < 0.5 else TrajectoryType.FAR_FROM_SUN


@dataclass
class FitnessAdaptiveTrajectorySelector(TrajectorySelector):
    """Worse planets explore more (escape), better planets exploit (near-sun)."""
    def select(self, i: int, Rnorm_i: float, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> TrajectoryType:
        ranks = np.argsort(np.argsort(fit))
        rank_ratio = ranks[i] / max(1, len(fit) - 1)
        escape_prob = 0.1 + 0.3 * rank_ratio
        if rng.random() < escape_prob:
            return TrajectoryType.ESCAPE
        return TrajectoryType.NEAR_SUN if Rnorm_i < 0.5 else TrajectoryType.FAR_FROM_SUN


# ══════════════════════════════════════════════════════════════════════════════
#                         MASS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

class MassComputer(ABC):
    @abstractmethod
    def compute(self, fit: np.ndarray, Sun_Score: float,
                rng: Generator, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class OriginalMassComputer(MassComputer):
    """Original KOA mass computation (Eq. 8-9)."""
    eps: float = 1e-300

    def compute(self, fit: np.ndarray, Sun_Score: float, rng: Generator,
                state: Any) -> Tuple[np.ndarray, np.ndarray]:
        N = len(fit)
        worstFitness = float(fit.max())
        denom = float((fit - worstFitness).sum())
        if abs(denom) < self.eps:
            denom = -self.eps
        MS = rng.random(N) * (Sun_Score - worstFitness) / denom
        m = (fit - worstFitness) / denom
        return MS, m


@dataclass
class RankBasedMassComputer(MassComputer):
    """Rank-based mass computation (robust to outliers)."""
    def compute(self, fit: np.ndarray, Sun_Score: float, rng: Generator,
                state: Any) -> Tuple[np.ndarray, np.ndarray]:
        N = len(fit)
        ranks = np.argsort(np.argsort(fit)).astype(float)
        m = (N - 1 - ranks) / max(1, N - 1)
        MS = rng.random(N) * m
        return MS, m


@dataclass
class SoftmaxMassComputer(MassComputer):
    """Softmax-based mass computation."""
    temperature: float = 1.0
    eps: float = 1e-10

    def compute(self, fit: np.ndarray, Sun_Score: float, rng: Generator,
                state: Any) -> Tuple[np.ndarray, np.ndarray]:
        neg_fit = -fit
        exp_fit = np.exp((neg_fit - neg_fit.max()) / self.temperature)
        m = exp_fit / (exp_fit.sum() + self.eps)
        MS = rng.random(len(fit)) * m
        return MS, m


# ══════════════════════════════════════════════════════════════════════════════
#                         GRAVITY SCHEDULING
# ══════════════════════════════════════════════════════════════════════════════

class GravityScheduler(ABC):
    @abstractmethod
    def get_M(self, evals: int, max_evals: int,
              fit: np.ndarray, state: Any) -> float:
        raise NotImplementedError


@dataclass
class ExponentialDecayScheduler(GravityScheduler):
    """Original KOA exponential decay (Eq. 12)."""
    M0: float = 0.1
    lam: float = 15.0

    def get_M(self, evals: int, max_evals: int, fit: np.ndarray, state: Any) -> float:
        return self.M0 * np.exp(-self.lam * (evals / max(1, max_evals)))


@dataclass
class LinearDecayScheduler(GravityScheduler):
    M0: float = 0.1
    M_min: float = 0.001

    def get_M(self, evals: int, max_evals: int, fit: np.ndarray, state: Any) -> float:
        progress = evals / max(1, max_evals)
        return self.M0 * (1.0 - progress) + self.M_min * progress


@dataclass
class AdaptiveDecayScheduler(GravityScheduler):
    """Faster decay when improving, slower when stuck."""
    M0: float = 0.1
    M_min: float = 0.001
    decay_rate: float = 0.99
    boost_rate: float = 1.1

    def __post_init__(self):
        self._current_M = self.M0
        self._prev_best = float('inf')

    def get_M(self, evals: int, max_evals: int, fit: np.ndarray, state: Any) -> float:
        best = float(fit.min())
        if best < self._prev_best:
            self._current_M *= self.decay_rate
        else:
            self._current_M = min(self.M0, self._current_M * self.boost_rate)
        self._prev_best = best
        self._current_M = max(self.M_min, self._current_M)
        return self._current_M


@dataclass
class RestartScheduler(GravityScheduler):
    """Reset M when stagnation detected."""
    M0: float = 0.1
    lam: float = 15.0
    stagnation_threshold: int = 50

    def __post_init__(self):
        self._stagnation_count = 0
        self._prev_best = float('inf')
        self._restart_offset = 0

    def get_M(self, evals: int, max_evals: int, fit: np.ndarray, state: Any) -> float:
        best = float(fit.min())
        if best < self._prev_best * 0.9999:
            self._stagnation_count = 0
            self._prev_best = best
        else:
            self._stagnation_count += 1
        if self._stagnation_count >= self.stagnation_threshold:
            self._restart_offset = evals
            self._stagnation_count = 0
        effective_evals = evals - self._restart_offset
        return self.M0 * np.exp(-self.lam * (effective_evals / max(1, max_evals)))
