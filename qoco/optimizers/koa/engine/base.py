"""
BasePMH2 — Lean framework for population-based metaheuristics.

Eval-budget driven, per-individual update loop with granular hooks.
Ported from KOA/src/algos/new/BasePMH2.py (subset: no ContinuousPMH).
"""
from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import inf
from typing import Callable, Generic, List, Optional, TypeVar

import numpy as np
from numpy.random import Generator

Individual = TypeVar('Individual')
State = TypeVar('State', bound='PMHState')


class BudgetExhausted(Exception):
    """Raised when evaluation budget is exhausted."""
    pass


@dataclass
class PMHConfig:
    """Configuration for a population-based metaheuristic."""
    dimension: int
    population_size: int = 25
    max_evaluations: int = 10000
    max_iterations: int = 1000
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class PMHResult(Generic[Individual]):
    """Result container for a PMH run."""
    best_position: Individual
    best_fitness: float
    evaluations: int
    wall_time: float
    history: List[float] = field(default_factory=list)


@dataclass
class PMHState:
    """Base class for iteration state. Subclass to add algorithm-specific fields."""
    t: int = 0


@dataclass
class BasePMH2(ABC, Generic[Individual, State]):
    """
    Abstract base class for population-based metaheuristics.
    Generic over Individual type and State type.

    Subclasses must implement:
      - init_population() -> List[Individual]
      - propose(i, state) -> Individual
      - constrain(x) -> Individual
    """
    config: PMHConfig
    objective_fn: Callable[[Individual], float]

    rng: Generator = field(init=False)
    pop: List[Individual] = field(init=False)
    fit: np.ndarray = field(init=False)
    best_pos: Optional[Individual] = field(init=False, default=None)
    best_fit: float = field(init=False, default=inf)
    evals: int = field(init=False, default=0)
    _raw_objective_fn: Callable[[Individual], float] = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)
        self._raw_objective_fn = self.objective_fn
        self.objective_fn = self._budget_checked_eval

    @abstractmethod
    def init_population(self) -> List[Individual]:
        raise NotImplementedError

    def prepare_state_begin(self, t: int, state: Optional[State]) -> State:
        return PMHState(t=t)  # type: ignore

    @abstractmethod
    def propose(self, i: int, state: State) -> Individual:
        raise NotImplementedError

    @abstractmethod
    def constrain(self, x: Individual) -> Individual:
        raise NotImplementedError

    def accept(self, i: int, x_old: Individual, f_old: float,
               x_new: Individual, f_new: float, state: State) -> bool:
        return f_new <= f_old

    def prepare_state_end(self, t: int, state: State) -> State:
        return state

    def stop_condition(self, t: int) -> bool:
        return self.evals >= self.config.max_evaluations or t >= self.config.max_iterations

    def _budget_checked_eval(self, x: Individual) -> float:
        if self.evals >= self.config.max_evaluations:
            raise BudgetExhausted()
        self.evals += 1
        return self._raw_objective_fn(x)

    def optimize(self) -> PMHResult[Individual]:
        start = time.time()
        history: List[float] = []

        try:
            self.pop = self.init_population()
            self.fit = np.array([self.objective_fn(x) for x in self.pop])

            best_idx = int(np.argmin(self.fit))
            self.best_fit = float(self.fit[best_idx])
            self.best_pos = self.pop[best_idx]
            history.append(self.best_fit)

            state: Optional[State] = None

            for t in itertools.count():
                if self.stop_condition(t):
                    break

                state = self.prepare_state_begin(t, state)

                for i in range(self.config.population_size):
                    x_old = self.pop[i]
                    f_old = self.fit[i]

                    x_new = self.constrain(self.propose(i, state))
                    f_new = self.objective_fn(x_new)

                    if self.accept(i, x_old, f_old, x_new, f_new, state):
                        self.pop[i] = x_new
                        self.fit[i] = f_new
                        if f_new < self.best_fit:
                            self.best_fit = f_new
                            self.best_pos = x_new

                state = self.prepare_state_end(t, state)
                history.append(self.best_fit)

                if self.config.verbose:
                    print(f"Iter {t}, Best: {self.best_fit:.6g}, Evals: {self.evals}")

        except BudgetExhausted:
            pass

        elapsed = time.time() - start
        return PMHResult(
            best_position=self.best_pos if self.best_pos is not None else self.pop[0],
            best_fitness=self.best_fit,
            evaluations=self.evals,
            wall_time=elapsed,
            history=history,
        )


class PMHUtils:
    """Static utility functions for PMH implementations."""

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1]. Returns zeros if constant."""
        ptp = np.ptp(arr)
        return (arr - arr.min()) / ptp if ptp > 0 else np.zeros_like(arr)
