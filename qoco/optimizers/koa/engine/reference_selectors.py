"""
Reference planet selection strategies for KOA.

Each planet uses two reference planets (a, b) to compute a differential direction.
Pluggable strategies allow tuning exploration/exploitation balance.

Ported from KOA/src/algos/new/reference_selectors.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from numpy.random import Generator


class ReferenceSelector(ABC):
    """Abstract base class for reference planet selection strategies."""

    @abstractmethod
    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        raise NotImplementedError

    def _random_except(self, N: int, rng: Generator, *exclude: int) -> int:
        candidates = [j for j in range(N) if j not in exclude]
        return rng.choice(candidates)

    def _random_pair_except(self, N: int, rng: Generator, i: int) -> Tuple[int, int]:
        candidates = [j for j in range(N) if j != i]
        selected = rng.choice(candidates, size=2, replace=False)
        return int(selected[0]), int(selected[1])


@dataclass
class RandomSelector(ReferenceSelector):
    """Original KOA strategy: completely random selection."""

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        return self._random_pair_except(len(pop), rng, i)


@dataclass
class FitnessBiasedSelector(ReferenceSelector):
    """Select references biased toward fitter planets."""
    eps: float = 1e-10

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        N = len(pop)
        inv_fit = 1.0 / (np.abs(fit) + self.eps)
        probs = inv_fit.copy()
        probs[i] = 0
        if probs.sum() < self.eps:
            return self._random_pair_except(N, rng, i)
        probs = probs / probs.sum()
        selected = rng.choice(N, size=2, replace=False, p=probs)
        a, b = int(selected[0]), int(selected[1])
        if fit[a] < fit[b]:
            a, b = b, a
        return a, b


@dataclass
class TournamentSelector(ReferenceSelector):
    """Tournament selection for reference planets."""
    tournament_size: int = 3

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        N = len(pop)
        candidates = [j for j in range(N) if j != i]

        def tournament_winner(exclude: List[int]) -> int:
            available = [j for j in candidates if j not in exclude]
            if len(available) == 0:
                return candidates[0] if candidates else 0
            k = min(self.tournament_size, len(available))
            tournament = rng.choice(available, size=k, replace=False)
            return int(tournament[np.argmin(fit[tournament])])

        b = tournament_winner([])
        a = self._random_except(N, rng, i, b)
        return a, b


@dataclass
class NeighborhoodSelector(ReferenceSelector):
    """Select from k-nearest neighbors in position space."""
    k: int = 5

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        N = len(pop)
        k = min(self.k, N - 1)
        distances = np.linalg.norm(pop - pop[i], axis=1)
        distances[i] = np.inf
        neighbors = np.argsort(distances)[:k]
        if len(neighbors) < 2:
            return self._random_pair_except(N, rng, i)
        b = int(neighbors[np.argmin(fit[neighbors])])
        other_neighbors = [n for n in neighbors if n != b]
        if other_neighbors:
            a = int(rng.choice(other_neighbors))
        else:
            a = self._random_except(N, rng, i, b)
        return a, b


@dataclass
class DEBestSelector(ReferenceSelector):
    """DE-inspired: best/1, current-to-best, rand-to-best."""
    strategy: str = "best/1"

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        N = len(pop)
        best_idx = int(np.argmin(fit))
        if self.strategy == "best/1":
            b = best_idx
            a = self._random_except(N, rng, i, best_idx)
        elif self.strategy == "current-to-best":
            a, b = i, best_idx
            if a == b:
                return self._random_pair_except(N, rng, i)
        elif self.strategy == "rand-to-best":
            b = best_idx
            a = self._random_except(N, rng, i, best_idx)
        else:
            return self._random_pair_except(N, rng, i)
        return a, b


@dataclass
class DECompositeSelector(ReferenceSelector):
    """Randomly choose between DE strategies."""
    strategies: Tuple[str, ...] = ("best/1", "rand-to-best", "rand/1")

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        strategy = rng.choice(self.strategies)
        return DEBestSelector(strategy=strategy).select(i, pop, fit, rng, state)


@dataclass
class AdaptiveSelector(ReferenceSelector):
    """Phase-dependent strategy switching: explore early, exploit late."""
    early_threshold: float = 0.3
    late_threshold: float = 0.7

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        progress = getattr(state, 'progress', 0.5)
        if hasattr(state, 't'):
            progress = min(1.0, state.t / 100.0)
        if progress < self.early_threshold:
            return RandomSelector().select(i, pop, fit, rng, state)
        elif progress < self.late_threshold:
            r = rng.random()
            if r < 0.4:
                return FitnessBiasedSelector().select(i, pop, fit, rng, state)
            elif r < 0.7:
                return TournamentSelector(tournament_size=3).select(i, pop, fit, rng, state)
            else:
                return DEBestSelector(strategy="rand-to-best").select(i, pop, fit, rng, state)
        else:
            r = rng.random()
            if r < 0.5:
                return NeighborhoodSelector(k=3).select(i, pop, fit, rng, state)
            elif r < 0.8:
                return DEBestSelector(strategy="current-to-best").select(i, pop, fit, rng, state)
            else:
                return TournamentSelector(tournament_size=5).select(i, pop, fit, rng, state)


@dataclass
class EnsembleSelector(ReferenceSelector):
    """Randomly choose from a set of selectors with equal probability."""
    selectors: Tuple[ReferenceSelector, ...] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.selectors is None:
            self.selectors = (
                RandomSelector(),
                FitnessBiasedSelector(),
                TournamentSelector(tournament_size=3),
                DEBestSelector(strategy="best/1"),
            )

    def select(self, i: int, pop: np.ndarray, fit: np.ndarray,
               rng: Generator, state: Any) -> Tuple[int, int]:
        selector = self.selectors[rng.integers(0, len(self.selectors))]
        return selector.select(i, pop, fit, rng, state)


SELECTOR_REGISTRY = {
    'random': RandomSelector,
    'fitness_biased': FitnessBiasedSelector,
    'tournament': TournamentSelector,
    'neighborhood': NeighborhoodSelector,
    'de_best': lambda: DEBestSelector(strategy="best/1"),
    'de_current_to_best': lambda: DEBestSelector(strategy="current-to-best"),
    'de_rand_to_best': lambda: DEBestSelector(strategy="rand-to-best"),
    'de_composite': DECompositeSelector,
    'adaptive': AdaptiveSelector,
    'ensemble': EnsembleSelector,
}


def get_selector(name: str, **kwargs) -> ReferenceSelector:
    """Get a selector by name with optional parameters."""
    if name not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown selector: {name}. Available: {list(SELECTOR_REGISTRY.keys())}")
    factory = SELECTOR_REGISTRY[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory(**kwargs)
