from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar
from abc import ABC

from qoco.core.converter import Converter

P = TypeVar("P")
M = TypeVar("M")
S = TypeVar("S")


class Reducer(Converter[P, P], ABC):
    """Exact subsetting of a problem (same type in/out)."""


@dataclass(frozen=True)
class CollapseMapping(Generic[S]):
    steps: list[Callable[[S], S]] = field(default_factory=list)

    def compose(self, other: "CollapseMapping[S]") -> "CollapseMapping[S]":
        return CollapseMapping(steps=[*self.steps, *other.steps])


@dataclass(frozen=True)
class Collapsed(Generic[P, M]):
    problem: P
    mapping: M


class Collapser(Converter[P, Collapsed[P, M]], ABC):
    """Approximate simplification of a problem with a mapping back."""
