from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

P = TypeVar("P")
R = TypeVar("R")


@dataclass(frozen=True)
class SolvedProblem(Generic[P, R]):
    problem: P
    solution: R
