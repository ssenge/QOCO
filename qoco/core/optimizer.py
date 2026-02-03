"""Abstract optimizer interface.

Optimizers use a Converter to transform a problem, then optimize it.
Includes always-on structured run logging.
"""

import time
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from .converter import Converter
from qoco.converters.identity import IdentityConverter
from qoco.core.problem import Problem
from qoco.core.solution import OptimizationResult, OptimizerRun, ProblemSummary, Solution

# Type variables
SummaryT = TypeVar("SummaryT", bound=ProblemSummary)
P = TypeVar("P", bound=Problem[SummaryT])  # Problem type
T = TypeVar("T")  # Converted problem type
SolutionT = TypeVar("SolutionT", bound=Solution)
RunT = TypeVar("RunT", bound=OptimizerRun)


@dataclass
class Optimizer(ABC, Generic[P, T, SolutionT, RunT, SummaryT]):
    """
    Abstract optimizer that converts a problem and optimizes it.

    Type parameters:
        P: Problem type (must extend Problem)
        T: Converted problem type
        R: Result type
    """

    name: str = ""
    converter: Converter[P, T] = field(default_factory=IdentityConverter)

    def optimize(self, problem: P) -> OptimizationResult[SolutionT, RunT, SummaryT]:
        """Convert problem and optimize, with timing.
        """

        ts_start = datetime.now(timezone.utc)
        converted = self.converter.convert(problem)
        solution, run = self._optimize(converted)
        ts_end = datetime.now(timezone.utc)

        summary = problem.summary() if hasattr(problem, "summary") else ProblemSummary()
        return OptimizationResult(
            solution=solution,
            run=replace(run, timestamp_start=ts_start, timestamp_end=ts_end),
            problem=summary,
        )

    @abstractmethod
    def _optimize(self, converted: T) -> tuple[SolutionT, RunT]:
        """Optimize the converted problem. Implement in subclass."""
        raise NotImplementedError

