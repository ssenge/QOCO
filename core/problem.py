"""
Abstract base class for optimization problems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Tuple, TypeVar

from qoco.core.solution import ProblemSummary

SummaryT = TypeVar("SummaryT", bound=ProblemSummary)


@dataclass
class Problem(ABC, Generic[SummaryT]):
    """Abstract base class for optimization problems."""

    name: str

    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the problem instance. Returns (is_valid, error_messages)."""
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> SummaryT:
        """Return a structured summary of the problem."""
        raise NotImplementedError

