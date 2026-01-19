"""
Abstract base class for optimization problems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Problem(ABC):
    """Abstract base class for optimization problems."""

    name: str

    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the problem instance. Returns (is_valid, error_messages)."""
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        """Return a text summary of the problem."""
        raise NotImplementedError

