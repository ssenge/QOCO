"""
Abstract converter interface.

Converters transform a problem into a specific representation (e.g., Pyomo model).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

# Type variables
P = TypeVar("P")  # Source type (Problem)
T = TypeVar("T")  # Target type (converted representation)


@dataclass
class Converter(ABC, Generic[P, T]):
    """
    Base class for problem converters.

    Type parameters:
        P: Source problem type
        T: Target representation type
    """

    @abstractmethod
    def convert(self, problem: P) -> T:
        """Convert problem to target representation."""
        raise NotImplementedError

