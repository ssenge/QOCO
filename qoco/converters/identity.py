from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from qoco.core.converter import Converter

T = TypeVar("T")


@dataclass
class IdentityConverter(Generic[T], Converter[T, T]):
    """Converter that returns the input unchanged."""

    def convert(self, problem: T) -> T:
        return problem

