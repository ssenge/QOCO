from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

P = TypeVar("P")


class Sampler(ABC, Generic[P]):
    @abstractmethod
    def sample(self) -> P:
        raise NotImplementedError
