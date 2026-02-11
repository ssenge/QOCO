"""Solution decoder abstractions for optimizer post-processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from qoco.core.solution import Solution

S = TypeVar("S", bound=Solution)

@dataclass
class ResultDecoder(ABC, Generic[S]):
    """Decode optimizer solutions to canonical variable naming."""

    @abstractmethod
    def decode(self, solution: S) -> S:
        raise NotImplementedError


@dataclass
class IdentityDecoder(ResultDecoder[S]):
    """No-op decoder used as default by all optimizers."""

    def decode(self, solution: S) -> S:
        return solution


@dataclass
class IndexToVarNameDecoder(ResultDecoder[Solution]):
    """Decode numeric variable keys using a positional name list."""

    names_by_index: list[str]

    def _decode_key(self, key: str) -> str:
        try:
            idx = int(key)
        except ValueError:
            return str(key)
        if idx < 0 or idx >= len(self.names_by_index):
            return str(key)
        return str(self.names_by_index[idx])

    def decode(self, solution: Solution) -> Solution:
        decoded_var_values = {
            self._decode_key(str(key)): value
            for key, value in dict(solution.var_values).items()
        }
        decoded_array_index = dict(solution.var_array_index)
        if "x" in decoded_array_index:
            decoded_array_index["x"] = list(self.names_by_index)
        return replace(solution, var_values=decoded_var_values, var_array_index=decoded_array_index)
