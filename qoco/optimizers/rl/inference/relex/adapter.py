from __future__ import annotations

from abc import ABC, abstractmethod

from qoco.core.solution import Solution
from qoco.optimizers.rl.shared.base import RLAdapter


class RelexAdapter(RLAdapter, ABC):
    """Adapter requirements for the relex (clone-style) runner."""

    @abstractmethod
    def is_done(self, batch): ...

    @abstractmethod
    def action_mask(self, batch): ...

    @abstractmethod
    def step(self, batch, action): ...

    @abstractmethod
    def to_solution(self, batch) -> Solution: ...

    @abstractmethod
    def clone_node_features(self, batch): ...

    @abstractmethod
    def clone_step_features(self, batch): ...

