from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from qoco.core.solution import Solution, Status
from qoco.optimizers.rl.shared.base import RLAdapter


class RL4COAdapter(RLAdapter, ABC):
    """Adapter requirements for the RL4CO-based runner."""

    @abstractmethod
    def build_rl4co_policy(self, embed_dim: int, num_heads: int, num_encoder_layers: int): ...

    @abstractmethod
    def build_rl4co_env(self): ...

    @abstractmethod
    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]: ...

    @abstractmethod
    def to_solution_from_actions(self, inst: Any, actions, status: Status, cost: float) -> Solution: ...

