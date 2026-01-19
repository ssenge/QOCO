from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from qoco.core.solution import Solution, Status


class EnvBackend(ABC):
    """Approach-agnostic environment backend.

    This is the *backend axis*: how we generate episodes, masks, rewards, and decode solutions.
    Approach adapters (relex / rl4co) wrap a backend and add policy-family hooks.
    """

    name: str

    @abstractmethod
    def load_test_instances(self, path, limit: int): ...

    @abstractmethod
    def optimal_cost(self, inst: Any) -> float: ...

    @abstractmethod
    def optimal_time_s(self, inst: Any) -> float: ...

    @abstractmethod
    def train_batch(self, batch_size: int, device: str): ...

    @abstractmethod
    def make_eval_batch(self, inst: Any, device: str): ...

    @abstractmethod
    def reset(self, batch): ...

    @abstractmethod
    def step(self, batch, action): ...

    @abstractmethod
    def is_done(self, batch) -> torch.Tensor: ...

    @abstractmethod
    def action_mask(self, batch) -> torch.Tensor: ...

    @abstractmethod
    def reward(self, batch) -> torch.Tensor: ...

    @abstractmethod
    def observe(self, batch) -> Mapping[str, torch.Tensor]:
        """Return policy-facing observation tensors.

        Required keys:
        - node_features: (B, n_nodes, node_feat_dim)
        - step_features: (B, step_feat_dim) or similar
        - action_mask: (B, n_nodes) boolean
        - done: (B,) boolean
        """

    @abstractmethod
    def score_eval_batch(self, batch) -> tuple[Status, float]: ...

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        raise NotImplementedError

    @abstractmethod
    def to_solution(self, batch) -> Solution: ...

