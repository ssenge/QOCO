from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import torch

from qoco.core.solution import Status


class ProblemAdapter(ABC):
    name: str

    @abstractmethod
    def load_test_instances(self, path: Path, limit: int) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def optimal_cost(self, inst: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def optimal_time_s(self, inst: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def train_batch(self, batch_size: int, device: str) -> Any:
        """Return a fresh training batch on `device`."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, batch: Any, action: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_done(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def action_mask(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def reward(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def make_eval_batch(self, inst: Any, device: str) -> Any:
        """Build a single-instance eval batch."""
        raise NotImplementedError

    @abstractmethod
    def score_eval_batch(self, batch: Any) -> tuple[Status, float]:
        """Return (status, cost) for an eval batch."""
        raise NotImplementedError

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        """Optional: compute (status, cost) directly from a reward tensor.

        Useful when a library returns only reward (e.g., RL4CO policy forward output).
        """
        raise NotImplementedError

    # Optional hooks for specific method families
    # ------------------------------------------
    # Methods should depend on these hooks instead of hard-coding problem types.

    def build_rl4co_policy(self, embed_dim: int, num_heads: int, num_encoder_layers: int):
        """Build and return an RL4CO AttentionModelPolicy compatible with this problem.

        Not all problems need to support this; methods that require it should fail fast if
        the adapter doesn't implement it.
        """
        raise NotImplementedError

    def clone_feature_specs(self) -> dict:
        """Return feature specs for the clone model.

        Expected keys:
          - n_nodes: int (number of action nodes)
          - node_feat_dim: int (feature dim per node)
          - step_feat_dim: int (feature dim per decoding step)
        """
        raise NotImplementedError

    def clone_node_features(self, batch: Any):
        """Return per-node features tensor of shape (B, n_nodes, node_feat_dim)."""
        raise NotImplementedError

    def clone_step_features(self, batch: Any):
        """Return per-step features tensor of shape (B, step_feat_dim)."""
        raise NotImplementedError
