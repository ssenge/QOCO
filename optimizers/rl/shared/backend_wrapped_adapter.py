from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping

import torch

from qoco.core.solution import Solution, Status
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter
from qoco.optimizers.rl.shared.backend import EnvBackend


@dataclass
class BackendWrappedAdapter(ProblemAdapter):
    """Turns an EnvBackend into a ProblemAdapter (delegation).

    Approach-specific adapters subclass this and add their hooks.
    """

    backend: EnvBackend

    @property
    def name(self) -> str:  # type: ignore[override]
        return str(getattr(self.backend, "name", self.backend.__class__.__name__))

    def load_test_instances(self, path: Path, limit: int) -> List[Any]:
        return self.backend.load_test_instances(path, limit)

    def optimal_cost(self, inst: Any) -> float:
        return self.backend.optimal_cost(inst)

    def optimal_time_s(self, inst: Any) -> float:
        return self.backend.optimal_time_s(inst)

    def train_batch(self, batch_size: int, device: str):
        return self.backend.train_batch(batch_size, device)

    def reset(self, batch):
        td = self.backend.reset(batch)
        obs = self.backend.observe(td)
        td["node_features"] = obs["node_features"]
        td["step_features"] = obs["step_features"]
        return td

    def step(self, batch, action):
        td = self.backend.step(batch, action)
        obs = self.backend.observe(td)
        td["node_features"] = obs["node_features"]
        td["step_features"] = obs["step_features"]
        return td

    def is_done(self, batch) -> torch.Tensor:
        return self.backend.is_done(batch)

    def action_mask(self, batch) -> torch.Tensor:
        return self.backend.action_mask(batch)

    def reward(self, batch) -> torch.Tensor:
        return self.backend.reward(batch)

    def make_eval_batch(self, inst: Any, device: str):
        td = self.backend.make_eval_batch(inst, device)
        obs = self.backend.observe(td)
        td["node_features"] = obs["node_features"]
        td["step_features"] = obs["step_features"]
        return td

    def observe(self, batch) -> Mapping[str, torch.Tensor]:
        return self.backend.observe(batch)

    def score_eval_batch(self, batch) -> tuple[Status, float]:
        return self.backend.score_eval_batch(batch)

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        return self.backend.score_from_reward(reward)

    def to_solution(self, batch) -> Solution:
        return self.backend.to_solution(batch)

    @property
    def generator(self):
        """Expose backend.generator for RL4CO dataset creation."""
        return getattr(self.backend, "generator")

