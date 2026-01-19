from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from qoco.optimizers.rl.training.core.types import TrainResult
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter


class Method(ABC):
    name: str

    @abstractmethod
    def train(
        self,
        problem: ProblemAdapter,
        train_seconds: float,
        device: str,
        eval_interval: int,
        eval_instances: Sequence | None = None,
        opt_costs: Sequence[float] | None = None,
        max_steps: int | None = None,
    ) -> TrainResult:
        raise NotImplementedError

    @abstractmethod
    def infer(self, problem: ProblemAdapter, eval_batch: Any) -> Any:
        raise NotImplementedError

    # Back-compat: older code used infer_greedy; keep as alias.
    def infer_greedy(self, problem: ProblemAdapter, eval_batch: Any) -> Any:
        return self.infer(problem, eval_batch)
