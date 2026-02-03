from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple

import lightning.pytorch as pl
import torch

from qoco.core.logging import RunLogger
from qoco.optimizers.rl.training.core.benchmark import evaluate_method_unlogged
from qoco.optimizers.rl.training.methods.base import Method
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter
from qoco.optimizers.rl.shared.base import PolicyMeta, save_policy_checkpoint


class EvalCurveLoggingCallback(pl.Callback):
    """Evaluate gap/feas at fixed training steps and log into the train run."""

    def __init__(
        self,
        *,
        logger: RunLogger,
        trainer: Method,
        problem: ProblemAdapter,
        instances: Sequence,
        opt_costs: Sequence[float],
        device: str,
        every: int = 1,
        on_metrics: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.method = trainer
        self.problem = problem
        self.instances = list(instances)
        self.opt_costs = list(opt_costs)
        self.device = str(device)
        self.every = int(every)
        self.on_metrics = on_metrics

        self._t0: float | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        step = int(getattr(trainer, "global_step", 0) or 0)
        if step <= 0:
            return
        if self.every <= 0 or (step % self.every) != 0:
            return

        t0 = self._t0 if self._t0 is not None else time.perf_counter()
        _ = float(time.perf_counter() - t0)

        metrics = evaluate_method_unlogged(
            trainer=self.method,
            problem=self.problem,
            instances=self.instances,
            device=self.device,
            opt_costs=self.opt_costs,
        )
        self.logger.log_metrics(step=step, metrics=metrics)
        if self.on_metrics is not None:
            self.on_metrics(metrics)


class BestCheckpointCallback(pl.Callback):
    """Save `best.pt` in the run dir whenever eval gap improves."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        get_checkpoint: Callable[[], Tuple[PolicyMeta, Mapping[str, torch.Tensor]]],
        require_feasible_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.get_checkpoint = get_checkpoint
        self.require_feasible_rate = float(require_feasible_rate)
        self._best_gap: float | None = None

    def maybe_save(self, metrics: dict[str, float]) -> None:
        gap = metrics.get("eval/gap_pct", None)
        feas = metrics.get("eval/feasible_rate", None)
        if gap is None or feas is None:
            return
        if float(feas) <= self.require_feasible_rate:
            return
        g = float(gap)
        if not (g == g):  # NaN
            return
        if self._best_gap is None or g < self._best_gap:
            meta, state = self.get_checkpoint()
            save_policy_checkpoint(self.checkpoint_path, meta=meta, state=state)
            self._best_gap = g

