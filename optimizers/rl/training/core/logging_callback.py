from __future__ import annotations

import time
from typing import Any

import lightning.pytorch as pl

from qoco.core.logging import JsonlLogger


class JsonlTrainingLoggerCallback(pl.Callback):
    def __init__(self, logger: JsonlLogger, *, every_n_steps: int = 10) -> None:
        self.logger = logger
        self.every_n_steps = int(every_n_steps)
        self._t0: float | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t0 = time.time()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        gs = int(getattr(trainer, "global_step", 0) or 0)
        if self.every_n_steps > 0 and (gs % self.every_n_steps) != 0:
            return
        metrics: dict[str, float] = {}
        for k, v in getattr(trainer, "callback_metrics", {}).items():
            key = str(k)
            if "/" not in key:
                key = f"train/{key}"
            if hasattr(v, "detach"):
                try:
                    metrics[key] = float(v.detach().cpu().item())
                except Exception:
                    continue
            elif isinstance(v, (int, float)):
                metrics[key] = float(v)
        metrics["train/wall_s"] = float(time.time() - self._t0) if self._t0 is not None else 0.0
        payload = {"step": gs, "metrics": metrics}
        if self._t0 is not None:
            payload["wall_s"] = float(time.time() - self._t0)
        self.logger.log("step", payload)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        gs = int(getattr(trainer, "global_step", 0) or 0)
        payload = {"steps": gs}
        if self._t0 is not None:
            payload["wall_s"] = float(time.time() - self._t0)
        self.logger.log("train_end", payload)

