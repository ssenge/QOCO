from __future__ import annotations

import logging
import time
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from qoco.optimizers.rl.training.methods.relex.algos import Algo
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter

logger = logging.getLogger("qoco.training")


class CloneModule(pl.LightningModule):
    def __init__(
        self,
        adapter: ProblemAdapter,
        model: torch.nn.Module,
        algo: Algo,
        lr: float,
        batch_size: int,
        train_data_size: int = 2_048,
        profile_warmup_steps: int = 20,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.model = model
        self.algo = algo
        self.lr = lr
        self.batch_size = batch_size
        self.train_data_size = train_data_size
        self.profile_warmup_steps = int(profile_warmup_steps)

        # Lightweight timing instrumentation for short-run analysis.
        self._t_fit_start: float | None = None
        self._t_fit_end: float | None = None
        self._t_baseline_setup_start: float | None = None
        self._t_baseline_setup_end: float | None = None
        self._t_first_batch_start: float | None = None
        self._t_first_opt_step_end: float | None = None
        self._t_warmup_step_end: float | None = None
        self._t_last_progress_log: float | None = None

    def training_step(self, batch, batch_idx):
        if self._t_first_batch_start is None:
            self._t_first_batch_start = time.time()
        t_batch_start = time.time()
        td = self.adapter.reset(self.adapter.train_batch(self.batch_size, device=self.device))
        t_batch_end = time.time()
        loss = self.algo.loss(model=self.model, adapter=self.adapter, td=td)
        t_loss_end = time.time()
        self.algo.baseline.maybe_update(step=self.global_step, model=self.model, adapter=self.adapter)
        self.log("train/loss", loss, prog_bar=False)

        now = time.time()
        if self._t_last_progress_log is None or (now - self._t_last_progress_log) >= 60.0:
            self._t_last_progress_log = now
            step = int(getattr(self.trainer, "global_step", 0) or 0)
            t0 = self._t_fit_start if self._t_fit_start is not None else now
            elapsed = float(now - t0)
            batch_s = float(t_batch_end - t_batch_start)
            loss_s = float(t_loss_end - t_batch_end)
            logger.info(
                "[progress] step=%s elapsed_s=%.1f batch_s=%.2f loss_s=%.2f",
                step,
                elapsed,
                batch_s,
                loss_s,
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        # dummy dataloader: we generate data in training_step
        dummy = TensorDataset(torch.zeros(self.train_data_size, 1))
        return DataLoader(dummy, batch_size=1, shuffle=False)

    def on_fit_start(self) -> None:
        self._t_fit_start = time.time()
        self._t_baseline_setup_start = time.time()
        self.algo.baseline.setup(
            model=self.model,
            adapter=self.adapter,
            device=self.device,
            batch_size=self.batch_size,
            dataset_size=self.train_data_size,
        )
        self._t_baseline_setup_end = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # `trainer.global_step` is incremented after optimizer step.
        if self._t_first_opt_step_end is None and int(self.trainer.global_step) >= 1:
            self._t_first_opt_step_end = time.time()
        ws = int(self.profile_warmup_steps)
        if ws > 0 and self._t_warmup_step_end is None and int(self.trainer.global_step) >= ws:
            self._t_warmup_step_end = time.time()

    def on_fit_end(self) -> None:
        self._t_fit_end = time.time()

    def profile_summary(self) -> dict[str, float | int | None]:
        """Return timing summary (seconds) for analysis/benchmarking."""
        t0 = self._t_fit_start
        out: dict[str, float | int | None] = {
            "warmup_steps": int(self.profile_warmup_steps),
        }
        if t0 is None:
            return out

        # (1) Baseline setup
        if self._t_baseline_setup_start is not None and self._t_baseline_setup_end is not None:
            out["baseline_setup_s"] = float(self._t_baseline_setup_end - self._t_baseline_setup_start)
        else:
            out["baseline_setup_s"] = None

        # (2) Time to first optimizer step
        if self._t_first_opt_step_end is not None:
            out["time_to_first_opt_step_s"] = float(self._t_first_opt_step_end - t0)
        else:
            out["time_to_first_opt_step_s"] = None

        # (3) Steady-state it/s after warmup step
        t_end = self._t_fit_end
        t_ws = self._t_warmup_step_end
        gs_end = int(getattr(self.trainer, "global_step", 0) or 0)
        ws = int(self.profile_warmup_steps)
        if t_end is not None and t_ws is not None and gs_end > ws:
            out["steady_it_per_s_after_warmup"] = float((gs_end - ws) / max(1e-9, (t_end - t_ws)))
        else:
            out["steady_it_per_s_after_warmup"] = None

        # Extra (helps interpret the above)
        if t_end is not None:
            out["fit_wall_s"] = float(t_end - t0)
        out["global_step_end"] = int(gs_end)
        return out
