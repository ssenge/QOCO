from __future__ import annotations

"""RL4CO-minlib method (implemented directly, no subpackage).

Prefer importing from here:
  from qoco.optimizers.rl.training.methods.rl4co import RL4COTrain
"""

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import torch

from qoco.optimizers.rl.shared.rl4co_switch import maybe_enable_local_rl4co

maybe_enable_local_rl4co()

from rl4co.utils.trainer import RL4COTrainer
from rl4co.models.rl import REINFORCE

from qoco.optimizers.rl.training.core.types import TrainResult
from qoco.optimizers.rl.training.core.utils import num_trainable_params
from qoco.optimizers.rl.training.core.logging_callback import JsonlTrainingLoggerCallback
from qoco.optimizers.rl.training.core.eval_curve_callback import EvalCurveLoggingCallback, BestCheckpointCallback
from qoco.optimizers.rl.training.methods.base import Method
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter
from qoco.core.logging import default_output_dir, start_run
from qoco.optimizers.rl.inference.rl4co.meta import RL4COPolicyMeta
from qoco.optimizers.rl.shared.base import save_policy_checkpoint


@dataclass
class RL4COTrain(Method):
    """Train an RL4CO AttentionModel policy (REINFORCE/PPO/etc.)."""

    name: str = "rl4co"

    embed_dim: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 3
    batch_size: int = 32
    lr: float = 1e-4

    # RL4CO pre-generates datasets in setup(), so keep moderate to avoid startup overhead.
    train_data_size: int = 2_048
    val_data_size: int = 64
    test_data_size: int = 64

    # Configurable RL4CO algorithm builder (e.g. partial(REINFORCE, baseline="rollout"))
    algo_factory: Callable[..., Any] = field(default_factory=lambda: partial(REINFORCE, baseline="rollout"), repr=False)

    module: object = field(default=None, init=False, repr=False)

    def build_module(self, problem: ProblemAdapter) -> Any:
        env = problem.build_rl4co_env()
        policy = problem.build_rl4co_policy(self.embed_dim, self.num_heads, self.num_encoder_layers)
        return self.algo_factory(
            env,
            policy,
            batch_size=self.batch_size,
            optimizer_kwargs={"lr": self.lr},
            train_data_size=self.train_data_size,
            val_data_size=self.val_data_size,
            test_data_size=self.test_data_size,
        )

    def train(
        self,
        problem: ProblemAdapter,
        train_seconds: float,
        device: str,
        eval_interval: int,
        eval_instances=None,
        opt_costs=None,
        max_steps: int | None = None,
    ) -> TrainResult:
        start = time.time()
        logger = start_run(
            name=self.name,
            kind="train",
            config={
                "method": self.name,
                "problem": getattr(problem, "name", problem.__class__.__name__),
                "batch_size": int(self.batch_size),
                "train_seconds": float(train_seconds),
                "device": str(device),
                "embed_dim": int(self.embed_dim),
                "num_heads": int(self.num_heads),
                "num_encoder_layers": int(self.num_encoder_layers),
                "max_steps": None if max_steps is None else int(max_steps),
            },
        )

        self.module = self.build_module(problem)

        n_params = num_trainable_params(self.module)

        # Force fp32 for consistency across methods/backends.
        precision = "32-true"

        if eval_interval > 0 and (eval_instances is None or opt_costs is None):
            raise ValueError("eval_interval > 0 requires eval_instances and opt_costs")

        ckpt_dir = logger.run_dir / "checkpoints"

        def _get_ckpt():
            meta = RL4COPolicyMeta(
                embed_dim=int(self.embed_dim),
                num_heads=int(self.num_heads),
                num_encoder_layers=int(self.num_encoder_layers),
            )
            # RL4CO module exposes `.policy`
            state = self.module.policy.state_dict()  # type: ignore[union-attr]
            return meta, state

        callbacks = [JsonlTrainingLoggerCallback(logger, every_n_steps=10)]
        if eval_interval > 0:
            best_cb = BestCheckpointCallback(
                checkpoint_path=ckpt_dir / "best.pt",
                get_checkpoint=_get_ckpt,
                require_feasible_rate=0.0,
            )
            callbacks.append(
                EvalCurveLoggingCallback(
                    logger=logger,
                    trainer=self,
                    problem=problem,
                    instances=eval_instances,
                    opt_costs=opt_costs,
                    device=device,
                    every=int(eval_interval),
                    on_metrics=best_cb.maybe_save,
                )
            )

        trainer_kwargs: dict[str, Any] = dict(
            accelerator=device,
            devices="auto",
            max_time={"seconds": float(train_seconds)},
            default_root_dir=str(default_output_dir() / "lightning"),
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            log_every_n_steps=50,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=callbacks,
            **({"precision": precision} if precision is not None else {}),
        )
        if max_steps is not None:
            trainer_kwargs["max_steps"] = int(max_steps)
        trainer = RL4COTrainer(**trainer_kwargs)
        trainer.fit(self.module)
        runtime_s = float(time.time() - start)

        final_meta, final_state = _get_ckpt()
        save_policy_checkpoint(ckpt_dir / "final.pt", meta=final_meta, state=final_state)

        tr = TrainResult(
            method=self.name,
            params=n_params,
            steps=trainer.global_step,
            runtime_s=runtime_s,
            it_per_s=(trainer.global_step / runtime_s) if runtime_s > 0 else 0.0,
            run_id=logger.run_dir.name,
            run_dir=str(logger.run_dir),
            curve=[],
        )
        logger.log(
            "run_end",
            {
                "status": "ok",
                "steps": int(tr.steps),
                "runtime_s": float(tr.runtime_s),
                "it_per_s": float(tr.it_per_s),
            },
        )
        logger.close()
        return tr

    @torch.no_grad()
    def infer(self, problem: ProblemAdapter, eval_batch):
        policy = self.module.policy.to(eval_batch.device)
        env = self.module.env

        policy.eval()
        out = policy(
            eval_batch,
            env=env,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )
        return out

