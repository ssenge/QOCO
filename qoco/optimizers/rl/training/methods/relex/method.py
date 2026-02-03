from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightning.pytorch as pl
import torch
import torch.nn as nn

from qoco.optimizers.rl.training.core.types import TrainResult
from qoco.optimizers.rl.training.core.utils import num_trainable_params
from qoco.optimizers.rl.training.core.logging_callback import JsonlTrainingLoggerCallback
from qoco.optimizers.rl.training.core.eval_curve_callback import EvalCurveLoggingCallback, BestCheckpointCallback
from qoco.optimizers.rl.training.methods.base import Method
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter
from qoco.optimizers.rl.training.methods.relex.model_standard import AMConfig, AttentionModel
from qoco.optimizers.rl.training.quantum.pqc_pool import PQCConfig
from qoco.optimizers.rl.training.methods.relex.algos import Algo, ReinforceAlgo
from qoco.optimizers.rl.training.methods.relex.lightning import CloneModule
from qoco.core.logging import default_output_dir, start_run
from qoco.optimizers.rl.inference.relex.meta import RelexAMConfig, RelexPolicyMeta
from qoco.optimizers.rl.shared.base import save_policy_checkpoint


@dataclass
class RelexTrain(Method):
    """Train a Relex/clone-style AttentionModel policy."""

    model: nn.Module
    name: str = "relex"
    batch_size: int = 32
    use_pqc: bool = False
    pqc: PQCConfig = PQCConfig()
    lr: float | None = None
    algo: Algo = field(default_factory=ReinforceAlgo)

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
        cfg = self.model.cfg
        logger = start_run(
            name=self.name,
            kind="train",
            config={
                "method": self.name,
                "problem": getattr(problem, "name", problem.__class__.__name__),
                "batch_size": int(self.batch_size),
                "train_seconds": float(train_seconds),
                "device": str(device),
                "use_pqc": bool(cfg.use_pqc),
                "pqc_mode": str(cfg.pqc.mode),
                "pqc_qubits": int(cfg.pqc.n_qubits),
                "pqc_layers": int(cfg.pqc.n_layers),
                "max_steps": None if max_steps is None else int(max_steps),
            },
        )
        module = CloneModule(
            adapter=problem,
            model=self.model,
            algo=self.algo,
            lr=(self.lr if self.lr is not None else cfg.lr),
            batch_size=self.batch_size,
        )

        if eval_interval > 0 and (eval_instances is None or opt_costs is None):
            raise ValueError("eval_interval > 0 requires eval_instances and opt_costs")

        ckpt_dir = logger.run_dir / "checkpoints"

        def _get_ckpt():
            cfg_model = cfg
            meta = RelexPolicyMeta(
                cfg=RelexAMConfig(
                    n_nodes=int(cfg_model.n_nodes),
                    node_feat_dim=int(cfg_model.node_feat_dim),
                    step_feat_dim=int(cfg_model.step_feat_dim),
                    embed_dim=int(cfg_model.embed_dim),
                    num_heads=int(cfg_model.num_heads),
                    num_layers=int(cfg_model.num_layers),
                    ff_hidden=int(cfg_model.ff_hidden),
                    lr=float(cfg_model.lr),
                    use_pqc=bool(cfg_model.use_pqc),
                    pqc=cfg_model.pqc,
                    model_name=str(self.model.model_name),
                )
            )
            return meta, module.model.state_dict()

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

        trainer_kwargs = dict(
            accelerator=device,
            devices="auto",
            precision="32-true",
            max_time={"seconds": float(train_seconds)},
            default_root_dir=str(default_output_dir() / "lightning"),
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            log_every_n_steps=50,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            gradient_clip_val=1.0,
            callbacks=callbacks,
        )
        if max_steps is not None:
            trainer_kwargs["max_steps"] = int(max_steps)
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(module)
        runtime_s = time.time() - start

        # Lightning already moved the module/model; keep it as-is
        self.model = module.model

        # Always write final checkpoint for inference-only runs.
        final_meta, final_state = _get_ckpt()
        save_policy_checkpoint(ckpt_dir / "final.pt", meta=final_meta, state=final_state)

        profile = module.profile_summary()

        tr = TrainResult(
            method=self.name,
            params=num_trainable_params(self.model),
            steps=trainer.global_step,
            runtime_s=runtime_s,
            it_per_s=(trainer.global_step / runtime_s) if runtime_s > 0 else 0.0,
            run_id=logger.run_dir.name,
            run_dir=str(logger.run_dir),
            curve=[{"profile": profile}],
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
        # If called before train(), this will crash (fine for internal PoC)
        self.model = self.model.to(eval_batch.device)
        self.model.eval()

        while not problem.is_done(eval_batch).all():
            mask = problem.action_mask(eval_batch)
            node_features = problem.clone_node_features(eval_batch)
            step_features = problem.clone_step_features(eval_batch)
            logits = self.model(node_features, step_features, mask)
            logits = logits.masked_fill(~mask, float("-inf"))
            action = logits.argmax(dim=-1)
            eval_batch = problem.step(eval_batch, action)

        return eval_batch
