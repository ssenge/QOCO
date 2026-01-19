from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from qoco.core.solution import Solution
from qoco.optimizers.rl.shared.base import PolicyRunner, load_policy_checkpoint
from qoco.optimizers.rl.inference.relex.adapter import RelexAdapter
from qoco.optimizers.rl.inference.relex.meta import RelexPolicyMeta
from qoco.optimizers.rl.training.methods.relex.model_standard import AMConfig, AttentionModel
from qoco.optimizers.rl.training.methods.relex.model_global_bias import GlobalBiasAttentionModel
from qoco.optimizers.rl.training.quantum.pqc_pool import (
    ClassicalBottleneckResidual,
    ClassicalExtraResidual,
    NoOpPQCBlock,
    PooledPQCReplace,
    PooledPQCResidual,
)


@dataclass
class RelexRunner(PolicyRunner[RelexAdapter]):
    model: torch.nn.Module

    @classmethod
    def load(cls, checkpoint_path: Path, *, device: str, adapter: RelexAdapter) -> "RelexRunner":
        ckpt = load_policy_checkpoint(checkpoint_path, map_location=device)
        meta = ckpt.meta
        if not isinstance(meta, RelexPolicyMeta):
            raise TypeError(f"Expected RelexPolicyMeta, got {type(meta)}")

        cfg = meta.cfg
        model_name = str(cfg.model_name)
        if model_name == "standard":
            pqc_block = NoOpPQCBlock()
            model = AttentionModel(
                AMConfig(
                    n_nodes=cfg.n_nodes,
                    node_feat_dim=cfg.node_feat_dim,
                    step_feat_dim=cfg.step_feat_dim,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    ff_hidden=cfg.ff_hidden,
                    lr=cfg.lr,
                    use_pqc=cfg.use_pqc,
                    pqc=cfg.pqc,
                ),
                pqc_block=pqc_block,
            )
        elif model_name == "global_bias":
            model = GlobalBiasAttentionModel(
                AMConfig(
                    n_nodes=cfg.n_nodes,
                    node_feat_dim=cfg.node_feat_dim,
                    step_feat_dim=cfg.step_feat_dim,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    ff_hidden=cfg.ff_hidden,
                    lr=cfg.lr,
                    use_pqc=cfg.use_pqc,
                    pqc=cfg.pqc,
                )
            )
        elif model_name == "pqc_residual":
            pqc_block = PooledPQCResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(
                AMConfig(
                    n_nodes=cfg.n_nodes,
                    node_feat_dim=cfg.node_feat_dim,
                    step_feat_dim=cfg.step_feat_dim,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    ff_hidden=cfg.ff_hidden,
                    lr=cfg.lr,
                    use_pqc=cfg.use_pqc,
                    pqc=cfg.pqc,
                ),
                pqc_block=pqc_block,
                model_name="pqc_residual",
            )
        elif model_name == "pqc_replace":
            pqc_block = PooledPQCReplace(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(
                AMConfig(
                    n_nodes=cfg.n_nodes,
                    node_feat_dim=cfg.node_feat_dim,
                    step_feat_dim=cfg.step_feat_dim,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    ff_hidden=cfg.ff_hidden,
                    lr=cfg.lr,
                    use_pqc=cfg.use_pqc,
                    pqc=cfg.pqc,
                ),
                pqc_block=pqc_block,
                model_name="pqc_replace",
            )
        elif model_name == "classic_bottleneck":
            pqc_block = ClassicalBottleneckResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(
                AMConfig(
                    n_nodes=cfg.n_nodes,
                    node_feat_dim=cfg.node_feat_dim,
                    step_feat_dim=cfg.step_feat_dim,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    ff_hidden=cfg.ff_hidden,
                    lr=cfg.lr,
                    use_pqc=cfg.use_pqc,
                    pqc=cfg.pqc,
                ),
                pqc_block=pqc_block,
                model_name="classic_bottleneck",
            )
        elif model_name == "classic_extra":
            pqc_block = ClassicalExtraResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(
                AMConfig(
                    n_nodes=cfg.n_nodes,
                    node_feat_dim=cfg.node_feat_dim,
                    step_feat_dim=cfg.step_feat_dim,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    ff_hidden=cfg.ff_hidden,
                    lr=cfg.lr,
                    use_pqc=cfg.use_pqc,
                    pqc=cfg.pqc,
                ),
                pqc_block=pqc_block,
                model_name="classic_extra",
            )
        else:
            raise ValueError(f"Unknown model_name: {cfg.model_name}")
        model.load_state_dict(ckpt.state)
        model.to(device)
        model.eval()
        return cls(model=model)

    def run(self, *, adapter: RelexAdapter, problem, device: str) -> Solution:
        td = adapter.make_eval_batch(problem, device=device)
        td = adapter.reset(td)
        with torch.no_grad():
            while not bool(adapter.is_done(td).all().item()):
                mask = adapter.action_mask(td)
                node_features = adapter.clone_node_features(td)
                step_features = adapter.clone_step_features(td)
                logits = self.model(node_features, step_features, mask)
                logits = logits.masked_fill(~mask, float("-inf"))
                action = logits.argmax(dim=-1)
                td = adapter.step(td, action)
        return adapter.to_solution(td)

