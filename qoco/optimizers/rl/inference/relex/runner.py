from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from qoco.core.solution import Solution
from qoco.optimizers.rl.shared.base import PolicyRunner, load_policy_checkpoint
from qoco.optimizers.rl.inference.relex.adapter import RelexAdapter
from qoco.optimizers.rl.inference.relex.meta import RelexPolicyMeta
from qoco.optimizers.rl.training.methods.relex.model_standard import AMConfig, AttentionModel, DualPathCache
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
        dual_path = bool(getattr(cfg, "dual_path", False))

        def _am_cfg(**overrides) -> AMConfig:
            base = dict(
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
                dual_path=dual_path,
            )
            base.update(overrides)
            return AMConfig(**base)

        if model_name == "standard":
            if cfg.use_pqc:
                pqc_block = PooledPQCResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            else:
                pqc_block = NoOpPQCBlock()
            model = AttentionModel(_am_cfg(), pqc_block=pqc_block)
        elif model_name == "global_bias":
            model = GlobalBiasAttentionModel(_am_cfg())
        elif model_name == "pqc_residual":
            pqc_block = PooledPQCResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(_am_cfg(), pqc_block=pqc_block, model_name="pqc_residual")
        elif model_name == "pqc_replace":
            pqc_block = PooledPQCReplace(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(_am_cfg(), pqc_block=pqc_block, model_name="pqc_replace")
        elif model_name == "classic_bottleneck":
            pqc_block = ClassicalBottleneckResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(_am_cfg(), pqc_block=pqc_block, model_name="classic_bottleneck")
        elif model_name == "classic_extra":
            pqc_block = ClassicalExtraResidual(embed_dim=int(cfg.embed_dim), cfg=cfg.pqc)
            model = AttentionModel(_am_cfg(), pqc_block=pqc_block, model_name="classic_extra")
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
            dec = cache = None
            can_cache = getattr(adapter, "static_node_features", True) or getattr(self.model, "dual_path", False)
            if can_cache:
                enc = getattr(self.model, "encoder", None)
                dec_mod = getattr(self.model, "decoder", None)
                pre = getattr(dec_mod, "precompute_cache", None) if dec_mod is not None else None
                if enc is not None and dec_mod is not None and pre is not None:
                    node_features0 = adapter.clone_node_features(td)
                    if getattr(enc, "_dual_path", False):
                        static_h = enc.encode_static(node_features0.shape[0], node_features0.device)
                        cache = DualPathCache(static_h=static_h, encoder=enc)
                    else:
                        h = enc(node_features0)
                        cache = pre(h)
                    dec = dec_mod

            while not bool(adapter.is_done(td).all().item()):
                mask = adapter.action_mask(td)
                step_features = adapter.clone_step_features(td)
                if dec is not None and cache is not None:
                    if isinstance(cache, DualPathCache):
                        node_features = adapter.clone_node_features(td)
                        h_dynamic = cache.encoder.encode_dynamic(node_features)
                        h = cache.static_h + h_dynamic
                        full_cache = dec.precompute_cache(h)
                        logits = dec(step_features, mask, full_cache)
                    else:
                        logits = dec(step_features, mask, cache)
                else:
                    node_features = adapter.clone_node_features(td)
                    logits = self.model(node_features, step_features, mask)
                logits = logits.masked_fill(~mask, float("-inf"))
                action = logits.argmax(dim=-1)
                td = adapter.step(td, action)
        return adapter.to_solution(td)

