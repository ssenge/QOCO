from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field

import torch

from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter
from qoco.optimizers.rl.training.methods.relex.baselines import Baseline, RolloutBaseline, WarmupBaseline
from qoco.optimizers.rl.training.methods.relex.model_standard import DualPathCache


class Algo(ABC):
    @abstractmethod
    def loss(self, model, adapter: ProblemAdapter, td) -> torch.Tensor:
        raise NotImplementedError


def _precompute_model_cache(model, node_features: torch.Tensor):
    """If the model exposes encoder/decoder with precompute_cache, encode once and reuse.

    For dual-path encoders, returns a ``DualPathCache`` containing the static
    embedding (GAN + PQC applied once).  The dynamic part is computed per step
    inside ``_logits_with_cache``.
    """
    enc = getattr(model, "encoder", None)
    dec = getattr(model, "decoder", None)
    pre = getattr(dec, "precompute_cache", None) if dec is not None else None
    if enc is None or dec is None or pre is None:
        return None, None
    if getattr(enc, "_dual_path", False):
        static_h = enc.encode_static(node_features.shape[0], node_features.device)
        return dec, DualPathCache(static_h=static_h, encoder=enc)
    h = enc(node_features)
    cache = pre(h)
    return dec, cache


def _logits_with_cache(model, node_features: torch.Tensor, step_features: torch.Tensor, mask: torch.Tensor, dec, cache):
    if dec is None or cache is None:
        return model(node_features, step_features, mask)
    if isinstance(cache, DualPathCache):
        h_dynamic = cache.encoder.encode_dynamic(node_features)
        h = cache.static_h + h_dynamic
        full_cache = dec.precompute_cache(h)
        return dec(step_features, mask, full_cache)
    return dec(step_features, mask, cache)


@dataclass
class ReinforceAlgo(Algo):
    """REINFORCE (Monte-Carlo policy gradient)."""

    baseline: Baseline = field(
        default_factory=lambda: WarmupBaseline(
            # NOTE: CloneModule uses a dummy DataLoader with train_data_size=2048 and batch_size=1,
            # so a "warmup epoch" may never complete in short wall-clock benchmarks (e.g. 2 minutes).
            # Use warmup_epochs=0 so the rollout baseline is actually active during short runs.
            inner=RolloutBaseline(update_every_steps=100, eval_batches=4),
            warmup_epochs=0,
            exp_beta=0.8,
        )
    )

    def loss(self, model, adapter: ProblemAdapter, td) -> torch.Tensor:
        device = td.device
        b = td.batch_size[0]

        total_logp = torch.zeros(b, device=device)

        init_td = td
        if self.baseline.requires_init_td:
            init_td = td.clone()

        dec = None
        cache = None
        can_cache = getattr(adapter, "static_node_features", True) or getattr(model, "dual_path", False)
        if can_cache:
            try:
                node_features0 = adapter.clone_node_features(td)
                dec, cache = _precompute_model_cache(model, node_features0)
            except Exception:
                dec, cache = None, None

        while not adapter.is_done(td).all():
            mask = adapter.action_mask(td)
            node_features = adapter.clone_node_features(td)
            step_features = adapter.clone_step_features(td)

            logits = _logits_with_cache(model, node_features, step_features, mask, dec, cache)
            logits = logits.masked_fill(~mask, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)

            all_masked = ~mask.any(dim=-1)
            if all_masked.any():
                for idx in all_masked.nonzero().squeeze(-1):
                    allowed = mask[idx]
                    k = int(allowed.sum().item())
                    probs[idx] = 0.0
                    if k <= 0:
                        # Shouldn't happen if the adapter provides a valid fallback mask,
                        # but be defensive to avoid NaNs / crashes.
                        probs[idx] = 1.0 / float(probs.shape[-1])
                    else:
                        probs[idx][allowed] = 1.0 / float(k)

            action = torch.multinomial(probs.clamp(min=1e-10), 1).squeeze(-1)
            total_logp += torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-10) * (
                ~adapter.is_done(td)
            ).float()

            td = adapter.step(td, action)

        rewards = adapter.reward(td)
        bl_val, bl_loss = self.baseline.eval(
            init_td=init_td,
            reward=rewards,
            model=model,
            adapter=adapter,
        )
        advantage = rewards - bl_val
        return -(advantage.detach() * total_logp).mean() + bl_loss


@dataclass
class PPO(Algo):
    """On-policy PPO with clipped objective and rollout baseline."""

    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    update_every_calls: int = 10
    baseline: Baseline = field(
        default_factory=lambda: WarmupBaseline(
            inner=RolloutBaseline(update_every_steps=100, eval_batches=4),
            warmup_epochs=0,
            exp_beta=0.8,
        )
    )
    _call_idx: int = 0
    _old_model: torch.nn.Module | None = None

    def _sync_old_model(self, model: torch.nn.Module) -> None:
        self._old_model = copy.deepcopy(model)
        self._old_model.eval()
        for p in self._old_model.parameters():
            p.requires_grad_(False)

    def _normalize_probs(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.nan_to_num(probs, nan=0.0)
        all_masked = ~mask.any(dim=-1)
        if all_masked.any():
            for idx in all_masked.nonzero().squeeze(-1):
                allowed = mask[idx]
                k = int(allowed.sum().item())
                probs[idx] = 0.0
                if k <= 0:
                    probs[idx] = 1.0 / float(probs.shape[-1])
                else:
                    probs[idx][allowed] = 1.0 / float(k)
        return probs

    def loss(self, model, adapter: ProblemAdapter, td) -> torch.Tensor:
        device = td.device
        b = td.batch_size[0]

        self._call_idx += 1
        if self._old_model is None or (self._call_idx % int(self.update_every_calls) == 0):
            self._sync_old_model(model)
        if self._old_model is None:
            raise RuntimeError("PPOAlgo old model not initialized")
        self._old_model.to(device)

        total_logp_new = torch.zeros(b, device=device)
        total_logp_old = torch.zeros(b, device=device)
        entropies: list[torch.Tensor] = []

        init_td = td
        if self.baseline.requires_init_td:
            init_td = td.clone()

        saved = []
        dec_new = None
        cache_new = None
        dec_old = None
        cache_old = None
        can_cache = getattr(adapter, "static_node_features", True) or getattr(model, "dual_path", False)
        if can_cache:
            try:
                node_features0 = adapter.clone_node_features(td)
                dec_new, cache_new = _precompute_model_cache(model, node_features0)
                dec_old, cache_old = _precompute_model_cache(self._old_model, node_features0)  # type: ignore[arg-type]
            except Exception:
                dec_new = cache_new = dec_old = cache_old = None

        while not adapter.is_done(td).all():
            mask = adapter.action_mask(td)
            node_features = adapter.clone_node_features(td)
            step_features = adapter.clone_step_features(td)
            alive = (~adapter.is_done(td)).float()

            logits = _logits_with_cache(model, node_features, step_features, mask, dec_new, cache_new)
            logits = logits.masked_fill(~mask, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            probs = self._normalize_probs(probs, mask)

            action = torch.multinomial(probs.clamp(min=1e-10), 1).squeeze(-1)
            entropies.append(-(probs * torch.log(probs + 1e-10)).sum(dim=-1))
            saved.append((node_features, step_features, mask, action, alive))

            td = adapter.step(td, action)

        rewards = adapter.reward(td)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        bl_val, bl_loss = self.baseline.eval(
            init_td=init_td,
            reward=rewards,
            model=model,
            adapter=adapter,
        )
        advantage = rewards - bl_val
        advantage = torch.nan_to_num(advantage, nan=0.0, posinf=0.0, neginf=0.0)
        adv = advantage.detach()

        for node_features, step_features, mask, action, alive in saved:
            logits_new = _logits_with_cache(model, node_features, step_features, mask, dec_new, cache_new).masked_fill(
                ~mask, float("-inf")
            )
            probs_new = torch.softmax(logits_new, dim=-1)
            probs_new = self._normalize_probs(probs_new, mask)

            logits_old = _logits_with_cache(
                self._old_model, node_features, step_features, mask, dec_old, cache_old  # type: ignore[arg-type]
            ).masked_fill(~mask, float("-inf"))
            probs_old = torch.softmax(logits_old, dim=-1)
            probs_old = self._normalize_probs(probs_old, mask)

            total_logp_new += torch.log(
                probs_new.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-10
            ) * alive
            total_logp_old += torch.log(
                probs_old.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-10
            ) * alive

        total_logp_new = torch.nan_to_num(total_logp_new, nan=0.0, posinf=0.0, neginf=0.0)
        total_logp_old = torch.nan_to_num(total_logp_old, nan=0.0, posinf=0.0, neginf=0.0)
        ratio = torch.exp(torch.clamp(total_logp_new - total_logp_old, min=-10.0, max=10.0))
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - float(self.clip_eps), 1.0 + float(self.clip_eps)) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()

        if entropies:
            entropy = torch.stack(entropies, dim=0).mean(dim=0)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            entropy_loss = -float(self.entropy_coef) * entropy.mean()
        else:
            entropy_loss = torch.zeros((), device=device)

        return policy_loss + bl_loss + entropy_loss
