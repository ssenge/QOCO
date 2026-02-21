from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import torch

from qoco.optimizers.rl.training.methods.relex.model_standard import DualPathCache


class Baseline(ABC):
    @property
    def requires_init_td(self) -> bool:
        return False

    def setup(
        self,
        model,
        adapter,
        device: torch.device,
        batch_size: int,
        dataset_size: int,
    ) -> None:
        return

    @abstractmethod
    def eval(self, init_td, reward: torch.Tensor, model, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def epoch_callback(
        self,
        model,
        adapter,
        device: torch.device,
        batch_size: int,
        dataset_size: int,
        epoch: int,
    ) -> None:
        return

    def maybe_update(
        self,
        step: int,
        model,
        adapter,
    ) -> None:
        return


@dataclass
class BatchMeanBaseline(Baseline):
    def eval(self, init_td, reward: torch.Tensor, model, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        bl = reward.mean()
        return bl.expand_as(reward), reward.new_zeros(())


def _greedy_rollout_reward(model, adapter, init_td: object) -> torch.Tensor:
    # IMPORTANT: some adapters (e.g. CSPAdapterSpec) keep episode state in external caches
    # keyed by episode_id, not inside the TensorDict. If we just clone `init_td` and roll out,
    # we may mutate/consume adapter caches and make subsequent rollouts inconsistent.
    #
    # Therefore, always start rollouts by calling `adapter.reset(...)` to obtain a fresh episode
    # (new episode_id + fresh caches) with the same underlying instance data.
    td0 = init_td.clone()
    td = adapter.reset(td0)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            dec = cache = None
            can_cache = getattr(adapter, "static_node_features", True) or getattr(model, "dual_path", False)
            if can_cache:
                enc = getattr(model, "encoder", None)
                dec_mod = getattr(model, "decoder", None)
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

            while not adapter.is_done(td).all():
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
                    logits = model(node_features, step_features, mask)
                logits = logits.masked_fill(~mask, float("-inf"))
                action = logits.argmax(dim=-1)
                td = adapter.step(td, action)
            return adapter.reward(td)
    finally:
        if was_training:
            model.train()


@dataclass
class RolloutBaseline(Baseline):
    dataset_size: int = 2_048
    update_every_steps: int = 200
    eval_batches: int = 2
    eval_batch_size: int | None = None

    policy: object = None
    mean: float | None = None
    _eval_tds: list[object] | None = None

    @property
    def requires_init_td(self) -> bool:
        return True

    def setup(
        self,
        model,
        adapter,
        device: torch.device,
        batch_size: int,
        dataset_size: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.policy = deepcopy(model).to(device)

        ebs = int(self.eval_batch_size) if self.eval_batch_size is not None else int(batch_size)
        n_batches = max(1, int(self.eval_batches))

        self._eval_tds = []
        for _ in range(n_batches):
            # Store *raw* problem batches. We'll call adapter.reset(...) inside rollouts
            # to ensure fresh episode caches each time.
            raw = adapter.train_batch(ebs, device=device)
            self._eval_tds.append(raw)

        r = self._mean_reward(model=self.policy, adapter=adapter)
        self.mean = float(r.item())

    def eval(self, init_td, reward: torch.Tensor, model, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        r = _greedy_rollout_reward(self.policy, adapter, init_td)
        return r, reward.new_zeros(())

    def _mean_reward(self, model, adapter) -> torch.Tensor:
        if not self._eval_tds:
            raise RuntimeError("RolloutBaseline not setup: missing cached eval batches")
        rs = []
        for raw in self._eval_tds:
            r = _greedy_rollout_reward(model, adapter, raw)
            rs.append(r.mean())
        return torch.stack(rs).mean()

    def maybe_update(
        self,
        model,
        adapter,
        step: int,
    ) -> None:
        every = int(self.update_every_steps)
        if every <= 0:
            return
        if int(step) % every != 0:
            return
        if self.policy is None or self.mean is None:
            return  # not set up yet

        device = next(model.parameters()).device
        candidate_mean_t = self._mean_reward(model=model, adapter=adapter)
        candidate_mean = float(candidate_mean_t.item())

        if candidate_mean > float(self.mean):
            self.policy = deepcopy(model).to(device)
            self.mean = candidate_mean


@dataclass
class ExponentialBaseline(Baseline):
    beta: float = 0.8
    v: torch.Tensor | None = None

    def eval(self, init_td, reward: torch.Tensor, model, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        m = reward.mean().detach()
        self.v = m if self.v is None else (self.beta * self.v + (1.0 - self.beta) * m)
        return self.v.expand_as(reward), reward.new_zeros(())


@dataclass
class WarmupBaseline(Baseline):
    inner: Baseline
    warmup_epochs: int = 1
    exp_beta: float = 0.8

    def __post_init__(self) -> None:
        self.warmup = ExponentialBaseline(beta=float(self.exp_beta))

    @property
    def requires_init_td(self) -> bool:
        return self.inner.requires_init_td

    def setup(
        self,
        model,
        adapter,
        device: torch.device,
        batch_size: int,
        dataset_size: int,
    ) -> None:
        self.inner.setup(
            model=model,
            adapter=adapter,
            device=device,
            batch_size=batch_size,
            dataset_size=dataset_size,
        )

    def eval(self, init_td, reward: torch.Tensor, model, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "epoch", 0) < int(self.warmup_epochs):
            return self.warmup.eval(init_td=init_td, reward=reward, model=model, adapter=adapter)
        return self.inner.eval(init_td=init_td, reward=reward, model=model, adapter=adapter)

    def epoch_callback(
        self,
        model,
        adapter,
        device: torch.device,
        batch_size: int,
        dataset_size: int,
        epoch: int,
    ) -> None:
        self.epoch = int(epoch)
        self.inner.epoch_callback(
            model=model,
            adapter=adapter,
            device=device,
            batch_size=batch_size,
            dataset_size=dataset_size,
            epoch=epoch,
        )

    def maybe_update(
        self,
        step: int,
        model,
        adapter,
    ) -> None:
        self.inner.maybe_update(step=step, model=model, adapter=adapter)

