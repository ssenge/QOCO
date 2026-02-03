from __future__ import annotations

from typing import Any, Optional

import torch
from tensordict import TensorDict

from qoco.optimizers.rl.shared.rl4co_switch import maybe_enable_local_rl4co

maybe_enable_local_rl4co()

from rl4co.envs import RL4COEnvBase


class AdapterEnv(RL4COEnvBase):
    """Generic RL4CO Env wrapper around our ProblemAdapter-style API.

    This removes per-problem RL4CO boilerplate: `_reset`, `_step`, `_get_action_mask`.
    Problems can keep tiny subclasses that only set `name = ...`.
    """

    # Subclasses should override this (RL4CO relies on it)
    name: str = "adapter"

    def __init__(self, adapter: Any, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter
        # RL4CO base classes expect `self.generator` for dataset creation.
        # Prefer adapter.generator if available, otherwise fall back to adapter.train_batch.
        gen = None
        try:
            gen = getattr(adapter, "generator", None)
        except Exception:
            gen = None
        if gen is not None:
            self.generator = gen  # type: ignore[attr-defined]
        else:
            self.generator = lambda batch_size: adapter.train_batch(  # type: ignore[attr-defined]
                int(batch_size[0] if isinstance(batch_size, list) else batch_size),
                device=str(self.device),
            )

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        if batch_size is None:
            batch_size = [32]
        elif isinstance(batch_size, int):
            batch_size = [batch_size]

        bs = int(batch_size[0])

        if td is None:
            # Prefer adapter.generator if present, otherwise fall back to adapter.train_batch
            if hasattr(self.adapter, "generator"):
                td_in = self.adapter.generator(bs)  # type: ignore[attr-defined]
            else:
                # adapter.train_batch wants a string device
                td_in = self.adapter.train_batch(bs, device=str(self.device))
        else:
            td_in = td

        return self.adapter.reset(td_in)

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        return self.adapter.step(td, action)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # RL4CO decoding may query masks on stale tds; adapters should recompute if needed.
        return self.adapter.action_mask(td)

    def _get_reward(self, td: TensorDict, actions) -> torch.Tensor:
        r = self.adapter.reward(td)
        if isinstance(r, torch.Tensor):
            return r
        return td.get("reward", torch.zeros(td.batch_size[0], device=td.device))

    def check_solution_validity(self, td: TensorDict, actions):
        return torch.ones(td.batch_size[0], dtype=torch.bool, device=td.device)

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        return None

