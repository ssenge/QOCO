from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class BaseTorchKernel(Protocol):
    """Minimal kernel interface for ML/RL decoding.

    Each concrete problem variant implements this to provide fast reset/step/mask/reward.
    """

    def reset(self, td: TensorDict, *, init_action_mask: bool = True) -> TensorDict: ...

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict: ...

    def action_mask(self, td: TensorDict) -> torch.Tensor: ...

    def reward(self, td: TensorDict) -> torch.Tensor: ...

