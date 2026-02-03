from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ToggleableBlock(nn.Module, ABC):
    """A dimension-preserving block that can be disabled as identity.

    Subclasses implement `_forward_enabled(x)`; when `enabled=False`, `forward(x)` returns `x`.
    """

    def __init__(self, *, enabled: bool):
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return self._forward_enabled(x)

    @abstractmethod
    def _forward_enabled(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

