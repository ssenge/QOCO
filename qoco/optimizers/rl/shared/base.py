"""
Shared RL-optimizer abstractions (inference-time).

Design goals:
- Generic and extensible (new policy families can be added under src/optimizers/rl/<family>/)
- ABCs (not Protocols)
- Fail-fast (minimal defensive code)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar

import torch

from qoco.core.solution import Solution


class PolicyMeta(ABC):
    """Base class for checkpoint meta objects."""


@dataclass(frozen=True)
class PolicyCheckpoint:
    """Minimal checkpoint: meta + state_dict (pure torch.save/load)."""

    meta: PolicyMeta
    state: Mapping[str, torch.Tensor]


def load_policy_checkpoint(path: Path, *, map_location: str) -> PolicyCheckpoint:
    """Load a policy checkpoint saved as {'meta': PolicyMeta, 'state': state_dict}."""
    # PyTorch 2.6+ defaults to weights_only=True, which disallows unpickling custom dataclasses.
    # We deliberately store `meta` as a dataclass (internal checkpoints), so we need full unpickling here.
    obj = torch.load(str(path), map_location=map_location, weights_only=False)
    meta = obj["meta"]
    state = obj["state"]
    return PolicyCheckpoint(meta=meta, state=state)


def save_policy_checkpoint(path: Path, *, meta: PolicyMeta, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "state": dict(state)}, str(path))


class RLAdapter(ABC):
    """Base adapter for RL inference (common minimal surface)."""

    @abstractmethod
    def make_eval_batch(self, inst: Any, device: str): ...

    @abstractmethod
    def reset(self, batch): ...

AdapterT = TypeVar("AdapterT", bound=RLAdapter)


class PolicyRunner(Generic[AdapterT], ABC):
    """Unified runner interface: load from checkpoint and produce a Solution."""

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_path: Path, *, device: str, adapter: AdapterT) -> "PolicyRunner[AdapterT]": ...

    @abstractmethod
    def run(self, *, adapter: AdapterT, problem: Any, device: str) -> Solution: ...

