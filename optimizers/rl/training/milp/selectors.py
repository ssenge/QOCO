from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pyomo.environ as pyo

from qoco.optimizers.rl.training.milp.pyomo_eval import CompiledNumericMILPKernel


class CandidateSelector:
    """Build a fixed-size action map for a MILP.

    Contract:
    - `prepare(...)` is called once per episode with the Pyomo model and compiled kernel.
    - `var_indices(step)` returns an array of length `n_actions`, filled with MILP var indices
      (kernel-space indices). Missing actions are encoded as -1.
    """

    n_actions: int

    def prepare(self, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel) -> "CandidateSelector":
        return self

    def var_indices(self, step: int) -> np.ndarray:
        raise NotImplementedError


def _iter_binary_var_ids(model: pyo.ConcreteModel) -> Iterable[int]:
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.is_binary():
            yield id(v)


@dataclass(frozen=True)
class AllBinarySelector(CandidateSelector):
    """Default: all binary variables become actions (can explode)."""

    _indices: np.ndarray

    @classmethod
    def from_model(cls, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel) -> "AllBinarySelector":
        idxs = [kernel.var_id_to_idx[vid] for vid in _iter_binary_var_ids(model) if vid in kernel.var_id_to_idx]
        arr = np.asarray(idxs, dtype=np.int64)
        return cls(_indices=arr)

    @property
    def n_actions(self) -> int:
        return int(self._indices.size)

    def var_indices(self, step: int) -> np.ndarray:
        return self._indices


@dataclass(frozen=True)
class PrefixBinarySelector(CandidateSelector):
    """All binary vars whose VarData name starts with `prefix`."""

    prefix: str
    _indices: np.ndarray

    @classmethod
    def from_model(
        cls, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel, prefix: str
    ) -> "PrefixBinarySelector":
        idxs: list[int] = []
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
            if not v.is_binary():
                continue
            if not str(v.name).startswith(prefix):
                continue
            vid = id(v)
            if vid in kernel.var_id_to_idx:
                idxs.append(kernel.var_id_to_idx[vid])
        arr = np.asarray(idxs, dtype=np.int64)
        return cls(prefix=str(prefix), _indices=arr)

    @property
    def n_actions(self) -> int:
        return int(self._indices.size)

    def var_indices(self, step: int) -> np.ndarray:
        return self._indices


@dataclass(frozen=True)
class ComponentIndexGatedSelector(CandidateSelector):
    """Gate a Var component by one index position equaling `step`.

    Example:
    - BinPacking: `component="x"`, `index_pos=1` (x[b, i]) with step=i => actions are bins b
    - SimpleCSP:  `component="x"`, `index_pos=1` (x[driver, segment]) with step=segment
    """

    component: str
    index_pos: int
    step_offset: int = 0
    _by_step: tuple[np.ndarray, ...] = ()
    _n_actions: int = 0

    @property
    def n_actions(self) -> int:
        return int(self._n_actions)

    def prepare(self, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel) -> "ComponentIndexGatedSelector":
        comp = getattr(model, self.component)
        if not isinstance(comp, pyo.Var):
            raise TypeError(f"model.{self.component} is not a Var component")

        tmp: dict[int, list[int]] = {}
        max_len = 0
        for idx in comp:
            v = comp[idx]
            if not v.is_binary():
                continue
            idx_tup = idx if isinstance(idx, tuple) else (idx,)
            step_val = int(idx_tup[int(self.index_pos)]) - int(self.step_offset)
            vid = id(v)
            if vid not in kernel.var_id_to_idx:
                continue
            tmp.setdefault(step_val, []).append(kernel.var_id_to_idx[vid])

        max_step = max(tmp.keys()) if tmp else -1
        by_step: list[np.ndarray] = []
        for s in range(max_step + 1):
            items = tmp.get(s, [])
            items_sorted = np.asarray(sorted(items), dtype=np.int64)
            by_step.append(items_sorted)
            max_len = max(max_len, int(items_sorted.size))

        return ComponentIndexGatedSelector(
            component=str(self.component),
            index_pos=int(self.index_pos),
            step_offset=int(self.step_offset),
            _by_step=tuple(by_step),
            _n_actions=int(max_len),
        )

    def var_indices(self, step: int) -> np.ndarray:
        s = int(step)
        if s < 0 or s >= len(self._by_step):
            return np.full((self.n_actions,), -1, dtype=np.int64)
        row = self._by_step[s]
        out = np.full((self.n_actions,), -1, dtype=np.int64)
        out[: int(row.size)] = row
        return out


@dataclass(frozen=True)
class CombinedSelector(CandidateSelector):
    """Concatenate multiple selectors into one padded action map."""

    selectors: Sequence[CandidateSelector]
    _offsets: np.ndarray
    _n_actions: int

    @classmethod
    def from_selectors(cls, selectors: Sequence[CandidateSelector]) -> "CombinedSelector":
        offs: list[int] = [0]
        for s in selectors:
            offs.append(offs[-1] + int(s.n_actions))
        return cls(selectors=tuple(selectors), _offsets=np.asarray(offs, dtype=np.int64), _n_actions=int(offs[-1]))

    @property
    def n_actions(self) -> int:
        return int(self._n_actions)

    def prepare(self, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel) -> "CombinedSelector":
        prepared = [s.prepare(model, kernel) for s in self.selectors]
        return CombinedSelector.from_selectors(prepared)

    def var_indices(self, step: int) -> np.ndarray:
        out = np.full((self.n_actions,), -1, dtype=np.int64)
        for i, s in enumerate(self.selectors):
            lo = int(self._offsets[i])
            hi = int(self._offsets[i + 1])
            out[lo:hi] = s.var_indices(step)
        return out


@dataclass(frozen=True)
class SelectorFactory:
    """Small helper to build selectors late (after model/kernel compilation)."""

    make: Callable[[pyo.ConcreteModel, CompiledNumericMILPKernel], CandidateSelector]

    def build(self, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel) -> CandidateSelector:
        return self.make(model, kernel)


@dataclass(frozen=True)
class PaddedSelector(CandidateSelector):
    """Pad/truncate another selector to a fixed global action size."""

    inner: CandidateSelector
    target_n_actions: int

    @property
    def n_actions(self) -> int:
        return int(self.target_n_actions)

    def prepare(self, model: pyo.ConcreteModel, kernel: CompiledNumericMILPKernel) -> "CandidateSelector":
        inner_p = self.inner.prepare(model, kernel)
        return PaddedSelector(inner=inner_p, target_n_actions=int(self.target_n_actions))

    def var_indices(self, step: int) -> np.ndarray:
        inner_idx = self.inner.var_indices(step)
        out = np.full((int(self.target_n_actions),), -1, dtype=np.int64)
        n = min(int(out.size), int(inner_idx.size))
        if n > 0:
            out[:n] = inner_idx[:n]
        return out

