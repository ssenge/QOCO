from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

from qoco.core.converter import Converter
from qoco.converters.decomposition.lagrangian import VarKey
from qoco.utils.pyomo.decomposition.rolling_horizon import RollingHorizonChunk


def _var_key(v: pyo.Var) -> VarKey:
    return VarKey(comp=v.parent_component().name, idx=v.index())


def _matches_prefix(v: pyo.Var, prefixes: Sequence[str]) -> bool:
    comp = v.parent_component().name
    name = v.name
    return any(comp.startswith(p) or name.startswith(p) for p in prefixes)


def _time_key(v: pyo.Var) -> object | None:
    idx = v.index()
    if idx is None:
        return None
    if isinstance(idx, tuple) and len(idx) > 0:
        return idx[0]
    return idx


@dataclass
class RollingHorizonSplit:
    chunks: List[RollingHorizonChunk]
    sub_models: List[pyo.ConcreteModel]
    prefixes: List[str]
    window: int
    overlap: int
    time_keys: List[object]


@dataclass
class RollingHorizonDecomposer(Converter[pyo.ConcreteModel, RollingHorizonSplit]):
    prefixes: List[str]
    window: int
    overlap: int
    default_fix_value: float = 0.0

    def convert(self, problem: pyo.ConcreteModel) -> RollingHorizonSplit:
        if not self.prefixes:
            raise ValueError("RollingHorizonDecomposer requires at least one prefix.")
        if self.window <= 0:
            raise ValueError("window must be > 0")
        if self.overlap < 0 or self.overlap >= self.window:
            raise ValueError("overlap must be >=0 and < window")

        model = problem.clone()

        # Collect split vars + time keys
        time_of: Dict[VarKey, object] = {}
        keys: List[object] = []
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
            if not _matches_prefix(v, self.prefixes):
                continue
            tk = _time_key(v)
            if tk is None:
                continue
            key = _var_key(v)
            time_of[key] = tk
            keys.append(tk)

        if not keys:
            raise ValueError("No variables matched prefixes with index-based keys.")

        try:
            time_keys = sorted(set(keys))
        except TypeError:
            time_keys = sorted(set(keys), key=str)

        windows = self._build_windows(time_keys=time_keys)
        chunks: List[RollingHorizonChunk] = []
        sub_models: List[pyo.ConcreteModel] = []

        for cid, (window_keys, core_keys, overlap_keys) in enumerate(windows):
            sub = model.clone()
            window_key_set = set(window_keys)

            # Fix split vars outside window
            for v in sub.component_data_objects(pyo.Var, active=True, descend_into=True):
                if not _matches_prefix(v, self.prefixes):
                    continue
                tk = _time_key(v)
                if tk is None:
                    continue
                if tk not in window_key_set:
                    if v.value is not None:
                        v.fix(float(pyo.value(v)))
                    else:
                        v.fix(float(self.default_fix_value))

            # Deactivate constraints that touch split vars outside window
            for con in sub.component_data_objects(pyo.Constraint, active=True, descend_into=True):
                vars_ = identify_variables(con.body, include_fixed=False)
                deactivate = False
                for v in vars_:
                    if not _matches_prefix(v, self.prefixes):
                        continue
                    tk = _time_key(v)
                    if tk is None:
                        continue
                    if tk not in window_key_set:
                        deactivate = True
                        break
                if deactivate:
                    con.deactivate()

            chunks.append(
                RollingHorizonChunk(
                    chunk_id=cid,
                    window_keys=list(window_keys),
                    core_keys=list(core_keys),
                    overlap_keys=list(overlap_keys),
                )
            )
            sub_models.append(sub)

        return RollingHorizonSplit(
            chunks=chunks,
            sub_models=sub_models,
            prefixes=list(self.prefixes),
            window=int(self.window),
            overlap=int(self.overlap),
            time_keys=list(time_keys),
        )

    def _build_windows(self, *, time_keys: List[object]) -> List[tuple[List[object], List[object], List[object]]]:
        stride = self.window - self.overlap
        out: List[tuple[List[object], List[object], List[object]]] = []
        i = 0
        n = len(time_keys)
        while i < n:
            window_keys = time_keys[i : min(n, i + self.window)]
            end = i + self.window
            is_last = end >= n
            if self.overlap > 0 and not is_last:
                core_keys = window_keys[: -self.overlap]
                overlap_keys = window_keys[-self.overlap :]
            else:
                core_keys = window_keys
                overlap_keys = window_keys[-self.overlap :] if self.overlap > 0 else []
            out.append((window_keys, core_keys, overlap_keys))
            if is_last:
                break
            i += stride
        return out

