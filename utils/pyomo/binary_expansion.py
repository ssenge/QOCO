from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import pyomo.environ as pyo


@dataclass(frozen=True)
class BinaryExpansionConfig:
    integer_step: float
    continuous_step: float
    max_bits: int | None
    tol: float


@dataclass(frozen=True)
class ExpandedVar:
    offset: float
    terms: Dict[str, float]
    upper_bound: float


def expand_var(
    v: pyo.Var,
    config: BinaryExpansionConfig,
    cache: Dict[str, ExpandedVar],
) -> ExpandedVar:
    if v.is_fixed():
        val = float(pyo.value(v))
        return ExpandedVar(offset=val, terms={}, upper_bound=val)

    name = v.name
    if name in cache:
        return cache[name]

    if v.is_binary():
        expanded = ExpandedVar(offset=0.0, terms={name: 1.0}, upper_bound=1.0)
        cache[name] = expanded
        return expanded

    lb, ub = v.bounds
    if lb is None or ub is None:
        raise ValueError(f"Missing bounds for non-binary variable: {name}")

    step = float(config.integer_step if v.is_integer() else config.continuous_step)
    if step <= 0:
        raise ValueError(f"Step must be > 0 for variable: {name}")

    span = float(ub - lb)
    if span < -config.tol:
        raise ValueError(f"Invalid bounds for {name}: ({lb}, {ub})")
    if abs(span) <= config.tol:
        expanded = ExpandedVar(offset=float(lb), terms={}, upper_bound=float(ub))
        cache[name] = expanded
        return expanded

    levels = int(math.ceil(span / step - config.tol))
    bits = int(math.ceil(math.log2(levels + 1)))
    if config.max_bits is not None and bits > config.max_bits:
        raise ValueError(f"Binary expansion for {name} exceeds max_bits={config.max_bits}")

    terms: Dict[str, float] = {}
    for i in range(bits):
        bit_name = f"{name}_bit{i}"
        terms[bit_name] = float(step) * float(2**i)

    expanded = ExpandedVar(offset=float(lb), terms=terms, upper_bound=float(ub))
    cache[name] = expanded
    return expanded
