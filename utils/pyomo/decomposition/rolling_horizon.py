from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

from qoco.converters.decomposition.lagrangian import VarKey


def _var_key(v: pyo.Var) -> VarKey:
    return VarKey(comp=v.parent_component().name, idx=v.index())


def _matches_prefix(v: pyo.Var, prefixes: Sequence[str]) -> bool:
    comp = v.parent_component().name
    name = v.name
    return any(comp.startswith(p) or name.startswith(p) for p in prefixes)


def _get_active_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    objs = [o for o in model.component_data_objects(pyo.Objective, active=True, descend_into=True)]
    if not objs:
        raise ValueError("No active Objective found on model.")
    if len(objs) > 1:
        raise ValueError("Multiple active Objectives found; cannot apply soft penalties deterministically.")
    return objs[0]


def extract_prefix_values(
    model: pyo.ConcreteModel, *, prefixes: Sequence[str]
) -> Dict[VarKey, float]:
    """Extract current values for variables matching prefixes."""
    values: Dict[VarKey, float] = {}
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if not _matches_prefix(v, prefixes):
            continue
        if v.value is None:
            continue
        values[_var_key(v)] = float(pyo.value(v))
    return values


def add_hard_boundary_constraints(
    model: pyo.ConcreteModel,
    *,
    prefixes: Sequence[str],
    values: Dict[VarKey, float],
    name_prefix: str = "rh_fix",
) -> int:
    """Add hard equality constraints for selected variables.

    Returns number of constraints added.
    """
    if not prefixes:
        return 0
    if not values:
        return 0
    cname = f"{name_prefix}_constraints"
    if not hasattr(model, cname):
        setattr(model, cname, pyo.ConstraintList())
    clist: pyo.ConstraintList = getattr(model, cname)

    added = 0
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if not _matches_prefix(v, prefixes):
            continue
        key = _var_key(v)
        if key not in values:
            continue
        clist.add(v == float(values[key]))
        added += 1
    return added


def add_soft_boundary_penalties(
    model: pyo.ConcreteModel,
    *,
    prefixes: Sequence[str],
    values: Dict[VarKey, float],
    penalty: float = 1.0,
    name_prefix: str = "rh_soft",
) -> int:
    """Add L1 soft penalties |x - value| via slack variables.

    Returns number of slack vars added.
    """
    if not prefixes:
        return 0
    if not values:
        return 0
    if penalty <= 0:
        raise ValueError("penalty must be > 0 for soft boundary constraints.")

    vname = f"{name_prefix}_slack"
    cname = f"{name_prefix}_constraints"
    if not hasattr(model, vname):
        setattr(model, vname, pyo.VarList(domain=pyo.NonNegativeReals))
    if not hasattr(model, cname):
        setattr(model, cname, pyo.ConstraintList())
    slacks: pyo.VarList = getattr(model, vname)
    clist: pyo.ConstraintList = getattr(model, cname)

    added = 0
    slack_vars: List[pyo.Var] = []
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if not _matches_prefix(v, prefixes):
            continue
        key = _var_key(v)
        if key not in values:
            continue
        val = float(values[key])
        s = slacks.add()
        clist.add(s >= v - val)
        clist.add(s >= val - v)
        slack_vars.append(s)
        added += 1

    if slack_vars:
        obj = _get_active_objective(model)
        expr = obj.expr + float(penalty) * sum(slack_vars)
        sense = obj.sense
        obj.deactivate()
        model.add_component(f"{name_prefix}_objective", pyo.Objective(expr=expr, sense=sense))
    return added


@dataclass(frozen=True)
class RollingHorizonChunk:
    chunk_id: int
    window_keys: List[object]
    core_keys: List[object]
    overlap_keys: List[object]

