from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

from qoco.core.qubo import QUBO


def _iter_active_vardata(model: pyo.ConcreteModel) -> Iterable[pyo.Var]:
    return model.component_data_objects(pyo.Var, active=True, descend_into=True)


def qubo_from_pyomo_model(model: pyo.ConcreteModel) -> QUBO:
    """Extract a QUBO matrix from a Pyomo model objective (constraints ignored).

    Requirements:
    - exactly 1 active objective
    - objective at most quadratic
    """
    vars_list = list(_iter_active_vardata(model))
    n = len(vars_list)
    if n == 0:
        return QUBO(Q=np.zeros((0, 0), dtype=np.float64), offset=0.0, var_map={}, metadata=None)

    var_to_idx = {id(v): i for i, v in enumerate(vars_list)}
    var_map = {v.name: i for i, v in enumerate(vars_list)}

    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if len(objs) != 1:
        raise ValueError(f"Expected exactly 1 active objective, found {len(objs)}")
    obj = objs[0]

    repn = generate_standard_repn(obj.expr, quadratic=True)
    Q = np.zeros((n, n), dtype=np.float64)
    offset = float(repn.constant) if repn.constant is not None else 0.0

    if repn.linear_vars is not None:
        for v, coef in zip(repn.linear_vars, repn.linear_coefs):
            idx = var_to_idx.get(id(v))
            if idx is None:
                continue
            Q[idx, idx] += float(coef)

    if repn.quadratic_vars is not None:
        for (v1, v2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
            i = var_to_idx.get(id(v1))
            j = var_to_idx.get(id(v2))
            if i is None or j is None:
                continue
            if i == j:
                Q[i, i] += float(coef)
            else:
                row, col = (i, j) if i < j else (j, i)
                Q[row, col] += float(coef)

    return QUBO(Q=Q, offset=float(offset), var_map=var_map, metadata=None)

