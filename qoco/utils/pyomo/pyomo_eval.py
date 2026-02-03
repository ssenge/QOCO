from __future__ import annotations

from typing import Mapping

import pyomo.environ as pyo


def eval_objective(model: pyo.ConcreteModel, var_values: Mapping[str, float]) -> float:
    for var in model.component_objects(pyo.Var, active=True):
        for idx in var:
            key = var[idx].name
            var[idx].value = var_values.get(key, 0)
    return float(pyo.value(model.obj))
