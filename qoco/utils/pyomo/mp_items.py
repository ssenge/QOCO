from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass(frozen=True)
class MPItem:
    label: str

    def add(self, ctx: object, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        return model


@dataclass(frozen=True)
class MPConstraint(MPItem):
    pass


@dataclass(frozen=True)
class MPObjective(MPItem):
    pass


@dataclass(frozen=True)
class MPExpression(MPItem):
    pass


@dataclass(frozen=True)
class MPVar(MPItem):
    pass


@dataclass(frozen=True)
class MPSet(MPItem):
    pass


@dataclass(frozen=True)
class MPParam(MPItem):
    pass
