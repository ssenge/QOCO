from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO
from qoco.pyomo_utils.qubo import qubo_from_pyomo_model


@dataclass
class PyomoToQuboConverter(Converter[pyo.ConcreteModel, QUBO]):
    def convert(self, problem: pyo.ConcreteModel) -> QUBO:
        return qubo_from_pyomo_model(problem)

