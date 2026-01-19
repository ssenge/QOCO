from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.pyomo_utils.decomposition.benders import BendersMapping, classic_benders


@dataclass
class BendersDecomposition:
    master: pyo.ConcreteModel
    sub: pyo.ConcreteModel
    mapping: BendersMapping


@dataclass
class ClassicBendersDecomposer(Converter[pyo.ConcreteModel, BendersDecomposition]):
    def convert(self, problem: pyo.ConcreteModel) -> BendersDecomposition:
        master, sub, mapping = classic_benders(problem)
        return BendersDecomposition(master=master, sub=sub, mapping=mapping)

