from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


@dataclass
class QuboToDictConverter(Converter[QUBO, dict[str, float]]):
    """Convert QUBO into a tuple-string coefficient dict.

    Mapping:
    - \"()\" -> offset
    - \"(i,)\" -> linear term for variable i
    - \"(i,j)\" -> quadratic term for i<j
    """

    tol: float = 0.0

    def convert(self, problem: QUBO) -> dict[str, float]:
        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])
        out: dict[str, float] = {}
        if float(problem.offset) != 0.0:
            out["()"] = float(problem.offset)
        for i in range(n):
            coef = float(Q[i, i])
            if abs(coef) > float(self.tol):
                out[f"({i},)"] = coef
        for i in range(n):
            for j in range(i + 1, n):
                coef = float(Q[i, j]) + float(Q[j, i])
                if abs(coef) > float(self.tol):
                    out[f"({i},{j})"] = coef
        return out
