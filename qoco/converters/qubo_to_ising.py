from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from qiskit.quantum_info import SparsePauliOp

import numpy as np

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


def _z_label(n: int, i: int) -> str:
    # Qiskit Pauli labels are big-endian: rightmost char is qubit 0.
    s = ["I"] * int(n)
    s[int(n) - 1 - int(i)] = "Z"
    return "".join(s)


def _zz_label(n: int, i: int, j: int) -> str:
    s = ["I"] * int(n)
    s[int(n) - 1 - int(i)] = "Z"
    s[int(n) - 1 - int(j)] = "Z"
    return "".join(s)


@dataclass
class QuboToIsingConverter(Converter[QUBO, tuple[Any, float]]):
    """Convert `QUBO` into an Ising operator + constant offset.

    Output operator is a Qiskit `SparsePauliOp` (typed as Any to avoid hard dependency).
    """

    tol: float = 0.0

    def convert(self, problem: QUBO) -> tuple[Any, float]:

        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])

        # Convention matches `QuboToQuadraticProgramConverter`:
        # effective quadratic coefficient for i<j is Q[i,j] + Q[j,i].
        diag = np.asarray(np.diag(Q), dtype=float)

        quad_pairs: list[tuple[tuple[int, int], float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                c = float(Q[i, j]) + float(Q[j, i])
                if c != 0.0:
                    quad_pairs.append(((i, j), c))

        offset = float(problem.offset)
        offset += 0.5 * float(np.sum(diag))
        offset += 0.25 * float(sum(c for (_ij, c) in quad_pairs))

        # h_i = -1/2 * w_i - 1/4 * sum_{j!=i} w_ij
        h = np.zeros((n,), dtype=float)
        h -= 0.5 * diag
        for (i, j), c in quad_pairs:
            h[i] -= 0.25 * float(c)
            h[j] -= 0.25 * float(c)

        # J_ij = 1/4 * w_ij
        terms: list[tuple[str, float]] = []
        tol = float(self.tol)

        for i in range(n):
            coef = float(h[i])
            if abs(coef) > tol:
                terms.append((_z_label(n, i), coef))

        for (i, j), c in quad_pairs:
            coef = 0.25 * float(c)
            if abs(coef) > tol:
                terms.append((_zz_label(n, i, j), coef))

        if not terms:
            # Avoid empty operator errors downstream.
            return SparsePauliOp.from_list([("I" * n, 0.0)]), float(offset)

        return SparsePauliOp.from_list(terms), float(offset)

