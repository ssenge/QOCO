from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class QLCBO:
    """Quadratic Linearly Constrained Binary Optimization.

    This is a compact container aligned with QCI's "sample-constraint" / QLCBO API:
    - Objective: symmetric matrix Q for x^T Q x (binary x)
    - Constraints: dense matrix of shape (m, n+1) encoding A|rhs for A x + rhs = 0
      (i.e., the RHS is the last column, typically negative b).
    """

    objective: np.ndarray  # shape (n,n)
    constraints: np.ndarray  # shape (m,n+1)
    var_names: Sequence[str] | None = None

    @property
    def n_vars(self) -> int:
        return int(self.objective.shape[0])

    @property
    def n_cons(self) -> int:
        return int(self.constraints.shape[0])

    # Pipeline stats helpers (mirrors QUBO, used by OptimizerPipeline).
    def nvariables(self) -> int:
        return int(self.n_vars)

    def nconstraints(self) -> int:
        return int(self.n_cons)

    def nobjectives(self) -> int:
        return 1

    def __post_init__(self) -> None:
        Q = np.asarray(self.objective, dtype=float)
        C = np.asarray(self.constraints, dtype=float)

        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"QLCBO objective must be square (got shape={Q.shape}).")
        n = int(Q.shape[0])

        if C.ndim != 2 or C.shape[1] != n + 1:
            raise ValueError(
                f"QLCBO constraints must have shape (m, n+1) with rhs last column "
                f"(got shape={C.shape}, n={n})."
            )

        object.__setattr__(self, "objective", Q)
        object.__setattr__(self, "constraints", C)

        if self.var_names is not None and len(self.var_names) != n:
            raise ValueError(f"var_names must have length n={n} (got {len(self.var_names)}).")

    def __str__(self) -> str:
        return f"QLCBO(n_vars={self.n_vars}, n_cons={self.n_cons})"

