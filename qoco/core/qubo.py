from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class QuboVarEncoding:
    """Encoding for an original variable into QUBO bits."""

    offset: float
    terms: Dict[str, float]
    upper_bound: float


@dataclass(frozen=True)
class QuboMetadata:
    """Optional metadata to decode QUBO solutions."""

    encodings: Dict[str, QuboVarEncoding]
    slack_prefixes: Tuple[str, ...] = ("slack_le", "slack_ge")


@dataclass(frozen=True)
class QUBO:
    """A QUBO in matrix form: minimize x^T Q x + offset.

    Convention:
    - Q is stored upper-triangular for i!=j: coefficient for x_i*x_j is in Q[min(i,j), max(i,j)].
    - Diagonal contains linear terms (and x_i^2 terms, which equal x_i for binary vars).
    - var_map maps a stable variable key -> index in x/Q.
    """

    Q: np.ndarray
    offset: float
    var_map: Dict[str, int]
    metadata: QuboMetadata | None = None

    @property
    def n_vars(self) -> int:
        return int(self.Q.shape[0])


@dataclass(frozen=True)
class QuboCoupling:
    """A cross-block coupling summary entry for the original QUBO."""

    i: int
    j: int
    coef: float


@dataclass(frozen=True)
class QuboPartition:
    """A partition of a QUBO into disjoint sub-QUBOs (local variable spaces).

    Note: the original full QUBO is intentionally NOT stored here (caller already has it).
    """

    blocks: List[List[int]]                  # disjoint global variable indices
    sub_qubos: List[QUBO]                    # one QUBO per block (local indexing)
    global_indices: List[np.ndarray]         # per block: local idx -> global idx
    couplings: List[QuboCoupling]            # cross-block couplings in original Q

