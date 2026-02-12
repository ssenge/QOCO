from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qoco.core.qubo import QUBO


def cvar_aggregation(*, alpha: float) -> Callable[[list[float]], float]:
    """Return a CVaR aggregation for QAOA energies (minimization).

    Qiskit QAOA can accept an `aggregation` callable that combines per-sample energies into
    a scalar objective. CVaR(alpha) uses the mean of the best alpha-fraction of energies.

    - alpha=1.0 -> mean energy (standard QAOA)
    - alpha small -> focus on the low-energy tail (higher chance of sampling good bitstrings)
    """

    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    def agg(values: list[float]) -> float:
        if not values:
            raise ValueError("no values")
        k = max(1, int(np.ceil(a * len(values))))
        best = np.sort(np.asarray(values, dtype=float))[:k]
        return float(np.mean(best))

    return agg


@dataclass(frozen=True)
class FourierInitialPoint:
    """Simple FQAOA-style initializer (angles schedule), not reduced-parameter optimization.

    Produces a full (gamma_1..gamma_p, beta_1..beta_p) vector from a small set of Fourier
    coefficients. This is useful as an initialization when you increase p.
    """

    gamma_cos: list[float]
    gamma_sin: list[float]
    beta_cos: list[float]
    beta_sin: list[float]

    def build(self, _qubo: QUBO, _n: int, reps: int) -> list[float]:
        p = int(reps)
        if p <= 0:
            return []

        def series(t: float, cos: list[float], sin: list[float]) -> float:
            out = 0.0
            for k, a in enumerate(cos, start=1):
                out += float(a) * float(np.cos(2.0 * np.pi * k * t))
            for k, b in enumerate(sin, start=1):
                out += float(b) * float(np.sin(2.0 * np.pi * k * t))
            return out

        gammas: list[float] = []
        betas: list[float] = []
        for ell in range(1, p + 1):
            t = float(ell) / float(p + 1)  # in (0,1)
            gammas.append(series(t, self.gamma_cos, self.gamma_sin))
            betas.append(series(t, self.beta_cos, self.beta_sin))

        return gammas + betas


def uniform_h_initial_state(*, n: int) -> QuantumCircuit:
    qc = QuantumCircuit(int(n))
    qc.h(range(int(n)))
    return qc


def bitstring_initial_state(*, bitstring: Iterable[int]) -> QuantumCircuit:
    bits = [int(b) for b in bitstring]
    qc = QuantumCircuit(len(bits))
    for i, b in enumerate(bits):
        if b:
            qc.x(i)
    return qc


def x_mixer_operator(*, n: int) -> SparsePauliOp:
    terms = []
    for i in range(int(n)):
        p = ["I"] * int(n)
        p[i] = "X"
        terms.append(("".join(p), 1.0))
    return SparsePauliOp.from_list(terms)


def xy_group_mixer_operator(*, n: int, groups: list[list[int]]) -> SparsePauliOp:
    """Hamming-weight preserving XY mixer on groups.

    For each group g, adds pairwise terms (X_i X_j + Y_i Y_j).
    """

    n = int(n)
    terms: list[tuple[str, float]] = []

    def add(pauli_i: str, pauli_j: str, i: int, j: int, w: float) -> None:
        p = ["I"] * n
        p[i] = pauli_i
        p[j] = pauli_j
        terms.append(("".join(p), float(w)))

    for g in groups:
        idx = [int(i) for i in g]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i = idx[a]
                j = idx[b]
                add("X", "X", i, j, 1.0)
                add("Y", "Y", i, j, 1.0)

    if not terms:
        return SparsePauliOp.from_list([("I" * n, 0.0)])
    return SparsePauliOp.from_list(terms)


def groups_from_qubo_metadata(qubo: QUBO, *, key: str = "groups") -> list[list[int]]:
    """Read groups from `qubo.metadata[key]` if present.

    Convention: groups is a list of groups, each group is a list of variable indices.
    """

    meta = getattr(qubo, "metadata", None)
    if not isinstance(meta, dict):
        return []
    raw = meta.get(key, None)
    if raw is None:
        return []
    return [[int(i) for i in grp] for grp in raw]

