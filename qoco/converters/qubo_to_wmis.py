from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO


def qubo_value(*, qubo: QUBO, x: np.ndarray) -> float:
    """Evaluate QUBO objective: x^T Q x + offset.

    Note: QUBO stores cross terms in upper triangle for i!=j, but we sum Q[i,j] + Q[j,i]
    for consistency with other converters in this codebase.
    """
    Q = np.asarray(qubo.Q, dtype=float)
    x = np.asarray(x, dtype=int).reshape(-1)
    n = int(Q.shape[0])
    if int(x.shape[0]) != n:
        raise ValueError(f"x has wrong shape: got={x.shape} expected=({n},)")

    value = float(qubo.offset)
    value += float(np.dot(np.diag(Q), x))
    for i in range(n):
        xi = int(x[i])
        if xi == 0:
            continue
        for j in range(i + 1, n):
            if int(x[j]) == 0:
                continue
            value += float(Q[i, j]) + float(Q[j, i])
    return float(value)


def _edge(u: int, v: int) -> tuple[int, int]:
    if u == v:
        raise ValueError("Self-loop edges are not allowed in WMIS.")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class WMISInstance:
    """A weighted maximum independent set instance.

    Vertex indices refer to positions in `vertex_weights`.
    """

    n_vars: int
    vertex_weights: np.ndarray
    edges: frozenset[tuple[int, int]]

    # Variable assignment vertices: per i -> (v(i,0), v(i,1))
    assign_vertices: tuple[tuple[int, int], ...]

    # Auxiliary vertices for couplings
    reward_vertices: dict[tuple[int, int], int]  # coef<0
    repulsive_vertices: dict[tuple[int, int], tuple[int, int]]  # coef>0 -> (a,b)

    base_weight: float
    constant_shift: float

    def is_independent(self, S: set[int]) -> bool:
        if not S:
            return True
        edges = self.edges
        for u, v in edges:
            if u in S and v in S:
                return False
        return True

    def weight(self, S: set[int]) -> float:
        return float(np.sum(self.vertex_weights[list(S)])) if S else 0.0

    def decode(self, S: set[int]) -> np.ndarray:
        x = np.zeros((self.n_vars,), dtype=int)
        for i, (v0, v1) in enumerate(self.assign_vertices):
            has0 = v0 in S
            has1 = v1 in S
            if has0 == has1:
                raise ValueError(f"Invalid independent set: variable {i} not uniquely assigned (has0={has0}, has1={has1})")
            x[i] = 1 if has1 else 0
        return x

    def canonical_independent_set(self, x: np.ndarray) -> set[int]:
        x = np.asarray(x, dtype=int).reshape(-1)
        if int(x.shape[0]) != int(self.n_vars):
            raise ValueError(f"x has wrong shape: got={x.shape} expected=({self.n_vars},)")

        S: set[int] = set()
        for i, (v0, v1) in enumerate(self.assign_vertices):
            S.add(v1 if int(x[i]) == 1 else v0)

        for (i, j), vid in self.reward_vertices.items():
            if int(x[i]) == 1 and int(x[j]) == 1:
                S.add(vid)

        for (i, j), (a, b) in self.repulsive_vertices.items():
            if int(x[i]) == 1 and int(x[j]) == 1:
                continue
            if int(x[i]) == 0:
                S.add(a)
            else:
                S.add(b)

        if not self.is_independent(S):
            raise ValueError("Internal error: canonical set is not independent.")
        return S


def _base_weight_for_variables(*, qubo: QUBO, tol: float) -> float:
    Q = np.asarray(qubo.Q, dtype=float)
    n = int(Q.shape[0])

    incident_abs = np.zeros((n,), dtype=float)
    for i in range(n):
        incident_abs[i] += abs(float(Q[i, i]))

    for i in range(n):
        for j in range(i + 1, n):
            coef = float(Q[i, j]) + float(Q[j, i])
            if abs(coef) <= tol:
                continue
            incident_abs[i] += abs(coef)
            incident_abs[j] += abs(coef)

    return float(np.max(incident_abs) + 1.0)


@dataclass
class QuboToWmisConverter(Converter[QUBO, WMISInstance]):
    """Convert a QUBO minimization into an equivalent WMIS maximization instance.

    Minimization objective:  x^T Q x + offset.
    WMIS objective: maximize sum of selected vertex weights.

    The conversion is exact up to a constant shift:
      wmis_weight(canonical_set(x)) = constant_shift - qubo_value(x)
    """

    tol: float = 0.0

    def convert(self, problem: QUBO) -> WMISInstance:
        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])
        tol = float(self.tol)

        base_weight = _base_weight_for_variables(qubo=problem, tol=tol)

        vertex_weights: list[float] = []
        edges: set[tuple[int, int]] = set()

        assign_vertices: list[tuple[int, int]] = []
        for i in range(n):
            v0 = len(vertex_weights)
            vertex_weights.append(float(base_weight))
            v1 = len(vertex_weights)
            vertex_weights.append(float(base_weight) - float(Q[i, i]))
            assign_vertices.append((v0, v1))
            edges.add(_edge(v0, v1))

        reward_vertices: dict[tuple[int, int], int] = {}
        repulsive_vertices: dict[tuple[int, int], tuple[int, int]] = {}

        positive_pair_constant = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                coef = float(Q[i, j]) + float(Q[j, i])
                if abs(coef) <= tol:
                    continue

                if coef < 0.0:
                    r = len(vertex_weights)
                    vertex_weights.append(abs(float(coef)))
                    reward_vertices[(i, j)] = r
                    vi0, _vi1 = assign_vertices[i]
                    vj0, _vj1 = assign_vertices[j]
                    edges.add(_edge(r, vi0))
                    edges.add(_edge(r, vj0))
                else:
                    a = len(vertex_weights)
                    vertex_weights.append(float(coef))
                    b = len(vertex_weights)
                    vertex_weights.append(float(coef))
                    repulsive_vertices[(i, j)] = (a, b)
                    _vi0, vi1 = assign_vertices[i]
                    _vj0, vj1 = assign_vertices[j]
                    edges.add(_edge(a, vi1))
                    edges.add(_edge(b, vj1))
                    edges.add(_edge(a, b))
                    positive_pair_constant += float(coef)

        constant_shift = float(n * base_weight + positive_pair_constant + float(problem.offset))

        return WMISInstance(
            n_vars=n,
            vertex_weights=np.asarray(vertex_weights, dtype=float),
            edges=frozenset(edges),
            assign_vertices=tuple(assign_vertices),
            reward_vertices=reward_vertices,
            repulsive_vertices=repulsive_vertices,
            base_weight=float(base_weight),
            constant_shift=float(constant_shift),
        )

