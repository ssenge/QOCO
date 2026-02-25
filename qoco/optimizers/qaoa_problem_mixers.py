"""
Problem-aware QAOA mixers for assignment problems.

Each problem type provides:
  - A mixer builder:  (qubo, n) -> SparsePauliOp
  - An initial state builder:  (qubo, n) -> QuantumCircuit

The XY group mixer preserves Hamming weight within each constraint group,
keeping the QAOA search inside the feasible subspace for that constraint family.
"""
from __future__ import annotations

import re
from typing import Callable, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qoco.core.qubo import QUBO
from qoco.optimizers.qaoa_rich_variants import xy_group_mixer_operator

BuildMixer = Callable[[QUBO, int], SparsePauliOp]
BuildInitialState = Callable[[QUBO, int], QuantumCircuit]

_XDS_RE = re.compile(r"^x\[\(?(\d+),\s*(\d+)\)?\]$")
_UD_RE = re.compile(r"^u\[\(?(\d+)\)?\]$")
_XAC_RE = re.compile(r"^x_ac\[\(?(\d+),\s*(\d+)\)?\]$")


def _parse_segment_v25_groups(qubo: QUBO) -> Tuple[dict[int, list[int]], list[int]]:
    """Parse Segment v2.5 var_map into coverage groups and u[d] qubit indices.

    Returns:
        coverage_groups: {segment_id: [qubit_indices for x[d,s]]}
        u_qubits: [qubit_indices for u[d]]
    """
    coverage_groups: dict[int, list[int]] = {}
    u_qubits: list[int] = []

    for name, idx in qubo.var_map.items():
        m = _XDS_RE.match(name)
        if m:
            seg = int(m.group(2))
            coverage_groups.setdefault(seg, []).append(idx)
            continue
        m = _UD_RE.match(name)
        if m:
            u_qubits.append(idx)

    return coverage_groups, u_qubits


def _parse_duty_daily_groups(qubo: QUBO) -> dict[int, list[int]]:
    """Parse Duty Daily ASC var_map into per-duty groups.

    Returns:
        duty_groups: {duty_id: [qubit_indices for x_ac[d,k]]}
    """
    duty_groups: dict[int, list[int]] = {}

    for name, idx in qubo.var_map.items():
        m = _XAC_RE.match(name)
        if m:
            duty = int(m.group(2))
            duty_groups.setdefault(duty, []).append(idx)

    return duty_groups


def _composite_mixer(
    n: int,
    xy_groups: list[list[int]],
    free_qubits: list[int],
    xy_weight: float = 1.0,
    free_weight: float = 1.0,
) -> SparsePauliOp:
    """XY group mixer on constrained groups + single-qubit X mixer on free qubits."""
    terms: list[tuple[str, float]] = []

    for g in xy_groups:
        if len(g) < 2:
            free_qubits.extend(g)
            continue
        pairs = [(g[a], g[b]) for a in range(len(g)) for b in range(a + 1, len(g))]
        for i, j in pairs:
            for pauli in ("X", "Y"):
                p = ["I"] * n
                p[i] = pauli
                p[j] = pauli
                terms.append(("".join(p), xy_weight))

    for q in free_qubits:
        p = ["I"] * n
        p[q] = "X"
        terms.append(("".join(p), free_weight))

    if not terms:
        return SparsePauliOp.from_list([("I" * n, 0.0)])
    return SparsePauliOp.from_list(terms)


def _one_hot_initial_state(
    n: int,
    groups: list[list[int]],
    free_qubits: list[int],
) -> QuantumCircuit:
    """One-hot per group (X on first qubit) + H on free qubits."""
    qc = QuantumCircuit(n)
    for g in groups:
        if g:
            qc.x(g[0])
    for q in free_qubits:
        qc.h(q)
    return qc


# ══════════════════════════════════════════════════════════════════════════════
#                    SEGMENT v2.5 BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def segment_v25_mixer(qubo: QUBO, n: int) -> SparsePauliOp:
    """XY group mixer on coverage groups + X mixer on u[d] qubits."""
    coverage_groups, u_qubits = _parse_segment_v25_groups(qubo)
    xy_groups = list(coverage_groups.values())
    return _composite_mixer(n, xy_groups, u_qubits)


def segment_v25_initial_state(qubo: QUBO, n: int) -> QuantumCircuit:
    """One-hot per coverage group + H on u[d] qubits."""
    coverage_groups, u_qubits = _parse_segment_v25_groups(qubo)
    return _one_hot_initial_state(n, list(coverage_groups.values()), u_qubits)


# ══════════════════════════════════════════════════════════════════════════════
#                    DUTY DAILY ASC BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def duty_daily_mixer(qubo: QUBO, n: int) -> SparsePauliOp:
    """XY group mixer on per-duty groups (at-most-one driver per duty).

    No ancilla qubits. Qubits not in any duty group get standard X mixing.
    """
    duty_groups = _parse_duty_daily_groups(qubo)
    grouped_qubits = {q for g in duty_groups.values() for q in g}
    free_qubits = [i for i in range(n) if i not in grouped_qubits]
    return _composite_mixer(n, list(duty_groups.values()), free_qubits)


def duty_daily_initial_state(qubo: QUBO, n: int) -> QuantumCircuit:
    """One-hot per duty group + H on free qubits.

    Starts each duty with one driver assigned (Hamming weight 1 per group),
    which the XY mixer preserves.
    """
    duty_groups = _parse_duty_daily_groups(qubo)
    grouped_qubits = {q for g in duty_groups.values() for q in g}
    free_qubits = [i for i in range(n) if i not in grouped_qubits]
    return _one_hot_initial_state(n, list(duty_groups.values()), free_qubits)
