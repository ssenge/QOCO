from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from qoco.core.qubo import QUBO


def _normalize_name(name: str) -> str:
    if "[" not in name or not name.endswith("]"):
        return name
    prefix, body = name.split("[", 1)
    body = body[:-1].strip()
    if not body:
        return name
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    parts = [p.strip() for p in body.split(",") if p.strip()]
    joined = ", ".join(parts)
    return f"{prefix}[({joined})]"


def decode_assignment(qubo: QUBO, x: np.ndarray) -> Dict[str, float]:
    if qubo.metadata is None:
        raise ValueError("QUBO metadata is missing; cannot decode assignment.")
    if x.shape[0] != qubo.n_vars:
        raise ValueError(f"Expected x of length {qubo.n_vars}, got {x.shape[0]}")

    decoded: Dict[str, float] = {}
    for name, enc in qubo.metadata.encodings.items():
        value = float(enc.offset)
        for bit_name, weight in enc.terms.items():
            idx = qubo.var_map.get(bit_name)
            if idx is None:
                continue
            value += float(weight) * float(x[idx])
        decoded[_normalize_name(name)] = value
    return decoded


def extract_binary_assignments(
    var_values: Dict[str, float],
    prefixes: Iterable[str] = ("x[", "y["),
) -> Dict[str, int]:
    assignments: Dict[str, int] = {}
    for name, val in var_values.items():
        if not name.startswith(tuple(prefixes)):
            continue
        assignments[_normalize_name(name)] = int(round(val))
    return assignments


def compare_assignments(
    mip_values: Dict[str, int],
    qubo_values: Dict[str, float],
) -> Tuple[int, int]:
    mismatches = 0
    total = 0
    for name, mip_val in mip_values.items():
        if name not in qubo_values:
            continue
        qubo_val = int(round(qubo_values[name]))
        total += 1
        if qubo_val != mip_val:
            mismatches += 1
    return total, mismatches
