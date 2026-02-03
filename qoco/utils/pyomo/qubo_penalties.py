from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PenaltyConfig:
    penalty_eq: float
    penalty_le: float
    penalty_ge: float
    require_integral: bool
    integrality_tol: float
    slack_step: float


def default_penalty_config(penalty: float) -> PenaltyConfig:
    return PenaltyConfig(
        penalty_eq=float(penalty),
        penalty_le=float(penalty),
        penalty_ge=float(penalty),
        require_integral=False,
        integrality_tol=1e-6,
        slack_step=1.0,
    )


def auto_penalty_from_linear_objective(terms: Dict[str, float]) -> float:
    lower = 0.0
    upper = 0.0
    for coef in terms.values():
        if coef >= 0:
            upper += coef
        else:
            lower += coef
    penalty = 1.0 + (upper - lower)
    return float(penalty if penalty > 0 else 1.0)


def ensure_square(Q: np.ndarray, n: int) -> np.ndarray:
    if Q.shape[0] == n:
        return Q
    out = np.zeros((n, n), dtype=np.float64)
    m = Q.shape[0]
    if m > 0:
        out[:m, :m] = Q
    return out


def is_integral(x: float, tol: float) -> bool:
    return abs(x - round(x)) <= tol


def require_integral(value: float, tol: float, label: str) -> None:
    if not is_integral(value, tol):
        raise ValueError(f"Non-integer {label} not supported: {value}")


def lhs_bounds(terms: Dict[str, float], constant: float) -> Tuple[float, float]:
    min_lhs = float(constant)
    max_lhs = float(constant)
    for coef in terms.values():
        if coef >= 0:
            max_lhs += coef
        else:
            min_lhs += coef
    return float(min_lhs), float(max_lhs)


def add_penalty_eq(
    *,
    Q: np.ndarray,
    var_map: Dict[str, int],
    terms: Dict[str, float],
    constant: float,
    rhs: float,
    penalty: float,
) -> float:
    if penalty <= 0:
        raise ValueError("Penalty must be > 0.")
    c = float(constant - rhs)
    names = list(terms.keys())
    coefs = [float(terms[n]) for n in names]

    for i, name_i in enumerate(names):
        idx_i = var_map[name_i]
        ai = coefs[i]
        Q[idx_i, idx_i] += penalty * (ai * ai + 2.0 * ai * c)
        for j in range(i + 1, len(names)):
            idx_j = var_map[names[j]]
            aj = coefs[j]
            row, col = (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)
            Q[row, col] += penalty * (2.0 * ai * aj)

    return float(penalty * (c * c))


def slack_terms(
    *,
    var_map: Dict[str, int],
    slack_max: float,
    config: PenaltyConfig,
    prefix: str,
) -> Dict[str, float]:
    if config.slack_step <= 0:
        raise ValueError("slack_step must be > 0.")
    steps = float(slack_max) / float(config.slack_step)
    if steps < 0:
        raise ValueError("Slack max must be >= 0.")
    if config.require_integral:
        require_integral(steps, config.integrality_tol, "slack_steps")

    smax = int(math.ceil(steps - config.integrality_tol))
    if smax <= 0:
        return {}

    bits = int(math.ceil(math.log2(smax + 1)))
    terms: Dict[str, float] = {}
    for _ in range(bits):
        name = f"{prefix}_{len(var_map)}"
        if name in var_map:
            raise ValueError(f"Duplicate slack var name: {name}")
        var_map[name] = len(var_map)
        terms[name] = float(config.slack_step) * float(2 ** (len(terms)))
    return terms


def add_constraint_le(
    *,
    terms: Dict[str, float],
    constant: float,
    rhs: float,
    Q: np.ndarray,
    var_map: Dict[str, int],
    config: PenaltyConfig,
    offset: float,
) -> Tuple[np.ndarray, float]:
    if config.require_integral:
        require_integral(rhs, config.integrality_tol, "rhs")
        require_integral(constant, config.integrality_tol, "constant")
        for coef in terms.values():
            require_integral(coef, config.integrality_tol, "coef")

    min_lhs, _ = lhs_bounds(terms, constant)
    slack_max = float(rhs - min_lhs)
    if slack_max < 0:
        raise ValueError("Infeasible <= constraint (rhs below minimum LHS).")

    slack = slack_terms(var_map=var_map, slack_max=slack_max, config=config, prefix="slack_le")
    Q = ensure_square(Q, len(var_map))
    all_terms = dict(terms)
    all_terms.update(slack)
    offset_add = add_penalty_eq(
        Q=Q,
        var_map=var_map,
        terms=all_terms,
        constant=constant,
        rhs=float(rhs),
        penalty=float(config.penalty_le),
    )
    return Q, float(offset + offset_add)


def add_constraint_ge(
    *,
    terms: Dict[str, float],
    constant: float,
    rhs: float,
    Q: np.ndarray,
    var_map: Dict[str, int],
    config: PenaltyConfig,
    offset: float,
) -> Tuple[np.ndarray, float]:
    if config.require_integral:
        require_integral(rhs, config.integrality_tol, "rhs")
        require_integral(constant, config.integrality_tol, "constant")
        for coef in terms.values():
            require_integral(coef, config.integrality_tol, "coef")

    _, max_lhs = lhs_bounds(terms, constant)
    slack_max = float(max_lhs - rhs)
    if slack_max < 0:
        raise ValueError("Infeasible >= constraint (rhs above maximum LHS).")

    slack = slack_terms(var_map=var_map, slack_max=slack_max, config=config, prefix="slack_ge")
    Q = ensure_square(Q, len(var_map))
    all_terms = dict(terms)
    for k, v in slack.items():
        all_terms[k] = all_terms.get(k, 0.0) - float(v)
    offset_add = add_penalty_eq(
        Q=Q,
        var_map=var_map,
        terms=all_terms,
        constant=constant,
        rhs=float(rhs),
        penalty=float(config.penalty_ge),
    )
    return Q, float(offset + offset_add)
