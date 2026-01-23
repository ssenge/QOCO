from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO, QuboMetadata, QuboVarEncoding
from qoco.pyomo_utils.binary_expansion import BinaryExpansionConfig, ExpandedVar, expand_var
from qoco.pyomo_utils.qubo_penalties import (
    PenaltyConfig,
    add_constraint_ge,
    add_constraint_le,
    auto_penalty_from_linear_objective,
    default_penalty_config,
    ensure_square,
)


@dataclass(frozen=True)
class PyomoToQuboConfig:
    penalties: PenaltyConfig | None = None
    expansion: BinaryExpansionConfig | None = None
    split_ranged: bool = True


@dataclass
class PyomoToQuboConverter(Converter[pyo.ConcreteModel, QUBO]):
    config: PyomoToQuboConfig | None = None

    def convert(self, model: pyo.ConcreteModel) -> QUBO:
        config = self._resolve_config(model)
        obj = _get_active_objective(model)
        if obj.sense != pyo.minimize:
            raise ValueError("PyomoToQuboConverter requires a minimization objective.")

        Q = np.zeros((0, 0), dtype=np.float64)
        var_map: Dict[str, int] = {}
        offset = 0.0
        expanded_cache: Dict[str, ExpandedVar] = {}
        bound_terms: list[tuple[Dict[str, float], float, float]] = []
        bound_seen: set[str] = set()

        # Objective (linear only)
        repn = generate_standard_repn(obj.expr, compute_values=True)
        if not repn.is_linear() or repn.is_quadratic():
            raise ValueError("PyomoToQuboConverter supports linear objectives only.")

        terms, const = _collect_linear_terms(
            var_map=var_map,
            Q=Q,
            linear_vars=list(repn.linear_vars or []),
            linear_coefs=list(repn.linear_coefs or []),
            constant=float(repn.constant) if repn.constant is not None else 0.0,
            expansion=config.expansion,
            expanded_cache=expanded_cache,
            bound_terms=bound_terms,
            bound_seen=bound_seen,
        )
        offset += const
        Q = _add_linear_terms(Q=Q, var_map=var_map, terms=terms)
        offset = float(offset)

        # Constraints (linear only)
        for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            if con.body is None:
                continue
            repn = generate_standard_repn(con.body, compute_values=True)
            if not repn.is_linear() or repn.is_quadratic():
                raise ValueError(f"Nonlinear constraint not supported: {con.name}")

            terms, constant = _collect_linear_terms(
                var_map=var_map,
                Q=Q,
                linear_vars=list(repn.linear_vars or []),
                linear_coefs=list(repn.linear_coefs or []),
                constant=float(repn.constant) if repn.constant is not None else 0.0,
                expansion=config.expansion,
                expanded_cache=expanded_cache,
                bound_terms=bound_terms,
                bound_seen=bound_seen,
            )

            lower = con.lower
            upper = con.upper

            if lower is not None and upper is not None:
                if not config.split_ranged:
                    raise ValueError(f"Ranged constraint requires split_ranged=True: {con.name}")
                Q, offset = add_constraint_le(
                    terms=terms,
                    constant=constant,
                    rhs=float(upper),
                    Q=Q,
                    var_map=var_map,
                    offset=offset,
                    config=config.penalties,
                )
                Q, offset = add_constraint_ge(
                    terms=terms,
                    constant=constant,
                    rhs=float(lower),
                    Q=Q,
                    var_map=var_map,
                    offset=offset,
                    config=config.penalties,
                )
                continue

            if upper is not None:
                Q, offset = add_constraint_le(
                    terms=terms,
                    constant=constant,
                    rhs=float(upper),
                    Q=Q,
                    var_map=var_map,
                    offset=offset,
                    config=config.penalties,
                )
            elif lower is not None:
                Q, offset = add_constraint_ge(
                    terms=terms,
                    constant=constant,
                    rhs=float(lower),
                    Q=Q,
                    var_map=var_map,
                    offset=offset,
                    config=config.penalties,
                )
            else:
                raise ValueError(f"Constraint without bounds: {con.name}")

            Q = ensure_square(Q, len(var_map))

        # Add bounds for expanded variables (upper bound only; lower bound via offset)
        for terms, constant, rhs in bound_terms:
            Q, offset = add_constraint_le(
                terms=terms,
                constant=constant,
                rhs=rhs,
                Q=Q,
                var_map=var_map,
                offset=offset,
                config=config.penalties,
            )

        encodings = {
            name: QuboVarEncoding(
                offset=float(expanded.offset),
                terms=dict(expanded.terms),
                upper_bound=float(expanded.upper_bound),
            )
            for name, expanded in expanded_cache.items()
        }
        metadata = QuboMetadata(encodings=encodings)
        return QUBO(Q=Q, offset=float(offset), var_map=dict(var_map), metadata=metadata)

    def _resolve_config(self, model: pyo.ConcreteModel) -> PyomoToQuboConfig:
        if self.config is None:
            config = PyomoToQuboConfig()
        else:
            config = self.config

        expansion = config.expansion or BinaryExpansionConfig(
            integer_step=1.0,
            continuous_step=1.0,
            max_bits=32,
            tol=1e-9,
        )

        penalties = config.penalties
        if penalties is None:
            repn = generate_standard_repn(_get_active_objective(model).expr, compute_values=True)
            linear_vars = list(repn.linear_vars or [])
            linear_coefs = list(repn.linear_coefs or [])
            terms = _linear_terms_for_penalty(linear_vars, linear_coefs)
            has_non_integral = _constraints_have_non_integral_coeffs(model)
            penalty = 1e5 if has_non_integral else auto_penalty_from_linear_objective(terms)
            penalties = default_penalty_config(penalty)

        return PyomoToQuboConfig(
            penalties=penalties,
            expansion=expansion,
            split_ranged=config.split_ranged,
        )


def _get_active_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    objs = [o for o in model.component_data_objects(pyo.Objective, active=True, descend_into=True)]
    if not objs:
        raise ValueError("No active objective found.")
    if len(objs) > 1:
        raise ValueError("Multiple active objectives found.")
    return objs[0]


def _ensure_var(var_map: Dict[str, int], Q: np.ndarray, name: str) -> None:
    if name in var_map:
        return
    var_map[name] = len(var_map)


def _collect_linear_terms(
    *,
    var_map: Dict[str, int],
    Q: np.ndarray,
    linear_vars: Iterable[pyo.Var],
    linear_coefs: Iterable[float],
    constant: float,
    expansion: BinaryExpansionConfig,
    expanded_cache: Dict[str, ExpandedVar],
    bound_terms: list[tuple[Dict[str, float], float, float]],
    bound_seen: set[str],
) -> Tuple[Dict[str, float], float]:
    terms: Dict[str, float] = {}
    const = float(constant)

    for v, c in zip(linear_vars, linear_coefs):
        coef = float(c)
        if v.is_fixed():
            const += coef * float(pyo.value(v))
            continue

        name = v.name
        expanded = expand_var(v, expansion, expanded_cache)
        const += coef * float(expanded.offset)

        for bit_name, bit_coef in expanded.terms.items():
            _ensure_var(var_map, Q, bit_name)
            terms[bit_name] = terms.get(bit_name, 0.0) + coef * float(bit_coef)

        if expanded.terms and name not in bound_seen:
            max_value = expanded.offset + sum(expanded.terms.values())
            if max_value > expanded.upper_bound + expansion.tol:
                bound_terms.append((dict(expanded.terms), float(expanded.offset), float(expanded.upper_bound)))
            bound_seen.add(name)

    return terms, const


def _linear_terms_for_penalty(
    linear_vars: Iterable[pyo.Var],
    linear_coefs: Iterable[float],
) -> Dict[str, float]:
    terms: Dict[str, float] = {}
    for v, c in zip(linear_vars, linear_coefs):
        coef = float(c)
        if v.is_fixed():
            continue
        if not v.is_binary():
            continue
        terms[v.name] = terms.get(v.name, 0.0) + coef
    return terms


def _constraints_have_non_integral_coeffs(model: pyo.ConcreteModel) -> bool:
    for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        if con.body is None:
            continue
        repn = generate_standard_repn(con.body, compute_values=True)
        for coef in list(repn.linear_coefs or []):
            if isinstance(coef, float) and not float(coef).is_integer():
                return True
        if con.lower is not None and isinstance(con.lower, float) and not float(con.lower).is_integer():
            return True
        if con.upper is not None and isinstance(con.upper, float) and not float(con.upper).is_integer():
            return True
    return False


def _add_linear_terms(
    *,
    Q: np.ndarray,
    var_map: Dict[str, int],
    terms: Dict[str, float],
) -> np.ndarray:
    Q = ensure_square(Q, len(var_map))
    for name, coef in terms.items():
        idx = var_map[name]
        Q[idx, idx] += float(coef)
    return Q

