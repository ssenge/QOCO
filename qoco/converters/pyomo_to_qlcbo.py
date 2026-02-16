from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from qoco.core.converter import Converter
from qoco.core.qlcbo import QLCBO


@dataclass(frozen=True)
class PyomoToQLCBOConfig:
    """Conversion options for Pyomo -> QLCBO."""

    # If True, split ranged constraints (lower <= body <= upper) into two constraints.
    split_ranged: bool = True

    # How to handle inequalities:
    # - "slack": introduce binary slack bits to convert <= / >= into equalities
    # - "reject": reject any inequality constraints
    ineq_mode: str = "slack"  # "slack" | "reject"

    # Slack encoding:
    slack_prefix: str = "slack"
    slack_max_bits: int = 32
    int_tol: float = 1e-9


@dataclass
class PyomoToQLCBOConverter(Converter[pyo.ConcreteModel, QLCBO]):
    """Convert a binary Pyomo MIQP (linear constraints, quadratic objective) to QLCBO.

    Supported:
    - Binary decision variables (Pyomo domain Binary). Fixed vars are folded into constants.
    - Objective: linear/quadratic in binary vars (minimize).
    - Constraints: linear equalities, and (optionally) inequalities via binary slack bits.

    Output:
    - QLCBO(objective=Q, constraints=A|rhs, var_names=[...])
    """

    config: PyomoToQLCBOConfig | None = None

    def convert(self, model: pyo.ConcreteModel) -> QLCBO:
        cfg = self.config or PyomoToQLCBOConfig()

        obj = _get_active_objective(model)
        if obj.sense != pyo.minimize:
            raise ValueError("PyomoToQLCBOConverter requires a minimization objective.")

        # 1) variables
        base_vars = _collect_binary_vars(model)
        var_names: list[str] = [v.name for v in base_vars]
        var_map: Dict[str, int] = {name: i for i, name in enumerate(var_names)}

        # 3) constraints (may add slack vars, so do this before objective matrix sizing)
        rows: list[np.ndarray] = []
        for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            if con.body is None:
                continue

            bounds = _constraint_bounds(con)
            if bounds.kind == "free":
                raise ValueError(f"Constraint without bounds: {con.name}")

            # Split ranged constraints if requested.
            if bounds.kind == "ranged":
                if not bool(cfg.split_ranged):
                    raise ValueError(f"Ranged constraint requires split_ranged=True: {con.name}")
                _add_constraint_row(
                    rows=rows,
                    con=con,
                    bound_kind="le",
                    bound_value=float(bounds.upper),
                    var_names=var_names,
                    var_map=var_map,
                    cfg=cfg,
                )
                _add_constraint_row(
                    rows=rows,
                    con=con,
                    bound_kind="ge",
                    bound_value=float(bounds.lower),
                    var_names=var_names,
                    var_map=var_map,
                    cfg=cfg,
                )
                continue

            if bounds.kind == "eq":
                _add_constraint_row(
                    rows=rows,
                    con=con,
                    bound_kind="eq",
                    bound_value=float(bounds.lower),
                    var_names=var_names,
                    var_map=var_map,
                    cfg=cfg,
                )
                continue

            if bounds.kind == "le":
                _add_constraint_row(
                    rows=rows,
                    con=con,
                    bound_kind="le",
                    bound_value=float(bounds.upper),
                    var_names=var_names,
                    var_map=var_map,
                    cfg=cfg,
                )
                continue

            if bounds.kind == "ge":
                _add_constraint_row(
                    rows=rows,
                    con=con,
                    bound_kind="ge",
                    bound_value=float(bounds.lower),
                    var_names=var_names,
                    var_map=var_map,
                    cfg=cfg,
                )
                continue

            raise ValueError(f"Unsupported constraint bound kind: {bounds.kind} ({con.name})")

        n = int(len(var_names))

        # 2) objective -> Q
        Q = np.zeros((n, n), dtype=float)
        repn = generate_standard_repn(obj.expr, quadratic=True, compute_values=True)
        if repn.nonlinear_expr is not None:
            raise ValueError("PyomoToQLCBOConverter supports linear/quadratic objectives only.")

        # Linear terms
        for v, c in zip(list(repn.linear_vars or []), list(repn.linear_coefs or [])):
            if v.is_fixed():
                continue
            _assert_binary(v)
            i = var_map.get(v.name, None)
            if i is None:
                raise ValueError(f"Objective references unknown variable: {v.name}")
            Q[int(i), int(i)] += float(c)

        # Quadratic terms
        for (v1, v2), c in zip(list(repn.quadratic_vars or []), list(repn.quadratic_coefs or [])):
            if v1.is_fixed() or v2.is_fixed():
                continue
            _assert_binary(v1)
            _assert_binary(v2)
            i = var_map.get(v1.name, None)
            j = var_map.get(v2.name, None)
            if i is None or j is None:
                raise ValueError(f"Objective references unknown variable(s): {v1.name}, {v2.name}")
            i = int(i)
            j = int(j)
            coef = float(c)
            if i == j:
                # x_i^2 == x_i for binary vars
                Q[i, i] += coef
            else:
                Q[i, j] += 0.5 * coef
                Q[j, i] += 0.5 * coef

        # 4) output format: constraints matrix A|rhs (rhs last column)
        if rows:
            C = np.vstack(rows).astype(float, copy=False)
        else:
            C = np.zeros((0, n + 1), dtype=float)

        return QLCBO(objective=Q, constraints=C, var_names=list(var_names))


# -----------------------------------------------------------------------------
# Validation utilities (kept in this module by request)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QLCBOValidationReport:
    ok: bool
    max_obj_abs_err: float
    num_samples: int
    num_constraints: int
    # Constraint feasibility is not evaluated here (random assignments are typically infeasible).
    # This field is reserved for internal conversion errors (e.g., slack vars leaked into objective).
    num_violations: int
    messages: Sequence[str]


def validate_pyomo_vs_qlcbo(
    *,
    model: pyo.ConcreteModel,
    qlcbo: QLCBO,
    num_samples: int = 50,
    seed: int = 0,
    tol: float = 1e-6,
) -> QLCBOValidationReport:
    """Randomized sanity-check for the objective mapping (and basic structural checks).

    Notes:
    - Random assignments will almost always violate constraints for nontrivial models, so this
      validator does NOT attempt to assess feasibility equivalence.
    - It focuses on verifying that the quadratic objective was mapped correctly into Q, and that
      any introduced slack variables do not leak into the objective.
    """

    rng = np.random.default_rng(int(seed))
    obj = _get_active_objective(model)

    repn_obj = generate_standard_repn(obj.expr, quadratic=True, compute_values=True)
    obj_const = float(repn_obj.constant) if repn_obj.constant is not None else 0.0

    # Map Pyomo binary vars by name -> VarData
    py_vars = {v.name: v for v in _collect_binary_vars(model)}

    names = list(qlcbo.var_names) if qlcbo.var_names is not None else [f"x{i}" for i in range(int(qlcbo.n_vars))]
    n = int(len(names))
    name_to_idx = {str(name): int(i) for i, name in enumerate(names)}

    max_obj_err = 0.0
    violations = 0
    messages: list[str] = []

    # Structural check: slack vars must not appear in the objective.
    # (They are an encoding artifact for inequalities and should only appear in constraints.)
    Q = np.asarray(qlcbo.objective, dtype=float)
    slack_prefix = f"{PyomoToQLCBOConfig().slack_prefix}_"
    slack_idx = [i for i, nm in enumerate(names) if str(nm).startswith(slack_prefix)]
    if slack_idx:
        leaked = False
        for i in slack_idx:
            if np.any(np.abs(Q[int(i), :]) > float(tol)) or np.any(np.abs(Q[:, int(i)]) > float(tol)):
                leaked = True
                break
        if leaked:
            violations += 1
            messages.append("Slack variables have nonzero coefficients in the objective Q.")

    for _ in range(int(num_samples)):
        # Build x vector (including slack bits; filled later)
        x = np.zeros((n,), dtype=int)

        # Sample original binary vars and assign into Pyomo model + QLCBO vector.
        for nm, v in py_vars.items():
            val = int(rng.integers(0, 2))
            v.set_value(val)
            if nm in name_to_idx:
                x[name_to_idx[nm]] = val

        # Objective compare (ignore constant offset)
        py_obj = float(pyo.value(obj.expr)) - obj_const
        q_obj = float(x @ (Q @ x))
        max_obj_err = max(max_obj_err, abs(py_obj - q_obj))

    ok = (violations == 0) and (max_obj_err <= float(tol))
    return QLCBOValidationReport(
        ok=bool(ok),
        max_obj_abs_err=float(max_obj_err),
        num_samples=int(num_samples),
        num_constraints=int(qlcbo.n_cons),
        num_violations=int(violations),
        messages=tuple(messages[:50]),
    )


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _get_active_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    objs = [o for o in model.component_data_objects(pyo.Objective, active=True, descend_into=True)]
    if not objs:
        raise ValueError("No active objective found.")
    if len(objs) > 1:
        raise ValueError("Multiple active objectives found.")
    return objs[0]


def _assert_binary(v: pyo.Var) -> None:
    if not (hasattr(v, "is_binary") and v.is_binary()):
        raise ValueError(f"Non-binary variable not supported: {getattr(v, 'name', v)}")


def _collect_binary_vars(model: pyo.ConcreteModel) -> list[pyo.Var]:
    out: list[pyo.Var] = []
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if var.is_fixed():
            continue
        if not var.is_binary():
            continue
        out.append(var)
    # stable order by name
    out.sort(key=lambda v: str(v.name))
    return out


@dataclass(frozen=True)
class _Bounds:
    kind: str  # "eq" | "le" | "ge" | "ranged" | "free"
    lower: float | None = None
    upper: float | None = None


def _constraint_bounds(con: pyo.Constraint) -> _Bounds:
    lower = con.lower
    upper = con.upper
    lo = float(pyo.value(lower)) if lower is not None else None
    up = float(pyo.value(upper)) if upper is not None else None
    if lo is None and up is None:
        return _Bounds(kind="free")
    if lo is not None and up is not None:
        if abs(float(lo) - float(up)) <= 0.0:
            return _Bounds(kind="eq", lower=float(lo), upper=float(up))
        return _Bounds(kind="ranged", lower=float(lo), upper=float(up))
    if up is not None:
        return _Bounds(kind="le", upper=float(up))
    return _Bounds(kind="ge", lower=float(lo))  # type: ignore[arg-type]


def _add_constraint_row(
    *,
    rows: list[np.ndarray],
    con: pyo.Constraint,
    bound_kind: str,  # "eq" | "le" | "ge"
    bound_value: float,
    var_names: list[str],
    var_map: Dict[str, int],
    cfg: PyomoToQLCBOConfig,
) -> None:
    repn = generate_standard_repn(con.body, compute_values=True)
    if not repn.is_linear() or repn.is_quadratic() or repn.nonlinear_expr is not None:
        raise ValueError(f"Nonlinear constraint not supported: {con.name}")

    linear_vars = list(repn.linear_vars or [])
    linear_coefs = list(repn.linear_coefs or [])
    constant = float(repn.constant) if repn.constant is not None else 0.0

    # Fold fixed vars into constant; verify all remaining are binary.
    terms: dict[int, float] = {}
    for v, c in zip(linear_vars, linear_coefs):
        coef = float(c)
        if v.is_fixed():
            constant += coef * float(pyo.value(v))
            continue
        _assert_binary(v)
        idx = var_map.get(v.name, None)
        if idx is None:
            raise ValueError(f"Constraint references unknown variable: {v.name} ({con.name})")
        terms[int(idx)] = terms.get(int(idx), 0.0) + coef

    # Prepare row (may be extended if we introduce slack vars)
    row_idx = int(len(rows))

    if bound_kind == "eq":
        # sum a_i x_i + constant == bound_value
        rhs = constant - float(bound_value)
        row = np.zeros((len(var_names) + 1,), dtype=float)
        for i, a in terms.items():
            row[int(i)] = float(a)
        row[-1] = float(rhs)
        rows.append(row)
        return

    if cfg.ineq_mode != "slack":
        raise ValueError(f"Inequality constraint not supported (ineq_mode={cfg.ineq_mode}): {con.name}")

    # Slack encoding for inequalities.
    if bound_kind == "le":
        # sum a_i x_i + constant + slack == ub
        ub = float(bound_value)
        # slack = ub - (constant + sum a_i x_i)
        slack_max = ub - _min_linear_value(terms, constant)
        slack_max_i = _as_int(slack_max, tol=float(cfg.int_tol))
        if slack_max_i < 0:
            slack_max_i = 0
        slack_cols = _ensure_slack_bits(
            slack_max=slack_max_i,
            prefix=str(cfg.slack_prefix),
            kind="le",
            row=row_idx,
            var_names=var_names,
            var_map=var_map,
            max_bits=int(cfg.slack_max_bits),
        )
        _pad_rows_in_place(rows, n_vars=int(len(var_names)))
        rhs = constant - ub
        row = np.zeros((len(var_names) + 1,), dtype=float)
        for i, a in terms.items():
            row[int(i)] = float(a)
        for col, weight in slack_cols:
            row[int(col)] += float(weight)
        row[-1] = float(rhs)
        rows.append(row)
        return

    if bound_kind == "ge":
        # sum a_i x_i + constant - slack == lb
        lb = float(bound_value)
        slack_max = _max_linear_value(terms, constant) - lb
        slack_max_i = _as_int(slack_max, tol=float(cfg.int_tol))
        if slack_max_i < 0:
            slack_max_i = 0
        slack_cols = _ensure_slack_bits(
            slack_max=slack_max_i,
            prefix=str(cfg.slack_prefix),
            kind="ge",
            row=row_idx,
            var_names=var_names,
            var_map=var_map,
            max_bits=int(cfg.slack_max_bits),
        )
        _pad_rows_in_place(rows, n_vars=int(len(var_names)))
        rhs = constant - lb
        row = np.zeros((len(var_names) + 1,), dtype=float)
        for i, a in terms.items():
            row[int(i)] = float(a)
        for col, weight in slack_cols:
            row[int(col)] -= float(weight)
        row[-1] = float(rhs)
        rows.append(row)
        return

    raise ValueError(f"Unsupported bound_kind={bound_kind} ({con.name})")


def _ensure_slack_bits(
    *,
    slack_max: int,
    prefix: str,
    kind: str,  # "le" | "ge"
    row: int,
    var_names: list[str],
    var_map: Dict[str, int],
    max_bits: int,
) -> list[tuple[int, int]]:
    """Add slack bit variables to var_names/var_map and return their (col, weight)."""
    if int(slack_max) <= 0:
        return []
    bits = int(ceil(log2(int(slack_max) + 1)))
    if bits <= 0:
        return []
    if bits > int(max_bits):
        raise ValueError(f"Slack requires {bits} bits > slack_max_bits={max_bits} (row={row}).")

    cols: list[tuple[int, int]] = []
    for b in range(bits):
        name = f"{prefix}_{kind}_{int(row)}_b{int(b)}"
        if name not in var_map:
            var_map[name] = int(len(var_names))
            var_names.append(name)
        col = int(var_map[name])
        cols.append((col, 1 << int(b)))
    return cols


def _pad_rows_in_place(rows: list[np.ndarray], *, n_vars: int) -> None:
    """Pad existing constraint rows to match the current variable count.

    Each row stores [A..., rhs]. When variables are added after previous rows were built
    (e.g., slack bits), those rows must be padded with zeros *before* rhs.
    """
    target_len = int(n_vars) + 1
    for i, r in enumerate(list(rows)):
        if int(r.shape[0]) == target_len:
            continue
        if int(r.shape[0]) > target_len:
            raise ValueError("Internal error: constraint row longer than current n_vars.")
        rhs = float(r[-1])
        a = np.asarray(r[:-1], dtype=float)
        pad = np.zeros((target_len - 1 - a.shape[0],), dtype=float)
        rows[i] = np.concatenate([a, pad, np.array([rhs], dtype=float)])


def _min_linear_value(terms: dict[int, float], constant: float) -> float:
    # x_i in {0,1} -> min contribution is min(0, a_i)
    return float(constant) + float(sum(min(0.0, float(a)) for a in terms.values()))


def _max_linear_value(terms: dict[int, float], constant: float) -> float:
    return float(constant) + float(sum(max(0.0, float(a)) for a in terms.values()))


def _as_int(x: float, *, tol: float) -> int:
    xi = int(round(float(x)))
    if abs(float(x) - float(xi)) > float(tol):
        raise ValueError(f"Expected integer value within tol={tol}, got {x}.")
    return int(xi)

