from __future__ import annotations

"""Generic Pyomo candidate evaluation utilities (no solve).

These helpers treat a `pyomo.environ.ConcreteModel` as the single source of truth:
- You set `.value` on the model variables for a candidate solution
- Then you evaluate objective and constraint satisfaction by calling `pyo.value(...)`

This module is intentionally problem-agnostic. Problem-specific code (e.g. mapping
"assignment" -> variable values) must live in the respective adapter/optimizer.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pyomo.environ as pyo
import numpy as np
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn


@dataclass(frozen=True)
class ConstraintViolation:
    name: str
    body: float
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class ConstraintCheck:
    feasible: bool
    violations: list[ConstraintViolation]

@dataclass(frozen=True)
class CompiledPyomoModel:
    """Cached handles for fast repeated evaluations (no solve)."""

    model: pyo.ConcreteModel
    constraints: list[Any]  # list[pyo.ConstraintData] (typed as Any to avoid Pyomo typing issues)
    objective: Any | None   # pyo.ObjectiveData | None
    fixed_vars: list[Any]   # list[pyo.VarData] where v.fixed
    var_to_constraints: dict[int, list[Any]]  # id(VarData) -> list[ConstraintData]


@dataclass(frozen=True)
class LinearConstraintRepn:
    lower: float | None
    upper: float | None
    constant: float
    var_ids: np.ndarray  # (k,) ids of VarData
    coefs: np.ndarray    # (k,) float coefficients


@dataclass(frozen=True)
class LinearObjectiveRepn:
    constant: float
    var_ids: np.ndarray
    coefs: np.ndarray
    var_coef: dict[int, float]


@dataclass(frozen=True)
class CompiledLinearModel:
    """Compiled linear representation for fast feasibility/objective evaluation.

    - Linear constraints are stored as arrays.
    - Nonlinear constraints (if any) are kept as Pyomo objects for fallback evaluation.
    """

    model: pyo.ConcreteModel
    fixed_vars: list[Any]
    linear_constraints: list[LinearConstraintRepn]
    nonlinear_constraints: list[Any]
    var_to_linear: dict[int, list[tuple[int, float]]]
    var_to_nonlinear: dict[int, list[Any]]
    objective_linear: LinearObjectiveRepn | None
    objective_obj: Any | None


@dataclass(frozen=True)
class CompiledLinearKernel:
    """A Pyomo-free (in the hot loop) kernel keyed by VarData ids.

    This keeps MILP-as-source-of-truth by compiling once from the Pyomo model.
    Inner-loop feasibility checks can then be done without touching VarData objects.
    """

    linear_constraints: list[LinearConstraintRepn]
    nonlinear_constraints: list[Any]
    # var_id -> [(constraint_idx, coef)]
    var_to_linear: dict[int, list[tuple[int, float]]]
    # var_id -> [nonlinear ConstraintData]
    var_to_nonlinear: dict[int, list[Any]]
    # fixed var ids and values (hard constraints)
    fixed_var_value: dict[int, float]
    # objective representation
    objective_constant: float
    objective_var_coef: dict[int, float]


def compile_linear_kernel(*, compiled: CompiledLinearModel) -> CompiledLinearKernel:
    fixed_var_value = {id(v): float(pyo.value(v)) for v in compiled.fixed_vars}
    objective_constant = 0.0
    objective_var_coef: dict[int, float] = {}
    if compiled.objective_linear is not None:
        objective_constant = float(compiled.objective_linear.constant)
        objective_var_coef = dict(compiled.objective_linear.var_coef)
    elif compiled.objective_obj is not None:
        # Fallback to 0 constant; objective will be evaluated via Pyomo if needed.
        objective_constant = 0.0
        objective_var_coef = {}

    return CompiledLinearKernel(
        linear_constraints=compiled.linear_constraints,
        nonlinear_constraints=compiled.nonlinear_constraints,
        var_to_linear=compiled.var_to_linear,
        var_to_nonlinear=compiled.var_to_nonlinear,
        fixed_var_value=fixed_var_value,
        objective_constant=objective_constant,
        objective_var_coef=objective_var_coef,
    )


@dataclass(frozen=True)
class CompiledNumericMILPKernel:
    """Pure-numeric MILP kernel (derived from Pyomo; no Pyomo objects in hot loop).

    Constraints and objective are linear. Nonlinear constraints/objective are detected
    at compile time and reported via `has_nonlinear`.
    """

    n_vars: int
    # var id <-> index
    var_id_to_idx: dict[int, int]
    idx_to_var_id: np.ndarray  # (n_vars,)
    # hard fixed values
    fixed_idx_value: dict[int, float]
    # constraints
    lower: np.ndarray  # (n_cons,) float with -inf for None
    upper: np.ndarray  # (n_cons,) float with +inf for None
    constant: np.ndarray  # (n_cons,)
    # sparse structure by var: idx -> [(cons_idx, coef)]
    var_to_cons: list[list[tuple[int, float]]]
    # objective
    obj_constant: float
    obj_var_coef: dict[int, float]  # var_idx -> coef
    # meta
    has_nonlinear: bool


def compile_numeric_milp_kernel(*, model: pyo.ConcreteModel) -> CompiledNumericMILPKernel:
    """Compile a Pyomo MILP into a numeric kernel.

    - Requires all active constraints to be linear (otherwise has_nonlinear=True).
    - Requires objective to be linear (otherwise has_nonlinear=True).
    """
    # Build variable index (only active vars)
    vars_list = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
    var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}
    idx_to_var_id = np.fromiter((id(v) for v in vars_list), dtype=np.int64, count=len(vars_list))

    fixed_idx_value: dict[int, float] = {}
    for v in vars_list:
        if v.fixed:
            fixed_idx_value[var_id_to_idx[id(v)]] = float(pyo.value(v))

    cons = list(model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
    n_cons = len(cons)
    lower = np.full((n_cons,), -np.inf, dtype=np.float64)
    upper = np.full((n_cons,), np.inf, dtype=np.float64)
    constant = np.zeros((n_cons,), dtype=np.float64)

    var_to_cons: list[list[tuple[int, float]]] = [[] for _ in range(len(vars_list))]

    has_nonlinear = False
    for cidx, con in enumerate(cons):
        if con.lower is not None:
            lower[cidx] = float(pyo.value(con.lower))
        if con.upper is not None:
            upper[cidx] = float(pyo.value(con.upper))

        repn = generate_standard_repn(con.body, compute_values=False)
        if not (repn.is_linear() and not repn.is_quadratic()):
            has_nonlinear = True
            continue

        constant[cidx] = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
        lv = list(repn.linear_vars) if repn.linear_vars is not None else []
        lc = list(repn.linear_coefs) if repn.linear_coefs is not None else []
        for v, coef in zip(lv, lc):
            vid = id(v)
            # Skip vars not in active list (shouldn't happen)
            if vid not in var_id_to_idx:
                continue
            vidx = var_id_to_idx[vid]
            # Fold fixed-variable contribution into the constant term
            fv = fixed_idx_value.get(vidx)
            if fv is not None:
                constant[cidx] += float(coef) * float(fv)
                continue
            var_to_cons[vidx].append((cidx, float(coef)))

    # Objective
    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    obj_constant = 0.0
    obj_var_coef: dict[int, float] = {}
    if len(objs) == 1:
        repn = generate_standard_repn(objs[0].expr, compute_values=False)
        if not (repn.is_linear() and not repn.is_quadratic()):
            has_nonlinear = True
        else:
            obj_constant = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
            lv = list(repn.linear_vars) if repn.linear_vars is not None else []
            lc = list(repn.linear_coefs) if repn.linear_coefs is not None else []
            for v, coef in zip(lv, lc):
                vid = id(v)
                if vid not in var_id_to_idx:
                    continue
                vidx = var_id_to_idx[vid]
                fv = fixed_idx_value.get(vidx)
                if fv is not None:
                    obj_constant += float(coef) * float(fv)
                    continue
                obj_var_coef[vidx] = float(coef)
    else:
        has_nonlinear = True

    return CompiledNumericMILPKernel(
        n_vars=len(vars_list),
        var_id_to_idx=var_id_to_idx,
        idx_to_var_id=idx_to_var_id,
        fixed_idx_value=fixed_idx_value,
        lower=lower,
        upper=upper,
        constant=constant,
        var_to_cons=var_to_cons,
        obj_constant=obj_constant,
        obj_var_coef=obj_var_coef,
        has_nonlinear=has_nonlinear,
    )


def init_lhs_numeric(*, kernel: CompiledNumericMILPKernel, x: np.ndarray) -> np.ndarray:
    """Compute constraint LHS for current x."""
    lhs = kernel.constant.copy()
    for var_idx in range(kernel.n_vars):
        xv = float(x[var_idx])
        if xv == 0.0:
            continue
        for cidx, coef in kernel.var_to_cons[var_idx]:
            lhs[cidx] += coef * xv
    return lhs


def apply_delta_numeric_inplace(*, kernel: CompiledNumericMILPKernel, lhs: np.ndarray, var_idx: int, delta: float) -> None:
    for cidx, coef in kernel.var_to_cons[int(var_idx)]:
        lhs[cidx] += float(coef) * float(delta)


def feasible_all_numeric(*, kernel: CompiledNumericMILPKernel, lhs: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(lhs >= (kernel.lower - tol)) and np.all(lhs <= (kernel.upper + tol)))


def feasible_candidates_numeric_numpy(
    *,
    kernel: CompiledNumericMILPKernel,
    lhs: np.ndarray,
    var_indices: np.ndarray,
    delta: float,
    base_value: float = 0.0,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> np.ndarray:
    """Vectorized feasibility for candidate var toggles (numpy)."""
    var_indices = np.asarray(var_indices, dtype=np.int64)
    n = int(var_indices.size)
    out = np.ones((n,), dtype=bool)

    # fixed mismatch
    for k in range(n):
        vidx = int(var_indices[k])
        fv = kernel.fixed_idx_value.get(vidx)
        if fv is None:
            continue
        if abs((float(base_value) + float(delta)) - float(fv)) > fixed_tol:
            out[k] = False

    # constraint->candidate contributions
    by_cidx: dict[int, tuple[list[int], list[float]]] = {}
    for k in range(n):
        if not out[k]:
            continue
        vidx = int(var_indices[k])
        for cidx, coef in kernel.var_to_cons[vidx]:
            ks, ds = by_cidx.get(cidx, ([], []))
            ks.append(k)
            ds.append(float(coef) * float(delta))
            by_cidx[cidx] = (ks, ds)

    for cidx, (ks, ds) in by_cidx.items():
        base = float(lhs[cidx])
        bodies = base + np.asarray(ds, dtype=np.float64)
        ok = (bodies >= (kernel.lower[cidx] - tol)) & (bodies <= (kernel.upper[cidx] + tol))
        out[np.asarray(ks, dtype=np.int64)] &= ok

    return out


def feasible_candidates_numeric_torch(
    *,
    kernel: CompiledNumericMILPKernel,
    lhs: np.ndarray,
    var_indices: np.ndarray,
    delta: float,
    base_value: float = 0.0,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> "Any":
    import torch

    var_indices = np.asarray(var_indices, dtype=np.int64)
    n = int(var_indices.size)
    out = torch.ones((n,), dtype=torch.bool)

    for k in range(n):
        vidx = int(var_indices[k])
        fv = kernel.fixed_idx_value.get(vidx)
        if fv is None:
            continue
        if abs((float(base_value) + float(delta)) - float(fv)) > fixed_tol:
            out[k] = False

    lhs_t = torch.tensor(lhs, dtype=torch.float64)
    for k in range(n):
        if not bool(out[k].item()):
            continue
        vidx = int(var_indices[k])
        # quick per-candidate check over impacted constraints
        ok = True
        for cidx, coef in kernel.var_to_cons[vidx]:
            body = float(lhs_t[cidx].item() + float(coef) * float(delta))
            if body < float(kernel.lower[cidx] - tol) or body > float(kernel.upper[cidx] + tol):
                ok = False
                break
        out[k] = bool(ok)

    return out


def apply_linear_delta_varid_inplace(*, kernel: CompiledLinearKernel, lhs: np.ndarray, var_id: int, delta: float) -> None:
    for cidx, coef in kernel.var_to_linear.get(int(var_id), []):
        lhs[cidx] += float(coef) * float(delta)


def feasible_candidate_varid(
    *,
    kernel: CompiledLinearKernel,
    lhs: np.ndarray,
    var_id: int,
    delta: float,
    base_value: float = 0.0,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> bool:
    """Feasible check for a single var toggle using only compiled linear data."""
    vid = int(var_id)
    if vid in kernel.fixed_var_value:
        if abs((float(base_value) + float(delta)) - kernel.fixed_var_value[vid]) > fixed_tol:
            return False

    # linear constraints impacted by var
    for cidx, coef in kernel.var_to_linear.get(vid, []):
        c = kernel.linear_constraints[cidx]
        body = float(lhs[cidx] + float(coef) * float(delta))
        if c.lower is not None and body < c.lower - tol:
            return False
        if c.upper is not None and body > c.upper + tol:
            return False

    # nonlinear fallback if any constraint depends on this var
    if kernel.nonlinear_constraints and kernel.var_to_nonlinear.get(vid):
        # We cannot evaluate nonlinear without VarData values, so signal slow path.
        return False

    return True


def feasible_candidates_varids_numpy(
    *,
    kernel: CompiledLinearKernel,
    lhs: np.ndarray,
    var_ids: np.ndarray,
    delta: float,
    base_value: float = 0.0,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> np.ndarray:
    """Vectorized feasibility for many candidate vars (numpy), keyed by var_id."""
    var_ids = np.asarray(var_ids, dtype=np.int64)
    n = int(var_ids.size)
    feasible = np.ones((n,), dtype=bool)

    # fixed mismatch pruning
    if kernel.fixed_var_value:
        for k in range(n):
            vid = int(var_ids[k])
            fv = kernel.fixed_var_value.get(vid)
            if fv is None:
                continue
            if abs((float(base_value) + float(delta)) - float(fv)) > fixed_tol:
                feasible[k] = False

    # constraint -> candidate contributions
    by_cidx: dict[int, tuple[list[int], list[float]]] = {}
    for k in range(n):
        if not feasible[k]:
            continue
        vid = int(var_ids[k])
        for cidx, coef in kernel.var_to_linear.get(vid, []):
            ks, ds = by_cidx.get(cidx, ([], []))
            ks.append(k)
            ds.append(float(coef) * float(delta))
            by_cidx[cidx] = (ks, ds)

    for cidx, (ks, ds) in by_cidx.items():
        c = kernel.linear_constraints[cidx]
        base = float(lhs[cidx])
        bodies = base + np.asarray(ds, dtype=np.float64)
        ok = np.ones((len(ks),), dtype=bool)
        if c.lower is not None:
            ok &= bodies >= (c.lower - tol)
        if c.upper is not None:
            ok &= bodies <= (c.upper + tol)
        feasible[np.asarray(ks, dtype=np.int64)] &= ok

    # nonlinear: mark unknown/slow-path candidates as infeasible for now
    if kernel.nonlinear_constraints:
        for k in range(n):
            if not feasible[k]:
                continue
            if kernel.var_to_nonlinear.get(int(var_ids[k])):
                feasible[k] = False

    return feasible


def feasible_candidates_varids_torch(
    *,
    kernel: CompiledLinearKernel,
    lhs: np.ndarray,
    var_ids: np.ndarray,
    delta: float,
    base_value: float = 0.0,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> "Any":
    """Torch version of feasible_candidates_varids_numpy (still CPU)."""
    import torch

    var_ids = np.asarray(var_ids, dtype=np.int64)
    n = int(var_ids.size)
    feasible = torch.ones((n,), dtype=torch.bool)

    if kernel.fixed_var_value:
        for k in range(n):
            vid = int(var_ids[k])
            fv = kernel.fixed_var_value.get(vid)
            if fv is None:
                continue
            if abs((float(base_value) + float(delta)) - float(fv)) > fixed_tol:
                feasible[k] = False

    by_cidx: dict[int, tuple[list[int], list[float]]] = {}
    for k in range(n):
        if not bool(feasible[k].item()):
            continue
        vid = int(var_ids[k])
        for cidx, coef in kernel.var_to_linear.get(vid, []):
            ks, ds = by_cidx.get(cidx, ([], []))
            ks.append(k)
            ds.append(float(coef) * float(delta))
            by_cidx[cidx] = (ks, ds)

    lhs_t = torch.tensor(lhs, dtype=torch.float64)
    for cidx, (ks, ds) in by_cidx.items():
        c = kernel.linear_constraints[cidx]
        base = lhs_t[cidx]
        bodies = base + torch.tensor(ds, dtype=torch.float64)
        ok = torch.ones((len(ks),), dtype=torch.bool)
        if c.lower is not None:
            ok &= bodies >= (c.lower - tol)
        if c.upper is not None:
            ok &= bodies <= (c.upper + tol)
        idxs = torch.tensor(ks, dtype=torch.long)
        feasible[idxs] &= ok

    if kernel.nonlinear_constraints:
        for k in range(n):
            if not bool(feasible[k].item()):
                continue
            if kernel.var_to_nonlinear.get(int(var_ids[k])):
                feasible[k] = False

    return feasible

def compile_model(*, model: pyo.ConcreteModel) -> CompiledPyomoModel:
    """Precompute constraint/objective/var handles so we don't traverse the model each call."""
    constraints = list(model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    objective = objs[0] if len(objs) == 1 else None
    fixed_vars = [v for v in model.component_data_objects(pyo.Var, active=True, descend_into=True) if v.fixed]
    var_to_constraints: dict[int, list[Any]] = {}
    for con in constraints:
        for v in identify_variables(con.body):
            var_to_constraints.setdefault(id(v), []).append(con)
    return CompiledPyomoModel(
        model=model,
        constraints=constraints,
        objective=objective,
        fixed_vars=fixed_vars,
        var_to_constraints=var_to_constraints,
    )


def compile_linear_model(*, model: pyo.ConcreteModel) -> CompiledLinearModel:
    """Compile constraints/objective to a linear representation when possible.

    Any constraint whose body is not linear becomes part of `nonlinear_constraints` and is
    evaluated via `pyo.value` as a fallback.
    """
    fixed_vars = [v for v in model.component_data_objects(pyo.Var, active=True, descend_into=True) if v.fixed]

    linear_constraints: list[LinearConstraintRepn] = []
    nonlinear_constraints: list[Any] = []
    var_to_linear: dict[int, list[tuple[int, float]]] = {}
    var_to_nonlinear: dict[int, list[Any]] = {}

    for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        repn = generate_standard_repn(con.body, compute_values=False)
        lower = None if con.lower is None else float(pyo.value(con.lower))
        upper = None if con.upper is None else float(pyo.value(con.upper))

        if repn.is_linear() and not repn.is_quadratic():
            # NOTE: Pyomo may move fixed-variable terms into `repn.constant` even when
            # `compute_values=False`, and that constant can still be a Pyomo expression.
            # We evaluate it numerically (fixed vars must have values).
            const = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
            vars_ = list(repn.linear_vars) if repn.linear_vars is not None else []
            coefs_ = list(repn.linear_coefs) if repn.linear_coefs is not None else []
            var_ids = np.fromiter((id(v) for v in vars_), dtype=np.int64, count=len(vars_))
            coefs = np.asarray(coefs_, dtype=np.float64)
            idx = len(linear_constraints)
            linear_constraints.append(
                LinearConstraintRepn(lower=lower, upper=upper, constant=const, var_ids=var_ids, coefs=coefs)
            )
            for vid, coef in zip(var_ids.tolist(), coefs.tolist()):
                var_to_linear.setdefault(int(vid), []).append((idx, float(coef)))
        else:
            nonlinear_constraints.append(con)
            for v in identify_variables(con.body):
                var_to_nonlinear.setdefault(id(v), []).append(con)

    # Objective
    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    objective_obj = objs[0] if len(objs) == 1 else None
    objective_linear: LinearObjectiveRepn | None = None
    if objective_obj is not None:
        repn = generate_standard_repn(objective_obj.expr, compute_values=False)
        if repn.is_linear() and not repn.is_quadratic():
            const = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
            vars_ = list(repn.linear_vars) if repn.linear_vars is not None else []
            coefs_ = list(repn.linear_coefs) if repn.linear_coefs is not None else []
            var_ids = np.fromiter((id(v) for v in vars_), dtype=np.int64, count=len(vars_))
            coefs = np.asarray(coefs_, dtype=np.float64)
            var_coef = {int(vid): float(c) for vid, c in zip(var_ids.tolist(), coefs.tolist())}
            objective_linear = LinearObjectiveRepn(constant=const, var_ids=var_ids, coefs=coefs, var_coef=var_coef)

    return CompiledLinearModel(
        model=model,
        fixed_vars=fixed_vars,
        linear_constraints=linear_constraints,
        nonlinear_constraints=nonlinear_constraints,
        var_to_linear=var_to_linear,
        var_to_nonlinear=var_to_nonlinear,
        objective_linear=objective_linear,
        objective_obj=objective_obj,
    )


def init_linear_lhs(*, compiled: CompiledLinearModel) -> np.ndarray:
    """Compute current LHS values for all compiled linear constraints."""
    lhs = np.empty((len(compiled.linear_constraints),), dtype=np.float64)
    for idx, c in enumerate(compiled.linear_constraints):
        v = c.constant
        # Sum coef * var.value
        # We need VarData objects; retrieve by scanning model vars by id is expensive.
        # Instead, use pyo.value on each var via stored ids by iterating model vars once:
        # Here we do a cheap fallback: compute via constraints' ids map using model component traversal.
        # For step-based RL, variables are typically initialized; we can treat missing ids as 0.
        lhs[idx] = v
    # Populate by iterating vars once and applying to all constraints they appear in
    # (much faster than nested loops).
    id_to_val: dict[int, float] = {}
    for var in compiled.model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if var.value is None:
            continue
        id_to_val[id(var)] = float(var.value)
    for idx, c in enumerate(compiled.linear_constraints):
        if c.var_ids.size == 0:
            continue
        vals = np.fromiter((id_to_val.get(int(vid), 0.0) for vid in c.var_ids.tolist()), dtype=np.float64, count=c.var_ids.size)
        lhs[idx] = c.constant + float(np.dot(c.coefs, vals))
    return lhs


def apply_linear_delta_inplace(*, compiled: CompiledLinearModel, lhs: np.ndarray, var: Any, delta: float) -> None:
    """Update LHS array in-place for setting var.value += delta."""
    for cidx, coef in compiled.var_to_linear.get(id(var), []):
        lhs[cidx] += coef * float(delta)


def feasible_linear_changed_vars(
    *,
    compiled: CompiledLinearModel,
    lhs: np.ndarray,
    changed_vars: Sequence[Any],
    deltas: Sequence[float],
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> bool:
    """Feasibility check using compiled linear constraints + nonlinear fallback.

    Assumes current `lhs` corresponds to the baseline state *before* applying deltas.
    """
    # Fixed mismatch short-circuit for changed vars (using base + delta)
    for v, d in zip(changed_vars, deltas):
        if not getattr(v, "fixed", False):
            continue
        fixed_val = float(pyo.value(v))
        base = 0.0 if v.value is None else float(v.value)
        if abs((base + float(d)) - fixed_val) > fixed_tol:
            return False

    # Accumulate per-constraint delta
    delta_by_cidx: dict[int, float] = {}
    impacted_nl: dict[int, Any] = {}

    for v, d in zip(changed_vars, deltas):
        dv = float(d)
        if dv == 0.0:
            continue
        for cidx, coef in compiled.var_to_linear.get(id(v), []):
            delta_by_cidx[cidx] = delta_by_cidx.get(cidx, 0.0) + coef * dv
        for con in compiled.var_to_nonlinear.get(id(v), []):
            impacted_nl[id(con)] = con

    # Check impacted linear constraints
    for cidx, dsum in delta_by_cidx.items():
        c = compiled.linear_constraints[cidx]
        body = float(lhs[cidx] + dsum)
        if c.lower is not None and body < c.lower - tol:
            return False
        if c.upper is not None and body > c.upper + tol:
            return False

    # Check impacted nonlinear constraints (if any) by temporarily applying deltas to var values.
    if impacted_nl:
        old_vals: list[tuple[Any, Any]] = []
        try:
            for v, d in zip(changed_vars, deltas):
                old_vals.append((v, v.value))
                base = 0.0 if v.value is None else float(v.value)
                v.value = base + float(d)

            for con in impacted_nl.values():
                body = float(pyo.value(con.body))
                lower = None if con.lower is None else float(pyo.value(con.lower))
                upper = None if con.upper is None else float(pyo.value(con.upper))
                if lower is not None and body < lower - tol:
                    return False
                if upper is not None and body > upper + tol:
                    return False
        finally:
            for v, old in old_vals:
                v.value = old

    return True


def objective_value_linear(*, compiled: CompiledLinearModel) -> float:
    """Evaluate objective with linear repn if available."""
    if compiled.objective_linear is None or compiled.objective_obj is None:
        if compiled.objective_obj is None:
            return eval_objective_value(model=compiled.model)
        return float(pyo.value(compiled.objective_obj))

    # Fast: constant + sum coef * var.value (from ids)
    id_to_val: dict[int, float] = {}
    for var in compiled.model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if var.value is None:
            continue
        id_to_val[id(var)] = float(var.value)
    vals = np.fromiter(
        (id_to_val.get(int(vid), 0.0) for vid in compiled.objective_linear.var_ids.tolist()),
        dtype=np.float64,
        count=compiled.objective_linear.var_ids.size,
    )
    return float(compiled.objective_linear.constant + np.dot(compiled.objective_linear.coefs, vals))


def feasible_candidates_linear_numpy(
    *,
    compiled: CompiledLinearModel,
    lhs: np.ndarray,
    candidates: Sequence[Any],
    delta: float,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> np.ndarray:
    """Vectorized linear feasibility check for many candidate vars (numpy).

    Assumes each candidate corresponds to setting var.value += delta (typically delta=+1 for binaries).
    Returns boolean mask of shape (len(candidates),).

    If the model has nonlinear constraints involving any candidate variable, we fall back to
    per-candidate checks for those candidates (still generic).
    """
    n = len(candidates)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    feasible = np.ones((n,), dtype=bool)

    # Fixed mismatch pruning
    for k, v in enumerate(candidates):
        if not getattr(v, "fixed", False):
            continue
        fixed_val = float(pyo.value(v))
        base = 0.0 if v.value is None else float(v.value)
        if abs((base + float(delta)) - fixed_val) > fixed_tol:
            feasible[k] = False

    # Build constraint -> candidate contributions
    by_cidx: dict[int, tuple[list[int], list[float]]] = {}
    for k, v in enumerate(candidates):
        if not feasible[k]:
            continue
        for cidx, coef in compiled.var_to_linear.get(id(v), []):
            ks, ds = by_cidx.get(cidx, ([], []))
            ks.append(k)
            ds.append(float(coef) * float(delta))
            by_cidx[cidx] = (ks, ds)

    for cidx, (ks, ds) in by_cidx.items():
        c = compiled.linear_constraints[cidx]
        base = float(lhs[cidx])
        bodies = base + np.asarray(ds, dtype=np.float64)
        ok = np.ones((len(ks),), dtype=bool)
        if c.lower is not None:
            ok &= bodies >= (c.lower - tol)
        if c.upper is not None:
            ok &= bodies <= (c.upper + tol)
        feasible[np.asarray(ks, dtype=np.int64)] &= ok

    # Nonlinear fallback (rare for MILPs)
    if compiled.nonlinear_constraints:
        for k, v in enumerate(candidates):
            if not feasible[k]:
                continue
            if not compiled.var_to_nonlinear.get(id(v)):
                continue
            # Full generic check (linear+nonlinear) for this candidate
            ok = feasible_linear_changed_vars(
                compiled=compiled,
                lhs=lhs,
                changed_vars=[v],
                deltas=[delta],
                tol=tol,
                fixed_tol=fixed_tol,
            )
            feasible[k] = bool(ok)

    return feasible


def feasible_candidates_linear_torch(
    *,
    compiled: CompiledLinearModel,
    lhs: np.ndarray,
    candidates: Sequence[Any],
    delta: float,
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> "Any":
    """Same as feasible_candidates_linear_numpy, but uses torch ops for the bulk comparisons.

    Note: this still runs on CPU because the underlying Pyomo model lives in Python.
    """
    import torch

    n = len(candidates)
    if n == 0:
        return torch.zeros((0,), dtype=torch.bool)

    feasible = torch.ones((n,), dtype=torch.bool)

    # Fixed mismatch pruning (still scalar python loop)
    for k, v in enumerate(candidates):
        if not getattr(v, "fixed", False):
            continue
        fixed_val = float(pyo.value(v))
        base = 0.0 if v.value is None else float(v.value)
        if abs((base + float(delta)) - fixed_val) > fixed_tol:
            feasible[k] = False

    by_cidx: dict[int, tuple[list[int], list[float]]] = {}
    for k, v in enumerate(candidates):
        if not bool(feasible[k].item()):
            continue
        for cidx, coef in compiled.var_to_linear.get(id(v), []):
            ks, ds = by_cidx.get(cidx, ([], []))
            ks.append(k)
            ds.append(float(coef) * float(delta))
            by_cidx[cidx] = (ks, ds)

    lhs_t = torch.tensor(lhs, dtype=torch.float64)
    for cidx, (ks, ds) in by_cidx.items():
        c = compiled.linear_constraints[cidx]
        base = lhs_t[cidx]
        bodies = base + torch.tensor(ds, dtype=torch.float64)
        ok = torch.ones((len(ks),), dtype=torch.bool)
        if c.lower is not None:
            ok &= bodies >= (c.lower - tol)
        if c.upper is not None:
            ok &= bodies <= (c.upper + tol)
        idxs = torch.tensor(ks, dtype=torch.long)
        feasible[idxs] &= ok

    if compiled.nonlinear_constraints:
        for k, v in enumerate(candidates):
            if not bool(feasible[k].item()):
                continue
            if not compiled.var_to_nonlinear.get(id(v)):
                continue
            ok = feasible_linear_changed_vars(
                compiled=compiled,
                lhs=lhs,
                changed_vars=[v],
                deltas=[delta],
                tol=tol,
                fixed_tol=fixed_tol,
            )
            feasible[k] = bool(ok)

    return feasible


def _fixed_var_violations_vars(*, fixed_vars: Sequence[Any], tol: float = 1e-9) -> list[ConstraintViolation]:
    """Treat fixed-variable mismatches as feasibility violations.

    Pyomo's constraint evaluation does not automatically enforce `var.fix(v)` if the user
    later sets `var.value` to a different number. For "candidate evaluation", we want
    fixed vars to behave like hard constraints.
    """
    out: list[ConstraintViolation] = []
    for v in fixed_vars:
        fixed_val = float(pyo.value(v))
        # If value is None, treat as mismatch (candidate incomplete)
        if v.value is None:
            out.append(ConstraintViolation(name=v.name + " (fixed)", body=float("nan"), lower=fixed_val, upper=fixed_val))
            continue
        val = float(v.value)
        if abs(val - fixed_val) > tol:
            out.append(ConstraintViolation(name=v.name + " (fixed)", body=val, lower=fixed_val, upper=fixed_val))
    return out


def eval_objective_value(*, model: pyo.ConcreteModel) -> float:
    """Evaluate the (single) active objective value at current variable values."""
    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if len(objs) != 1:
        raise ValueError(f"Expected exactly 1 active objective, found {len(objs)}")
    return float(pyo.value(objs[0]))


def eval_objective_value_compiled(*, compiled: CompiledPyomoModel) -> float:
    """Evaluate objective using cached objective handle."""
    if compiled.objective is None:
        # fall back (either 0 or multiple objectives)
        return eval_objective_value(model=compiled.model)
    return float(pyo.value(compiled.objective))


def check_constraints(*, model: pyo.ConcreteModel, tol: float = 1e-6) -> ConstraintCheck:
    """Check all active constraints in the model at current variable values.

    This works generically for any model structure (indexed constraints, ConstraintList,
    nested Blocks, etc.) by iterating over `ConstraintData` objects.
    """
    violations: list[ConstraintViolation] = []

    # Fixed variables are effectively hard constraints for candidate evaluation
    fixed_vars = [v for v in model.component_data_objects(pyo.Var, active=True, descend_into=True) if v.fixed]
    violations.extend(_fixed_var_violations_vars(fixed_vars=fixed_vars))

    for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        body = float(pyo.value(con.body))
        lower = None if con.lower is None else float(pyo.value(con.lower))
        upper = None if con.upper is None else float(pyo.value(con.upper))

        if lower is not None and body < lower - tol:
            violations.append(ConstraintViolation(name=con.name, body=body, lower=lower, upper=upper))
        if upper is not None and body > upper + tol:
            violations.append(ConstraintViolation(name=con.name, body=body, lower=lower, upper=upper))

    return ConstraintCheck(feasible=(len(violations) == 0), violations=violations)


def check_constraints_compiled(*, compiled: CompiledPyomoModel, tol: float = 1e-6) -> ConstraintCheck:
    """Same as check_constraints(), but avoids re-traversing the model."""
    violations: list[ConstraintViolation] = []
    violations.extend(_fixed_var_violations_vars(fixed_vars=compiled.fixed_vars))

    for con in compiled.constraints:
        body = float(pyo.value(con.body))
        lower = None if con.lower is None else float(pyo.value(con.lower))
        upper = None if con.upper is None else float(pyo.value(con.upper))

        if lower is not None and body < lower - tol:
            violations.append(ConstraintViolation(name=con.name, body=body, lower=lower, upper=upper))
        if upper is not None and body > upper + tol:
            violations.append(ConstraintViolation(name=con.name, body=body, lower=lower, upper=upper))

    return ConstraintCheck(feasible=(len(violations) == 0), violations=violations)


@dataclass(frozen=True)
class CandidateScore:
    feasible: bool
    objective: float | None
    violations: list[ConstraintViolation]


def score_candidate(*, model: pyo.ConcreteModel, tol: float = 1e-6) -> CandidateScore:
    """Compute (feasible?, objective) for the current variable values."""
    chk = check_constraints(model=model, tol=tol)
    if not chk.feasible:
        return CandidateScore(feasible=False, objective=None, violations=chk.violations)
    return CandidateScore(feasible=True, objective=eval_objective_value(model=model), violations=[])


def score_candidate_compiled(*, compiled: CompiledPyomoModel, tol: float = 1e-6) -> CandidateScore:
    chk = check_constraints_compiled(compiled=compiled, tol=tol)
    if not chk.feasible:
        return CandidateScore(feasible=False, objective=None, violations=chk.violations)
    return CandidateScore(feasible=True, objective=eval_objective_value_compiled(compiled=compiled), violations=[])


def feasible_candidate(*, model: pyo.ConcreteModel, tol: float = 1e-6) -> bool:
    """Fast boolean feasibility check for current variable values."""
    return check_constraints(model=model, tol=tol).feasible


def feasible_candidate_compiled(*, compiled: CompiledPyomoModel, tol: float = 1e-6) -> bool:
    return check_constraints_compiled(compiled=compiled, tol=tol).feasible


def feasible_candidate_compiled_changed_vars(
    *,
    compiled: CompiledPyomoModel,
    changed_vars: Sequence[Any],
    tol: float = 1e-6,
    fixed_tol: float = 1e-9,
) -> bool:
    """Feasibility check that only re-evaluates constraints impacted by changed vars.

    This is generic and assumes the caller has *already* updated `.value` on the changed vars
    (and potentially other vars). We validate:
    - fixed var mismatches for the changed vars themselves (fast short-circuit)
    - all constraints that reference any of the changed vars
    """
    # Fixed mismatch short-circuit for changed vars
    for v in changed_vars:
        if not getattr(v, "fixed", False):
            continue
        fixed_val = float(pyo.value(v))
        if v.value is None:
            return False
        if abs(float(v.value) - fixed_val) > fixed_tol:
            return False

    # Union of impacted constraints
    impacted: dict[int, Any] = {}
    for v in changed_vars:
        for con in compiled.var_to_constraints.get(id(v), []):
            impacted[id(con)] = con

    for con in impacted.values():
        body = float(pyo.value(con.body))
        lower = None if con.lower is None else float(pyo.value(con.lower))
        upper = None if con.upper is None else float(pyo.value(con.upper))

        if lower is not None and body < lower - tol:
            return False
        if upper is not None and body > upper + tol:
            return False

    return True


# Optional helper: generic value assignment (only used if caller provides a mapping)
def set_values(
    *,
    model: pyo.ConcreteModel,
    items: Iterable[tuple[Any, float]],
) -> None:
    """Set `.value` for a sequence of (VarData, value) pairs."""
    for var, val in items:
        var.value = float(val)

