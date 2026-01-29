from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn


@dataclass(frozen=True)
class BendersCheck:
    ok: bool
    reasons: List[str]
    n_vars: int
    n_master_vars: int
    n_sub_vars: int
    n_constraints: int


@dataclass(frozen=True)
class VarKey:
    comp: str
    idx: object | None


@dataclass(frozen=True)
class CutRow:
    con_name: str
    rhs: float
    x_coef: Dict[VarKey, float]


@dataclass(frozen=True)
class BendersMapping:
    master_vars: List[VarKey]
    sub_constraint_names: List[str]
    cut_rows: List[CutRow]
    theta_name: str
    cuts_name: str


def _active_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if len(objs) != 1:
        raise ValueError(f"Expected exactly 1 active objective, found {len(objs)}")
    return objs[0]


def check_classic_benders(model: pyo.ConcreteModel) -> BendersCheck:
    reasons: List[str] = []

    vars_ = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
    cons = list(model.component_data_objects(pyo.Constraint, active=True, descend_into=True))

    master = [v for v in vars_ if not bool(v.is_continuous())]
    sub = [v for v in vars_ if bool(v.is_continuous())]

    try:
        obj = _active_objective(model)
        repn = generate_standard_repn(obj.expr, compute_values=False)
        if not (repn.is_linear() and not repn.is_quadratic()):
            reasons.append("objective is not linear")
    except Exception as e:
        reasons.append(f"objective check failed: {type(e).__name__}")

    for con in cons:
        repn = generate_standard_repn(con.body, compute_values=False)
        if not (repn.is_linear() and not repn.is_quadratic()):
            reasons.append(f"nonlinear constraint: {con.name}")
            break

    ok = len(reasons) == 0
    return BendersCheck(
        ok=ok,
        reasons=reasons,
        n_vars=len(vars_),
        n_master_vars=len(master),
        n_sub_vars=len(sub),
        n_constraints=len(cons),
    )


def classic_benders(model: pyo.ConcreteModel) -> Tuple[pyo.ConcreteModel, pyo.ConcreteModel, BendersMapping]:
    chk = check_classic_benders(model)
    if not chk.ok:
        raise ValueError("Model not eligible for classic Benders:\n- " + "\n- ".join(chk.reasons))

    master = model.clone()
    sub = model.clone()

    master_vars = [v for v in master.component_data_objects(pyo.Var, active=True, descend_into=True) if not v.is_continuous()]
    master_var_keys = [VarKey(comp=v.parent_component().name, idx=v.index()) for v in master_vars]
    master_key_set = set(master_var_keys)

    sub_vars = [v for v in sub.component_data_objects(pyo.Var, active=True, descend_into=True) if v.is_continuous()]
    sub_key_set = {VarKey(comp=v.parent_component().name, idx=v.index()) for v in sub_vars}

    obj_m = _active_objective(master)
    repn_m = generate_standard_repn(obj_m.expr, compute_values=False)
    const = float(pyo.value(repn_m.constant)) if repn_m.constant is not None else 0.0
    lv = list(repn_m.linear_vars) if repn_m.linear_vars is not None else []
    lc = list(repn_m.linear_coefs) if repn_m.linear_coefs is not None else []

    x_terms = []
    for v, c in zip(lv, lc):
        k = VarKey(comp=v.parent_component().name, idx=v.index())
        if k in master_key_set:
            x_terms.append(float(c) * v)

    # Sub objective: y-only, and compute theta lower bound from variable bounds.
    obj_s = _active_objective(sub)
    repn_s = generate_standard_repn(obj_s.expr, compute_values=False)
    lv_s = list(repn_s.linear_vars) if repn_s.linear_vars is not None else []
    lc_s = list(repn_s.linear_coefs) if repn_s.linear_coefs is not None else []

    y_expr = 0.0
    theta_lb: float | None = 0.0
    for v, c in zip(lv_s, lc_s):
        k = VarKey(comp=v.parent_component().name, idx=v.index())
        if k in sub_key_set:
            y_expr = y_expr + float(c) * v
            coef = float(c)
            if coef >= 0.0:
                lb = v.lb
                if lb is None:
                    theta_lb = None
                elif theta_lb is not None:
                    theta_lb += coef * float(pyo.value(lb))
            else:
                ub = v.ub
                if ub is None:
                    theta_lb = None
                elif theta_lb is not None:
                    theta_lb += coef * float(pyo.value(ub))

    obj_s.deactivate()
    sub.obj_benders = pyo.Objective(expr=y_expr, sense=pyo.minimize)

    obj_m.deactivate()
    master.theta = pyo.Var(domain=pyo.Reals)
    if theta_lb is None:
        raise ValueError("classic_benders requires finite theta lower bound (derive from variable bounds)")
    master.theta.setlb(float(theta_lb))
    master.obj_benders = pyo.Objective(expr=const + sum(x_terms) + master.theta, sense=pyo.minimize)
    master.benders_cuts = pyo.ConstraintList()

    # Relax integrality in sub so duals exist; master vars remain and will be fixed during solves.
    pyo.TransformationFactory("core.relax_integer_vars").apply_to(sub)

    # Keep only constraints involving any y; deactivate x-only constraints in sub.
    sub_constraints: List[pyo.Constraint] = []
    for con in sub.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        repn = generate_standard_repn(con.body, compute_values=False)
        vars_in = list(repn.linear_vars) if repn.linear_vars is not None else []
        has_y = any(VarKey(comp=v.parent_component().name, idx=v.index()) in sub_key_set for v in vars_in)
        if not has_y:
            con.deactivate()
            continue
        sub_constraints.append(con)

    cut_rows: List[CutRow] = []
    sub_constraint_names: List[str] = []
    for con in sub_constraints:
        lower = con.lower
        upper = con.upper
        if lower is not None and upper is not None:
            body = con.body
            con.deactivate()
            con1 = pyo.Constraint(expr=body <= upper)
            con2 = pyo.Constraint(expr=(-body) <= (-lower))
            name1 = con.name + "_le"
            name2 = con.name + "_ge"
            sub.add_component(name1, con1)
            sub.add_component(name2, con2)
            targets = [con1, con2]
            names = [name1, name2]
        elif upper is not None:
            targets = [con]
            names = [con.name]
        elif lower is not None:
            body = con.body
            con.deactivate()
            conn = pyo.Constraint(expr=(-body) <= (-lower))
            name = con.name + "_ge"
            sub.add_component(name, conn)
            targets = [conn]
            names = [name]
        else:
            raise ValueError(f"Constraint {con.name} has no bounds")

        for cobj, cname in zip(targets, names):
            repn = generate_standard_repn(cobj.body, compute_values=False)
            constant = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
            up = cobj.upper
            if up is None:
                raise ValueError(f"Normalized constraint {cname} has no upper bound")
            rhs = float(pyo.value(up)) - constant

            x_coef: Dict[VarKey, float] = {}
            lv = list(repn.linear_vars) if repn.linear_vars is not None else []
            lc = list(repn.linear_coefs) if repn.linear_coefs is not None else []
            for v, coef in zip(lv, lc):
                k = VarKey(comp=v.parent_component().name, idx=v.index())
                if k in master_key_set:
                    x_coef[k] = x_coef.get(k, 0.0) + float(coef)

            cut_rows.append(CutRow(con_name=cname, rhs=rhs, x_coef=x_coef))
            sub_constraint_names.append(cname)

    mapping = BendersMapping(
        master_vars=master_var_keys,
        sub_constraint_names=sub_constraint_names,
        cut_rows=cut_rows,
        theta_name="theta",
        cuts_name="benders_cuts",
    )
    return master, sub, mapping

