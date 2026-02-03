from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn

from qoco.core.converter import Converter


@dataclass(frozen=True)
class VarKey:
    comp: str
    idx: object | None


@dataclass(frozen=True)
class RelaxedConstraint:
    name: str
    sense: str  # "le" | "ge" | "eq"
    rhs: float
    constant: float
    coef: Dict[VarKey, float]


@dataclass
class LagrangianSplit:
    """One-time Lagrangian decomposition artifact.

    - `group_models[g]` is a standalone Pyomo subproblem (local constraints only).
    - `group_vars[g]` enumerates the (relaxed) decision vars owned by that group.
    - `relaxed` are the coupling constraints removed from all subproblems and handled via multipliers.
    """

    group_models: List[pyo.ConcreteModel]
    group_vars: List[List[VarKey]]
    relaxed: List[RelaxedConstraint]


@dataclass
class ClassicLagrangianDecomposer(Converter[pyo.ConcreteModel, LagrangianSplit]):
    """Fully automatic coupling-constraint relaxation + per-group subproblems.

    Grouping policy: first index element of each variable (idx[0]) if tuple, else single group.
    Coupling constraints: active linear constraints that touch variables from >1 group.
    """

    def convert(self, problem: pyo.ConcreteModel) -> LagrangianSplit:
        model = problem.clone()

        # Build var -> group mapping (only discrete vars participate)
        vars_ = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
        key_of = {id(v): VarKey(comp=v.parent_component().name, idx=v.index()) for v in vars_}

        group_id_of: Dict[int, int] = {}
        group_key_to_id: Dict[object, int] = {}
        for v in vars_:
            if v.is_continuous():
                continue
            idx = v.index()
            gk = idx[0] if isinstance(idx, tuple) and len(idx) > 0 else 0
            gid = group_key_to_id.get(gk)
            if gid is None:
                gid = len(group_key_to_id)
                group_key_to_id[gk] = gid
            group_id_of[id(v)] = gid

        n_groups = max(1, len(group_key_to_id))

        # Identify coupling constraints and compile relaxed constraints coefficient maps.
        relaxed: List[RelaxedConstraint] = []
        relaxed_names: set[str] = set()

        for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            repn = generate_standard_repn(con.body, compute_values=False)
            if not (repn.is_linear() and not repn.is_quadratic()):
                raise ValueError(f"LagrangianDecomposer requires linear constraints; got nonlinear: {con.name}")

            groups_touched: set[int] = set()
            lv = list(repn.linear_vars) if repn.linear_vars is not None else []
            lc = list(repn.linear_coefs) if repn.linear_coefs is not None else []
            for v in lv:
                gid = group_id_of.get(id(v))
                if gid is not None:
                    groups_touched.add(int(gid))

            if len(groups_touched) <= 1:
                continue

            # Coupling -> relax
            relaxed_names.add(con.name)

            lower = con.lower
            upper = con.upper
            if lower is not None and upper is not None:
                sense = "eq"
                rhs = float(pyo.value(upper))
            elif upper is not None:
                sense = "le"
                rhs = float(pyo.value(upper))
            elif lower is not None:
                sense = "ge"
                rhs = float(pyo.value(lower))
            else:
                raise ValueError(f"Constraint {con.name} has no bounds")

            constant = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
            coef: Dict[VarKey, float] = {}
            for v, c in zip(lv, lc):
                k = key_of[id(v)]
                coef[k] = coef.get(k, 0.0) + float(c)

            relaxed.append(RelaxedConstraint(name=con.name, sense=sense, rhs=rhs, constant=constant, coef=coef))

        # Build per-group submodels: deactivate coupling constraints, keep only local constraints, fix non-group vars.
        group_models: List[pyo.ConcreteModel] = []
        group_vars: List[List[VarKey]] = []

        for gid in range(n_groups):
            sub = model.clone()
            # deactivate all relaxed constraints
            for con in sub.component_data_objects(pyo.Constraint, active=True, descend_into=True):
                if con.name in relaxed_names:
                    con.deactivate()

            # Fix vars not in this group (discrete only) to 0
            owned: List[VarKey] = []
            for v in sub.component_data_objects(pyo.Var, active=True, descend_into=True):
                if v.is_continuous():
                    continue
                idx = v.index()
                gk = idx[0] if isinstance(idx, tuple) and len(idx) > 0 else 0
                gid2 = group_key_to_id.get(gk, 0)
                if int(gid2) != int(gid):
                    v.fix(0)
                else:
                    owned.append(VarKey(comp=v.parent_component().name, idx=v.index()))

            group_models.append(sub)
            group_vars.append(owned)

        return LagrangianSplit(group_models=group_models, group_vars=group_vars, relaxed=relaxed)

