from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn
from qiskit_optimization import QuadraticProgram

from qoco.core.converter import Converter


def _sanitize_name(name: str) -> str:
    # Qiskit variable names should be simple identifiers.
    out = name.replace("[", "_").replace("]", "").replace(",", "_").replace(" ", "")
    out = out.replace("(", "_").replace(")", "")
    out = out.replace(".", "_")
    out = out.replace("-", "_")
    return out


@dataclass
class PyomoMIPToQuadraticProgramConverter(Converter[pyo.ConcreteModel, QuadraticProgram]):
    """Convert a Pyomo MIP/MIQP (linear constraints, quadratic-or-less objective) to Qiskit QuadraticProgram."""

    def convert(self, problem: pyo.ConcreteModel) -> QuadraticProgram:
        model = problem
        qp = QuadraticProgram(name=getattr(model, "name", "pyomo_mip"))

        # 1) Collect variables (VarData)
        vars_list = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
        var_to_name: Dict[int, str] = {}
        names_in_use: set[str] = set()

        for v in vars_list:
            raw = str(v.name)
            nm = _sanitize_name(raw)
            if nm in names_in_use:
                # deterministic disambiguation
                k = 1
                while f"{nm}_{k}" in names_in_use:
                    k += 1
                nm = f"{nm}_{k}"
            names_in_use.add(nm)
            var_to_name[id(v)] = nm

            lb = v.lb
            ub = v.ub
            if v.is_binary():
                qp.binary_var(name=nm)
            elif v.is_integer():
                qp.integer_var(name=nm, lowerbound=float(lb) if lb is not None else 0.0, upperbound=float(ub) if ub is not None else 1e9)
            else:
                qp.continuous_var(name=nm, lowerbound=float(lb) if lb is not None else -1e9, upperbound=float(ub) if ub is not None else 1e9)

        # 2) Objective (single active)
        objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
        if len(objs) != 1:
            raise ValueError(f"Expected exactly 1 active objective, found {len(objs)}")
        obj = objs[0]
        repn = generate_standard_repn(obj.expr, quadratic=True, compute_values=False)
        if repn.is_nonlinear():
            raise ValueError("Objective is nonlinear beyond quadratic")

        linear: Dict[str, float] = {}
        quad: Dict[Tuple[str, str], float] = {}
        constant = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0

        lv = list(repn.linear_vars) if repn.linear_vars is not None else []
        lc = list(repn.linear_coefs) if repn.linear_coefs is not None else []
        for v, c in zip(lv, lc):
            nm = var_to_name.get(id(v))
            if nm is None:
                continue
            linear[nm] = linear.get(nm, 0.0) + float(c)

        qv = list(repn.quadratic_vars) if repn.quadratic_vars is not None else []
        qc = list(repn.quadratic_coefs) if repn.quadratic_coefs is not None else []
        for (v1, v2), c in zip(qv, qc):
            n1 = var_to_name.get(id(v1))
            n2 = var_to_name.get(id(v2))
            if n1 is None or n2 is None:
                continue
            key = (n1, n2) if n1 <= n2 else (n2, n1)
            quad[key] = quad.get(key, 0.0) + float(c)

        sense = obj.sense
        if sense == pyo.minimize:
            qp.minimize(constant=constant, linear=linear, quadratic=quad)
        else:
            qp.maximize(constant=constant, linear=linear, quadratic=quad)

        # 3) Linear constraints (split ranged constraints)
        def add_linear_con(name: str, rep, sense: str, rhs: float) -> None:
            terms: Dict[str, float] = {}
            lv2 = list(rep.linear_vars) if rep.linear_vars is not None else []
            lc2 = list(rep.linear_coefs) if rep.linear_coefs is not None else []
            const2 = float(pyo.value(rep.constant)) if rep.constant is not None else 0.0
            for v, c in zip(lv2, lc2):
                nm = var_to_name.get(id(v))
                if nm is None:
                    continue
                terms[nm] = terms.get(nm, 0.0) + float(c)
            # Move constant to rhs
            qp.linear_constraint(linear=terms, sense=sense, rhs=float(rhs) - const2, name=name)

        cons = list(model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
        for con in cons:
            rep = generate_standard_repn(con.body, quadratic=False, compute_values=False)
            if rep.is_nonlinear() or rep.is_quadratic():
                raise ValueError(f"Nonlinear constraint not supported: {con.name}")
            lb = con.lower
            ub = con.upper
            if lb is not None and ub is not None:
                lbv = float(pyo.value(lb))
                ubv = float(pyo.value(ub))
                if abs(lbv - ubv) <= 1e-12:
                    add_linear_con(con.name, rep, "==", lbv)
                else:
                    add_linear_con(con.name + "_ge", rep, ">=", lbv)
                    add_linear_con(con.name + "_le", rep, "<=", ubv)
            elif ub is not None:
                add_linear_con(con.name, rep, "<=", float(pyo.value(ub)))
            elif lb is not None:
                add_linear_con(con.name, rep, ">=", float(pyo.value(lb)))
            else:
                raise ValueError(f"Constraint has no bounds: {con.name}")

        return qp

