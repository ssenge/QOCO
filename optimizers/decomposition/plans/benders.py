from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pyomo.environ as pyo

from qoco.converters.decomposition.benders import BendersDecomposition
from qoco.core.optimizer import Optimizer
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.multistage import StagePlan, StageResult, StageTask
from qoco.utils.pyomo.decomposition.benders import VarKey


@dataclass
class BendersState:
    decomp: BendersDecomposition
    master_solver: Optimizer[pyo.ConcreteModel, object, Solution]
    sub_solver: Optimizer[pyo.ConcreteModel, object, Solution]
    tol: float = 1e-6

    phase: str = "master"  # master | sub
    it: int = 0
    last_master_obj: float | None = None
    last_master_x: Dict[VarKey, float] | None = None
    last_sub_obj: float | None = None

    def theta_value(self) -> float:
        return float(pyo.value(getattr(self.decomp.master, self.decomp.mapping.theta_name)))


@dataclass(frozen=True)
class ClassicBendersPlan(StagePlan[BendersState, pyo.ConcreteModel, str]):
    def next_task(self, state: BendersState) -> Optional[StageTask[pyo.ConcreteModel, str]]:
        if state.phase == "master":
            return StageTask(tag="master", problem=state.decomp.master, solver=state.master_solver)
        if state.phase == "sub":
            if state.last_master_x is None:
                raise RuntimeError("benders: missing master solution")
            for k, val in state.last_master_x.items():
                comp = state.decomp.sub.find_component(k.comp)
                if comp is None:
                    raise RuntimeError(f"benders: sub var component not found: {k.comp}")
                v = comp if k.idx is None else comp[k.idx]
                v.fix(float(val))
            if not hasattr(state.decomp.sub, "dual"):
                state.decomp.sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            return StageTask(tag="sub", problem=state.decomp.sub, solver=state.sub_solver)
        raise RuntimeError(f"benders: unknown phase: {state.phase}")

    def apply(self, state: BendersState, result: StageResult[pyo.ConcreteModel, str]) -> BendersState:
        if result.task.tag == "master":
            if result.solution.status not in (Status.OPTIMAL, Status.FEASIBLE):
                raise RuntimeError(f"benders: master solve failed: {result.solution.status}")
            xvals: Dict[VarKey, float] = {}
            for k in state.decomp.mapping.master_vars:
                comp = state.decomp.master.find_component(k.comp)
                if comp is None:
                    raise RuntimeError(f"benders: master var component not found: {k.comp}")
                v = comp if k.idx is None else comp[k.idx]
                xvals[k] = float(pyo.value(v))
            state.last_master_obj = float(pyo.value(state.decomp.master.obj_benders))
            state.last_master_x = xvals
            state.phase = "sub"
            return state

        if result.task.tag == "sub":
            if result.solution.status not in (Status.OPTIMAL, Status.FEASIBLE):
                raise RuntimeError(f"benders: sub solve failed: {result.solution.status}")

            duals: Dict[str, float] = {}
            for cname in state.decomp.mapping.sub_constraint_names:
                con = state.decomp.sub.find_component(cname)
                if con is None:
                    raise RuntimeError(f"benders: sub constraint not found: {cname}")
                dv = state.decomp.sub.dual.get(con, None)  # type: ignore[attr-defined]
                if dv is None:
                    raise RuntimeError(f"benders: missing dual for constraint: {cname}")
                duals[cname] = float(dv)

            state.last_sub_obj = float(pyo.value(state.decomp.sub.obj_benders))

            theta = getattr(state.decomp.master, state.decomp.mapping.theta_name)
            cuts = getattr(state.decomp.master, state.decomp.mapping.cuts_name)
            expr = 0.0
            xcoef: Dict[VarKey, float] = {}
            for row in state.decomp.mapping.cut_rows:
                pi = float(duals[row.con_name])
                expr += pi * float(row.rhs)
                for k, a in row.x_coef.items():
                    xcoef[k] = xcoef.get(k, 0.0) - pi * float(a)
            for k, coef in xcoef.items():
                comp = state.decomp.master.find_component(k.comp)
                if comp is None:
                    raise RuntimeError(f"benders: master var component not found for cut: {k.comp}")
                v = comp if k.idx is None else comp[k.idx]
                expr = expr + float(coef) * v
            cuts.add(theta >= expr)

            state.it += 1
            state.phase = "master"
            return state

        raise RuntimeError("benders: unexpected stage tag")

    def converged(self, state: BendersState) -> bool:
        if state.last_sub_obj is None:
            return False
        if state.phase != "master":
            return False
        return bool(abs(state.theta_value() - float(state.last_sub_obj)) <= float(state.tol))

    def to_solution(self, state: BendersState) -> Solution:
        if state.last_master_obj is None or state.last_sub_obj is None:
            return Solution(status=Status.INFEASIBLE, objective=float("inf"), var_values={}, info={"iters": int(state.it)})
        total = float(state.last_master_obj - state.theta_value() + float(state.last_sub_obj))
        return Solution(status=Status.FEASIBLE, objective=total, var_values={}, info={"iters": int(state.it)})

