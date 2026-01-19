from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pyomo.environ as pyo

from qoco.converters.decomposition.lagrangian import LagrangianSplit, VarKey
from qoco.core.optimizer import Optimizer
from qoco.core.solution import Solution, Status
from qoco.optimizers.decomposition.multistage import StagePlan, StageResult, StageTask


def _active_objective(model: pyo.ConcreteModel) -> pyo.Objective:
    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if len(objs) != 1:
        raise ValueError(f"Expected exactly 1 active objective, found {len(objs)}")
    return objs[0]

def _active_objective_value(model: pyo.ConcreteModel) -> float:
    return float(pyo.value(_active_objective(model)))


@dataclass
class LagrangianState:
    split: LagrangianSplit
    sub_solver: Optimizer[pyo.ConcreteModel, object, Solution]
    step_size: float
    max_iters: int

    it: int = 0
    gid: int = 0
    x: Dict[VarKey, float] | None = None
    lambdas: Dict[str, float] | None = None
    _iter_sub_obj_sum: float = 0.0
    best_dual_bound: float | None = None

    def __post_init__(self) -> None:
        if self.x is None:
            self.x = {}
        if self.lambdas is None:
            self.lambdas = {rc.name: 0.0 for rc in self.split.relaxed}


@dataclass(frozen=True)
class ClassicLagrangianPlan(StagePlan[LagrangianState, pyo.ConcreteModel, int]):
    def next_task(self, state: LagrangianState) -> Optional[StageTask[pyo.ConcreteModel, int]]:
        if int(state.it) >= int(state.max_iters):
            return None
        if int(state.gid) >= len(state.split.group_models):
            return None
        sub = state.split.group_models[int(state.gid)].clone()
        _apply_lagrangian_objective(state=state, sub=sub)
        return StageTask(tag=int(state.gid), problem=sub, solver=state.sub_solver)

    def apply(self, state: LagrangianState, result: StageResult[pyo.ConcreteModel, int]) -> LagrangianState:
        if result.solution.status not in (Status.OPTIMAL, Status.FEASIBLE):
            raise RuntimeError(f"lagrangian: sub solve failed for group {result.task.tag}: {result.solution.status}")

        gid = int(result.task.tag)
        sub = result.task.problem
        # IMPORTANT: HiGHSOptimizer reports `model.obj` by name, but we create a new active
        # objective for Lagrangian. Use the active objective value here.
        state._iter_sub_obj_sum += _active_objective_value(sub)
        for k in state.split.group_vars[gid]:
            comp = sub.find_component(k.comp)
            if comp is None:
                continue
            v = comp if k.idx is None else comp[k.idx]
            state.x[k] = float(pyo.value(v))  # type: ignore[index]

        state.gid += 1
        if int(state.gid) >= len(state.split.group_models):
            dual = _dual_bound(state)
            if state.best_dual_bound is None or float(dual) > float(state.best_dual_bound):
                state.best_dual_bound = float(dual)
            _update_multipliers(state)
            state.gid = 0
            state.it += 1
            state._iter_sub_obj_sum = 0.0
        return state

    def converged(self, state: LagrangianState) -> bool:
        return bool(int(state.it) >= int(state.max_iters))

    def to_solution(self, state: LagrangianState) -> Solution:
        vv = {f"{k.comp}{'' if k.idx is None else str(k.idx)}": float(v) for k, v in (state.x or {}).items()}
        dual = state.best_dual_bound if state.best_dual_bound is not None else _dual_bound(state)
        return Solution(
            status=Status.FEASIBLE,
            objective=float(dual),
            var_values=vv,
            info={"iters": int(state.it), "lambdas": dict(state.lambdas or {}), "note": "objective is best Lagrangian dual bound"},
        )


def _apply_lagrangian_objective(*, state: LagrangianState, sub: pyo.ConcreteModel) -> None:
    obj = _active_objective(sub)
    base = obj.expr
    obj.deactivate()

    expr = base
    for rc in state.split.relaxed:
        lam = float((state.lambdas or {}).get(rc.name, 0.0))
        if lam == 0.0:
            continue
        term = 0.0
        for k, a in rc.coef.items():
            comp = sub.find_component(k.comp)
            if comp is None:
                continue
            v = comp if k.idx is None else comp[k.idx]
            term = term + float(a) * v
        lhs = float(rc.constant) + term
        # IMPORTANT: Do NOT include +/- rhs inside each subproblem (would be counted once per group).
        # We add only the lhs contribution here, and handle the rhs once in the global dual bound.
        if rc.sense == "le":
            expr = expr + lam * lhs
        elif rc.sense == "ge":
            expr = expr - lam * lhs
        else:  # eq
            expr = expr + lam * lhs
    sub.obj_lagrangian = pyo.Objective(expr=expr, sense=obj.sense)


def _update_multipliers(state: LagrangianState) -> None:
    step = float(state.step_size)
    x = state.x or {}
    lambdas = state.lambdas or {}
    for rc in state.split.relaxed:
        body = float(rc.constant)
        for k, a in rc.coef.items():
            body += float(a) * float(x.get(k, 0.0))
        if rc.sense == "le":
            viol = max(0.0, body - float(rc.rhs))
            lambdas[rc.name] = max(0.0, float(lambdas.get(rc.name, 0.0)) + step * viol)
        elif rc.sense == "ge":
            viol = max(0.0, float(rc.rhs) - body)
            lambdas[rc.name] = max(0.0, float(lambdas.get(rc.name, 0.0)) + step * viol)
        else:
            viol = body - float(rc.rhs)
            lambdas[rc.name] = float(lambdas.get(rc.name, 0.0)) + step * viol
    state.lambdas = lambdas


def _dual_bound(state: LagrangianState) -> float:
    """Compute L(λ) for minimization from current iteration's subproblem sums.

    With our construction:
    - le/eq: sub objs include +λ*lhs_g, so L = sum_sub - λ*rhs
    - ge:    sub objs include -λ*lhs_g, so L = sum_sub + λ*rhs
    """
    s = float(state._iter_sub_obj_sum)
    lambdas = state.lambdas or {}
    for rc in state.split.relaxed:
        lam = float(lambdas.get(rc.name, 0.0))
        if rc.sense == "ge":
            s += lam * float(rc.rhs)
        else:
            s -= lam * float(rc.rhs)
    return float(s)

