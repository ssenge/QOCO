from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pyomo.environ as pyo

from qoco.core.converter import Converter
from qoco.core.problem import Problem
from qoco.core.solution import ProblemSummary


@dataclass(frozen=True)
class Operation:
    machine: int
    duration: float


@dataclass
class JobShopScheduling(Problem[ProblemSummary]):
    name: str
    jobs: List[List[Operation]]

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not self.jobs:
            errors.append("no jobs provided")
            return False, errors
        for j, ops in enumerate(self.jobs):
            if not ops:
                errors.append(f"job {j} has no operations")
            for op in ops:
                if op.duration <= 0:
                    errors.append(f"job {j} has non-positive duration")
                if op.machine < 0:
                    errors.append(f"job {j} has negative machine id")
        return len(errors) == 0, errors

    def summary(self) -> ProblemSummary:
        return ProblemSummary()

    class MILPConverter(Converter["JobShopScheduling", pyo.ConcreteModel]):
        def convert(self, problem: "JobShopScheduling") -> pyo.ConcreteModel:
            jobs = problem.jobs
            n_jobs = len(jobs)

            op_keys: list[tuple[int, int]] = []
            for j, ops in enumerate(jobs):
                for o in range(len(ops)):
                    op_keys.append((j, o))

            durations = {(j, o): float(jobs[j][o].duration) for j, o in op_keys}
            machines = {(j, o): int(jobs[j][o].machine) for j, o in op_keys}

            total_time = sum(durations.values())
            model = pyo.ConcreteModel(name=getattr(problem, "name", None) or "JobShopScheduling")

            model.J = pyo.RangeSet(0, n_jobs - 1)
            model.OP = pyo.Set(dimen=2, initialize=op_keys)

            model.p = pyo.Param(model.OP, initialize=durations)
            model.m = pyo.Param(model.OP, initialize=machines)

            model.start = pyo.Var(model.OP, domain=pyo.NonNegativeReals)
            model.cmax = pyo.Var(domain=pyo.NonNegativeReals)

            def precedence_rule(mdl, j, o):
                if o + 1 >= len(jobs[j]):
                    return pyo.Constraint.Skip
                return mdl.start[j, o + 1] >= mdl.start[j, o] + mdl.p[j, o]

            model.precedence = pyo.Constraint(model.OP, rule=precedence_rule)

            # Pairs of operations on same machine
            pairs: list[tuple[int, int, int, int]] = []
            for (j1, o1) in op_keys:
                for (j2, o2) in op_keys:
                    if (j1, o1) >= (j2, o2):
                        continue
                    if machines[(j1, o1)] == machines[(j2, o2)]:
                        pairs.append((j1, o1, j2, o2))

            model.PAIRS = pyo.Set(dimen=4, initialize=pairs)
            model.y = pyo.Var(model.PAIRS, domain=pyo.Binary)

            def disjunctive_left(mdl, j1, o1, j2, o2):
                return mdl.start[j1, o1] + mdl.p[j1, o1] <= mdl.start[j2, o2] + total_time * (1 - mdl.y[j1, o1, j2, o2])

            def disjunctive_right(mdl, j1, o1, j2, o2):
                return mdl.start[j2, o2] + mdl.p[j2, o2] <= mdl.start[j1, o1] + total_time * mdl.y[j1, o1, j2, o2]

            model.machine_left = pyo.Constraint(model.PAIRS, rule=disjunctive_left)
            model.machine_right = pyo.Constraint(model.PAIRS, rule=disjunctive_right)

            def cmax_rule(mdl, j, o):
                return mdl.cmax >= mdl.start[j, o] + mdl.p[j, o]

            model.cmax_def = pyo.Constraint(model.OP, rule=cmax_rule)
            model.obj = pyo.Objective(expr=model.cmax, sense=pyo.minimize)
            return model
