from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Sequence, TypeVar

from qoco.core.optimizer import Optimizer
from qoco.core.solution import Solution


StateT = TypeVar("StateT")
ProblemT = TypeVar("ProblemT")
TagT = TypeVar("TagT")


@dataclass(frozen=True)
class StageTask(Generic[ProblemT, TagT]):
    tag: TagT
    problem: ProblemT
    solver: Optimizer[ProblemT, object, Solution]


@dataclass(frozen=True)
class StageResult(Generic[ProblemT, TagT]):
    task: StageTask[ProblemT, TagT]
    solution: Solution


class StagePlan(Protocol[StateT, ProblemT, TagT]):
    def next_task(self, state: StateT) -> Optional[StageTask[ProblemT, TagT]]: ...
    def apply(self, state: StateT, result: StageResult[ProblemT, TagT]) -> StateT: ...
    def converged(self, state: StateT) -> bool: ...
    def to_solution(self, state: StateT) -> Solution: ...


@dataclass
class MultiStageSolver(Generic[StateT, ProblemT, TagT]):
    plan: StagePlan[StateT, ProblemT, TagT]
    state: StateT
    max_steps: int = 10_000
    time_limit_s: float | None = None

    def run(self) -> Solution:
        t0 = time.perf_counter()
        st = self.state

        for _ in range(int(self.max_steps)):
            if self.time_limit_s is not None and (time.perf_counter() - t0) >= float(self.time_limit_s):
                break
            if self.plan.converged(st):
                break

            task = self.plan.next_task(st)
            if task is None:
                break

            sol = task.solver.optimize(task.problem, log=False)
            st = self.plan.apply(st, StageResult(task=task, solution=sol))

        self.state = st
        return self.plan.to_solution(st)


@dataclass(frozen=True)
class TwoStageSolverV2(Generic[StateT, ProblemT, TagT]):
    """Thin wrapper around MultiStageSolver for 2-stage plans."""

    plan: StagePlan[StateT, ProblemT, TagT]
    state: StateT
    max_steps: int = 10_000
    time_limit_s: float | None = None

    def run(self) -> Solution:
        return MultiStageSolver(
            plan=self.plan,
            state=self.state,
            max_steps=int(self.max_steps),
            time_limit_s=self.time_limit_s,
        ).run()

