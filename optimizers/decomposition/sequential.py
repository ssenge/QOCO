from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar

from qoco.core.solution import Solution
from qoco.optimizers.decomposition.multistage import MultiStageSolver, StagePlan, StageResult, StageTask


StateT = TypeVar("StateT")
ProblemT = TypeVar("ProblemT")
TagT = TypeVar("TagT")


@dataclass
class _SeqState(Generic[ProblemT, TagT]):
    tasks: list[StageTask[ProblemT, TagT]]
    results: list[StageResult[ProblemT, TagT]]
    i: int = 0


@dataclass(frozen=True)
class SequentialSolver(Generic[ProblemT, TagT]):
    """Solve a fixed list of stage tasks once, in order, and combine.

    This is the explicit "SequentialSolver" we discussed: no iteration logic, no schemes,
    just run tasks 0..K-1, collect results, and combine into a Solution.
    """

    tasks: Sequence[StageTask[ProblemT, TagT]]
    combine: Callable[[Sequence[StageResult[ProblemT, TagT]]], Solution]
    time_limit_s: float | None = None

    def run(self) -> Solution:
        st = _SeqState(tasks=list(self.tasks), results=[], i=0)
        plan = _SequentialPlan(combine=self.combine)
        return MultiStageSolver(plan=plan, state=st, max_steps=len(st.tasks) + 1, time_limit_s=self.time_limit_s).run()


@dataclass(frozen=True)
class _SequentialPlan(StagePlan[_SeqState[ProblemT, TagT], ProblemT, TagT]):
    combine: Callable[[Sequence[StageResult[ProblemT, TagT]]], Solution]

    def next_task(self, state: _SeqState[ProblemT, TagT]) -> Optional[StageTask[ProblemT, TagT]]:
        if int(state.i) >= len(state.tasks):
            return None
        return state.tasks[int(state.i)]

    def apply(self, state: _SeqState[ProblemT, TagT], result: StageResult[ProblemT, TagT]) -> _SeqState[ProblemT, TagT]:
        state.results.append(result)
        state.i += 1
        return state

    def converged(self, state: _SeqState[ProblemT, TagT]) -> bool:
        return bool(int(state.i) >= len(state.tasks))

    def to_solution(self, state: _SeqState[ProblemT, TagT]) -> Solution:
        return self.combine(tuple(state.results))

