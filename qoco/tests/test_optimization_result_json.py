from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from qoco.core.problem import Problem
from qoco.core.solution import ProblemSummary, Status
from qoco.optimizers.const import ConstOptimizer


@dataclass
class DummyProblem(Problem[ProblemSummary]):
    name: str = "dummy"

    def validate(self):
        return True, []

    def summary(self) -> ProblemSummary:
        return ProblemSummary(num_vars=0, num_constraints=0)


def test_optimization_result_json() -> None:
    problem = DummyProblem()
    optimizer = ConstOptimizer(value=42.0)
    result = optimizer.optimize(problem)
    assert result.solution.status == Status.OPTIMAL
    json_bytes = TypeAdapter(type(result)).dump_json(result)
    json_str = json_bytes.decode("utf-8")
    print(json_str)
