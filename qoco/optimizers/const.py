"""
Constant optimizer - dummy implementation for testing.
"""

from dataclasses import dataclass
from typing import Generic

from qoco.core.problem import Problem
from qoco.core.optimizer import Optimizer, P
from qoco.core.converter import Converter
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status


class NullConverter(Converter[Problem, None]):
    """Converter that returns None (for dummy optimizer)."""
    
    def convert(self, problem: Problem) -> None:
        return None


@dataclass
class ConstOptimizer(Generic[P], Optimizer[P, None, Solution, OptimizerRun, ProblemSummary]):
    """Dummy optimizer that always returns a constant value."""
    name: str = "Const"
    converter: Converter[P, None] = None
    value: float = 0.0
    
    def __post_init__(self):
        if self.converter is None:
            self.converter = NullConverter()
    
    def _optimize(self, converted: None) -> tuple[Solution, OptimizerRun]:
        solution = Solution(
            status=Status.OPTIMAL,
            objective=self.value,
            var_values={},
        )
        run = OptimizerRun(name=self.name)
        return solution, run
