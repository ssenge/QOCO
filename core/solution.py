"""
Solution class for optimization results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, Generic, Iterable, Optional, TypeVar
import ast
from pydantic import TypeAdapter

try:
    from pydantic.dataclasses import dataclass as pydantic_dataclass
except Exception:  # pragma: no cover - fallback when pydantic is unavailable
    pydantic_dataclass = dataclass


class Status(Enum):
    """Solver termination status."""

    OPTIMAL = auto()  # Proven optimal
    FEASIBLE = auto()  # Feasible but not proven optimal
    INFEASIBLE = auto()  # Proven infeasible
    UNKNOWN = auto()  # Solver didn't find anything / error


SummaryT = TypeVar("SummaryT", bound="ProblemSummary")
SolutionT = TypeVar("SolutionT", bound="Solution")
RunT = TypeVar("RunT", bound="OptimizerRun")


@pydantic_dataclass
class ProblemSummary:
    num_vars: int | None = None
    num_constraints: int | None = None


@pydantic_dataclass
class OptimizerRun:
    name: str
    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None
    optimizer_timestamp_start: datetime | None = None
    optimizer_timestamp_end: datetime | None = None

    def tts(self) -> float | None:
        if self.timestamp_start is None or self.timestamp_end is None:
            return None
        return (self.timestamp_end - self.timestamp_start).total_seconds()

    def optimizer_tts(self) -> float | None:
        if self.optimizer_timestamp_start is None or self.optimizer_timestamp_end is None:
            return None
        return (self.optimizer_timestamp_end - self.optimizer_timestamp_start).total_seconds()


@pydantic_dataclass
class OptimizationResult(Generic[SolutionT, RunT, SummaryT]):
    solution: SolutionT
    run: RunT
    problem: SummaryT

    def write(self, path: str | Path) -> None:
        target = Path(path)
        payload = TypeAdapter(type(self)).dump_json(self).decode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")


@pydantic_dataclass
class Solution:
    """
    Generic solution from an optimizer.

    Attributes:
        status: Solver termination status
        objective: Objective function value
        var_values: Variable name -> value mapping
        known_optimal_value: Optional a-priori optimal value (if known)
    """

    status: Status
    objective: float
    var_values: Dict[str, Any]
    # Optional structured variable arrays (useful for n-dim variables).
    # The canonical interchange format remains `var_values`.
    var_arrays: Dict[str, Any] = field(default_factory=dict)
    # Optional index mapping for non-dense variable arrays:
    # var_name -> list of Pyomo indices in the array's flattened order (or nested order by convention).
    var_array_index: Dict[str, Any] = field(default_factory=dict)
    known_optimal_value: Optional[float] = None

    @property
    def feasible(self) -> bool:
        """Whether the solution is feasible."""
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    def __repr__(self) -> str:
        status_str = "✓ " + self.status.name if self.feasible else "✗ " + self.status.name
        return f"Solution({status_str}, obj={self.objective:.4f}, vars={len(self.var_values)})"

    def equals(
        self,
        other: object,
        objective_tol: float = 1e-8,
        var_names: Iterable[str] | None = None,
    ) -> bool:
        if not isinstance(other, Solution):
            return False
        if abs(float(self.objective) - float(other.objective)) > float(objective_tol):
            return False
        if var_names is None:
            var_names = self.var_values.keys()
        return all(
            name in other.var_values and other.var_values[name] == self.var_values[name]
            for name in var_names
        )
    
    def equals_common_vars(self, other: object) -> bool:
        var_names = set(self.var_values.keys()) & set(other.var_values.keys())
        return self.equals(other, var_names=var_names)

    def __eq__(self, other: object) -> bool:
        return self.equals_common_vars(other)

    @staticmethod
    def _parse_index(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    def selected_indices(self, var_name: str, threshold: float = 0.5) -> list[Any]:
        prefix = f"{var_name}["
        out: list[Any] = []
        for key, value in self.var_values.items():
            if not key.startswith(prefix) or not key.endswith("]"):
                continue
            if value is None or value < threshold:
                continue
            idx_text = key[len(prefix) : -1]
            out.append(self._parse_index(idx_text))
        return out


@pydantic_dataclass
class InfoSolution(Solution):
    info: Dict[str, Any] = field(default_factory=dict)

