"""
Solution class for optimization results.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, Generic, Iterable, TypeVar
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
    metadata: dict[str, Any] | None = None

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
    metadata: dict[str, Any] | None = None

    def write(self, path: str | Path) -> None:
        def _normalize(obj: Any) -> Any:
            """Make values JSON-serializable for log output.

            Logging should be robust even when solutions/metadata contain numpy objects.
            """
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, datetime):
                # ISO 8601 (pydantic uses this convention too).
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {str(k): _normalize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_normalize(v) for v in obj]
            if isinstance(obj, Enum):
                return obj.name

            # Numpy support (optional).
            try:
                import numpy as _np  # type: ignore

                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, _np.generic):
                    return obj.item()
            except Exception:
                pass

            # Last resort: stringify to avoid crashing long experiments.
            return str(obj)

        target = Path(path)
        # Write in a readability-first order: problem/run first, then solution (var_values), then metadata.
        # JSON object order is not semantically relevant, but it matters a lot for scanning large `.jsonl` logs.
        dumped = TypeAdapter(type(self)).dump_python(self, mode="python")
        ordered = {
            "problem": dumped.get("problem"),
            "run": dumped.get("run"),
            "solution": dumped.get("solution"),
            "metadata": dumped.get("metadata"),
        }
        payload = json.dumps(_normalize(ordered), ensure_ascii=False, separators=(",", ":"))
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



