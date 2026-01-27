"""
Solution class for optimization results.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, Optional
import ast


class Status(Enum):
    """Solver termination status."""

    OPTIMAL = auto()  # Proven optimal
    FEASIBLE = auto()  # Feasible but not proven optimal
    INFEASIBLE = auto()  # Proven infeasible
    UNKNOWN = auto()  # Solver didn't find anything / error


@dataclass
class Solution:
    """
    Generic solution from an optimizer.

    Attributes:
        status: Solver termination status
        objective: Objective function value
        var_values: Variable name -> value mapping
        tts: Time to solution (seconds), set by Optimizer.optimize()
        info: Additional solver-specific information (e.g., gap, iterations)
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
    tts: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Whether the solution is feasible."""
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    def __repr__(self) -> str:
        status_str = "✓ " + self.status.name if self.feasible else "✗ " + self.status.name
        time_str = f", tts={self.tts:.3f}s" if self.tts is not None else ""
        return f"Solution({status_str}, obj={self.objective:.4f}, vars={len(self.var_values)}{time_str})"

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

