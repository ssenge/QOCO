"""
Solution class for optimization results.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional
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

