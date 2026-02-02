"""
Simulated Annealing optimizer for QUBO matrices.
"""

from dataclasses import dataclass, field
from typing import Generic, Optional, Sequence
import numpy as np

from qoco.core.optimizer import Optimizer, P
from qoco.core.converter import Converter
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter
from ..core.qubo import QUBO


@dataclass
class SimulatedAnnealingOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """
    Simulated Annealing for QUBO matrices.
    
    QUBO form: minimize x^T Q x + offset, where x ∈ {0,1}^n
    
    Attributes:
        converter: Returns (Q, offset, var_map)
        T_initial: Starting temperature
        T_final: Ending temperature
        cooling_rate: Multiplicative cooling factor per iteration
        max_iter: Maximum iterations
        seed: Random seed for reproducibility
    """
    name: str = "SimulatedAnnealing"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    T_initial: float = 100.0
    T_final: float = 0.01
    cooling_rate: float = 0.9995
    max_iter: int = 10000
    seed: Optional[int] = None
    
    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        Q = np.asarray(qubo.Q, dtype=float)
        offset = float(qubo.offset)
        var_map = dict(qubo.var_map)
        n = int(Q.shape[0])
        
        # Set random seed
        rng = np.random.default_rng(self.seed)
        
        # Initialize random solution
        x = rng.integers(0, 2, n)
        current_cost = self._compute_cost(Q, x, offset)
        
        best_x = x.copy()
        best_cost = current_cost
        
        T = self.T_initial
        iterations = 0
        accepts = 0
        
        for iterations in range(1, self.max_iter + 1):
            if T < self.T_final:
                break
            
            # Pick random bit to flip
            j = rng.integers(n)
            
            # Compute cost change for flipping bit j
            delta = self._compute_delta(Q, x, j)
            
            # Metropolis criterion
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x[j] = 1 - x[j]
                current_cost += delta
                accepts += 1
                
                if current_cost < best_cost:
                    best_x = x.copy()
                    best_cost = current_cost
            
            # Cool down
            T *= self.cooling_rate
        
        # Build var_values from best_x
        idx_to_name: list[str] = [""] * n
        if var_map:
            inv = {int(idx): str(name) for name, idx in var_map.items()}
            for k in range(n):
                idx_to_name[k] = inv.get(k, str(k))
        else:
            for k in range(n):
                idx_to_name[k] = str(k)
        var_values = {idx_to_name[k]: int(best_x[k]) for k in range(n)}
        
        solution = Solution(
            status=Status.FEASIBLE,  # SA doesn't prove optimality
            objective=best_cost,
            var_values=var_values,
            var_arrays={"x": best_x.copy()},
            var_array_index={"x": list(idx_to_name)},
        )
        return solution, OptimizerRun(name=self.name)
    
    def _compute_cost(self, Q: np.ndarray, x: np.ndarray, offset: float) -> float:
        """Compute x^T Q x + offset."""
        return float(x @ Q @ x + offset)
    
    def _compute_delta(self, Q: np.ndarray, x: np.ndarray, j: int) -> float:
        """
        Compute cost change for flipping bit j.
        
        If x[j] = 0 → 1: Δ = Q[j,j] + 2 * Σₖ Q[j,k] * x[k]  (k ≠ j)
        If x[j] = 1 → 0: Δ = -Q[j,j] - 2 * Σₖ Q[j,k] * x[k]  (k ≠ j)
        
        Simplifies to: Δ = (1 - 2*x[j]) * (Q[j,j] + 2 * Σₖ Q[j,k] * x[k])
        But we need to handle symmetric Q properly.
        """
        n = len(x)
        
        # Sum of interactions with other bits (row + column for symmetric)
        interaction = 0.0
        for k in range(n):
            if k != j:
                # Q is upper triangular, so check both Q[j,k] and Q[k,j]
                interaction += (Q[j, k] + Q[k, j]) * x[k]
        
        if x[j] == 0:
            # Flipping 0 → 1
            delta = Q[j, j] + interaction
        else:
            # Flipping 1 → 0
            delta = -Q[j, j] - interaction
        
        return delta

