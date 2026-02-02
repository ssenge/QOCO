"""
Branch-and-Bound optimizer for QUBO matrices.

Uses exact or heuristic bounds at each node to prune the search tree.
Guarantees optimal solution (within node limit).
"""

from dataclasses import dataclass, field
from typing import Generic, Optional, List
from itertools import product
import heapq
import numpy as np

from qoco.core.optimizer import Optimizer, P
from qoco.core.converter import Converter
from qoco.core.solution import InfoSolution, OptimizerRun, ProblemSummary, Status
from qoco.converters.identity import IdentityConverter
from ..core.qubo import QUBO


@dataclass(order=True)
class BnBNode:
    """A node in the B&B tree, ordered by lower bound."""
    lower_bound: float
    # Non-comparable fields
    fixed_vars: Dict[int, int] = field(compare=False)
    upper_bound: float = field(default=float('inf'), compare=False)
    solution: Optional[np.ndarray] = field(default=None, compare=False)


@dataclass
class BranchAndBoundOptimizer(Generic[P], Optimizer[P, QUBO, InfoSolution, OptimizerRun, ProblemSummary]):
    """
    Branch-and-Bound for QUBO with exact/heuristic bounding.
    
    Guarantees optimal solution for small problems (within node limit).
    
    Attributes:
        converter: Converts problem to (Q, offset, var_map)
        max_nodes: Maximum B&B nodes to explore
        max_vars_exact: Max vars for exact subproblem solve (brute force)
        branch_strategy: "first_free" or "random"
        seed: Random seed
        verbose: Print progress every N nodes (0 = silent)
    """
    name: str = "BranchAndBound"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    max_nodes: int = 10000
    max_vars_exact: int = 20  # 2^20 = ~1M combinations (increased default)
    branch_strategy: str = "first_free"
    seed: Optional[int] = None
    verbose: int = 100  # Print every N nodes
    
    def _optimize(self, qubo: QUBO) -> tuple[InfoSolution, OptimizerRun]:
        Q = np.asarray(qubo.Q, dtype=float)
        offset = float(qubo.offset)
        var_map = dict(qubo.var_map)
        n = int(Q.shape[0])
        
        if n == 0:
            solution = InfoSolution(status=Status.OPTIMAL, objective=offset, var_values={})
            return solution, OptimizerRun(name=self.name)
        
        rng = np.random.default_rng(self.seed)
        
        # Initialize
        best_ub = float('inf')
        best_solution = None
        nodes_explored = 0
        nodes_pruned = 0
        
        # Priority queue (min-heap by lower bound)
        root = BnBNode(lower_bound=-float('inf'), fixed_vars={})
        open_nodes = [root]
        
        while open_nodes and nodes_explored < self.max_nodes:
            # Pop node with lowest LB (best-first search)
            node = heapq.heappop(open_nodes)
            nodes_explored += 1
            
            # Progress output
            if self.verbose > 0 and nodes_explored % self.verbose == 0:
                print(f"  B&B: nodes={nodes_explored}, pruned={nodes_pruned}, "
                      f"open={len(open_nodes)}, best_ub={best_ub:.2f}, node_lb={node.lower_bound:.2f}")
            
            # Build reduced QUBO
            Q_red, offset_red, free_vars, fixed_contrib = self._reduce_qubo(
                Q, offset, node.fixed_vars
            )
            n_free = len(free_vars)
            
            # Leaf node: all variables fixed
            if n_free == 0:
                obj = offset_red + fixed_contrib
                if obj < best_ub:
                    best_ub = obj
                    best_solution = self._expand_solution(
                        np.array([], dtype=int), node.fixed_vars, n
                    )
                continue
            
            # Compute lower bound
            lb, lb_solution = self._compute_lower_bound(Q_red, offset_red, free_vars, rng)
            lb += fixed_contrib
            
            # Prune if LB >= best UB
            if lb >= best_ub - 1e-9:
                nodes_pruned += 1
                continue
            
            # Compute upper bound (heuristic)
            ub, ub_sol_reduced = self._compute_upper_bound(Q_red, offset_red, n_free, rng)
            ub += fixed_contrib
            
            # Update best if improved
            if ub < best_ub:
                best_ub = ub
                best_solution = self._expand_solution(ub_sol_reduced, node.fixed_vars, n, free_vars)
            
            # If gap is zero at this node, no need to branch
            if abs(ub - lb) < 1e-9:
                continue
            
            # Branch: select variable and create children
            branch_idx = self._select_branch_var(free_vars, rng)
            branch_var = free_vars[branch_idx]
            
            for val in [0, 1]:
                child_fixed = node.fixed_vars.copy()
                child_fixed[branch_var] = val
                child = BnBNode(lower_bound=lb, fixed_vars=child_fixed)
                heapq.heappush(open_nodes, child)
        
        # Build var_values
        idx_to_name: list[str] = [""] * n
        if var_map:
            inv = {int(idx): str(name) for name, idx in var_map.items()}
            for k in range(n):
                idx_to_name[k] = inv.get(k, str(k))
        else:
            for k in range(n):
                idx_to_name[k] = str(k)

        var_values = {}
        if best_solution is not None:
            for k in range(n):
                var_values[idx_to_name[k]] = int(best_solution[k])
        
        # Compute final gap
        final_lb = -float('inf')
        if open_nodes:
            final_lb = open_nodes[0].lower_bound  # Best remaining LB
        gap = (best_ub - final_lb) / max(abs(best_ub), 1e-10) if best_ub < float('inf') else float('inf')
        
        # Determine status
        if not open_nodes or gap < 1e-9:
            status = Status.OPTIMAL
            gap = 0.0
        elif best_solution is not None:
            status = Status.FEASIBLE
        else:
            status = Status.UNKNOWN
        
        solution = InfoSolution(
            status=status,
            objective=best_ub if best_ub < float('inf') else float('inf'),
            var_values=var_values,
            var_arrays={"x": best_solution.copy()} if best_solution is not None else {},
            var_array_index={"x": list(idx_to_name)} if best_solution is not None else {},
            info={
                "nodes_explored": nodes_explored,
                "nodes_pruned": nodes_pruned,
                "gap": gap,
                "open_nodes": len(open_nodes),
            }
        )
        return solution, OptimizerRun(name=self.name)
    
    def _reduce_qubo(self, Q: np.ndarray, offset: float, fixed_vars: dict[int, int]) -> tuple[np.ndarray, float, list[int], float]:
        """
        Create reduced QUBO with fixed variables eliminated.
        
        Returns:
            Q_reduced: Reduced Q matrix for free variables
            offset_reduced: New offset
            free_vars: List of original indices that are free
            fixed_contrib: Contribution from fixed variables
        """
        n = Q.shape[0]
        free_vars = [i for i in range(n) if i not in fixed_vars]
        n_free = len(free_vars)
        
        if n_free == 0:
            # All fixed: compute objective directly
            fixed_contrib = offset
            for i, vi in fixed_vars.items():
                fixed_contrib += Q[i, i] * vi
                for j, vj in fixed_vars.items():
                    if i < j:
                        fixed_contrib += (Q[i, j] + Q[j, i]) * vi * vj
            return np.zeros((0, 0)), 0.0, [], fixed_contrib
        
        # Build reduced Q
        Q_red = np.zeros((n_free, n_free))
        offset_red = offset
        fixed_contrib = 0.0
        
        # Fixed-fixed interactions -> constant
        for i, vi in fixed_vars.items():
            fixed_contrib += Q[i, i] * vi
            for j, vj in fixed_vars.items():
                if i < j:
                    fixed_contrib += (Q[i, j] + Q[j, i]) * vi * vj
        
        # Free-free interactions -> reduced Q
        for li, gi in enumerate(free_vars):
            Q_red[li, li] = Q[gi, gi]
            for lj in range(li + 1, n_free):
                gj = free_vars[lj]
                Q_red[li, lj] = Q[gi, gj] + Q[gj, gi]
        
        # Fixed-free interactions -> modify linear terms
        for li, gi in enumerate(free_vars):
            for fi, fv in fixed_vars.items():
                if fv == 1:
                    Q_red[li, li] += Q[gi, fi] + Q[fi, gi]
        
        return Q_red, offset_red, free_vars, fixed_contrib
    
    def _compute_lower_bound(self, Q: np.ndarray, offset: float, free_vars: list[int], rng) -> tuple[float, np.ndarray]:
        """Compute lower bound via exact solve (if small) or relaxation."""
        n = Q.shape[0]
        
        if n <= self.max_vars_exact:
            # Exact brute force
            return self._solve_exact(Q, offset)
        else:
            # For larger problems: use a simple lower bound
            # LB = offset + sum of negative diagonal entries (each x can be 0 or 1)
            # This is a weak but valid lower bound
            lb = offset
            for i in range(n):
                if Q[i, i] < 0:
                    lb += Q[i, i]  # If diagonal is negative, x=1 helps
                # Also consider off-diagonal: if Q[i,j] < 0, both being 1 helps
                for j in range(i + 1, n):
                    q_ij = Q[i, j] + Q[j, i]
                    if q_ij < 0:
                        lb += q_ij
            return lb, np.zeros(n, dtype=int)
    
    def _compute_upper_bound(self, Q: np.ndarray, offset: float, n: int, rng) -> tuple[float, np.ndarray]:
        """Compute upper bound via heuristic (SA)."""
        if n == 0:
            return offset, np.array([], dtype=int)
        
        if n <= self.max_vars_exact:
            # If small enough, solve exactly
            return self._solve_exact(Q, offset)
        
        # Otherwise use SA
        return self._solve_sa(Q, offset, rng)
    
    def _solve_exact(self, Q: np.ndarray, offset: float) -> tuple[float, np.ndarray]:
        """Solve QUBO exactly via brute force."""
        n = Q.shape[0]
        if n == 0:
            return offset, np.array([], dtype=int)
        
        best_cost = float('inf')
        best_x = None
        
        for bits in product([0, 1], repeat=n):
            x = np.array(bits, dtype=int)
            cost = float(x @ Q @ x + offset)
            if cost < best_cost:
                best_cost = cost
                best_x = x
        
        return best_cost, best_x
    
    def _solve_sa(self, Q: np.ndarray, offset: float, rng) -> tuple[float, np.ndarray]:
        """Solve QUBO approximately via SA."""
        n = Q.shape[0]
        x = rng.integers(0, 2, n)
        
        def cost(x):
            return float(x @ Q @ x + offset)
        
        def delta(x, j):
            interaction = sum((Q[j, k] + Q[k, j]) * x[k] for k in range(n) if k != j)
            return (Q[j, j] + interaction) if x[j] == 0 else (-Q[j, j] - interaction)
        
        best_x, best_cost = x.copy(), cost(x)
        current_cost = best_cost
        T = 50.0
        
        for _ in range(2000):
            j = rng.integers(n)
            d = delta(x, j)
            if d < 0 or rng.random() < np.exp(-d / max(T, 0.01)):
                x[j] = 1 - x[j]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            T *= 0.997
        
        return best_cost, best_x
    
    def _select_branch_var(self, free_vars: list[int], rng) -> int:
        """Select which free variable to branch on."""
        if self.branch_strategy == "first_free":
            return 0
        elif self.branch_strategy == "random":
            return rng.integers(len(free_vars))
        else:
            return 0
    
    def _expand_solution(self, sol_reduced: np.ndarray, fixed_vars: dict[int, int], n: int, free_vars: list[int] | None = None) -> np.ndarray:
        """Expand reduced solution to full solution."""
        x = np.zeros(n, dtype=int)
        
        # Set fixed variables
        for i, v in fixed_vars.items():
            x[i] = v
        
        # Set free variables from reduced solution
        if free_vars is not None and len(sol_reduced) > 0:
            for li, gi in enumerate(free_vars):
                x[gi] = sol_reduced[li]
        
        return x

