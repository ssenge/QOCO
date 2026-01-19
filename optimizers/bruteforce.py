"""
Brute force optimizer - enumerate all binary variable assignments.

Works on any Pyomo ConcreteModel with binary variables.
Only suitable for tiny instances (< 20 binary variables).
"""

from dataclasses import dataclass
from typing import Generic
from itertools import product

import pyomo.environ as pyo

from qoco.core.problem import Problem
from qoco.core.optimizer import Optimizer, P
from qoco.core.converter import Converter
from qoco.core.solution import Solution, Status


@dataclass
class BruteForceOptimizer(Generic[P], Optimizer[P, pyo.ConcreteModel, Solution]):
    """
    Brute force solver - enumerate all 2^n binary variable combinations.
    
    Works on any Pyomo model with binary variables.
    Only for testing with tiny instances!
    """
    converter: Converter[P, pyo.ConcreteModel]
    max_vars: int = 20  # Safety limit: 2^20 = ~1M iterations
    
    def _optimize(self, model: pyo.ConcreteModel) -> Solution:
        # Collect all binary variables, separating fixed and free
        free_vars = []
        fixed_vars = []
        for var in model.component_objects(pyo.Var, active=True):
            for idx in var:
                v = var[idx]
                if v.domain == pyo.Binary:
                    if v.is_fixed():
                        fixed_vars.append((var, idx))
                    else:
                        free_vars.append((var, idx))
        
        n_free = len(free_vars)
        
        if n_free > self.max_vars:
            raise ValueError(
                f"Too many binary variables ({n_free}) for brute force. "
                f"Max is {self.max_vars}. Use a real solver!"
            )
        
        if n_free == 0:
            # No free variables, just evaluate
            var_values = self._extract_all_vars(model, free_vars, fixed_vars)
            return Solution(
                status=Status.OPTIMAL if self._is_feasible(model) else Status.INFEASIBLE,
                objective=pyo.value(model.obj),
                var_values=var_values,
            )
        
        best_obj = float('inf')
        best_bits = None
        
        # Enumerate all 2^n combinations
        for bits in product([0, 1], repeat=n_free):
            # Set variable values
            for (var, idx), val in zip(free_vars, bits):
                var[idx].set_value(val)
            
            # Check feasibility
            if not self._is_feasible(model):
                continue
            
            # Evaluate objective
            try:
                obj_val = pyo.value(model.obj)
            except:
                continue
            
            if obj_val < best_obj:
                best_obj = obj_val
                best_bits = bits
        
        # Restore best solution
        if best_bits is not None:
            for (var, idx), val in zip(free_vars, best_bits):
                var[idx].set_value(val)
            var_values = self._extract_all_vars(model, free_vars, fixed_vars)
            return Solution(
                status=Status.OPTIMAL,  # Brute force is exhaustive
                objective=best_obj,
                var_values=var_values,
            )
        else:
            return Solution(
                status=Status.INFEASIBLE,
                objective=float('inf'),
                var_values={},
            )
    
    def _extract_all_vars(self, model: pyo.ConcreteModel, free_vars, fixed_vars) -> dict:
        """Extract all variable values (both free and fixed)."""
        var_values = {}
        for var, idx in free_vars + fixed_vars:
            var_values[f"{var.name}[{idx}]"] = pyo.value(var[idx])
        return var_values
    
    def _is_feasible(self, model: pyo.ConcreteModel) -> bool:
        """Check if all constraints are satisfied."""
        for con in model.component_objects(pyo.Constraint, active=True):
            for idx in con:
                c = con[idx]
                if c.active:
                    try:
                        body_val = pyo.value(c.body)
                        
                        # Check lower bound
                        if c.lower is not None:
                            lb = pyo.value(c.lower)
                            if body_val < lb - 1e-6:
                                return False
                        
                        # Check upper bound
                        if c.upper is not None:
                            ub = pyo.value(c.upper)
                            if body_val > ub + 1e-6:
                                return False
                                
                    except:
                        return False
        
        return True
