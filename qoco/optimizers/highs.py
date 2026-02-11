"""
HiGHS optimizer - high-performance MILP/LP solver.

Uses Pyomo's appsi_highs interface to the HiGHS solver.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generic, Optional

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from qoco.core.problem import Problem
from qoco.core.optimizer import Optimizer, P
from qoco.core.converter import Converter
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter


@dataclass
class HiGHSOptimizer(Generic[P], Optimizer[P, pyo.ConcreteModel, Solution, OptimizerRun, ProblemSummary]):
    """
    HiGHS MILP/LP solver via Pyomo.
    
    Attributes:
        converter: Converts problem to Pyomo model
        time_limit: Max solve time in seconds (None = no limit)
        mip_gap: Relative MIP gap tolerance (e.g., 0.01 = 1%)
        verbose: Print solver output
    """
    name: str = "HiGHS"
    converter: Converter[P, pyo.ConcreteModel] = field(default_factory=IdentityConverter)
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    verbose: bool = False
    capture_duals: bool = False
    
    def _optimize(self, model: pyo.ConcreteModel) -> tuple[Solution, OptimizerRun]:
        if self.capture_duals and not hasattr(model, "dual"):
            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Create solver
        solver = pyo.SolverFactory('appsi_highs')
        
        # Set options
        if self.time_limit is not None:
            solver.options['time_limit'] = self.time_limit
        if self.mip_gap is not None:
            solver.options['mip_rel_gap'] = self.mip_gap
        
        optimizer_timestamp_start = datetime.now(timezone.utc)
        result = solver.solve(model, tee=self.verbose)
        optimizer_timestamp_end = datetime.now(timezone.utc)
        
        # Map termination condition to Status
        tc = result.solver.termination_condition
        if tc == TerminationCondition.optimal:
            status = Status.OPTIMAL
        elif tc in [TerminationCondition.feasible, TerminationCondition.maxTimeLimit]:
            status = Status.FEASIBLE
        elif tc == TerminationCondition.infeasible:
            status = Status.INFEASIBLE
        else:
            status = Status.UNKNOWN
        
        # Extract variable values
        var_values = {}
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            for var in model.component_objects(pyo.Var, active=True):
                for idx in var:
                    v = var[idx]
                    val = pyo.value(v)
                    if val is not None:
                        var_values[v.name] = val
        
        # Get objective value
        try:
            obj_val = pyo.value(model.obj) if status in (Status.OPTIMAL, Status.FEASIBLE) else float('inf')
        except:
            obj_val = float('inf')
            status = Status.UNKNOWN
        
        solution = Solution(
            status=status,
            objective=obj_val,
            var_values=var_values,
        )
        return solution, OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
        )
