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
        
        # IMPORTANT: appsi_highs raises if no feasible solution exists and we try to load it.
        # We first solve with load_solutions=False to safely get termination condition.
        optimizer_timestamp_start = datetime.now(timezone.utc)
        result = solver.solve(model, tee=self.verbose, load_solutions=False)
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

        # Only load a solution into the Pyomo model when we expect one to exist.
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            try:
                solver.solve(model, tee=False, load_solutions=True)
            except Exception:
                status = Status.UNKNOWN
        
        # Extract variable values.
        #
        # IMPORTANT: Pyomo variables can be scalar or indexed. Iterating `for idx in var`
        # crashes for scalar vars. Using component_data_objects yields VarData for both.
        var_values: dict[str, float] = {}
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
                val = pyo.value(v)
                if val is None:
                    continue
                var_values[str(v.name)] = float(val)

        # Get objective value (do not assume an objective named `obj`).
        obj_val = float("inf")
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            objectives = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
            if objectives:
                try:
                    obj_val = float(pyo.value(objectives[0]))
                except Exception:
                    obj_val = float("inf")
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
