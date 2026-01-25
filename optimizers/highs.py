"""
HiGHS optimizer - high-performance MILP/LP solver.

Uses Pyomo's appsi_highs interface to the HiGHS solver.
"""

from dataclasses import dataclass, field
from typing import Generic, Optional

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from qoco.core.problem import Problem
from qoco.core.optimizer import Optimizer, P
from qoco.core.converter import Converter
from qoco.core.solution import Solution, Status
from qoco.converters.identity import IdentityConverter


@dataclass
class HiGHSOptimizer(Generic[P], Optimizer[P, pyo.ConcreteModel, Solution]):
    """
    HiGHS MILP/LP solver via Pyomo.
    
    Attributes:
        converter: Converts problem to Pyomo model
        time_limit: Max solve time in seconds (None = no limit)
        mip_gap: Relative MIP gap tolerance (e.g., 0.01 = 1%)
        verbose: Print solver output
    """
    converter: Converter[P, pyo.ConcreteModel] = field(default_factory=IdentityConverter)
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    verbose: bool = False
    capture_duals: bool = False
    
    def _optimize(self, model: pyo.ConcreteModel) -> Solution:
        if self.capture_duals and not hasattr(model, "dual"):
            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Create solver
        solver = pyo.SolverFactory('appsi_highs')
        
        # Set options
        if self.time_limit is not None:
            solver.options['time_limit'] = self.time_limit
        if self.mip_gap is not None:
            solver.options['mip_rel_gap'] = self.mip_gap
        
        # Solve
        result = solver.solve(model, tee=self.verbose)
        
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
                        var_values[f"{var.name}[{idx}]"] = val
        
        # Get objective value
        try:
            obj_val = pyo.value(model.obj) if status in (Status.OPTIMAL, Status.FEASIBLE) else float('inf')
        except:
            obj_val = float('inf')
            status = Status.UNKNOWN
        
        info: dict[str, object] = {}
        if self.capture_duals and status in (Status.OPTIMAL, Status.FEASIBLE):
            duals: dict[str, float] = {}
            for con in model.component_objects(pyo.Constraint, active=True):
                if con.is_indexed():
                    for idx in con:
                        cdata = con[idx]
                        dv = model.dual.get(cdata, None)
                        if dv is not None:
                            duals[f"{con.name}[{idx}]"] = float(dv)
                else:
                    dv = model.dual.get(con, None)
                    if dv is not None:
                        duals[con.name] = float(dv)
            if duals:
                info["duals"] = duals

        return Solution(
            status=status,
            objective=obj_val,
            var_values=var_values,
            info=info,
        )
