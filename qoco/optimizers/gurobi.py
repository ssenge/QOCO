"""
Gurobi optimizer - commercial MILP/LP solver.

Uses Pyomo's gurobi interface.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generic, Optional

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter


@dataclass
class GurobiOptimizer(Generic[P], Optimizer[P, pyo.ConcreteModel, Solution, OptimizerRun, ProblemSummary]):
    """
    Gurobi MILP/LP solver via Pyomo.

    Attributes:
        converter: Converts problem to Pyomo model
        time_limit: Max solve time in seconds (None = no limit)
        mip_gap: Relative MIP gap tolerance (e.g., 0.01 = 1%)
        log_file: Path to write the Gurobi log (None = no file)
        verbose: Print solver output
    """
    name: str = "Gurobi"
    converter: Converter[P, pyo.ConcreteModel] = field(default_factory=IdentityConverter)
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    log_file: Optional[str] = None
    verbose: bool = False
    capture_duals: bool = False
    options: dict[str, object] = field(default_factory=dict)

    def _optimize(self, model: pyo.ConcreteModel) -> tuple[Solution, OptimizerRun]:
        if self.capture_duals and not hasattr(model, "dual"):
            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        solver = pyo.SolverFactory("gurobi")

        if self.time_limit is not None:
            solver.options["TimeLimit"] = self.time_limit
        if self.mip_gap is not None:
            solver.options["MIPGap"] = self.mip_gap
        if self.log_file is not None:
            solver.options["LogFile"] = self.log_file
        for key, value in self.options.items():
            solver.options[str(key)] = value

        optimizer_timestamp_start = datetime.now(timezone.utc)
        result = solver.solve(model, tee=self.verbose)
        optimizer_timestamp_end = datetime.now(timezone.utc)

        tc = result.solver.termination_condition
        if tc == TerminationCondition.optimal:
            status = Status.OPTIMAL
        elif tc in [TerminationCondition.feasible, TerminationCondition.maxTimeLimit]:
            status = Status.FEASIBLE
        elif tc == TerminationCondition.infeasible:
            status = Status.INFEASIBLE
        else:
            status = Status.UNKNOWN

        var_values = {}
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            for var in model.component_objects(pyo.Var, active=True):
                for idx in var:
                    value = pyo.value(var[idx])
                    if value is not None:
                        var_values[var[idx].name] = value

        obj_val = float("inf")
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            objectives = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
            obj_val = pyo.value(objectives[0]) if objectives else float("inf")

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
