"""
Gurobi optimizer - commercial MILP/LP solver.

Uses Pyomo's gurobi interface.
"""

from dataclasses import dataclass, field
from typing import Generic, Optional

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import InfoSolution, OptimizerRun, ProblemSummary, Status
from qoco.converters.identity import IdentityConverter


@dataclass
class GurobiOptimizer(Generic[P], Optimizer[P, pyo.ConcreteModel, InfoSolution, OptimizerRun, ProblemSummary]):
    """
    Gurobi MILP/LP solver via Pyomo.

    Attributes:
        converter: Converts problem to Pyomo model
        time_limit: Max solve time in seconds (None = no limit)
        mip_gap: Relative MIP gap tolerance (e.g., 0.01 = 1%)
        verbose: Print solver output
    """
    name: str = "Gurobi"
    converter: Converter[P, pyo.ConcreteModel] = field(default_factory=IdentityConverter)
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    verbose: bool = False
    capture_duals: bool = False

    def _optimize(self, model: pyo.ConcreteModel) -> tuple[InfoSolution, OptimizerRun]:
        if self.capture_duals and not hasattr(model, "dual"):
            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        solver = pyo.SolverFactory("gurobi")

        if self.time_limit is not None:
            solver.options["TimeLimit"] = self.time_limit
        if self.mip_gap is not None:
            solver.options["MIPGap"] = self.mip_gap

        result = solver.solve(model, tee=self.verbose)

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

        try:
            obj_val = pyo.value(model.obj) if status in (Status.OPTIMAL, Status.FEASIBLE) else float("inf")
        except Exception:
            obj_val = float("inf")
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

        solution = InfoSolution(
            status=status,
            objective=obj_val,
            var_values=var_values,
            info=info,
        )
        return solution, OptimizerRun(name=self.name)
