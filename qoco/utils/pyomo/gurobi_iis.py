from __future__ import annotations

import pyomo.environ as pyo


def write_gurobi_iis(
    model: pyo.ConcreteModel,
    path: str,
    solver_name: str = "gurobi_direct",
    symbolic_solver_labels: bool = True,
) -> None:
    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(
        model,
        tee=False,
        load_solutions=False,
        symbolic_solver_labels=symbolic_solver_labels,
    )
    gurobi_model = solver._solver_model
    gurobi_model.computeIIS()
    gurobi_model.write(path)
