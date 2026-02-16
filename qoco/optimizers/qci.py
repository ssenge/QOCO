from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, Optional

import numpy as np

from qoco.converters.identity import IdentityConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


def _qubo_to_qci_polynomial_file(*, qubo: QUBO, zero_tol: float) -> dict[str, Any]:
    """Convert a QUBO into QCI Dirac polynomial-file JSON (degree <= 2).

    QCI expects:
      file = {"file_name": "...", "file_config": {"polynomial": {...}}}
    with polynomial data encoded as:
      data = [{"idx": [..], "val": coef}, ...]

    Notes:
    - Indices are 1-based (QCI convention).
    - For a linear term x_i with max_degree=2, we encode idx=[0, i].
    - Constant offset is omitted.
    """
    Q = np.asarray(qubo.Q, dtype=float)
    n = int(Q.shape[0])
    if n == 0:
        poly = {"num_variables": 0, "min_degree": 1, "max_degree": 1, "data": []}
        return {"file_name": "qoco_qubo_polynomial", "file_config": {"polynomial": poly}}

    tol = float(zero_tol)
    terms: list[tuple[list[int], float]] = []

    for i in range(n):
        coef = float(Q[i, i])
        if abs(coef) > tol:
            terms.append(([0, int(i) + 1], coef))

    for i in range(n):
        for j in range(i + 1, n):
            coef = float(Q[i, j]) + float(Q[j, i])
            if abs(coef) > tol:
                terms.append(([int(i) + 1, int(j) + 1], coef))

    has_linear = any(idx[0] == 0 for idx, _c in terms)
    has_quadratic = any(idx[0] != 0 for idx, _c in terms)
    if has_quadratic:
        max_degree = 2
        min_degree = 1 if has_linear else 2
    else:
        max_degree = 1
        min_degree = 1

    data: list[dict[str, Any]] = []
    for idx, coef in terms:
        if int(max_degree) == 1:
            if idx[0] == 0:
                data.append({"idx": [int(idx[1])], "val": float(coef)})
            else:
                data.append({"idx": [int(idx[0])], "val": float(coef)})
        else:
            data.append({"idx": [int(idx[0]), int(idx[1])], "val": float(coef)})

    poly = {
        "num_variables": int(n),
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "data": data,
    }
    return {"file_name": "qoco_qubo_polynomial", "file_config": {"polynomial": poly}}


@dataclass
class QCIOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """QCI Dirac optimizer via `qci-client` (cloud API).

    This optimizer submits the QUBO as a polynomial file and runs the Dirac-3 integer
    solver job type (`sample-hamiltonian-integer`) with binary levels (num_levels=2).
    """

    name: str = "QCI-Dirac"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)

    url: str = "https://api.qci-prod.com"
    api_token: str = ""
    timeout: Optional[float] = None

    device_type: str = "dirac-3"
    job_type: str = "sample-hamiltonian-integer"

    num_samples: int = 10
    relaxation_schedule: int = 1

    job_name: str | None = None
    job_tags: list[str] | None = None

    zero_tol: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.api_token).strip():
            raise ValueError("QCIOptimizer requires api_token (set QCI_TOKEN env var and pass it explicitly).")

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        try:
            from qci_client import JobStatus, QciClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('Missing dependency: install `qci-client` to use QCIOptimizer.') from exc

        n = int(qubo.n_vars)
        file_payload = _qubo_to_qci_polynomial_file(qubo=qubo, zero_tol=float(self.zero_tol))

        client = QciClient(url=str(self.url), api_token=str(self.api_token), timeout=self.timeout)

        optimizer_timestamp_start = datetime.now(timezone.utc)
        file_response = client.upload_file(file=file_payload)
        file_id = str(file_response["file_id"])

        job_params = {
            "device_type": str(self.device_type),
            "num_samples": int(self.num_samples),
            "relaxation_schedule": int(self.relaxation_schedule),
            "num_levels": [2] * int(n),
        }
        job_body = client.build_job_body(
            job_type=str(self.job_type),
            job_params=job_params,
            polynomial_file_id=file_id,
            job_name=self.job_name,
            job_tags=self.job_tags,
        )
        job_response = client.process_job(job_body=job_body)
        optimizer_timestamp_end = datetime.now(timezone.utc)

        status_raw = str(job_response.get("status", ""))
        if status_raw != JobStatus.COMPLETED.value:
            return Solution(status=Status.UNKNOWN, objective=float("inf"), var_values={}), OptimizerRun(
                name=self.name,
                optimizer_timestamp_start=optimizer_timestamp_start,
                optimizer_timestamp_end=optimizer_timestamp_end,
            )

        results = dict(job_response.get("results") or {})
        energies = list(results.get("energies") or [])
        solutions = list(results.get("solutions") or [])
        if not energies or not solutions:
            return Solution(status=Status.UNKNOWN, objective=float("inf"), var_values={}), OptimizerRun(
                name=self.name,
                optimizer_timestamp_start=optimizer_timestamp_start,
                optimizer_timestamp_end=optimizer_timestamp_end,
            )

        best_i = int(np.argmin(np.asarray(energies, dtype=float)))
        energy = float(energies[best_i])
        x_list = list(solutions[best_i])
        if len(x_list) != int(n):
            return Solution(status=Status.UNKNOWN, objective=float("inf"), var_values={}), OptimizerRun(
                name=self.name,
                optimizer_timestamp_start=optimizer_timestamp_start,
                optimizer_timestamp_end=optimizer_timestamp_end,
            )

        x = np.asarray([int(v) for v in x_list], dtype=int)
        names = _names_by_index(dict(qubo.var_map), int(n))
        var_values = {names[i]: int(x[i]) for i in range(int(n))}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=float(energy),
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
        )
        run = OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
        )
        return solution, run

