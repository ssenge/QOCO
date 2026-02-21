from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, Optional

import numpy as np

from qoco.converters.identity import IdentityConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.qlcbo import QLCBO
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
class _QCIBaseOptimizer(Generic[P]):
    """Shared QCI client config + helpers (internal)."""

    url: str = "https://api.qci-prod.com"
    api_token: str = ""
    timeout: Optional[float] = None
    job_name: str | None = None
    job_tags: list[str] | None = None

    def _client(self):
        try:
            from qci_client import QciClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: install `qci-client` to use QCI optimizers.") from exc
        return QciClient(url=str(self.url), api_token=str(self.api_token), timeout=self.timeout)

    def _require_token(self) -> None:
        if not str(self.api_token).strip():
            raise ValueError("QCI optimizer requires api_token (set QCI_TOKEN env var and pass it explicitly).")


@dataclass
class QCIQuboOptimizer(Generic[P], _QCIBaseOptimizer[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """QCI optimizer for QUBOs via Dirac-3 integer Hamiltonian sampling.

    Implementation detail: submits the QUBO as a polynomial file and runs
    `sample-hamiltonian-integer` with binary levels (num_levels=2).
    """

    name: str = "QCI-Dirac-QUBO"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)

    device_type: str = "dirac-3"
    job_type: str = "sample-hamiltonian-integer"
    num_samples: int = 10
    relaxation_schedule: int = 1
    zero_tol: float = 0.0

    def __post_init__(self) -> None:
        self._require_token()
        rs = int(self.relaxation_schedule)
        if rs < 1 or rs > 4:
            raise ValueError(f"QCIQuboOptimizer relaxation_schedule must be in [1, 4], got {rs}.")

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        from qci_client import JobStatus
        if not isinstance(qubo, QUBO):
            raise TypeError(
                f"QCIQuboOptimizer expects a QUBO model, got {type(qubo).__name__}. "
                "Use a QUBO runner (e.g. SegmentQUBORunnerV2_5) or convert your model to QUBO first."
            )

        n = int(qubo.n_vars)
        file_payload = _qubo_to_qci_polynomial_file(qubo=qubo, zero_tol=float(self.zero_tol))
        client = self._client()

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


def _qlcbo_to_qci_files(*, qlcbo: QLCBO) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a QLCBO into QCI objective/constraints file payloads (dense)."""
    Q = np.asarray(qlcbo.objective, dtype=float)
    C = np.asarray(qlcbo.constraints, dtype=float)
    n = int(Q.shape[0])
    m = int(C.shape[0])

    obj_payload = {
        "file_name": "qoco_qlcbo_objective.json",
        # qci-client expects numpy arrays (or scipy sparse), not Python lists
        "file_config": {"objective": {"num_variables": int(n), "data": Q}},
    }
    cons_payload = {
        "file_name": "qoco_qlcbo_constraints.json",
        "file_config": {
            # qci-client expects numpy arrays (or scipy sparse), not Python lists
            "constraints": {"num_constraints": int(m), "num_variables": int(n), "data": C}
        },
    }
    return obj_payload, cons_payload


@dataclass
class QCIQlcboOptimizer(Generic[P], _QCIBaseOptimizer[P], Optimizer[P, QLCBO, Solution, OptimizerRun, ProblemSummary]):
    """QCI optimizer for QLCBO via `sample-constraint` (linear constraints kept explicit)."""

    name: str = "QCI-Dirac-QLCBO"
    converter: Converter[P, QLCBO] = field(default_factory=IdentityConverter)

    # NOTE: qci-client currently restricts sample-constraint to "qubit" devices.
    device_type: str = "dirac-1"
    job_type: str = "sample-constraint"

    num_samples: int = 10
    alpha: float = 2.0
    atol: float = 1e-10

    def __post_init__(self) -> None:
        self._require_token()

    def _optimize(self, qlcbo: QLCBO) -> tuple[Solution, OptimizerRun]:
        from qci_client import JobStatus
        if not isinstance(qlcbo, QLCBO):
            raise TypeError(
                f"QCIQlcboOptimizer expects a QLCBO model, got {type(qlcbo).__name__}. "
                "For Segment v2.5+LC on QCI, use SegmentQLCBORunnerV2_5_LC (Pyomo->QLCBO) "
                "instead of a QUBO runner."
            )

        client = self._client()
        obj_payload, cons_payload = _qlcbo_to_qci_files(qlcbo=qlcbo)

        optimizer_timestamp_start = datetime.now(timezone.utc)
        cons_resp = client.upload_file(file=cons_payload)
        obj_resp = client.upload_file(file=obj_payload)
        constraints_file_id = str(cons_resp["file_id"])
        objective_file_id = str(obj_resp["file_id"])

        job_params = {
            "device_type": str(self.device_type),
            "num_samples": int(self.num_samples),
            "alpha": float(self.alpha),
            "atol": float(self.atol),
        }
        job_body = client.build_job_body(
            job_type=str(self.job_type),
            job_params=job_params,
            constraints_file_id=constraints_file_id,
            objective_file_id=objective_file_id,
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
        solutions = list(results.get("solutions") or [])
        feas = list(results.get("feasibilities") or [])
        energies = list(results.get("energies") or [])
        obj_vals = list(results.get("objective_values") or [])
        if not solutions:
            return Solution(status=Status.UNKNOWN, objective=float("inf"), var_values={}), OptimizerRun(
                name=self.name,
                optimizer_timestamp_start=optimizer_timestamp_start,
                optimizer_timestamp_end=optimizer_timestamp_end,
            )

        feas_arr = np.asarray(feas, dtype=bool) if feas else None
        energies_arr = np.asarray(energies, dtype=float) if energies else None

        if feas_arr is not None and feas_arr.any() and energies_arr is not None and len(energies_arr) == len(feas_arr):
            cand = np.where(feas_arr)[0]
            best_i = int(cand[int(np.argmin(energies_arr[cand]))])
            status = Status.FEASIBLE
        else:
            best_i = 0
            status = Status.UNKNOWN

        x_list = list(solutions[best_i])
        n = int(qlcbo.n_vars)
        if len(x_list) != n:
            return Solution(status=Status.UNKNOWN, objective=float("inf"), var_values={}), OptimizerRun(
                name=self.name,
                optimizer_timestamp_start=optimizer_timestamp_start,
                optimizer_timestamp_end=optimizer_timestamp_end,
            )

        x = np.asarray([int(v) for v in x_list], dtype=int)
        names = list(qlcbo.var_names) if qlcbo.var_names is not None else [f"x{i+1}" for i in range(n)]
        var_values = {names[i]: int(x[i]) for i in range(n)}

        # Prefer objective_values if present; otherwise fall back to energy.
        if obj_vals and len(obj_vals) > best_i:
            objective = float(obj_vals[best_i])
        elif energies and len(energies) > best_i:
            objective = float(energies[best_i])
        else:
            objective = float("inf")

        solution = Solution(
            status=status,
            objective=objective,
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
            metadata={
                "energy": float(energies[best_i]) if energies and len(energies) > best_i else None,
                "feasible": bool(feas[best_i]) if feas and len(feas) > best_i else None,
            },
        )
        run = OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
        )
        return solution, run


# Backwards-compatible alias (old name).
QCIOptimizer = QCIQuboOptimizer

