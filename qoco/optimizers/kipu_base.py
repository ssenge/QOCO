from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from planqk.service.client import PlanqkServiceClient

from qoco.core.converter import Converter
from qoco.core.decoder import IdentityDecoder, IndexToVarNameDecoder, ResultDecoder
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


@dataclass
class KipuBaseOptimizer(Optimizer[P, dict[str, Any], Solution, OptimizerRun, ProblemSummary]):
    """Base optimizer for Kipu Quantum Hub services."""

    name: str = "Kipu"
    service_endpoint: str = ""
    consumer_key: str = ""
    consumer_secret: str = ""
    token_endpoint: str = "https://gateway.hub.kipu-quantum.com/token"
    shots: int = 1000
    num_greedy_passes: int = 0
    return_circuit: bool = False
    execute_circuit: bool = True
    # Optional service-specific settings (used by some Kipu optimizers like Miray managed).
    variant: str | None = None
    backend_name: str | None = None
    use_session: bool | None = None
    num_iterations: int | None = None
    converter: Converter[P, dict[str, Any]] | None = None

    def _extract_problem_payload(self, converted: dict[str, Any]) -> tuple[dict[str, float], dict[str, int]]:
        if "problem" in converted:
            return dict(converted["problem"]), dict(converted.get("var_map", {}))
        return dict(converted), {}

    def build_decoder(self, problem: P, converted: dict[str, Any]) -> ResultDecoder[Solution]:
        _, var_map = self._extract_problem_payload(converted)
        if not var_map:
            return IdentityDecoder()
        n = max((int(index) for index in var_map.values()), default=-1) + 1
        return IndexToVarNameDecoder(names_by_index=_names_by_index(var_map, n))

    def _optimize(self, converted: dict[str, Any]) -> tuple[Solution, OptimizerRun]:
        problem_dict, _ = self._extract_problem_payload(converted)
        client = PlanqkServiceClient(
            str(self.service_endpoint),
            str(self.consumer_key),
            str(self.consumer_secret),
            str(self.token_endpoint),
        )
        request: dict[str, Any] = {
            "problem": dict(problem_dict),
            "problem_type": "binary",
            "shots": int(self.shots),
            "num_greedy_passes": int(self.num_greedy_passes),
            "return_circuit": bool(self.return_circuit),
            "execute_circuit": bool(self.execute_circuit),
        }
        if self.variant is not None:
            request["variant"] = str(self.variant)
        if self.backend_name is not None:
            request["backend_name"] = str(self.backend_name)
        if self.use_session is not None:
            request["use_session"] = bool(self.use_session)
        if self.num_iterations is not None:
            request["num_iterations"] = int(self.num_iterations)
        execution = client.run(request=request)
        result = execution.result()
        if hasattr(result, "dict"):
            payload = result.dict()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {}

        status = payload.get("_embedded", {}).get("status", {}).get("status")
        if status == "FAILED":
            raise ValueError("Kipu execution failed; check service logs")

        inner = payload.get("processed_result")
        if not isinstance(inner, dict):
            inner = payload.get("result", payload)
        if not isinstance(inner, dict):
            raise ValueError("Kipu result missing result payload")

        cost = float(inner.get("cost"))
        mapped = inner.get("mapped_solution", {})
        mapped_int = {int(k): int(v) for k, v in dict(mapped).items()}

        n = max(mapped_int.keys(), default=-1) + 1
        x = np.zeros((n,), dtype=int)
        for i in range(n):
            x[i] = int(mapped_int.get(i, 0))

        names = _names_by_index({}, n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=float(cost),
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
        )
        return solution, OptimizerRun(name=self.name)
