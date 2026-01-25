from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from planqk.service.client import PlanqkServiceClient

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import Solution, Status
from qoco.converters.identity import IdentityConverter


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


@dataclass
class IllayOptimizer(Optimizer[P, dict[str, float], Solution]):
    """Solve QUBOs using Kipu Quantum Illay service (simulator)."""

    converter: Converter[P, dict[str, float]] = field(default_factory=IdentityConverter)
    service_endpoint: str
    consumer_key: str
    consumer_secret: str
    token_endpoint: str = "https://gateway.hub.kipu-quantum.com/token"
    shots: int = 1000
    num_greedy_passes: int = 0
    return_circuit: bool = False
    execute_circuit: bool = True

    def _optimize(self, problem_dict: dict[str, float]) -> Solution:
        client = PlanqkServiceClient(
            str(self.service_endpoint),
            str(self.consumer_key),
            str(self.consumer_secret),
            str(self.token_endpoint),
        )
        request = {
            "problem": dict(problem_dict),
            "problem_type": "binary",
            "shots": int(self.shots),
            "num_greedy_passes": int(self.num_greedy_passes),
            "return_circuit": bool(self.return_circuit),
            "execute_circuit": bool(self.execute_circuit),
        }
        execution = client.run(request=request)
        result = execution.result()
        if hasattr(result, "dict"):
            payload = result.dict()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {}

        status = (
            payload.get("_embedded", {})
            .get("status", {})
            .get("status")
        )
        if status == "FAILED":
            raise ValueError("Illay execution failed; check service logs")

        inner = payload.get("processed_result")
        if not isinstance(inner, dict):
            inner = payload.get("result", payload)
        if not isinstance(inner, dict):
            raise ValueError("Illay result missing result payload")

        cost = float(inner.get("cost"))
        mapped = inner.get("mapped_solution", {})
        mapped_int = {int(k): int(v) for k, v in dict(mapped).items()}

        n = max(mapped_int.keys(), default=-1) + 1
        x = np.zeros((n,), dtype=int)
        for i in range(n):
            x[i] = int(mapped_int.get(i, 0))

        names = _names_by_index({}, n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        info = {
            "service_endpoint": str(self.service_endpoint),
            "execution_id": getattr(execution, "id", None),
        }
        return Solution(
            status=Status.FEASIBLE,
            objective=float(cost),
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
            info=info,
        )
