from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qoco.converters.qubo_to_dict import QuboToDictConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import P
from qoco.optimizers.kipu_base import KipuBaseOptimizer

MIRAY_SIM_DEFAULT_ENDPOINT = (
    "https://gateway.hub.kipu-quantum.com/kipu-quantum/"
    "miray-advanced-quantum-optimizer---simulator/1.0.0"
)

MIRAY_DEFAULT_ENDPOINT = (
    "https://gateway.hub.kipu-quantum.com/kipu-quantum/"
    "miray-advanced-quantum-optimizer/1.0.0"
)


@dataclass
class MiraySimOptimizer(KipuBaseOptimizer):
    """Solve QUBOs using Kipu Quantum Miray simulator."""

    name: str = "KipuMiraySimulator"

    service_endpoint: str = MIRAY_SIM_DEFAULT_ENDPOINT
    converter: Converter[P, dict[str, Any]] = field(default_factory=QuboToDictConverter)
    return_circuit: bool | None = False
    execute_circuit: bool | None = True


@dataclass
class MirayOptimizer(KipuBaseOptimizer):
    """Solve QUBOs using Kipu Quantum Miray managed service (real hardware).

    Per the official Marketplace API the only accepted fields are:
    problem, problem_type, shots, num_greedy_passes, num_iterations,
    backend_name, use_session.  IBM credentials are managed server-side.
    """

    name: str = "KipuMiray"

    service_endpoint: str = MIRAY_DEFAULT_ENDPOINT
    converter: Converter[P, dict[str, Any]] = field(default_factory=QuboToDictConverter)
    problem_type: str = "binary"
