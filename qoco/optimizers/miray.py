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
    return_circuit: bool = False
    execute_circuit: bool = True


@dataclass
class MirayOptimizer(KipuBaseOptimizer):
    """Solve QUBOs using Kipu Quantum Miray managed service (real hardware)."""

    name: str = "KipuMiray"

    service_endpoint: str = MIRAY_DEFAULT_ENDPOINT
    converter: Converter[P, dict[str, Any]] = field(default_factory=QuboToDictConverter)
    return_circuit: bool = False
    execute_circuit: bool = True
