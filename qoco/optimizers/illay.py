from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qoco.converters.qubo_to_dict import QuboToDictConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import P
from qoco.optimizers.kipu_base import KipuBaseOptimizer

ILLAY_DEFAULT_ENDPOINT = (
    "https://gateway.hub.kipu-quantum.com/kipu-quantum/"
    "illay-base-quantum-optimizer/1.0.0"
)


@dataclass
class IllaySimOptimizer(KipuBaseOptimizer):
    """Solve QUBOs using Kipu Quantum Illay service (simulator)."""

    name: str = "KipuIllaySimulator"

    service_endpoint: str = ILLAY_DEFAULT_ENDPOINT
    converter: Converter[P, dict[str, Any]] = field(default_factory=QuboToDictConverter)
    return_circuit: bool = False
    execute_circuit: bool = True
