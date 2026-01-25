from __future__ import annotations

from dataclasses import dataclass, field

from qoco.converters.identity import IdentityConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import P
from qoco.core.solution import Solution
from qoco.optimizers.kipu_base import KipuBaseOptimizer


@dataclass
class IllayOptimizer(KipuBaseOptimizer):
    """Solve QUBOs using Kipu Quantum Illay service (simulator)."""

    converter: Converter[P, dict[str, float]] = field(default_factory=IdentityConverter)
