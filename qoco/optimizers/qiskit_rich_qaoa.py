from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qoco.core.qubo import QUBO
from qoco.optimizers.qiskit_qaoa import QiskitQAOAOptimizer

BuildInitialState = Callable[[QUBO, int], Any | None]
BuildMixer = Callable[[QUBO, int], Any | None]
BuildCallback = Callable[[QUBO, int], Any | None]
BuildAggregation = Callable[[QUBO, int], Any | None]
BuildInitialPoint = Callable[[QUBO, int, int], list[float] | None]


@dataclass
class QiskitRichQAOAOptimizer(QiskitQAOAOptimizer[Any]):
    """QAOA optimizer with size-aware builders (called after MIQP -> QUBO conversion)."""

    build_initial_state: BuildInitialState = field(default_factory=lambda: (lambda _qubo, _n: None))
    build_mixer: BuildMixer = field(default_factory=lambda: (lambda _qubo, _n: None))
    build_callback: BuildCallback = field(default_factory=lambda: (lambda _qubo, _n: None))
    build_aggregation: BuildAggregation = field(default_factory=lambda: (lambda _qubo, _n: None))
    build_initial_point: BuildInitialPoint = field(default_factory=lambda: (lambda _qubo, _n, _reps: None))

    def __post_init__(self) -> None:
        self.ansatz_hook = self._ansatz_hook

    def _ansatz_hook(self, qubo: QUBO, n: int) -> tuple[dict[str, Any], list[float] | None]:
        qaoa_kwargs = {
            "initial_state": self.build_initial_state(qubo, n),
            "mixer": self.build_mixer(qubo, n),
            "callback": self.build_callback(qubo, n),
            "aggregation": self.build_aggregation(qubo, n),
        }
        initial_point = self.build_initial_point(qubo, n, int(self.reps))
        return qaoa_kwargs, initial_point

