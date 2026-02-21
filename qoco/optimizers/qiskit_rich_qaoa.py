from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qoco.core.qubo import QUBO
from qoco.optimizers.qiskit_qaoa import QiskitQAOAOptimizer
from qoco.optimizers.qiskit_reporting import format_qaoa_config

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
        initial_point = self.build_initial_point(qubo, n, self.reps)
        return qaoa_kwargs, initial_point

    def config_metadata(self) -> dict[str, Any]:
        base = dict(super().config_metadata())
        base.update(
            {
                "postproc": type(self.postproc).__name__,
                "build_initial_state": getattr(self.build_initial_state, "__name__", type(self.build_initial_state).__name__),
                "build_mixer": getattr(self.build_mixer, "__name__", type(self.build_mixer).__name__),
                "build_aggregation": getattr(self.build_aggregation, "__name__", type(self.build_aggregation).__name__),
                "build_initial_point": getattr(self.build_initial_point, "__name__", type(self.build_initial_point).__name__),
            }
        )
        return base

    def format_config(self) -> str:
        return format_qaoa_config(self.config_metadata())

