from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.rl.shared.base import PolicyRunner, RLAdapter


AdapterT = TypeVar("AdapterT", bound=RLAdapter)


@dataclass(kw_only=True)
class MLPolicyOptimizer(Generic[P, AdapterT], Optimizer[P, Any, Solution, OptimizerRun, ProblemSummary]):
    """Inference-only Optimizer wrapper for ML policies."""

    name: str = "MLPolicy"
    adapter: AdapterT
    checkpoint_path: Path
    runner_cls: type[PolicyRunner[AdapterT]]
    converter: Converter[P, Any] = field(default_factory=IdentityConverter)
    device: str = "cpu"

    _runner: PolicyRunner[AdapterT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._runner = self.runner_cls.load(self.checkpoint_path, device=self.device, adapter=self.adapter)

    def _optimize(self, converted: Any) -> tuple[Solution, OptimizerRun]:
        sol = self._runner.run(adapter=self.adapter, problem=converted, device=self.device)
        return sol, OptimizerRun(name=self.name)

