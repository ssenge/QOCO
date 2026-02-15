from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Generic, TypeVar

import optuna

from qoco.tuning.report import TunerReport
from qoco.tuning.enums import Direction

T = TypeVar("T")


@dataclass(kw_only=True)
class ParamTuner(Generic[T]):
    """Base class for Optuna-based parameter tuners (in-memory only).

    Optuna requires an objective(trial) -> float callable. We keep that internal and
    delegate trial evaluation to `_evaluate(trial)`.
    """

    seed: int = 0
    n_trials: int = 50
    direction: Direction = Direction.MINIMIZE
    sampler_factory: Callable[[int], optuna.samplers.BaseSampler] = field(
        default_factory=lambda: (lambda seed: optuna.samplers.TPESampler(seed=seed))
    )
    catch: tuple[type[BaseException], ...] = (Exception,)

    def _evaluate(self, trial: optuna.Trial) -> float:
        raise NotImplementedError

    def _n_qubits(self) -> int:
        return 0

    def run(self) -> TunerReport:
        t0 = perf_counter()

        sampler = self.sampler_factory(self.seed)
        study = optuna.create_study(direction=self.direction.optuna_value, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            return float(self._evaluate(trial))

        study.optimize(objective, n_trials=self.n_trials, catch=self.catch)

        t1 = perf_counter()
        return TunerReport(study=study, n_qubits=self._n_qubits(), duration_s=t1 - t0)

