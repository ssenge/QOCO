from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import optuna
from optuna.trial import TrialState


@dataclass(frozen=True)
class TunerReport:
    study: optuna.Study
    n_qubits: int
    duration_s: float

    def __str__(self) -> str:
        best = self.study.best_trial
        best_value = best.value
        best_params = dict(best.params)
        best_user_attrs = dict(best.user_attrs)

        trials = list(self.study.trials)
        n_total = len(trials)
        n_complete = sum(1 for t in trials if t.state is TrialState.COMPLETE)
        n_fail = sum(1 for t in trials if t.state is TrialState.FAIL)

        dur = timedelta(seconds=self.duration_s)
        return "\n".join(
            [
                "TunerReport:",
                f"  qubits={self.n_qubits}",
                f"  trials_total={n_total} completed={n_complete} failed={n_fail}",
                f"  duration={dur}",
                f"  best_objective={best_value}",
                f"  best_params={best_params}",
                f"  best_user_attrs={best_user_attrs}",
            ]
        )

