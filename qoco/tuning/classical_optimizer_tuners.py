from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import optuna
from qiskit_algorithms.optimizers import (
    ADAM,
    CG,
    COBYLA,
    GradientDescent,
    ISRES,
    L_BFGS_B,
    NELDER_MEAD,
    POWELL,
    QNSPSA,
    SLSQP,
    SPSA,
    TNC,
)


class ClassicalOptimizerParamTuner:
    def suggest(self, trial: optuna.Trial):
        raise NotImplementedError


def _cat(trial: optuna.Trial, name: str, values: Sequence):
    return trial.suggest_categorical(name, values)


@dataclass
class CobylaParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (20, 40, 80, 120)
    rhobeg: Sequence[float] = (0.2, 0.5, 1.0)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "cobyla_maxiter", self.maxiter)
        rhobeg = _cat(trial, "cobyla_rhobeg", self.rhobeg)
        return COBYLA(maxiter=maxiter, rhobeg=rhobeg)


@dataclass
class SpsaParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (20, 40, 80, 120)
    last_avg: Sequence[int] = (1, 5, 10)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "spsa_maxiter", self.maxiter)
        last_avg = _cat(trial, "spsa_last_avg", self.last_avg)
        return SPSA(maxiter=maxiter, last_avg=last_avg)


@dataclass
class QnspsaParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (20, 40, 80, 120)

    def suggest(self, trial: optuna.Trial):
        # Fidelity is required by QNSPSA. Qiskit expects a callable that returns an iterable
        # when evaluated on a batch of points (internal vectorization). We return a list of
        # ones with matching batch size.
        fidelity = lambda *points, **_kwargs: [1.0 for _ in points]
        maxiter = _cat(trial, "qnspsa_maxiter", self.maxiter)
        return QNSPSA(fidelity=fidelity, maxiter=maxiter)


@dataclass
class AdamParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (100, 200, 500)
    lr: Sequence[float] = (1e-3, 3e-3, 1e-2)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "adam_maxiter", self.maxiter)
        lr = _cat(trial, "adam_lr", self.lr)
        return ADAM(maxiter=maxiter, lr=lr)


@dataclass
class NelderMeadParamTuner(ClassicalOptimizerParamTuner):
    maxfev: Sequence[int] = (200, 500, 1000)

    def suggest(self, trial: optuna.Trial):
        maxfev = _cat(trial, "nelder_mead_maxfev", self.maxfev)
        return NELDER_MEAD(maxfev=maxfev)


@dataclass
class PowellParamTuner(ClassicalOptimizerParamTuner):
    maxfev: Sequence[int] = (200, 500, 1000)

    def suggest(self, trial: optuna.Trial):
        maxfev = _cat(trial, "powell_maxfev", self.maxfev)
        return POWELL(maxfev=maxfev)


@dataclass
class SlsqpParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (20, 50, 100)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "slsqp_maxiter", self.maxiter)
        return SLSQP(maxiter=maxiter)


@dataclass
class LbfgsbParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (100, 300, 800)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "lbfgsb_maxiter", self.maxiter)
        return L_BFGS_B(maxiter=maxiter)


@dataclass
class TncParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (20, 50, 100)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "tnc_maxiter", self.maxiter)
        return TNC(maxiter=maxiter)


@dataclass
class CgParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (20, 50, 100)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "cg_maxiter", self.maxiter)
        return CG(maxiter=maxiter)


@dataclass
class IsresParamTuner(ClassicalOptimizerParamTuner):
    max_evals: Sequence[int] = (200, 500, 1000)

    def suggest(self, trial: optuna.Trial):
        max_evals = _cat(trial, "isres_max_evals", self.max_evals)
        return ISRES(max_evals=max_evals)


@dataclass
class GradientDescentParamTuner(ClassicalOptimizerParamTuner):
    maxiter: Sequence[int] = (50, 100, 200)
    learning_rate: Sequence[float] = (0.001, 0.01, 0.05)

    def suggest(self, trial: optuna.Trial):
        maxiter = _cat(trial, "gd_maxiter", self.maxiter)
        learning_rate = _cat(trial, "gd_lr", self.learning_rate)
        return GradientDescent(maxiter=maxiter, learning_rate=learning_rate)

