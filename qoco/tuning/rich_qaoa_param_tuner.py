from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import optuna
from enum import Enum as _Enum
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit.circuit.library import QAOAAnsatz

from qoco.core.qubo import QUBO
from qoco.optimizers.qaoa_rich_variants import (
    FourierInitialPoint,
    cvar_aggregation,
    uniform_h_initial_state,
    x_mixer_operator,
    y_mixer_operator,
    xy_group_mixer_operator,
)
from qoco.optimizers.qiskit_qaoa import CostPrimitive, TranspileStrategy
from qoco.optimizers.qiskit_rich_qaoa import QiskitRichQAOAOptimizer
from qoco.postproc.qubo_postproc import (
    LocalSearchPostProcessor,
    MostFrequentPostProcessor,
    NoOpPostProcessor,
    TakeBestPostProcessor,
)
from qoco.postproc.pce_postproc import PCEPostProcessor, SafePCEPostProcessor
from qoco.tuning.classical_optimizer_tuners import ClassicalOptimizerParamTuner
from qoco.tuning.enums import (
    AggregationChoice,
    BackendChoice,
    CallbackChoice,
    InitialPointChoice,
    InitialStateChoice,
    MitigationProfile,
    MixerChoice,
    PostProcChoice,
)
from qoco.tuning.param_tuner import ParamTuner
from qoco.tuning.mitigation import apply_mitigation_profile


@dataclass(frozen=True)
class RichQAOASearchSpace:
    reps: Sequence[int] = (1, 2, 3, 4, 5)
    shots: Sequence[int] = (128, 256, 512)
    optimization_level: Sequence[int] = (0, 1, 2, 3)
    cost_scale: Sequence[float | None] = (None, 0.5, 1.0, 2.0)

    transpile_strategy: Sequence[TranspileStrategy] = (TranspileStrategy.DEFAULT,)
    cost_primitive: Sequence[CostPrimitive] = (CostPrimitive.SAMPLER,)
    return_distribution: Sequence[bool | None] = (None,)
    mitigation_profile: Sequence[MitigationProfile] = (MitigationProfile.NONE,)

    # Lean defaults (explicitly set to QAOA's standard choices):
    initial_state: Sequence[InitialStateChoice] = (InitialStateChoice.UNIFORM_H,)
    mixer: Sequence[MixerChoice] = (MixerChoice.X_MIXER,)
    aggregation: Sequence[AggregationChoice] = (AggregationChoice.NONE, AggregationChoice.CVAR)
    cvar_alpha: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0)

    initial_point: Sequence[InitialPointChoice] = (InitialPointChoice.NONE, InitialPointChoice.CONSTANT, InitialPointChoice.FOURIER)
    constant_initial_point: Sequence[float] = (0.05, 0.1, 0.2)

    callback: Sequence[CallbackChoice] = (CallbackChoice.NONE,)

    postproc: Sequence[PostProcChoice] = (
        PostProcChoice.NOOP,
        PostProcChoice.TAKE_BEST,
        PostProcChoice.MOST_FREQUENT,
        PostProcChoice.LOCAL_SEARCH,
        # PostProcChoice.PCE is supported, but not enabled by default.
    )
    local_search_k_start: Sequence[int] = (1, 2, 3, 5, 8)
    local_search_max_iters: Sequence[int] = (50, 100, 200, 400)


def _cat(trial: optuna.Trial, name: str, values: Sequence[Any]) -> Any:
    # Optuna stores categorical choices in persistent storage. Enums are not stable
    # (they can't be round-tripped safely), so we map them via their .name strings.
    if values and all(isinstance(v, _Enum) for v in values):
        enum_cls = type(values[0])
        names = [v.name for v in values]
        chosen = trial.suggest_categorical(name, names)
        return enum_cls[chosen]
    return trial.suggest_categorical(name, values)


@dataclass
class RichQAOAParamTuner(ParamTuner[QUBO]):
    qubo: QUBO

    backend_choice: BackendChoice = BackendChoice.AER
    backend: Any | None = None  # optional explicit backend

    classical_optimizer_tuners: Sequence[ClassicalOptimizerParamTuner] = field(default_factory=tuple)
    space: RichQAOASearchSpace = field(default_factory=RichQAOASearchSpace)

    # Weighted objective returned to Optuna:
    #   objective_weight * base_qubo_obj
    #   + depth_weight * circuit_depth
    #   + swap_weight * n_swaps
    #   + twoq_weight * n_2q_ops
    #
    # Use tiny weights for depth/swap/twoq if you want them as deterministic tie-breakers only.
    objective_weight: float = 1.0
    depth_weight: float = 1.0
    swap_weight: float = 1.0
    twoq_weight: float = 1.0

    # Fixed optimizer knobs (still "covered" by being configurable here)
    transpiler: Any | None = None
    sampler: Any | None = None

    # Optional group definitions for XY group mixers.
    # Groups are lists of qubit indices (0..n-1) and are encoding-specific.
    mixer_groups: list[list[int]] | None = None
    xy_group_weight: float = 1.0

    def _n_qubits(self) -> int:
        return self.qubo.n_vars

    def _make_backend(self) -> Any | None:
        if self.backend is not None:
            return self.backend
        if self.backend_choice == BackendChoice.AER:
            return None
        if self.backend_choice == BackendChoice.FAKE_KYIV:
            return FakeKyiv()
        if self.backend_choice == BackendChoice.CUSTOM:
            raise ValueError("backend_choice=CUSTOM requires backend to be provided")
        raise ValueError(f"Unknown backend_choice: {self.backend_choice}")

    def _run_once(self, *, opt: QiskitRichQAOAOptimizer, qubo: QUBO) -> tuple[Any, int, int, int]:
        op, _offset = opt.encode_qubo(qubo)
        if opt.cost_scale is not None:
            op = op * opt.cost_scale

        n = qubo.n_vars
        hook_kwargs, _hook_initial_point = opt.ansatz_hook(qubo, n)
        ansatz = QAOAAnsatz(
            cost_operator=op,
            reps=opt.reps,
            initial_state=hook_kwargs.get("initial_state"),
            mixer_operator=hook_kwargs.get("mixer"),
        )
        ansatz.measure_all()

        if opt.transpiler is not None:
            compiled = opt.transpiler.run(ansatz)
        elif opt.backend is not None:
            layout_method: str | None = None
            routing_method: str | None = None
            if opt.transpile_strategy == TranspileStrategy.SABRE:
                layout_method = "sabre"
                routing_method = "sabre"
            elif opt.transpile_strategy == TranspileStrategy.SABRE_LAYOUT:
                layout_method = "sabre"
            elif opt.transpile_strategy == TranspileStrategy.SABRE_ROUTING:
                routing_method = "sabre"
            compiled = generate_preset_pass_manager(
                target=opt.backend.target,
                optimization_level=opt.optimization_level,
                layout_method=layout_method,
                routing_method=routing_method,
            ).run(ansatz)
        else:
            compiled = generate_preset_pass_manager(
                backend=AerSimulator(),
                optimization_level=opt.optimization_level,
            ).run(ansatz)

        depth = int(compiled.depth())
        ops = compiled.count_ops()
        swaps = int(ops.get("swap", 0))
        n_2q = int(sum(1 for _inst, qargs, _cargs in compiled.data if len(qargs) == 2))
        res = opt.optimize(qubo)
        return res, depth, swaps, n_2q

    def _evaluate(self, trial: optuna.Trial) -> float:
        if not self.classical_optimizer_tuners:
            raise ValueError("classical_optimizer_tuners must be provided (top-10 list)")

        reps = _cat(trial, "reps", self.space.reps)
        shots = _cat(trial, "shots", self.space.shots)
        optimization_level = _cat(trial, "optimization_level", self.space.optimization_level)
        cost_scale = _cat(trial, "cost_scale", self.space.cost_scale)
        cost_scale_f = cost_scale

        transpile_strategy = _cat(trial, "transpile_strategy", self.space.transpile_strategy)
        cost_primitive = _cat(trial, "cost_primitive", self.space.cost_primitive)
        return_distribution = _cat(trial, "return_distribution", self.space.return_distribution)
        mitigation_profile = _cat(trial, "mitigation_profile", self.space.mitigation_profile)

        # Choose a classical optimizer tuner, then let it suggest its own params.
        idx = trial.suggest_int("classical_optimizer_idx", 0, len(self.classical_optimizer_tuners) - 1)
        cot = self.classical_optimizer_tuners[idx]
        classical_optimizer = cot.suggest(trial)
        trial.set_user_attr("classical_optimizer", cot.__class__.__name__)

        initial_state = _cat(trial, "initial_state", self.space.initial_state)
        mixer = _cat(trial, "mixer", self.space.mixer)
        aggregation = _cat(trial, "aggregation", self.space.aggregation)
        initial_point = _cat(trial, "initial_point", self.space.initial_point)
        postproc = _cat(trial, "postproc", self.space.postproc)
        callback = _cat(trial, "callback", self.space.callback)

        if aggregation == AggregationChoice.CVAR:
            alpha = _cat(trial, "cvar_alpha", self.space.cvar_alpha)
            build_aggregation = lambda _qubo, _n: cvar_aggregation(alpha=alpha)
        else:
            build_aggregation = lambda _qubo, _n: None

        if initial_state == InitialStateChoice.UNIFORM_H:
            build_initial_state = lambda _qubo, n: uniform_h_initial_state(n=n)
        else:
            build_initial_state = lambda _qubo, _n: None

        if mixer == MixerChoice.X_MIXER:
            build_mixer = lambda _qubo, n: x_mixer_operator(n=n)
        elif mixer == MixerChoice.Y_MIXER:
            build_mixer = lambda _qubo, n: y_mixer_operator(n=n)
        elif mixer == MixerChoice.XY_GROUP_RING:
            groups = list(self.mixer_groups or [])
            w = float(self.xy_group_weight)
            build_mixer = lambda _qubo, n: xy_group_mixer_operator(n=n, groups=groups, topology="ring", weight=w)
        elif mixer == MixerChoice.XY_GROUP_COMPLETE:
            groups = list(self.mixer_groups or [])
            w = float(self.xy_group_weight)
            build_mixer = lambda _qubo, n: xy_group_mixer_operator(n=n, groups=groups, topology="complete", weight=w)
        else:
            build_mixer = lambda _qubo, _n: None

        if callback == CallbackChoice.NONE:
            build_callback = lambda _qubo, _n: None
        else:
            build_callback = lambda _qubo, _n: None

        if initial_point == InitialPointChoice.CONSTANT:
            v = _cat(trial, "constant_initial_point", self.space.constant_initial_point)
            build_initial_point = lambda _qubo, _n, r: [v] * (2 * r)
        elif initial_point == InitialPointChoice.FOURIER:
            fq = FourierInitialPoint(gamma_cos=[0.2], gamma_sin=[0.0], beta_cos=[0.1], beta_sin=[0.0])
            build_initial_point = lambda q, n, r: fq.build(q, n, r)
        else:
            build_initial_point = lambda _qubo, _n, _r: None

        if postproc == PostProcChoice.TAKE_BEST:
            post_processor = TakeBestPostProcessor()
        elif postproc == PostProcChoice.MOST_FREQUENT:
            post_processor = MostFrequentPostProcessor()
        elif postproc == PostProcChoice.LOCAL_SEARCH:
            k_start = _cat(trial, "ls_k_start", self.space.local_search_k_start)
            max_iters = _cat(trial, "ls_max_iters", self.space.local_search_max_iters)
            post_processor = LocalSearchPostProcessor(k_start=k_start, max_iters=max_iters)
        elif postproc == PostProcChoice.PCE:
            post_processor = SafePCEPostProcessor(inner=PCEPostProcessor())
        else:
            post_processor = NoOpPostProcessor()

        opt = QiskitRichQAOAOptimizer(
            name="QAOA-Optuna",
            reps=reps,
            classical_optimizer=classical_optimizer,
            shots=shots,
            seed=self.seed,
            backend=self._make_backend(),
            optimization_level=optimization_level,
            cost_scale=cost_scale_f,
            transpiler=self.transpiler,
            sampler=self.sampler,
            transpile_strategy=transpile_strategy,
            cost_primitive=cost_primitive,
            return_distribution=return_distribution,
            build_initial_state=build_initial_state,
            build_mixer=build_mixer,
            build_callback=build_callback,
            build_aggregation=build_aggregation,
            build_initial_point=build_initial_point,
            postproc=post_processor,
        )

        if mitigation_profile != MitigationProfile.NONE:
            if opt.sampler is None:
                raise ValueError("mitigation_profile requires an explicit sampler to be provided")
            apply_mitigation_profile(sampler=opt.sampler, profile=mitigation_profile)

        trial.set_user_attr("n_qubits", self.qubo.n_vars)
        res, depth, swaps, n_2q = self._run_once(opt=opt, qubo=self.qubo)
        base_obj = res.solution.objective
        trial.set_user_attr("base_objective", base_obj)
        trial.set_user_attr("circuit_depth", depth)
        trial.set_user_attr("circuit_swaps", swaps)
        trial.set_user_attr("circuit_2q_ops", n_2q)
        if res.run.metadata is not None and "qaoa_config" in res.run.metadata:
            trial.set_user_attr("qaoa_config", res.run.metadata["qaoa_config"])

        return (
            self.objective_weight * base_obj
            + self.depth_weight * depth
            + self.swap_weight * swaps
            + self.twoq_weight * n_2q
        )

