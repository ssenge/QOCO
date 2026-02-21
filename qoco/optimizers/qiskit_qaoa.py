from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, Optional

import numpy as np

from qoco.converters.qubo_to_ising import QuboToIsingConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter
from qoco.optimizers.qiskit_reporting import sample_batch_objective_stats
from qoco.postproc.qubo_postproc import NoOpPostProcessor, PostProcessor, Sample, SampleBatch


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(n)]
    inv = {idx: name for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(n)]


def _to_bitstring_msb(*, value: int, n: int) -> str:
    return format(value, f"0{n}b")


def _precompute_z_masks(op: Any) -> tuple[list[int], np.ndarray]:
    """Extract integer Z-bitmasks and real coefficients from a diagonal Pauli operator."""
    z_masks: list[int] = []
    coeffs = np.empty(len(op), dtype=np.float64)
    for idx, (pauli, coeff) in enumerate(zip(op.paulis, op.coeffs)):
        mask = 0
        for i, has_z in enumerate(pauli.z):
            if has_z:
                mask |= 1 << i
        z_masks.append(mask)
        coeffs[idx] = coeff.real
    return z_masks, coeffs


def _eval_diagonal_pauli_cost(state_int: int, z_masks: list[int], coeffs: np.ndarray) -> float:
    """Evaluate ⟨state|H|state⟩ for a diagonal Pauli Hamiltonian on a computational-basis state."""
    total = 0.0
    for mask, c in zip(z_masks, coeffs):
        parity = bin(state_int & mask).count("1") % 2
        total += c * (1 - 2 * parity)
    return total


def _aggregate_costs(
    energies: np.ndarray,
    probs: np.ndarray,
    counts: list[int],
    aggregation: Any,
) -> float:
    """Aggregate per-bitstring energies into a scalar cost.

    aggregation=None  → expectation value
    aggregation=float → CVaR with that alpha
    aggregation=callable → expand to per-shot list and call
    """
    if aggregation is None:
        return float(np.dot(probs, energies))
    if isinstance(aggregation, (int, float)):
        alpha = float(aggregation)
        order = np.argsort(energies)
        sorted_e = energies[order]
        sorted_p = probs[order]
        cum_p = 0.0
        total_e = 0.0
        for e, p in zip(sorted_e, sorted_p):
            take = min(p, alpha - cum_p)
            total_e += e * take
            cum_p += take
            if cum_p >= alpha - 1e-12:
                break
        return total_e / alpha
    expanded: list[float] = []
    for e, c in zip(energies, counts):
        expanded.extend([float(e)] * int(c))
    return float(aggregation(expanded))

def _default_classical_optimizer() -> Any:
    try:
        from qiskit_algorithms.optimizers import COBYLA
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "QiskitQAOAOptimizer requires 'qiskit-algorithms' to construct the default COBYLA optimizer. "
            "Install it or pass classical_optimizer explicitly."
        ) from e
    return COBYLA(maxiter=1)


class TranspileStrategy(Enum):
    DEFAULT = "default"
    SABRE = "sabre"
    SABRE_LAYOUT = "sabre_layout"
    SABRE_ROUTING = "sabre_routing"


class CostPrimitive(Enum):
    SAMPLER = "sampler"
    ESTIMATOR = "estimator"


@dataclass
class QiskitQAOAOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Solve QUBOs using QAOAAnsatz + SamplerV2 + classical outer loop."""

    name: str = "QiskitQAOA"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_ising: Converter[QUBO, tuple[Any, float]] = field(default_factory=QuboToIsingConverter)

    reps: int = 1
    classical_optimizer: Any = field(default_factory=_default_classical_optimizer)

    # Execution / primitives
    sampler: Any | None = None
    estimator: Any | None = None
    shots: int | None = None
    seed: Optional[int] = 0

    # Hardware / compilation configuration
    backend: Any | None = None
    optimization_level: int = 1
    transpiler: Any | None = None  # PassManager-like; if None we build a sensible default
    transpile_strategy: TranspileStrategy = TranspileStrategy.DEFAULT
    cost_primitive: CostPrimitive = CostPrimitive.SAMPLER
    return_distribution: bool | None = None

    # QAOA customization + post-processing
    initial_point: list[float] | None = None
    cost_scale: float | None = None
    postproc: PostProcessor = field(default_factory=NoOpPostProcessor)
    ansatz_hook: Callable[[QUBO, int], tuple[dict[str, Any], list[float] | None]] = field(
        default_factory=lambda: (lambda _qubo, _n: ({}, None))
    )

    def config_metadata(self) -> dict[str, Any]:
        backend = type(self.backend).__name__ if self.backend is not None else None
        classical = type(self.classical_optimizer).__name__ if self.classical_optimizer is not None else None
        return {
            "optimizer_type": type(self).__name__,
            "name": self.name,
            "backend": backend,
            "reps": self.reps,
            "shots": self.shots,
            "seed": self.seed,
            "optimization_level": self.optimization_level,
            "transpile_strategy": self.transpile_strategy.value,
            "cost_primitive": self.cost_primitive.value,
            "return_distribution": self.return_distribution,
            "cost_scale": self.cost_scale,
            "classical_optimizer": classical,
        }

    def encode_qubo(self, qubo: QUBO) -> tuple[Any, float]:
        return self.qubo_to_ising.convert(qubo)

    def make_sampler(self) -> Any:
        if self.sampler is not None:
            return self.sampler
        from qiskit_aer.primitives import SamplerV2

        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.shots is not None:
            kwargs["default_shots"] = self.shots
        return SamplerV2(**kwargs)  # Aer fallback

    def make_estimator(self) -> Any:
        if self.estimator is not None:
            return self.estimator
        from qiskit_aer.primitives import EstimatorV2

        return EstimatorV2()

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        op, _offset = self.encode_qubo(qubo)
        if self.cost_scale is not None:
            op = op * self.cost_scale

        n = qubo.n_vars
        hook_kwargs, hook_initial_point = self.ansatz_hook(qubo, n)

        ansatz = QAOAAnsatz(
            cost_operator=op,
            reps=self.reps,
            initial_state=hook_kwargs.get("initial_state"),
            mixer_operator=hook_kwargs.get("mixer"),
        )
        ansatz.measure_all()

        if self.transpiler is not None:
            transpiled = self.transpiler.run(ansatz)
        elif self.backend is not None:
            layout_method: str | None = None
            routing_method: str | None = None
            if self.transpile_strategy == TranspileStrategy.SABRE:
                layout_method = "sabre"
                routing_method = "sabre"
            elif self.transpile_strategy == TranspileStrategy.SABRE_LAYOUT:
                layout_method = "sabre"
            elif self.transpile_strategy == TranspileStrategy.SABRE_ROUTING:
                routing_method = "sabre"

            transpiled = generate_preset_pass_manager(
                target=self.backend.target,
                optimization_level=self.optimization_level,
                layout_method=layout_method,
                routing_method=routing_method,
            ).run(ansatz)
        else:
            from qiskit_aer import AerSimulator

            transpiled = generate_preset_pass_manager(
                backend=AerSimulator(),
                optimization_level=self.optimization_level,
            ).run(ansatz)

        z_masks, coeffs = _precompute_z_masks(op)
        aggregation = hook_kwargs.get("aggregation")
        callback_fn = hook_kwargs.get("callback")
        cost_trace: list[float] = []
        final_distribution_enabled = (
            self.return_distribution
            if self.return_distribution is not None
            else (self.cost_primitive == CostPrimitive.SAMPLER)
        )

        initial_point = hook_initial_point if hook_initial_point is not None else self.initial_point
        if initial_point is None:
            rng = np.random.default_rng(self.seed if self.seed is not None else 0)
            initial_point = rng.uniform(0, 2 * np.pi, transpiled.num_parameters).tolist()

        eval_count = [0]

        def objective(params: np.ndarray) -> float:
            if self.cost_primitive == CostPrimitive.ESTIMATOR:
                estimator = self.make_estimator()
                isa_op = op.apply_layout(transpiled.layout)
                est_job = estimator.run([(transpiled, isa_op, list(params))])
                est_result = est_job.result()[0]
                cost = float(est_result.data.evs)
            else:
                sampler = self.make_sampler()
                job = sampler.run([(transpiled, list(params))])
                result = job.result()[0]
                counts = result.data.meas.get_int_counts()
                total_shots = sum(counts.values())

                state_energies: list[float] = []
                state_counts: list[int] = []
                for state_int, count in counts.items():
                    state_energies.append(_eval_diagonal_pauli_cost(state_int, z_masks, coeffs))
                    state_counts.append(count)

                energies = np.array(state_energies)
                probs = np.array(state_counts, dtype=np.float64) / total_shots
                cost = _aggregate_costs(energies, probs, state_counts, aggregation)
            cost_trace.append(float(cost))

            eval_count[0] += 1
            if callback_fn is not None and callable(callback_fn):
                callback_fn(eval_count[0], params, cost, None)
            return cost

        optimizer_timestamp_start = datetime.now(timezone.utc)
        if self.seed is not None:
            np.random.seed(self.seed)
        opt_result = self.classical_optimizer.minimize(objective, np.array(initial_point))
        optimizer_timestamp_end = datetime.now(timezone.utc)

        # Final sampling with optimal parameters (needed to produce a Solution).
        sampler = self.make_sampler()
        final_job = sampler.run([(transpiled, list(opt_result.x))])
        final_result = final_job.result()[0]
        final_counts = final_result.data.meas.get_int_counts()
        total_shots = sum(final_counts.values())

        samples = [
            Sample(_to_bitstring_msb(value=state_int, n=n), weight=count / total_shots)
            for state_int, count in final_counts.items()
        ]
        batch = SampleBatch(n=n, samples=samples, bit_order="qiskit")
        pre_stats = sample_batch_objective_stats(qubo=qubo, batch=batch)

        post = self.postproc.run(qubo=qubo, batch=batch)
        x = np.asarray(post.x, dtype=int).reshape(-1)
        x_list = x.tolist()
        obj = post.objective

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: x_list[i] for i in range(n)}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=obj,
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
        )

        run_meta: dict[str, Any] = {"cost_trace": cost_trace, "qaoa_config": self.config_metadata()}
        if cost_trace:
            run_meta["nfev"] = len(cost_trace)
            run_meta["best_cost"] = min(cost_trace)
            run_meta["best_iter"] = cost_trace.index(min(cost_trace)) + 1
            run_meta["final_cost"] = cost_trace[-1]

        run_meta.update(dict(pre_stats))
        run_meta["post_objective"] = obj
        run_meta["post_bitstring"] = post.bitstring
        if "pre_best_objective" in pre_stats:
            run_meta["post_delta_vs_pre_best"] = obj - pre_stats["pre_best_objective"]
        if final_distribution_enabled:
            run_meta["final_shots"] = total_shots
            run_meta["final_counts_int"] = {k: v for k, v in final_counts.items()}
            run_meta["final_distribution_int"] = {k: v / total_shots for k, v in final_counts.items()}
            try:
                counts_bin = final_result.data.meas.get_counts()
                run_meta["final_distribution_bin"] = {
                    k: v / total_shots for k, v in counts_bin.items()
                }
            except Exception:
                pass
        return solution, OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
            metadata=run_meta,
        )

