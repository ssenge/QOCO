from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generic, Optional

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qoco.converters.qubo_to_ising import QuboToIsingConverter
from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter
from qoco.postproc.qubo_postproc import NoOpPostProcessor, PostProcessor, Sample, SampleBatch


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


def _bitstring_to_x(bitstring: str, n: int) -> np.ndarray:
    s = bitstring.strip()
    if s.startswith("0b"):
        s = s[2:]
    if len(s) != int(n):
        raise ValueError("bitstring length mismatch")
    return np.array([1 if ch == "1" else 0 for ch in s], dtype=int)


def _to_bitstring_msb(*, value: int, n: int) -> str:
    return format(int(value), f"0{int(n)}b")


def _normalize_bitstring(*, bitstring: Any, n: int) -> str:
    try:
        value = int(bitstring)
    except Exception:
        s = str(bitstring).strip()
        if s.startswith("0b"):
            s = s[2:]
        s = "".join(ch for ch in s if ch in ("0", "1"))
        if not s:
            raise ValueError("cannot parse bitstring")
        if len(s) < int(n):
            s = s.zfill(int(n))
        if len(s) > int(n):
            s = s[-int(n) :]
        return s
    return _to_bitstring_msb(value=value, n=int(n))


def _sample_batch_from_qaoa_result(*, result: Any, n: int) -> SampleBatch:
    eigenstate = getattr(result, "eigenstate", None)
    if eigenstate is not None:
        binary_probabilities = getattr(eigenstate, "binary_probabilities", None)
        if callable(binary_probabilities):
            samples = [
                Sample(_normalize_bitstring(bitstring=k, n=int(n)), weight=float(w))
                for k, w in binary_probabilities().items()
            ]
            if samples:
                return SampleBatch(n=int(n), samples=samples, bit_order="qiskit")

        items = getattr(eigenstate, "items", None)
        if callable(items):
            samples = [
                Sample(_to_bitstring_msb(value=int(k), n=int(n)), weight=float(w))
                for k, w in eigenstate.items()
            ]
            if samples:
                return SampleBatch(n=int(n), samples=samples, bit_order="qiskit")

    bm = getattr(result, "best_measurement", None)
    if bm is None:
        raise ValueError("QAOA result has no eigenstate and no best_measurement")
    bitstring = _normalize_bitstring(bitstring=bm["bitstring"], n=int(n))
    return SampleBatch(n=int(n), samples=[Sample(bitstring, weight=1.0)], bit_order="qiskit")


def _drop_idle_wires(circuit: Any) -> Any:
    """Drop idle qubits/clbits from a circuit.

    Qiskit 2.0's backend-native compilation can materialize circuits on full device width even
    if the logical circuit uses only `n` qubits. QAOA requires ansatz width == operator width.
    """

    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.circuit import Qubit, Clbit

    dag = circuit_to_dag(circuit)
    idle = list(dag.idle_wires())
    idle_qubits = [w for w in idle if isinstance(w, Qubit)]
    idle_clbits = [w for w in idle if isinstance(w, Clbit)]
    if idle_qubits:
        dag.remove_qubits(*idle_qubits)
    if idle_clbits:
        dag.remove_clbits(*idle_clbits)
    return dag_to_circuit(dag)


@dataclass(frozen=True)
class _DropIdleWiresTranspiler:
    base: Any

    def run(self, circuit: Any, **kwargs: Any) -> Any:
        out = self.base.run(circuit, **kwargs)
        return _drop_idle_wires(out)


def _is_ibm_runtime_primitive(obj: Any) -> bool:
    # Heuristic: IBM runtime primitives live in `qiskit_ibm_runtime.*`.
    # We avoid importing qiskit_ibm_runtime here so Aer-only installs still work.
    return type(obj).__module__.startswith("qiskit_ibm_runtime")


@dataclass
class QiskitQAOAOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Solve QUBOs using Qiskit's QAOA.

    Design constraints (per project convention):
    - Do not hardcode basis gates or backend-specific gate replacements here.
    - Backend-specific compilation should be configured outside (by passing `backend` and/or `transpiler`).
    """

    name: str = "QiskitQAOA"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_ising: Converter[QUBO, tuple[Any, float]] = field(default_factory=QuboToIsingConverter)
    qubo_to_qp: Converter[QUBO, Any] = field(default_factory=QuboToQuadraticProgramConverter)
    use_quadratic_program: bool = False

    reps: int = 1
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=50))

    # Execution / primitives
    sampler: Any | None = None
    shots: int | None = None
    seed: Optional[int] = 0

    # Hardware / compilation configuration
    backend: Any | None = None
    optimization_level: int = 1
    transpiler: Any | None = None  # PassManager-like; if None we build a sensible default

    # QAOA customization + post-processing
    initial_point: list[float] | None = None
    qaoa_kwargs: dict[str, Any] = field(default_factory=dict)
    cost_scale: float | None = None
    postproc: PostProcessor = field(default_factory=NoOpPostProcessor)
    ansatz_hook: Callable[[QUBO, int], tuple[dict[str, Any], list[float] | None]] = field(
        default_factory=lambda: (lambda _qubo, _n: ({}, None))
    )

    def encode_qubo(self, qubo: QUBO) -> tuple[Any, float]:
        if self.use_quadratic_program:
            qp = self.qubo_to_qp.convert(qubo)
            return qp.to_ising()
        return self.qubo_to_ising.convert(qubo)

    def make_sampler(self) -> Any:
        if self.sampler is not None:
            return self.sampler
        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if self.shots is not None:
            kwargs["default_shots"] = int(self.shots)
        return SamplerV2(**kwargs)  # Aer fallback

    def make_qaoa(
        self,
        *,
        sampler: Any,
        transpiler: Any | None,
        qaoa_kwargs: dict[str, Any],
        initial_point: list[float] | None,
    ) -> QAOA:
        qaoa = QAOA(
            sampler=sampler,
            optimizer=self.classical_optimizer,
            reps=int(self.reps),
            transpiler=transpiler,
            **dict(qaoa_kwargs),
        )
        if initial_point is not None:
            qaoa.initial_point = list(initial_point)
        return qaoa

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        op, offset = self.encode_qubo(qubo)
        if self.cost_scale is not None:
            op = op * float(self.cost_scale)
            offset = float(offset) * float(self.cost_scale)

        n = int(qubo.n_vars)
        sampler = self.make_sampler()

        transpiler: Any | None
        if self.transpiler is not None:
            transpiler = self.transpiler
        elif self.backend is None:
            transpiler = generate_preset_pass_manager(
                backend=AerSimulator(),
                optimization_level=int(self.optimization_level),
            )
        else:
            pm = generate_preset_pass_manager(
                backend=self.backend,
                optimization_level=int(self.optimization_level),
                initial_layout=list(range(n)) if n > 0 else None,
            )
            transpiler = _DropIdleWiresTranspiler(pm)

        hook_kwargs, hook_initial_point = self.ansatz_hook(qubo, n)
        qaoa_kwargs = dict(self.qaoa_kwargs)
        qaoa_kwargs.update(dict(hook_kwargs))
        initial_point = hook_initial_point if hook_initial_point is not None else self.initial_point

        qaoa = self.make_qaoa(
            sampler=sampler,
            transpiler=transpiler,
            qaoa_kwargs=qaoa_kwargs,
            initial_point=initial_point,
        )
        if self.seed is not None:
            # Qiskit algorithms use numpy global RNG internally; seed is best-effort.
            np.random.seed(int(self.seed))

        optimizer_timestamp_start = datetime.now(timezone.utc)
        res = qaoa.compute_minimum_eigenvalue(op)
        optimizer_timestamp_end = datetime.now(timezone.utc)

        batch = _sample_batch_from_qaoa_result(result=res, n=n)
        post = self.postproc.run(qubo=qubo, batch=batch)
        x = np.asarray(post.x, dtype=int).reshape(-1)
        obj = float(post.objective)

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=float(obj),
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
        )
        return solution, OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
        )

