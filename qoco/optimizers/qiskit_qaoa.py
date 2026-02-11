from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, Optional

import numpy as np
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
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
            samples = [Sample(_to_bitstring_msb(value=int(k), n=int(n)), weight=float(w)) for k, w in eigenstate.items()]
            if samples:
                return SampleBatch(n=int(n), samples=samples, bit_order="qiskit")

    bm = getattr(result, "best_measurement", None)
    if bm is None:
        raise ValueError("QAOA result has no eigenstate and no best_measurement")
    bitstring = _normalize_bitstring(bitstring=bm["bitstring"], n=int(n))
    return SampleBatch(n=int(n), samples=[Sample(bitstring, weight=1.0)], bit_order="qiskit")


_STANDARD_GATE_NAMES = set(get_standard_gate_name_mapping().keys())


@dataclass
class QiskitQAOAOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Solve QUBOs using Qiskit's QAOA.

    Notes:
    - By default, uses Aer SamplerV2 if no sampler is provided.
    - If a backend is provided (e.g. IBM QPU via runtime), the intended workflow is to
      let the primitive/runtime handle hardware compilation. We therefore do NOT use
      `generate_preset_pass_manager(backend=...)` by default, because it can inflate
      circuits to full device width (ancilla allocation), which breaks qiskit-algorithms'
      circuit/observable consistency checks.
    - For Aer, we DO use `generate_preset_pass_manager(backend=AerSimulator(), ...)` by
      default to ensure composite instructions are unrolled into supported gates.
    - By default, converts QUBO -> Ising directly (SparsePauliOp + offset). If
      `use_quadratic_program=True`, uses Qiskit's QuadraticProgram.to_ising() path instead.
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

    # Hardware-aware transpilation
    backend: Any | None = None
    optimization_level: int = 1
    transpiler: Any | None = None  # a Qiskit PassManager-like object

    # QAOA customization
    initial_point: list[float] | None = None
    qaoa_kwargs: dict[str, Any] = field(default_factory=dict)
    cost_scale: float | None = None
    postproc: PostProcessor = field(default_factory=NoOpPostProcessor)

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

    def make_transpiler(self) -> Any:
        if self.transpiler is not None:
            return self.transpiler
        backend = self.backend

        # Default: local Aer simulation.
        if backend is None:
            if self.sampler is None:
                return generate_preset_pass_manager(
                    backend=AerSimulator(),
                    optimization_level=int(self.optimization_level),
                )
            return generate_preset_pass_manager(optimization_level=int(self.optimization_level))

        # Hardware: keep circuit width at n, but decompose composite instructions into
        # ISA-compatible gates.
        ops = list(getattr(getattr(backend, "target", None), "operation_names", []))
        basis_gates = [str(name) for name in ops if str(name) in _STANDARD_GATE_NAMES]
        if not basis_gates:
            basis_gates = ["rz", "sx", "x", "cx"]
        return generate_preset_pass_manager(
            optimization_level=int(self.optimization_level),
            basis_gates=basis_gates,
        )

    def make_qaoa(self) -> QAOA:
        qaoa = QAOA(
            sampler=self.make_sampler(),
            optimizer=self.classical_optimizer,
            reps=int(self.reps),
            transpiler=self.make_transpiler(),
            **dict(self.qaoa_kwargs),
        )
        if self.initial_point is not None:
            qaoa.initial_point = list(self.initial_point)
        return qaoa

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        op, offset = self.encode_qubo(qubo)
        if self.cost_scale is not None:
            op = op * float(self.cost_scale)
            offset = float(offset) * float(self.cost_scale)

        qaoa = self.make_qaoa()

        optimizer_timestamp_start = datetime.now(timezone.utc)
        res = qaoa.compute_minimum_eigenvalue(op)
        optimizer_timestamp_end = datetime.now(timezone.utc)
        n = int(qubo.n_vars)
        batch = _sample_batch_from_qaoa_result(result=res, n=n)
        post = self.postproc.run(qubo=qubo, batch=batch)
        x = np.asarray(post.x, dtype=int).reshape(-1)
        obj = float(post.objective)

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        solution = Solution(
            status=Status.FEASIBLE,  # let's cross fingers ;-)
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

