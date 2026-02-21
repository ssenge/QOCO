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

from qoco.converters.identity import IdentityConverter
from qoco.converters.qubo_to_ising import QuboToIsingConverter
from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.optimizers.qiskit_qaoa import _names_by_index, _to_bitstring_msb
from qoco.postproc.qubo_postproc import NoOpPostProcessor, PostProcessor, Sample, SampleBatch


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
        if len(s) < n:
            s = s.zfill(n)
        if len(s) > n:
            s = s[-n:]
        return s
    return _to_bitstring_msb(value=value, n=n)


def _sample_batch_from_qaoa_result(*, result: Any, n: int) -> SampleBatch:
    eigenstate = getattr(result, "eigenstate", None)
    if eigenstate is not None:
        binary_probabilities = getattr(eigenstate, "binary_probabilities", None)
        if callable(binary_probabilities):
            samples = [
                Sample(_normalize_bitstring(bitstring=k, n=n), weight=float(w))
                for k, w in binary_probabilities().items()
            ]
            if samples:
                return SampleBatch(n=n, samples=samples, bit_order="qiskit")

        items = getattr(eigenstate, "items", None)
        if callable(items):
            samples = [Sample(_to_bitstring_msb(value=int(k), n=n), weight=float(w)) for k, w in eigenstate.items()]
            if samples:
                return SampleBatch(n=n, samples=samples, bit_order="qiskit")

    bm = getattr(result, "best_measurement", None)
    if bm is None:
        raise ValueError("QAOA result has no eigenstate and no best_measurement")
    bitstring = _normalize_bitstring(bitstring=bm["bitstring"], n=n)
    return SampleBatch(n=n, samples=[Sample(bitstring, weight=1.0)], bit_order="qiskit")


def _drop_idle_wires(circuit: Any) -> Any:
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


def _connected_subgraph(cm: Any, n: int) -> list[int]:
    from collections import deque

    graph = cm.graph.to_undirected()
    all_nodes = sorted(graph.node_indices())
    if not all_nodes:
        raise ValueError("Coupling map has no qubits.")
    visited: list[int] = []
    seen: set[int] = set()
    queue: deque[int] = deque([all_nodes[0]])
    seen.add(all_nodes[0])
    while queue and len(visited) < n:
        node = queue.popleft()
        visited.append(node)
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(visited) < n:
        raise ValueError(f"Backend coupling map has only {len(visited)} reachable qubits but {n} are required.")
    return visited[:n]


def _make_fixed_width_transpiler(*, backend: Any, n: int, optimization_level: int) -> Any:
    target = backend.target
    full_cm = target.build_coupling_map()
    qubits = _connected_subgraph(full_cm, n)
    sub_cm = full_cm.reduce(qubits)
    basis_gates = sorted(target.operation_names)

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        coupling_map=sub_cm,
        basis_gates=basis_gates,
        initial_layout=list(range(n)),
    )
    return _FixedWidthTranspiler(base=pm, expected_width=n)


@dataclass(frozen=True)
class _FixedWidthTranspiler:
    base: Any
    expected_width: int

    def run(self, circuit: Any, **kwargs: Any) -> Any:
        out = self.base.run(circuit, **kwargs)
        out = _drop_idle_wires(out)
        if out.num_qubits != self.expected_width:
            raise ValueError(
                f"Transpilation produced {out.num_qubits} qubits but QAOA requires exactly {self.expected_width} "
                f"(= QUBO n_vars). The backend transpiler widened the circuit beyond the operator width."
            )
        return out


@dataclass
class QiskitLegacyQAOAOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Legacy QAOA path based on `qiskit_algorithms.minimum_eigensolvers.QAOA`.

    Kept for compatibility with wrappers in `qiskit_optimization` that require a
    `MinimumEigensolver` instance (warm-start, recursive minimum eigen).
    """

    name: str = "QiskitLegacyQAOA"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_ising: Converter[QUBO, tuple[Any, float]] = field(default_factory=QuboToIsingConverter)
    qubo_to_qp: Converter[QUBO, Any] = field(default_factory=QuboToQuadraticProgramConverter)
    use_quadratic_program: bool = False

    reps: int = 1
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=50))

    sampler: Any | None = None
    shots: int | None = None
    seed: Optional[int] = 0

    backend: Any | None = None
    optimization_level: int = 1
    transpiler: Any | None = None

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
            kwargs["seed"] = self.seed
        if self.shots is not None:
            kwargs["default_shots"] = self.shots
        return SamplerV2(**kwargs)

    def make_qaoa(self, *, qubo: QUBO, n: int) -> QAOA:
        sampler = self.make_sampler()

        if self.transpiler is not None:
            transpiler = self.transpiler
        elif self.backend is None:
            transpiler = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=self.optimization_level)
        else:
            transpiler = _make_fixed_width_transpiler(
                backend=self.backend,
                n=n,
                optimization_level=self.optimization_level,
            )

        hook_kwargs, hook_initial_point = self.ansatz_hook(qubo, n)
        qaoa_kwargs = dict(self.qaoa_kwargs)
        qaoa_kwargs.update(dict(hook_kwargs))
        initial_point = hook_initial_point if hook_initial_point is not None else self.initial_point

        qaoa = QAOA(
            sampler=sampler,
            optimizer=self.classical_optimizer,
            reps=self.reps,
            transpiler=transpiler,
            **dict(qaoa_kwargs),
        )
        if initial_point is not None:
            qaoa.initial_point = list(initial_point)
        return qaoa

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        op, offset = self.encode_qubo(qubo)
        if self.cost_scale is not None:
            op = op * self.cost_scale
            offset = offset * self.cost_scale

        n = qubo.n_vars
        qaoa = self.make_qaoa(qubo=qubo, n=n)

        if self.seed is not None:
            np.random.seed(self.seed)

        optimizer_timestamp_start = datetime.now(timezone.utc)
        res = qaoa.compute_minimum_eigenvalue(op)
        optimizer_timestamp_end = datetime.now(timezone.utc)

        batch = _sample_batch_from_qaoa_result(result=res, n=n)
        post = self.postproc.run(qubo=qubo, batch=batch)
        x = np.asarray(post.x, dtype=int).reshape(-1)
        obj = post.objective

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=obj,
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
        )
        return solution, OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
        )

