from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Generic, Literal, Optional

import numpy as np
from qiskit import QuantumCircuit  # type: ignore
from qiskit.circuit import ParameterVector  # type: ignore
from qiskit.quantum_info import SparsePauliOp, Statevector  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore
from qiskit_aer.primitives import SamplerV2  # type: ignore
from qiskit_algorithms.optimizers import COBYLA  # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore

from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter
from qoco.postproc.qubo_postproc import evaluate_qubo, greedy_1flip_local_search

PauliBasis = Literal["X", "Y", "Z"]
CorrelatorEvaluation = Literal["shots", "statevector"]


@dataclass(frozen=True)
class PauliCorrelationOperator:
    """k-body operator of the paper's Π^(k) family.

    Each logical variable is represented as a k-body correlator in a single basis (X or Y or Z),
    supported on `targets`.
    """

    basis: PauliBasis
    targets: tuple[int, ...]


@dataclass(frozen=True)
class PauliCorrelationSchema:
    n_qubits: int
    k: int
    operators: list[PauliCorrelationOperator]

    @property
    def capacity(self) -> int:
        return 3 * int(_n_choose_k(self.n_qubits, self.k))


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return int(np.math.comb(int(n), int(k)))


def _build_paper_schema(*, n_qubits: int, k: int, num_vars: int) -> PauliCorrelationSchema:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if k <= 0 or k > n_qubits:
        raise ValueError("k must satisfy 1 <= k <= n_qubits")
    if num_vars <= 0:
        raise ValueError("num_vars must be positive")

    supports = list(combinations(range(int(n_qubits)), int(k)))
    ops: list[PauliCorrelationOperator] = []
    for basis in ("X", "Y", "Z"):
        for t in supports:
            ops.append(PauliCorrelationOperator(basis=basis, targets=tuple(int(i) for i in t)))

    if int(num_vars) > len(ops):
        raise ValueError(
            f"Too many variables for (n={n_qubits}, k={k}) PCE. "
            f"Need num_vars={num_vars}, capacity={len(ops)}."
        )
    return PauliCorrelationSchema(n_qubits=int(n_qubits), k=int(k), operators=ops[: int(num_vars)])


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


def _qubo_to_ising_coeffs(*, qubo: QUBO, tol: float = 0.0) -> tuple[np.ndarray, list[tuple[int, int, float]], float]:
    """Return (h, J_edges, offset) for E(z) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j.

    Here z_i in {+1,-1} relates to QUBO bits x_i in {0,1} via z = 1 - 2x.
    """
    Q = np.asarray(qubo.Q, dtype=float)
    n = int(Q.shape[0])
    diag = np.asarray(np.diag(Q), dtype=float)

    quad_pairs: list[tuple[tuple[int, int], float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(Q[i, j]) + float(Q[j, i])
            if abs(c) > float(tol):
                quad_pairs.append(((int(i), int(j)), float(c)))

    offset = float(qubo.offset)
    offset += 0.5 * float(np.sum(diag))
    offset += 0.25 * float(sum(c for (_ij, c) in quad_pairs))

    h = np.zeros((n,), dtype=float)
    h -= 0.5 * diag
    for (i, j), c in quad_pairs:
        h[int(i)] -= 0.25 * float(c)
        h[int(j)] -= 0.25 * float(c)

    J_edges: list[tuple[int, int, float]] = []
    for (i, j), c in quad_pairs:
        J = 0.25 * float(c)
        if abs(J) > float(tol):
            J_edges.append((int(i), int(j), float(J)))

    return h, J_edges, float(offset)


def _apply_measurement_basis(*, qc: QuantumCircuit, basis: PauliBasis) -> None:
    if basis == "Z":
        return
    if basis == "X":
        qc.h(range(qc.num_qubits))
        return
    if basis == "Y":
        for q in range(qc.num_qubits):
            qc.sdg(q)
            qc.h(q)
        return
    raise ValueError(f"Unknown basis: {basis}")


def _bit_eig(bit: str) -> int:
    return 1 if bit == "0" else -1


def _targets_product(*, bitstring: str, targets: tuple[int, ...]) -> int:
    # Qiskit bitstrings are big-endian in display, little-endian by qubit index: rightmost is qubit 0.
    prod = 1
    for q in targets:
        prod *= _bit_eig(bitstring[-1 - int(q)])
    return int(prod)


def _estimate_correlators(
    sampler: Any,
    transpiler: Any,
    ansatz: QuantumCircuit,
    params: list[Any],
    values: np.ndarray,
    schema: PauliCorrelationSchema,
    shots: int,
) -> np.ndarray:
    if len(params) != int(values.shape[0]):
        raise ValueError("parameter length mismatch")

    idx_by_basis: dict[PauliBasis, list[int]] = {"X": [], "Y": [], "Z": []}
    for i, op in enumerate(schema.operators):
        idx_by_basis[op.basis].append(int(i))

    out = np.zeros((len(schema.operators),), dtype=float)
    bindings = {p: float(values[i]) for i, p in enumerate(params)}
    base = ansatz.assign_parameters(bindings, inplace=False)

    for basis, indices in idx_by_basis.items():
        if not indices:
            continue
        qc = base.copy()
        _apply_measurement_basis(qc=qc, basis=basis)
        qc.measure_all()
        qc = transpiler.run(qc)

        pub = sampler.run([qc], shots=int(shots)).result()[0]
        counts = pub.data.meas.get_counts()
        total = float(sum(int(v) for v in counts.values())) or float(shots)

        for idx in indices:
            targets = schema.operators[int(idx)].targets
            s = 0.0
            for bitstring, c in counts.items():
                s += float(int(c)) * float(_targets_product(bitstring=str(bitstring), targets=targets))
            out[int(idx)] = float(s / total)

    return out


def _pauli_label(n_qubits: int, basis: PauliBasis, targets: tuple[int, ...]) -> str:
    # Qiskit Pauli labels are big-endian: rightmost char is qubit 0.
    s = ["I"] * int(n_qubits)
    for q in targets:
        s[int(n_qubits) - 1 - int(q)] = str(basis)
    return "".join(s)


def _estimate_correlators_statevector(
    ansatz: QuantumCircuit,
    params: list[Any],
    values: np.ndarray,
    schema: PauliCorrelationSchema,
) -> np.ndarray:
    if len(params) != int(values.shape[0]):
        raise ValueError("parameter length mismatch")
    bindings = {p: float(values[i]) for i, p in enumerate(params)}
    qc = ansatz.assign_parameters(bindings, inplace=False)
    sv = Statevector.from_instruction(qc)
    out = np.zeros((len(schema.operators),), dtype=float)
    for i, op in enumerate(schema.operators):
        label = _pauli_label(n_qubits=int(schema.n_qubits), basis=op.basis, targets=op.targets)
        obs = SparsePauliOp.from_list([(label, 1.0)])
        out[int(i)] = float(np.real(sv.expectation_value(obs)))
    return out


@dataclass
class QiskitPauliCorrelationEncodingOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Pauli Correlation Encoding (PCE) optimizer (Sciorilli et al., 2024).

    MVP implementation:
    - Paper-style Π^(k) schema with 3 measurement settings (X/Y/Z).
    - Nonlinear surrogate loss using z_i = tanh(alpha * <Π_i>).
    - Classical black-box optimizer (COBYLA).
    - Decode via sign(<Π_i>) and refine with greedy 1-flip local search.
    """

    name: str = "QiskitPCE"
    converter: Any = field(default_factory=IdentityConverter)

    # PCE encoding
    n_qubits: int = 10
    k: int = 2

    # Ansatz / training
    layers: int = 2
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=30))
    seed: Optional[int] = 0
    num_restarts: int = 1

    # Loss shaping (paper-inspired)
    alpha: float | None = None
    reg_strength: float = 0.0

    # Sampling / primitives
    sampler: Any | None = None
    shots_per_setting: int = 200
    evaluation: CorrelatorEvaluation = "shots"
    optimization_level: int = 1
    transpiler: Any | None = None

    # Output refinement
    refine_local_search: bool = True
    refine_local_search_max_iters: int = 200

    def make_sampler(self) -> Any:
        if self.sampler is not None:
            return self.sampler
        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        kwargs["default_shots"] = int(self.shots_per_setting)
        return SamplerV2(**kwargs)

    def make_transpiler(self) -> Any:
        if self.transpiler is not None:
            return self.transpiler
        return generate_preset_pass_manager(backend=AerSimulator(), optimization_level=int(self.optimization_level))

    def _build_ansatz(self, *, n_qubits: int, layers: int) -> tuple[QuantumCircuit, list[Any]]:
        rng = np.random.default_rng(self.seed)
        n = int(n_qubits)
        L = int(layers)
        params = ParameterVector("θ", L * n * 2)
        qc = QuantumCircuit(n)
        t = 0
        for layer in range(L):
            for q in range(n):
                qc.ry(params[t + 0], q)
                qc.rz(params[t + 1], q)
                t += 2
            start = 0 if layer % 2 == 0 else 1
            for q in range(start, n - 1, 2):
                qc.cx(q, q + 1)
        init = 0.01 * rng.standard_normal(int(len(params))).astype(float)
        return qc, list(params), init

    def _loss(
        self,
        *,
        h: np.ndarray,
        J_edges: list[tuple[int, int, float]],
        offset: float,
        corr: np.ndarray,
    ) -> float:
        alpha = float(self._effective_alpha())
        z = np.tanh(alpha * np.asarray(corr, dtype=float))
        e = float(offset) + float(np.dot(h, z))
        for i, j, Jij in J_edges:
            e += float(Jij) * float(z[int(i)] * z[int(j)])
        if float(self.reg_strength) > 0.0:
            e += float(self.reg_strength) * float(np.mean(z * z) ** 2)
        return float(e)

    def _effective_alpha(self) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        n = int(self.n_qubits)
        k = int(self.k)
        return float(max(1.0, n ** max(1, (k // 2))))

    def _decode_x(self, corr: np.ndarray) -> np.ndarray:
        z = np.sign(np.asarray(corr, dtype=float))
        z[z == 0.0] = 1.0
        x = (1.0 - z) / 2.0
        return np.asarray(np.rint(x), dtype=int)

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        if self.seed is not None:
            np.random.seed(int(self.seed))

        n_vars = int(qubo.n_vars)
        schema = _build_paper_schema(n_qubits=int(self.n_qubits), k=int(self.k), num_vars=n_vars)
        ansatz, params, x0 = self._build_ansatz(n_qubits=int(schema.n_qubits), layers=int(self.layers))

        h, J_edges, ising_offset = _qubo_to_ising_coeffs(qubo=qubo, tol=0.0)
        if h.shape[0] != n_vars:
            raise ValueError("Ising coefficient size mismatch")

        sampler = self.make_sampler()
        transpiler = self.make_transpiler()

        def objective(theta: np.ndarray) -> float:
            values = np.asarray(theta, dtype=float).reshape(-1)
            if self.evaluation == "statevector":
                corr = _estimate_correlators_statevector(ansatz=ansatz, params=params, values=values, schema=schema)
            else:
                corr = _estimate_correlators(
                    sampler=sampler,
                    transpiler=transpiler,
                    ansatz=ansatz,
                    params=params,
                    values=values,
                    schema=schema,
                    shots=int(self.shots_per_setting),
                )
            return self._loss(h=h, J_edges=J_edges, offset=float(ising_offset), corr=corr)

        optimizer_timestamp_start = datetime.now(timezone.utc)
        best_res = None
        best_fun = float("inf")
        rng = np.random.default_rng(self.seed)
        for r in range(int(max(1, self.num_restarts))):
            if r == 0:
                x_init = np.asarray(x0, dtype=float)
            else:
                x_init = 0.01 * rng.standard_normal(int(len(x0))).astype(float)
            res = self.classical_optimizer.minimize(fun=objective, x0=x_init)  # type: ignore[no-untyped-call]
            f = float(getattr(res, "fun", float("inf")))
            if f < best_fun:
                best_fun = f
                best_res = res
        optimizer_timestamp_end = datetime.now(timezone.utc)

        res = best_res
        best_theta = np.asarray(getattr(res, "x", x0) if res is not None else x0, dtype=float).reshape(-1)
        if self.evaluation == "statevector":
            corr = _estimate_correlators_statevector(ansatz=ansatz, params=params, values=best_theta, schema=schema)
        else:
            corr = _estimate_correlators(
                sampler=sampler,
                transpiler=transpiler,
                ansatz=ansatz,
                params=params,
                values=best_theta,
                schema=schema,
                shots=int(self.shots_per_setting),
            )
        x = self._decode_x(corr=corr)

        if int(x.shape[0]) != n_vars:
            raise ValueError("decoded x has wrong length")

        if self.refine_local_search:
            x2, _obj2, _it = greedy_1flip_local_search(
                qubo=qubo,
                x0=x,
                max_iters=int(self.refine_local_search_max_iters),
                first_improvement=True,
            )
            x = np.asarray(x2, dtype=int)

        obj = float(evaluate_qubo(qubo=qubo, x=x))
        names = _names_by_index(dict(qubo.var_map), n_vars)
        var_values = {names[i]: int(x[i]) for i in range(n_vars)}

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

