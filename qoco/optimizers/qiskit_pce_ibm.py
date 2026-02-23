from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Generic, Optional

import numpy as np

from qoco.converters.identity import IdentityConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.postproc.qubo_postproc import evaluate_qubo, greedy_1flip_local_search


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(n)]
    inv = {idx: name for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(n)]


def _qubo_to_ising_coeffs(*, qubo: QUBO, tol: float = 0.0) -> tuple[np.ndarray, list[tuple[int, int, float]], float]:
    Q = np.asarray(qubo.Q, dtype=float)
    n = Q.shape[0]
    diag = np.diag(Q).astype(float)

    quad_pairs: list[tuple[tuple[int, int], float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(Q[i, j]) + float(Q[j, i])
            if abs(c) > tol:
                quad_pairs.append(((i, j), c))

    offset = float(qubo.offset)
    offset += 0.5 * float(np.sum(diag))
    offset += 0.25 * float(sum(c for (_ij, c) in quad_pairs))

    h = np.zeros((n,), dtype=float)
    h -= 0.5 * diag
    for (i, j), c in quad_pairs:
        h[i] -= 0.25 * c
        h[j] -= 0.25 * c

    J_edges: list[tuple[int, int, float]] = []
    for (i, j), c in quad_pairs:
        J = 0.25 * c
        if abs(J) > tol:
            J_edges.append((i, j, float(J)))

    return h, J_edges, float(offset)


def _min_pce_qubits(*, num_vars: int, k: int) -> int:
    import math

    n = k
    while 3 * math.comb(n, k) < num_vars:
        n += 1
    return n


def _build_pce_group(*, pauli: str, count: int, n_qubits: int, k: int) -> list[Any]:
    from qiskit.quantum_info import SparsePauliOp

    out: list[Any] = []
    for idx, combo in enumerate(combinations(range(n_qubits), k)):
        if idx >= count:
            break
        label = ["I"] * n_qubits
        for q in combo:
            label[q] = pauli
        out.append(SparsePauliOp.from_list([("".join(label)[::-1], 1.0)]))
    return out


def _build_ibmstyle_schema(*, n_qubits: int, k: int, num_vars: int) -> tuple[list[list[Any]], list[list[int]]]:
    """Return (groups, group_var_indices) for commuting X/Y/Z measurement groups.

    Operators are assigned in the tutorial's canonical order:
      X over all k-combos, then Y, then Z, truncated to `num_vars`.
    """
    from qiskit.quantum_info import SparsePauliOp

    supports = list(combinations(range(n_qubits), k))
    ops_linear: list[tuple[str, Any]] = []
    for basis in ("X", "Y", "Z"):
        for combo in supports:
            label = ["I"] * n_qubits
            for q in combo:
                label[q] = basis
            ops_linear.append((basis, SparsePauliOp.from_list([("".join(label)[::-1], 1.0)])))
            if len(ops_linear) >= num_vars:
                break
        if len(ops_linear) >= num_vars:
            break

    if len(ops_linear) != num_vars:
        raise ValueError(f"PCE schema capacity too small: got={len(ops_linear)} needed={num_vars}")

    groups: list[list[Any]] = [[], [], []]  # X, Y, Z
    indices: list[list[int]] = [[], [], []]
    b2g = {"X": 0, "Y": 1, "Z": 2}
    for var_i, (basis, op) in enumerate(ops_linear):
        gi = b2g[basis]
        groups[gi].append(op)
        indices[gi].append(int(var_i))
    return groups, indices


@dataclass
class QiskitIbmStylePCEOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """IBM-tutorial-style PCE for general QUBOs.

    Key characteristics (aligned with IBM tutorial):
    - efficient_su2 ansatz
    - EstimatorV2 for correlator expectation values
    - relaxed tanh(alpha * <Π_i>) decoding surrogate + regularization
    - decode via sign(<Π_i>) + optional 1-flip local search refinement
    """

    name: str = "QiskitPCE-IBMStyle"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)

    k: int = 2
    n_qubits: int | None = None
    ansatz_reps: int = 2

    alpha: float | None = None
    reg_strength: float = 0.5

    estimator: Any | None = None
    backend: Any | None = None
    optimization_level: int = 1

    classical_maxiter: int = 50
    seed: Optional[int] = 0

    refine_local_search: bool = True
    refine_local_search_max_iters: int = 200

    def make_estimator(self) -> Any:
        if self.estimator is not None:
            return self.estimator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator

        return AerEstimator()

    def _effective_alpha(self, *, n_qubits: int) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        return float(n_qubits)

    def _loss(
        self,
        *,
        h: np.ndarray,
        J_edges: list[tuple[int, int, float]],
        offset: float,
        corr: np.ndarray,
        alpha: float,
        n_vars: int,
    ) -> float:
        z = np.tanh(float(alpha) * np.asarray(corr, dtype=float))

        e = float(offset) + float(np.dot(h, z))
        for i, j, Jij in J_edges:
            e += float(Jij) * float(z[i] * z[j])

        if float(self.reg_strength) > 0.0:
            nu = float(len(J_edges)) / 2.0 + float(n_vars - 1) / 4.0
            reg = float(np.mean(z * z) ** 2)
            e += float(self.reg_strength) * float(nu) * reg
        return float(e)

    def _decode_x(self, corr: np.ndarray) -> np.ndarray:
        z = np.sign(np.asarray(corr, dtype=float))
        z[z == 0.0] = 1.0
        x = (1.0 - z) / 2.0
        return np.asarray(np.rint(x), dtype=int)

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        from qiskit.circuit.library import efficient_su2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from scipy.optimize import minimize as scipy_minimize

        n_vars = qubo.n_vars
        n_qubits = self.n_qubits if self.n_qubits is not None else _min_pce_qubits(num_vars=n_vars, k=self.k)
        k = self.k
        alpha = self._effective_alpha(n_qubits=n_qubits)
        pce_groups, group_var_indices = _build_ibmstyle_schema(n_qubits=n_qubits, k=k, num_vars=n_vars)

        qc = efficient_su2(n_qubits, ["ry", "rz"], reps=self.ansatz_reps)
        if self.backend is not None:
            pm = generate_preset_pass_manager(target=self.backend.target, optimization_level=self.optimization_level)
        else:
            from qiskit_aer import AerSimulator

            pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=self.optimization_level)
        qc_t = pm.run(qc)

        pce_groups = [[op.apply_layout(qc_t.layout) for op in grp] for grp in pce_groups]

        h, J_edges, ising_offset = _qubo_to_ising_coeffs(qubo=qubo, tol=0.0)
        estimator = self.make_estimator()

        cost_trace: list[float] = []

        def correlators(params: np.ndarray) -> np.ndarray:
            pubs: list[tuple[Any, Any, list[float]]] = []
            pub_group_ids: list[int] = []
            for gi, grp in enumerate(pce_groups):
                if grp:
                    pubs.append((qc_t, grp, list(params)))
                    pub_group_ids.append(int(gi))
            results = estimator.run(pubs).result()
            corr = np.zeros((n_vars,), dtype=float)
            for r, gi in zip(results, pub_group_ids):
                evs = [float(ev) for ev in r.data.evs]
                idxs = group_var_indices[gi]
                if len(evs) != len(idxs):
                    raise ValueError(f"PCE group size mismatch: group={gi} evs={len(evs)} idxs={len(idxs)}")
                for vi, ev in zip(idxs, evs):
                    corr[int(vi)] = float(ev)
            return corr

        def loss_fn(params: np.ndarray) -> float:
            corr = correlators(params)
            val = self._loss(h=h, J_edges=J_edges, offset=ising_offset, corr=corr, alpha=alpha, n_vars=n_vars)
            cost_trace.append(float(val))
            return float(val)

        rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        x0 = rng.uniform(0.0, 1.0, qc_t.num_parameters)

        optimizer_timestamp_start = datetime.now(timezone.utc)
        res = scipy_minimize(loss_fn, x0, method="COBYLA", options={"maxiter": self.classical_maxiter})
        optimizer_timestamp_end = datetime.now(timezone.utc)

        corr_final = correlators(np.asarray(res.x, dtype=float))
        x = self._decode_x(corr_final)

        if self.refine_local_search:
            x2, _obj2, _it = greedy_1flip_local_search(
                qubo=qubo,
                x0=x,
                max_iters=self.refine_local_search_max_iters,
                first_improvement=True,
            )
            x = np.asarray(x2, dtype=int)

        obj = float(evaluate_qubo(qubo=qubo, x=x))
        names = _names_by_index(dict(qubo.var_map), n_vars)
        var_values = {names[i]: int(x[i]) for i in range(n_vars)}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=obj,
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
        )

        run_meta: dict[str, Any] = {
            "pce_style": "ibm_tutorial",
            "n_vars": n_vars,
            "n_qubits": n_qubits,
            "k": k,
            "ansatz": "efficient_su2",
            "ansatz_reps": self.ansatz_reps,
            "alpha": alpha,
            "reg_strength": self.reg_strength,
            "classical_maxiter": self.classical_maxiter,
            "cost_trace": list(cost_trace),
        }

        return solution, OptimizerRun(
            name=self.name,
            optimizer_timestamp_start=optimizer_timestamp_start,
            optimizer_timestamp_end=optimizer_timestamp_end,
            metadata=run_meta,
        )

