from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional

import numpy as np
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qoco.converters.qubo_to_ising import QuboToIsingConverter
from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution, Status
from qoco.converters.identity import IdentityConverter


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


_STANDARD_GATE_NAMES = set(get_standard_gate_name_mapping().keys())


@dataclass
class QiskitVQEOptimizer(Generic[P], Optimizer[P, QUBO, Solution, OptimizerRun, ProblemSummary]):
    """Qiskit VQE on QUBOs.

    Note: Qiskit Optimization's `MinimumEigenOptimizer` does not accept estimator-based VQE
    in our stack, so we run VQE directly on the Ising operator and then sample the optimized
    circuit to extract a bitstring solution.
    """

    name: str = "QiskitVQE"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_ising: Converter[QUBO, tuple[Any, float]] = field(default_factory=QuboToIsingConverter)
    qubo_to_qp: Converter[QUBO, Any] = field(default_factory=QuboToQuadraticProgramConverter)
    use_quadratic_program: bool = False

    reps: int = 1
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=100))

    # Execution / primitives
    estimator: Any | None = None
    sampler: Any | None = None
    shots: int = 2048
    seed: Optional[int] = 0

    # Hardware-aware transpilation
    backend: Any | None = None
    optimization_level: int = 1
    transpiler: Any | None = None  # a Qiskit PassManager-like object

    # VQE customization
    vqe_kwargs: dict[str, Any] = field(default_factory=dict)
    cost_scale: float | None = None

    def encode_qubo(self, qubo: QUBO) -> tuple[Any, float]:
        if self.use_quadratic_program:
            qp = self.qubo_to_qp.convert(qubo)
            return qp.to_ising()
        return self.qubo_to_ising.convert(qubo)

    def make_estimator(self) -> Any:
        if self.estimator is not None:
            return self.estimator
        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        return EstimatorV2(**kwargs)

    def make_sampler(self) -> Any:
        if self.sampler is not None:
            return self.sampler
        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        return SamplerV2(**kwargs)

    def make_transpiler(self) -> Any:
        if self.transpiler is not None:
            return self.transpiler

        backend = self.backend
        if backend is None:
            if self.estimator is None and self.sampler is None:
                return generate_preset_pass_manager(
                    backend=AerSimulator(),
                    optimization_level=int(self.optimization_level),
                )
            return generate_preset_pass_manager(optimization_level=int(self.optimization_level))

        ops = list(getattr(getattr(backend, "target", None), "operation_names", []))
        basis_gates = [str(name) for name in ops if str(name) in _STANDARD_GATE_NAMES]
        if not basis_gates:
            basis_gates = ["rz", "sx", "x", "cx"]
        return generate_preset_pass_manager(
            optimization_level=int(self.optimization_level),
            basis_gates=basis_gates,
        )

    def make_vqe(self, n_qubits: int) -> VQE:
        ansatz = efficient_su2(num_qubits=int(n_qubits), reps=int(self.reps))
        return VQE(
            estimator=self.make_estimator(),
            ansatz=ansatz,
            optimizer=self.classical_optimizer,
            transpiler=self.make_transpiler(),
            **dict(self.vqe_kwargs),
        )

    def _optimize(self, qubo: QUBO) -> tuple[Solution, OptimizerRun]:
        if self.seed is not None:
            np.random.seed(int(self.seed))
        n = int(qubo.n_vars)
        op, offset = self.encode_qubo(qubo)
        if self.cost_scale is not None:
            op = op * float(self.cost_scale)
            offset = float(offset) * float(self.cost_scale)

        vqe = self.make_vqe(n_qubits=n)
        res = vqe.compute_minimum_eigenvalue(op)

        params = dict(res.optimal_parameters)
        qc = res.optimal_circuit.assign_parameters(params, inplace=False)
        qc.measure_all()
        qc = self.make_transpiler().run(qc)

        pub = self.make_sampler().run([qc], shots=int(self.shots)).result()[0]
        counts = pub.data.meas.get_counts()

        Q = np.asarray(qubo.Q, dtype=float)
        best_obj = float("inf")
        best_x = np.zeros((n,), dtype=int)

        for bitstring, _c in counts.items():
            # qiskit bitstring is little-endian: rightmost bit is qubit 0
            x = np.array([1 if bitstring[-1 - i] == "1" else 0 for i in range(n)], dtype=int)
            obj = float(x @ Q @ x + float(qubo.offset))
            if obj < best_obj:
                best_obj = obj
                best_x = x

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(best_x[i]) for i in range(n)}

        solution = Solution(
            status=Status.FEASIBLE,
            objective=float(best_obj),
            var_values=var_values,
            var_arrays={"x": best_x},
            var_array_index={"x": list(names)},
        )
        return solution, OptimizerRun(name=self.name)

