from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qoco.converters.qubo_to_qiskit_qp import QuboToQuadraticProgramConverter
from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import InfoSolution, OptimizerRun, ProblemSummary, Status
from qoco.converters.identity import IdentityConverter


@dataclass
class QiskitVQEOptimizer(Generic[P], Optimizer[P, QUBO, InfoSolution, OptimizerRun, ProblemSummary]):
    """Qiskit VQE on QUBOs.

    Note: Qiskit Optimization's `MinimumEigenOptimizer` does not accept estimator-based VQE
    in our stack, so we run VQE directly on the Ising operator and then sample the optimized
    circuit to extract a bitstring solution.
    """

    name: str = "QiskitVQE"
    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_qp: Any = field(default_factory=QuboToQuadraticProgramConverter)

    reps: int = 1
    classical_optimizer: Any = field(default_factory=lambda: COBYLA(maxiter=100))
    estimator: Any = field(default_factory=EstimatorV2)
    sampler: Any = field(default_factory=SamplerV2)
    pass_manager: Any = field(default_factory=lambda: generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"]))
    shots: int = 2048
    seed: Optional[int] = 0

    def _optimize(self, qubo: QUBO) -> tuple[InfoSolution, OptimizerRun]:
        if self.seed is not None:
            np.random.seed(int(self.seed))
        n = int(qubo.n_vars)
        qp = self.qubo_to_qp.convert(qubo)
        op, offset = qp.to_ising()

        ansatz = efficient_su2(num_qubits=n, reps=int(self.reps))
        vqe = VQE(estimator=self.estimator, ansatz=ansatz, optimizer=self.classical_optimizer, transpiler=self.pass_manager)
        res = vqe.compute_minimum_eigenvalue(op)

        params = dict(res.optimal_parameters)
        qc = res.optimal_circuit.assign_parameters(params, inplace=False)
        qc.measure_all()
        qc = self.pass_manager.run(qc)

        pub = self.sampler.run([qc], shots=int(self.shots)).result()[0]
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

        if not qubo.var_map:
            names = [str(i) for i in range(n)]
        else:
            inv = {int(idx): str(name) for name, idx in dict(qubo.var_map).items()}
            names = [inv.get(i, str(i)) for i in range(n)]
        var_values = {names[i]: int(best_x[i]) for i in range(n)}

        solution = InfoSolution(
            status=Status.FEASIBLE,
            objective=float(best_obj),
            var_values=var_values,
            var_arrays={"x": best_x},
            var_array_index={"x": list(names)},
            info={"solver": "qiskit.VQE", "reps": int(self.reps), "vqe_eigenvalue": float(np.real(res.eigenvalue)), "ising_offset": float(offset)},
        )
        return solution, OptimizerRun(name=self.name)

