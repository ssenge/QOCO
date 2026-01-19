from __future__ import annotations

"""Tiny pooled PQC block for end-to-end pipeline testing.

Design goal: test the full classical->quantum->classical chain with minimal overhead.

We intentionally run the PQC on a *pooled* embedding (one vector per instance) rather
than per node/per decoding step.
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PQCConfig:
    n_qubits: int = 2
    n_layers: int = 1
    mode: str = "variational"
    # We use 2 features per qubit (ry + rz) for embedding.
    features_per_qubit: int = 2


def _build_estimator():
    """Return an Estimator *V2* primitive compatible with EstimatorQNN.

    Use Aer in `statevector` simulation mode for speed (still exact/deterministic).
    """
    from qiskit_aer.primitives import EstimatorV2  # type: ignore

    return EstimatorV2(options={"backend_options": {"method": "statevector"}})


def _build_sampler():
    """Return a Sampler *V2* primitive compatible with SamplerQNN."""
    from qiskit_aer.primitives import SamplerV2  # type: ignore

    return SamplerV2(options={"backend_options": {"method": "statevector"}})


def _build_small_pqc(cfg: PQCConfig):
    """Return (qc, input_params, weight_params, observables)."""
    from qiskit import QuantumCircuit  # type: ignore
    from qiskit.circuit import ParameterVector  # type: ignore
    from qiskit.quantum_info import SparsePauliOp  # type: ignore

    n = int(cfg.n_qubits)
    mode = str(cfg.mode).lower()
    use_variational = mode != "hadamard"
    use_reuploading = mode == "reuploading"
    x = ParameterVector("x", n * int(cfg.features_per_qubit)) if use_variational else ParameterVector("x", 0)
    theta = ParameterVector("θ", int(cfg.n_layers) * n * 2) if use_variational else ParameterVector("θ", 0)

    qc = QuantumCircuit(n)

    # Superposition
    qc.h(range(n))

    if use_variational:
        def apply_embedding() -> None:
            for q in range(n):
                qc.ry(x[cfg.features_per_qubit * q + 0], q)
                qc.rz(x[cfg.features_per_qubit * q + 1], q)

        def apply_entanglement() -> None:
            for q in range(n - 1):
                qc.cx(q, q + 1)
            if n > 1:
                qc.cx(n - 1, 0)

        if not use_reuploading:
            apply_embedding()

        # Variational layers: RY/RZ + entanglement (+ optional reuploading)
        t = 0
        for _ in range(int(cfg.n_layers)):
            if use_reuploading:
                apply_embedding()
            for q in range(n):
                qc.ry(theta[t + 0], q)
                qc.rz(theta[t + 1], q)
                t += 2
            apply_entanglement()

    # Readout: Z expectation per qubit (n outputs)
    observables = []
    for q in range(n):
        pauli = ["I"] * n
        # Convention doesn't matter much for a smoke test; keep consistent.
        pauli[n - 1 - q] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli), 1.0)]))

    return qc, list(x), list(theta), observables


class NoOpPQCBlock(nn.Module):
    """Identity block used for non-PQC baselines."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h


class PooledPQCBase(nn.Module):
    """Apply a PQC to pooled embeddings and broadcast to nodes."""

    def __init__(
        self,
        *,
        embed_dim: int,
        cfg: PQCConfig = PQCConfig(),
        angle_scale: float = float(np.pi),
        init_weight_scale: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cfg = cfg
        self.angle_scale = float(angle_scale)

        n_qubits = int(cfg.n_qubits)

        # Classical projections around the PQC
        self.down = None
        self.up = nn.Linear(n_qubits, self.embed_dim)

        self._hadamard_only = str(cfg.mode).lower() == "hadamard"
        self._pqc_input_dim = 0
        self.q = None
        if not self._hadamard_only:
            from qiskit_machine_learning.connectors import TorchConnector  # type: ignore
            from qiskit_machine_learning.neural_networks import EstimatorQNN  # type: ignore

            qc, input_params, weight_params, observables = _build_small_pqc(cfg)
            estimator = _build_estimator()

            qnn = EstimatorQNN(
                circuit=qc,
                observables=observables,
                input_params=input_params,
                weight_params=weight_params,
                estimator=estimator,
            )

            init_w = init_weight_scale * np.random.randn(len(weight_params)).astype(np.float64)
            self.q = TorchConnector(qnn, initial_weights=init_w)

            self._pqc_input_dim = int(len(input_params))
            if self._pqc_input_dim > 0:
                self.down = nn.Linear(self.embed_dim, self._pqc_input_dim)

        # Counters (debug/analysis). Not persisted in checkpoints.
        self.register_buffer("_pqc_forward_calls", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("_pqc_backward_calls", torch.zeros((), dtype=torch.long), persistent=False)

    def reset_counters(self) -> None:
        self._pqc_forward_calls.zero_()
        self._pqc_backward_calls.zero_()

    @property
    def pqc_forward_calls(self) -> int:
        return int(self._pqc_forward_calls.item())

    @property
    def pqc_backward_calls(self) -> int:
        return int(self._pqc_backward_calls.item())

    def _count_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._pqc_backward_calls.add_(1)
        return grad

    def _compute_delta(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected (B,N,D) tensor, got shape {tuple(h.shape)}")

        self._pqc_forward_calls.add_(1)
        g = h.mean(dim=1)  # (B, D)

        if self._hadamard_only:
            z = torch.zeros((g.shape[0], int(self.cfg.n_qubits)), dtype=h.dtype, device=h.device)
        else:
            if self._pqc_input_dim > 0 and self.down is not None:
                x = self.down(g).tanh() * self.angle_scale
            else:
                x = torch.empty((g.shape[0], 0), dtype=g.dtype, device=g.device)

            q_dtype = torch.float32 if x.device.type == "mps" else torch.float64
            z = self.q(x.to(dtype=q_dtype))  # type: ignore[operator]
        if z.requires_grad:
            z.register_hook(self._count_backward)
        z = z.to(dtype=h.dtype)

        delta = self.up(z)  # (B, D)
        return delta.unsqueeze(1)


class PooledPQCResidual(PooledPQCBase):
    """Apply a PQC residual to pooled embeddings.

    Input/Output: h of shape (B, N, D) -> (B, N, D)
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self._compute_delta(h)


class PooledPQCReplace(PooledPQCBase):
    """Replace embeddings with PQC output (non-residual)."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self._compute_delta(h)


class PooledPQCSamplerResidual(nn.Module):
    """SamplerQNN-based residual with a low-dimensional output."""

    def __init__(
        self,
        *,
        embed_dim: int,
        cfg: PQCConfig = PQCConfig(),
        angle_scale: float = float(np.pi),
        init_weight_scale: float = 0.01,
        output_dim: int = 2,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cfg = cfg
        self.angle_scale = float(angle_scale)
        self.output_dim = int(output_dim)

        from qiskit_machine_learning.connectors import TorchConnector  # type: ignore
        from qiskit_machine_learning.neural_networks import SamplerQNN  # type: ignore

        qc, input_params, weight_params, _ = _build_small_pqc(cfg)
        sampler = _build_sampler()

        # Option A: map bitstrings to parity class (even/odd) -> 2 outputs.
        def interpret(bitstring: int) -> int:
            return int(bin(int(bitstring)).count("1") % 2)

        qnn = SamplerQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            sampler=sampler,
            interpret=interpret,
            output_shape=self.output_dim,
        )

        init_w = init_weight_scale * np.random.randn(len(weight_params)).astype(np.float64)
        self.q = TorchConnector(qnn, initial_weights=init_w)

        self._pqc_input_dim = int(len(input_params))
        self.down = None
        if self._pqc_input_dim > 0:
            self.down = nn.Linear(self.embed_dim, self._pqc_input_dim)

        self.up = nn.Linear(self.output_dim, self.embed_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected (B,N,D) tensor, got shape {tuple(h.shape)}")
        g = h.mean(dim=1)
        if self._pqc_input_dim > 0 and self.down is not None:
            x = self.down(g).tanh() * self.angle_scale
        else:
            x = torch.empty((g.shape[0], 0), dtype=g.dtype, device=g.device)
        q_dtype = torch.float32 if x.device.type == "mps" else torch.float64
        z = self.q(x.to(dtype=q_dtype))  # type: ignore[operator]
        z = z.to(dtype=h.dtype)
        delta = self.up(z)
        return h + delta.unsqueeze(1)


class ClassicalBottleneckResidual(nn.Module):
    """Classical residual block with a PQC-matched bottleneck."""

    def __init__(self, *, embed_dim: int, cfg: PQCConfig):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cfg = cfg
        bottleneck_dim = int(cfg.n_qubits) * int(cfg.features_per_qubit)
        self.down = nn.Linear(self.embed_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, self.embed_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected (B,N,D) tensor, got shape {tuple(h.shape)}")
        g = h.mean(dim=1)
        z = torch.relu(self.down(g))
        delta = self.up(z)
        return h + delta.unsqueeze(1)


class ClassicalExtraResidual(nn.Module):
    """Classical residual block with extra capacity vs PQC bottleneck."""

    def __init__(self, *, embed_dim: int, cfg: PQCConfig, expansion_factor: int = 4):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cfg = cfg
        bottleneck_dim = int(cfg.n_qubits) * int(cfg.features_per_qubit)
        hidden_dim = int(bottleneck_dim) * int(expansion_factor)
        self.down = nn.Linear(self.embed_dim, bottleneck_dim)
        self.mid = nn.Linear(bottleneck_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, self.embed_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected (B,N,D) tensor, got shape {tuple(h.shape)}")
        g = h.mean(dim=1)
        z = torch.relu(self.down(g))
        z = torch.relu(self.mid(z))
        delta = self.up(z)
        return h + delta.unsqueeze(1)

