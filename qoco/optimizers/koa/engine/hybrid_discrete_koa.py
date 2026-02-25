"""
HybridDiscreteKOA — Discrete KOA with variational quantum circuit (VQC) planets.

Extends DiscreteKOA with a mixed population:
  - Classical planets: integer assignment vectors, updated by discrete KOA moves.
  - Quantum planets: parameterized quantum circuits (PQC/VQC) that, when measured,
    produce candidate assignments. Circuit parameters are tuned classically (SPSA).

No QUBO or cost Hamiltonian — the objective is a classical black-box function.
The quantum circuit only generates candidate solutions; fitness is evaluated classically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from qoco.optimizers.koa.engine.discrete_koa import (
    DiscreteConfig, DiscreteKOA, DiscreteKOAState, DiscreteProblem,
)
from qoco.optimizers.koa.engine.base import PMHResult


# ══════════════════════════════════════════════════════════════════════════════
#                          QUANTUM PLANET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VQCPlanetConfig:
    """Configuration for a VQC planet."""
    n_qubits: int
    n_layers: int = 2
    spsa_a: float = 0.1
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    shots: int = 1


@dataclass
class VQCPlanet:
    """
    A variational quantum circuit that samples candidate assignments.

    Hardware-efficient ansatz: layers of RY rotations + CX entanglement.
    Measurement -> bitstring -> decoded to integer assignment via `decoder`.

    Parameters updated with SPSA. No QUBO — fitness is classical.
    """
    config: VQCPlanetConfig
    decoder: Callable[[np.ndarray], np.ndarray]
    rng: Generator

    params: np.ndarray = field(init=False)
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        n_params = self.config.n_qubits * self.config.n_layers * 2
        self.params = self.rng.uniform(0, 2 * np.pi, size=n_params)

    def _simulate_circuit(self, params: np.ndarray) -> np.ndarray:
        """Statevector simulation of hardware-efficient ansatz: RY, RZ layers + CX chain."""
        n = self.config.n_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

        param_idx = 0
        for _layer in range(self.config.n_layers):
            for q in range(n):
                theta = params[param_idx]
                param_idx += 1
                state = _apply_ry(state, n, q, theta)
            for q in range(n):
                phi = params[param_idx]
                param_idx += 1
                state = _apply_rz(state, n, q, phi)
            for q in range(n - 1):
                state = _apply_cx(state, n, q, q + 1)

        probs = np.abs(state) ** 2
        probs /= probs.sum()
        outcome = self.rng.choice(2**n, p=probs)
        return np.array([(outcome >> (n - 1 - q)) & 1 for q in range(n)], dtype=np.int64)

    def sample(self) -> np.ndarray:
        """Sample one assignment from the VQC."""
        bitstring = self._simulate_circuit(self.params)
        return self.decoder(bitstring)

    def sample_best(self, objective_fn: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """Sample `shots` assignments and return the best one."""
        best_x: Optional[np.ndarray] = None
        best_f = float("inf")
        for _ in range(self.config.shots):
            x = self.sample()
            f = objective_fn(x)
            if f < best_f:
                best_f = f
                best_x = x
        return best_x, best_f  # type: ignore[return-value]

    def update_params(self, objective_fn: Callable[[np.ndarray], float]) -> None:
        """One SPSA step to improve circuit parameters."""
        self._step += 1
        a_k = self.config.spsa_a / (self._step + 1) ** self.config.spsa_alpha
        c_k = self.config.spsa_c / (self._step + 1) ** self.config.spsa_gamma

        delta = self.rng.choice([-1.0, 1.0], size=len(self.params))
        params_plus = self.params + c_k * delta
        params_minus = self.params - c_k * delta

        bs_plus = self._simulate_circuit(params_plus)
        bs_minus = self._simulate_circuit(params_minus)
        f_plus = objective_fn(self.decoder(bs_plus))
        f_minus = objective_fn(self.decoder(bs_minus))

        gradient_estimate = (f_plus - f_minus) / (2.0 * c_k * delta)
        self.params -= a_k * gradient_estimate


# ══════════════════════════════════════════════════════════════════════════════
#                    STATEVECTOR GATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_single_qubit_gate(state: np.ndarray, n: int, qubit: int,
                             gate: np.ndarray) -> np.ndarray:
    new_state = np.zeros_like(state)
    for i in range(2**n):
        bit = (i >> (n - 1 - qubit)) & 1
        other_bits = i & ~(1 << (n - 1 - qubit))
        i0 = other_bits
        i1 = other_bits | (1 << (n - 1 - qubit))
        if bit == 0:
            new_state[i0] += gate[0, 0] * state[i0] + gate[0, 1] * state[i1]
            new_state[i1] += gate[1, 0] * state[i0] + gate[1, 1] * state[i1]
    return new_state


def _apply_ry(state: np.ndarray, n: int, qubit: int, theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    gate = np.array([[c, -s], [s, c]])
    return _apply_single_qubit_gate(state, n, qubit, gate)


def _apply_rz(state: np.ndarray, n: int, qubit: int, phi: float) -> np.ndarray:
    gate = np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])
    return _apply_single_qubit_gate(state, n, qubit, gate)


def _apply_cx(state: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    new_state = state.copy()
    for i in range(2**n):
        ctrl_bit = (i >> (n - 1 - control)) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << (n - 1 - target))
            new_state[i], new_state[j] = state[j], state[i]
    return new_state


# ══════════════════════════════════════════════════════════════════════════════
#                       HYBRID DISCRETE KOA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HybridDiscreteKOA(DiscreteKOA):
    """
    Discrete KOA with mixed classical + quantum population.

    The first `n_quantum` planets are backed by VQC circuits.
    Classical planets: updated by discrete KOA moves.
    Quantum planets: sample from circuit, evaluate, update via SPSA.
    """
    n_quantum: int = 3
    vqc_config: VQCPlanetConfig = field(default_factory=lambda: VQCPlanetConfig(n_qubits=8))
    bitstring_decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None

    _vqc_planets: List[VQCPlanet] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_quantum > self.config.population_size:
            raise ValueError("n_quantum must be <= population_size")

    def init_population(self) -> list[np.ndarray]:
        pop = super().init_population()

        if self.bitstring_decoder is None:
            raise ValueError("bitstring_decoder must be provided for hybrid mode")

        self._vqc_planets = [
            VQCPlanet(
                config=self.vqc_config,
                decoder=self.bitstring_decoder,
                rng=self.rng,
            )
            for _ in range(self.n_quantum)
        ]

        for qi in range(self.n_quantum):
            pop[qi] = self._vqc_planets[qi].sample()

        return pop

    def propose(self, i: int, s: DiscreteKOAState) -> np.ndarray:
        if i < self.n_quantum:
            return self._quantum_propose(i, s)
        return super().propose(i, s)

    def _quantum_propose(self, i: int, s: DiscreteKOAState) -> np.ndarray:
        vqc = self._vqc_planets[i]
        candidate = vqc.sample()
        vqc.update_params(self.objective_fn)
        return candidate
