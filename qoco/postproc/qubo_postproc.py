from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np

from qoco.core.qubo import QUBO

BitOrder = Literal["qiskit", "msb"]


@dataclass(frozen=True)
class Sample:
    bitstring: str
    weight: float = 1.0  # count or probability; only relative ordering matters


@dataclass(frozen=True)
class SampleBatch:
    n: int
    samples: list[Sample]
    bit_order: BitOrder = "qiskit"


@dataclass(frozen=True)
class PostProcResult:
    x: np.ndarray
    objective: float
    bitstring: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


def bitstring_to_x(*, bitstring: str, n: int, bit_order: BitOrder) -> np.ndarray:
    s = str(bitstring).strip()
    if s.startswith("0b"):
        s = s[2:]
    # Be tolerant: some backends drop leading zeros or include formatting noise.
    s = "".join(ch for ch in s if ch in ("0", "1"))
    if len(s) < int(n):
        s = s.zfill(int(n))
    if len(s) > int(n):
        s = s[-int(n) :]

    if bit_order == "qiskit":
        # Little-endian: rightmost bit is qubit 0.
        return np.asarray([1 if s[-1 - i] == "1" else 0 for i in range(int(n))], dtype=int)
    if bit_order == "msb":
        # Most-significant bit is qubit 0.
        return np.asarray([1 if s[i] == "1" else 0 for i in range(int(n))], dtype=int)
    raise ValueError(f"unknown bit_order: {bit_order}")


def evaluate_qubo(*, qubo: QUBO, x: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    Q = np.asarray(qubo.Q, dtype=float)
    return float(xx @ Q @ xx + float(qubo.offset))


class PostProcessor(ABC):
    @abstractmethod
    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        raise NotImplementedError


@dataclass
class NoOpPostProcessor(PostProcessor):
    """Default post-processor: return the most frequent sample (no refinement)."""

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        return MostFrequentPostProcessor().run(qubo=qubo, batch=batch)


@dataclass
class TakeBestPostProcessor(PostProcessor):
    """Pick the best objective among provided samples."""

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        best_obj = float("inf")
        best_x = np.zeros((int(batch.n),), dtype=int)
        best_bitstring: str | None = None

        for s in batch.samples:
            x = bitstring_to_x(bitstring=s.bitstring, n=int(batch.n), bit_order=batch.bit_order)
            obj = evaluate_qubo(qubo=qubo, x=x)
            if obj < best_obj:
                best_obj = float(obj)
                best_x = np.asarray(x, dtype=int)
                best_bitstring = str(s.bitstring)

        return PostProcResult(x=best_x, objective=float(best_obj), bitstring=best_bitstring, info={"method": "best"})


@dataclass
class MostFrequentPostProcessor(PostProcessor):
    """Pick the most frequent (highest weight) sample and evaluate it."""

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        if not batch.samples:
            raise ValueError("empty sample batch")
        s = max(batch.samples, key=lambda t: float(t.weight))
        x = bitstring_to_x(bitstring=s.bitstring, n=int(batch.n), bit_order=batch.bit_order)
        obj = evaluate_qubo(qubo=qubo, x=x)
        return PostProcResult(
            x=np.asarray(x, dtype=int),
            objective=float(obj),
            bitstring=str(s.bitstring),
            info={"method": "most_frequent", "weight": float(s.weight)},
        )


def _delta_single_flip(*, Q: np.ndarray, x: np.ndarray, i: int) -> float:
    # Objective is x^T Q x + offset. For bit flip at i, using dense formula:
    # Δ = (1 - 2 x_i) * (Q_ii + 2 * sum_{j!=i} Q_ij x_j)
    # Works for general (not-necessarily symmetric) Q.
    xi = float(x[int(i)])
    s = float(Q[int(i), int(i)])
    s += 2.0 * float(np.dot(Q[int(i), :], x) - Q[int(i), int(i)] * xi)
    return float((1.0 - 2.0 * xi) * s)


def greedy_1flip_local_search(
    *,
    qubo: QUBO,
    x0: np.ndarray,
    max_iters: int = 200,
    first_improvement: bool = True,
) -> tuple[np.ndarray, float, int]:
    Q = np.asarray(qubo.Q, dtype=float)
    x = np.asarray(x0, dtype=int).copy()
    obj = evaluate_qubo(qubo=qubo, x=x)

    n = int(x.shape[0])
    it = 0
    while it < int(max_iters):
        it += 1
        best_delta = 0.0
        best_i = -1
        for i in range(n):
            delta = _delta_single_flip(Q=Q, x=x.astype(float), i=i)
            if delta < best_delta:
                best_delta = float(delta)
                best_i = int(i)
                if first_improvement:
                    break
        if best_i < 0:
            break
        x[best_i] = 1 - int(x[best_i])
        obj = float(obj + best_delta)
    return x, float(obj), int(it)


@dataclass
class LocalSearchPostProcessor(PostProcessor):
    """Greedy 1-flip local search seeded from sample batch."""

    k_start: int = 5
    max_iters: int = 200
    first_improvement: bool = True

    def _improve(self, *, qubo: QUBO, x0: np.ndarray) -> tuple[np.ndarray, float, int]:
        return greedy_1flip_local_search(
            qubo=qubo,
            x0=x0,
            max_iters=int(self.max_iters),
            first_improvement=bool(self.first_improvement),
        )

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        if not batch.samples:
            raise ValueError("empty sample batch")

        seeds = sorted(batch.samples, key=lambda t: float(t.weight), reverse=True)[: int(self.k_start)]

        best_obj = float("inf")
        best_x = np.zeros((int(batch.n),), dtype=int)
        best_bitstring: str | None = None
        total_iters = 0

        for s in seeds:
            x0 = bitstring_to_x(bitstring=s.bitstring, n=int(batch.n), bit_order=batch.bit_order)
            x, obj, it = self._improve(qubo=qubo, x0=x0)
            total_iters += int(it)
            if obj < best_obj:
                best_obj = float(obj)
                best_x = np.asarray(x, dtype=int)
                best_bitstring = str(s.bitstring)

        return PostProcResult(
            x=best_x,
            objective=float(best_obj),
            bitstring=best_bitstring,
            info={
                "method": "local_search",
                "k_start": int(self.k_start),
                "max_iters": int(self.max_iters),
                "first_improvement": bool(self.first_improvement),
                "total_iters": int(total_iters),
            },
        )


@dataclass
class PCEPostProcessor(PostProcessor):
    """Pauli Correlation Encoding (PCE) post-processing (not implemented yet)."""

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        raise NotImplementedError("PCE post-processing not implemented yet")

