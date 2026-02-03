from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from qoco.core.converter import Converter
from qoco.core.qubo import QUBO, QuboCoupling, QuboPartition


@dataclass
class QuboDecomposer(Converter[QUBO, QuboPartition]):
    n_partitions: int = 4
    partition_method: str = "contiguous"  # contiguous | interleaved | random
    seed: int | None = None

    def convert(self, problem: QUBO) -> QuboPartition:
        Q = np.asarray(problem.Q, dtype=float)
        n = int(Q.shape[0])
        if n == 0:
            return QuboPartition(blocks=[], sub_qubos=[], global_indices=[], couplings=[])

        rng = np.random.default_rng(self.seed)
        blocks = self._partition_indices(n=n, rng=rng)

        # Build per-block sub QUBOs
        sub_qubos: List[QUBO] = []
        global_indices: List[np.ndarray] = []
        for block in blocks:
            idx = np.asarray(block, dtype=np.int64)
            m = int(idx.size)
            Qsub = np.zeros((m, m), dtype=float)
            for a in range(m):
                i = int(idx[a])
                Qsub[a, a] = float(Q[i, i])
                for b in range(a + 1, m):
                    j = int(idx[b])
                    # preserve upper-tri convention by folding both directions
                    Qsub[a, b] = float(Q[i, j]) + float(Q[j, i])
            # Offset: keep the original offset only once; caller should account for it separately.
            # Here we set offset=0.0 per subproblem and let the combiner add the global offset.
            sub_qubos.append(QUBO(Q=Qsub, offset=0.0, var_map={}))
            global_indices.append(idx)

        # Cross-block couplings
        block_of = np.full((n,), -1, dtype=np.int64)
        for b, block in enumerate(blocks):
            for i in block:
                block_of[int(i)] = int(b)

        couplings: List[QuboCoupling] = []
        for i in range(n):
            bi = int(block_of[i])
            for j in range(i + 1, n):
                bj = int(block_of[j])
                if bi == bj:
                    continue
                coef = float(Q[i, j]) + float(Q[j, i])
                if coef != 0.0:
                    couplings.append(QuboCoupling(i=int(i), j=int(j), coef=float(coef)))

        return QuboPartition(blocks=blocks, sub_qubos=sub_qubos, global_indices=global_indices, couplings=couplings)

    def _partition_indices(self, *, n: int, rng: np.random.Generator) -> List[List[int]]:
        K = int(min(max(1, int(self.n_partitions)), n))
        if self.partition_method == "contiguous":
            indices = list(range(n))
            group_size = n // K
            out: List[List[int]] = []
            for k in range(K):
                a = k * group_size
                b = (k + 1) * group_size if k < K - 1 else n
                out.append(indices[a:b])
            return out
        if self.partition_method == "interleaved":
            out = [[] for _ in range(K)]
            for i in range(n):
                out[i % K].append(i)
            return out
        if self.partition_method == "random":
            indices = list(range(n))
            rng.shuffle(indices)
            group_size = n // K
            out: List[List[int]] = []
            for k in range(K):
                a = k * group_size
                b = (k + 1) * group_size if k < K - 1 else n
                out.append(indices[a:b])
            return out
        raise ValueError(f"Unknown partition_method: {self.partition_method}")

