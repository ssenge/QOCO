from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qoco.core.sampler import Sampler
from qoco.examples.problems.tsp.problem import TSP


@dataclass
class RandomTSPSampler(Sampler[TSP]):
    sizes: Sequence[int] = (20, 30, 40, 50)
    seed: int = 0
    coord_scale: float = 1.0

    def sample(self) -> TSP:
        rng = np.random.default_rng(int(self.seed))
        n = int(rng.choice(self.sizes)) if self.sizes else 20
        coords = rng.random((n, 2), dtype=np.float64) * float(self.coord_scale)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        self.seed += 1
        return TSP(name=f"tsp_random_{n}", dist=dist.tolist())


@dataclass
class ClusteredTSPSampler(Sampler[TSP]):
    sizes: Sequence[int] = (20, 30, 40, 50)
    seed: int = 0
    coord_scale: float = 1.0
    n_clusters: int = 3
    cluster_spread: float = 0.08

    def sample(self) -> TSP:
        rng = np.random.default_rng(int(self.seed))
        n = int(rng.choice(self.sizes)) if self.sizes else 20
        k = max(1, int(self.n_clusters))
        centers = rng.random((k, 2), dtype=np.float64) * float(self.coord_scale)
        assignments = rng.integers(0, k, size=int(n))
        noise = rng.normal(scale=float(self.cluster_spread), size=(n, 2))
        coords = centers[assignments] + noise
        coords = np.clip(coords, 0.0, float(self.coord_scale))
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        self.seed += 1
        return TSP(name=f"tsp_clustered_{n}", dist=dist.tolist())


@dataclass
class RandomDiGraphTSPSampler(Sampler[TSP]):
    sizes: Sequence[int] = (20, 30, 40, 50)
    seed: int = 0
    coord_scale: float = 1.0
    edge_prob: float = 0.3
    weight_range: tuple[float, float] = (1.0, 10.0)
    missing_penalty: float = 1e6

    def sample(self) -> TSP:
        rng = np.random.default_rng(int(self.seed))
        n = int(rng.choice(self.sizes)) if self.sizes else 20
        coords = rng.random((n, 2), dtype=np.float64) * float(self.coord_scale)
        dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))

        # Build a directed graph with a guaranteed Hamiltonian cycle.
        edge_mask = rng.random((n, n)) < float(self.edge_prob)
        np.fill_diagonal(edge_mask, False)
        perm = rng.permutation(n)
        for i in range(n):
            edge_mask[perm[i], perm[(i + 1) % n]] = True

        # Optional additional random weights on top of geometry.
        lo, hi = float(self.weight_range[0]), float(self.weight_range[1])
        noise = rng.uniform(lo, hi, size=(n, n))
        dist = dist + noise
        dist = np.where(edge_mask, dist, float(self.missing_penalty))
        np.fill_diagonal(dist, 0.0)

        inst = TSP(name=f"tsp_digraph_{n}", dist=dist.tolist(), edge_mask=edge_mask.astype(bool).tolist())
        self.seed += 1
        return inst
