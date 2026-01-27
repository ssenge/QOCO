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
