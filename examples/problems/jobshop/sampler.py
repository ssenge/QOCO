from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qoco.core.sampler import Sampler
from qoco.examples.problems.jobshop.problem import JobShopScheduling, Operation


@dataclass
class RandomJobShopSampler(Sampler[JobShopScheduling]):
    sizes: Sequence[tuple[int, int, int]] = ((5, 5, 5), (8, 5, 5), (10, 5, 5))
    seed: int = 0
    duration_range: tuple[int, int] = (1, 10)
    machine_mode: str = "permute"

    def sample(self) -> JobShopScheduling:
        rng = np.random.default_rng(int(self.seed))
        if self.sizes:
            n_jobs, n_machines, n_ops = (int(v) for v in rng.choice(self.sizes))
        else:
            n_jobs, n_machines, n_ops = 5, 5, 5

        lo, hi = (int(self.duration_range[0]), int(self.duration_range[1]))
        hi = max(lo + 1, hi)
        jobs = []
        for j in range(int(n_jobs)):
            if self.machine_mode == "permute" and n_ops <= n_machines:
                machines = rng.permutation(n_machines)[:n_ops]
            else:
                machines = rng.integers(0, n_machines, size=int(n_ops))
            durations = rng.integers(lo, hi, size=int(n_ops))
            ops = [Operation(machine=int(m), duration=float(d)) for m, d in zip(machines, durations)]
            jobs.append(ops)

        self.seed += 1
        name = f"jobshop_random_{int(n_jobs)}x{int(n_ops)}"
        return JobShopScheduling(name=name, jobs=jobs)
