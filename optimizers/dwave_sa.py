from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Optional

import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler

from qoco.core.converter import Converter
from qoco.core.optimizer import Optimizer, P
from qoco.core.qubo import QUBO
from qoco.core.solution import Solution, Status
from qoco.converters.qubo_to_bqm import QuboToBQMConverter
from qoco.converters.identity import IdentityConverter


def _names_by_index(var_map: dict[str, int], n: int) -> list[str]:
    if not var_map:
        return [str(i) for i in range(int(n))]
    inv = {int(idx): str(name) for name, idx in var_map.items()}
    return [inv.get(i, str(i)) for i in range(int(n))]


@dataclass
class DWaveSimulatedAnnealingOptimizer(Generic[P], Optimizer[P, QUBO, Solution]):
    """Solve QUBOs using D-Wave's SimulatedAnnealingSampler (classical)."""

    converter: Converter[P, QUBO] = field(default_factory=IdentityConverter)
    qubo_to_bqm: Converter[QUBO, dimod.BinaryQuadraticModel] = field(default_factory=QuboToBQMConverter)

    num_reads: int = 128
    num_sweeps: int = 2_000
    seed: Optional[int] = None

    def _optimize(self, qubo: QUBO) -> Solution:
        bqm = self.qubo_to_bqm.convert(qubo)
        sampler = SimulatedAnnealingSampler()

        ss = sampler.sample(
            bqm,
            num_reads=int(self.num_reads),
            num_sweeps=int(self.num_sweeps),
            seed=None if self.seed is None else int(self.seed),
        )
        best = ss.first
        sample: dict[int, int] = dict(best.sample)
        energy = float(best.energy)

        n = int(qubo.n_vars)
        x = np.zeros((n,), dtype=int)
        for i in range(n):
            x[i] = int(sample.get(i, 0))

        names = _names_by_index(dict(qubo.var_map), n)
        var_values = {names[i]: int(x[i]) for i in range(n)}

        return Solution(
            status=Status.FEASIBLE,
            objective=float(energy),
            var_values=var_values,
            var_arrays={"x": x},
            var_array_index={"x": list(names)},
            info={"sampler": "dwave.samplers.SimulatedAnnealingSampler", "num_reads": int(self.num_reads), "num_sweeps": int(self.num_sweeps)},
        )

