from __future__ import annotations

import numpy as np

from qoco.core.qubo import QUBO


def qubo_cost(qubo: QUBO, x: np.ndarray) -> float:
    return float(x @ qubo.Q @ x + qubo.offset)
