"""
Types for the KOA optimizer integration with qoco.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from qoco.optimizers.koa.engine.discrete_koa import DiscreteProblem


@dataclass
class DiscreteKOAProblem:
    """
    Converted problem representation for the KOA optimizer.

    Bundles the DiscreteProblem (init/distance/random_slot_value),
    the scalar objective function, and a decoder that maps the best
    integer array back to qoco var_values.
    """
    problem: DiscreteProblem
    objective_fn: Callable[[np.ndarray], float]
    decode_to_var_values: Callable[[np.ndarray], Dict[str, Any]]
