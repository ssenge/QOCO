from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qoco.converters.decomposition.column_generation import PricingResult, PricingStrategy, SetPartitionMaster
from qoco.core.solution import InfoSolution


@dataclass
class ColumnGenState:
    master: SetPartitionMaster
    pricing: PricingStrategy
    tol: float = 1e-6
    it: int = 0
    last_master_solution: InfoSolution | None = None
    last_pricing_result: PricingResult | None = None
    last_duals: dict[Any, float] | None = None
    partition_keys: list[Any] | None = None
