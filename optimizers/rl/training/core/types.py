from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainResult:
    method: str
    params: int
    steps: int
    runtime_s: float
    it_per_s: float
    run_id: str | None = None
    run_dir: str | None = None
    # Optional training curve points (currently unused by ML2; kept as a JSON-friendly structure).
    curve: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalResult:
    method: str
    feasibility: float
    gap_pct: float
    infer_total_s: float
    infer_ms_per_instance: float
    run_id: str | None = None
    run_dir: str | None = None


