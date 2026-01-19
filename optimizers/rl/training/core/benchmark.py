from __future__ import annotations

import logging
import time
from typing import Dict, List, Sequence

import numpy as np

from qoco.core.solution import Status
from qoco.core.logging import start_run
logger = logging.getLogger("qoco.training")

from qoco.optimizers.rl.training.core.metrics import feasibility_rate, mean_gap_pct
from qoco.optimizers.rl.training.core.types import EvalResult
from qoco.optimizers.rl.training.methods.base import Method
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter


def evaluate_method_unlogged(
    *,
    trainer: Method,
    problem: ProblemAdapter,
    instances: Sequence,
    device: str,
    opt_costs: Sequence[float],
) -> dict[str, float]:
    """Evaluate without starting a new run (for in-training curves)."""
    adapter = problem
    feasible: List[bool] = []
    costs: List[float] = []
    times: List[float] = []

    for inst in instances:
        eval_batch = adapter.make_eval_batch(inst, device=device)
        start = time.time()
        out = trainer.infer(problem, eval_batch)
        elapsed = time.time() - start
        if isinstance(out, dict) and "reward" in out:
            f, c = adapter.score_from_reward(out["reward"])
        else:
            f, c = adapter.score_eval_batch(out)
        if isinstance(f, Status):
            feasible.append(f is Status.FEASIBLE)
        else:
            feasible.append(bool(f))
        costs.append(c)
        times.append(elapsed)

    infer_total_s = float(np.sum(times))
    infer_ms = infer_total_s / max(1, len(instances)) * 1000.0

    return {
        "eval/feasible_rate": float(feasibility_rate(feasible)),
        "eval/gap_pct": float(mean_gap_pct(costs, opt_costs, feasible)),
        "eval/infer_total_s": float(infer_total_s),
        "eval/infer_ms_per_instance": float(infer_ms),
    }


def evaluate_method(
    *,
    trainer: Method,
    problem: ProblemAdapter,
    instances: Sequence,
    device: str,
    opt_costs: Sequence[float] | None = None,
) -> EvalResult:
    logger = start_run(
        name=f"eval-{trainer.name}",
        kind="eval",
        config={
            "method": trainer.name,
            "problem": getattr(problem, "name", problem.__class__.__name__),
            "n_instances": int(len(instances)),
            "device": str(device),
        },
    )
    adapter = problem
    if hasattr(adapter, "reset_debug"):
        try:
            adapter.reset_debug()  # type: ignore[attr-defined]
        except Exception:
            pass
    feasible: List[bool] = []
    costs: List[float] = []
    times: List[float] = []

    if opt_costs is None:
        opt_costs = [adapter.optimal_cost(i) for i in instances]

    for inst in instances:
        eval_batch = adapter.make_eval_batch(inst, device=device)
        start = time.time()
        out = trainer.infer(problem, eval_batch)
        elapsed = time.time() - start
        if isinstance(out, dict) and "reward" in out:
            f, c = adapter.score_from_reward(out["reward"])
        else:
            f, c = adapter.score_eval_batch(out)
        if isinstance(f, Status):
            feasible.append(f is Status.FEASIBLE)
        else:
            feasible.append(bool(f))
        costs.append(c)
        times.append(elapsed)

    infer_total_s = float(np.sum(times))
    infer_ms = infer_total_s / max(1, len(instances)) * 1000.0

    if hasattr(adapter, "debug_summary"):
        try:
            dbg = adapter.debug_summary()  # type: ignore[attr-defined]
            if isinstance(dbg, dict) and dbg:
                logger.info("[adapter debug] %s", dbg)
        except Exception:
            pass

    res = EvalResult(
        method=trainer.name,
        feasibility=feasibility_rate(feasible),
        gap_pct=mean_gap_pct(costs, opt_costs, feasible),
        infer_total_s=infer_total_s,
        infer_ms_per_instance=float(infer_ms),
        run_id=logger.run_dir.name,
        run_dir=str(logger.run_dir),
    )
    logger.log(
        "eval",
        {
            "metrics": {
                "eval/feasible_rate": float(res.feasibility),
                "eval/gap_pct": float(res.gap_pct),
                "eval/infer_total_s": float(res.infer_total_s),
                "eval/infer_ms_per_instance": float(res.infer_ms_per_instance),
            }
        },
    )
    logger.log("run_end", {"status": "ok"})
    logger.close()
    return res
