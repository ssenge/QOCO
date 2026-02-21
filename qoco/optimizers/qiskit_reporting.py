from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from qoco.core.qubo import QUBO
from qoco.postproc.qubo_postproc import SampleBatch, bitstring_to_x, evaluate_qubo


def _to_bitstring(*, value: int, n: int) -> str:
    return format(value, f"0{n}b")


def format_cost_trace(*, cost_trace: Sequence[float], head: int = 10, tail: int = 10) -> str:
    trace = [float(v) for v in cost_trace]
    if not trace:
        return "cost_trace: <empty>"
    best = min(trace)
    best_i = trace.index(best) + 1
    shown_head = trace[: max(0, head)]
    shown_tail = trace[-max(0, tail) :]

    lines = [f"cost_trace: n={len(trace)} best={best:.6f} at iter={best_i}"]
    if shown_head:
        lines.append(f"  head={['{:.6f}'.format(v) for v in shown_head]}")
    if shown_tail and shown_tail != shown_head:
        lines.append(f"  tail={['{:.6f}'.format(v) for v in shown_tail]}")
    return "\n".join(lines)

def format_qaoa_config(config: Mapping[str, Any]) -> str:
    if not config:
        return "qaoa_config: <missing>"
    lines = ["qaoa_config:"]
    order = [
        "optimizer_type",
        "name",
        "backend",
        "reps",
        "shots",
        "seed",
        "optimization_level",
        "transpile_strategy",
        "cost_primitive",
        "return_distribution",
        "cost_scale",
        "classical_optimizer",
        "postproc",
        "build_initial_state",
        "build_mixer",
        "build_aggregation",
        "build_initial_point",
    ]
    for key in order:
        if key in config:
            lines.append(f"  {key}={config.get(key)}")
    extra = sorted(k for k in config.keys() if k not in set(order))
    for key in extra:
        lines.append(f"  {key}={config.get(key)}")
    return "\n".join(lines)

def format_distribution_int(
    *,
    distribution_int: Mapping[int, float],
    n: int,
    top_k: int = 10,
) -> str:
    if not distribution_int:
        return "distribution: <empty>"
    k = max(1, top_k)
    items = sorted(((int(s), float(p)) for s, p in distribution_int.items()), key=lambda kv: kv[1], reverse=True)[: k]
    lines = [f"distribution_int: top_k={len(items)}"]
    for state_int, prob in items:
        lines.append(f"  p={prob:.6f}  int={state_int:<8}  bits={_to_bitstring(value=state_int, n=n)}")
    return "\n".join(lines)


def sample_batch_objective_stats(*, qubo: QUBO, batch: SampleBatch) -> dict[str, Any]:
    if not batch.samples:
        raise ValueError("empty sample batch")

    best_obj = float("inf")
    best_bitstring: str | None = None

    for s in batch.samples:
        x = bitstring_to_x(bitstring=s.bitstring, n=batch.n, bit_order=batch.bit_order)
        obj = float(evaluate_qubo(qubo=qubo, x=np.asarray(x, dtype=int)))
        if obj < best_obj:
            best_obj = obj
            best_bitstring = s.bitstring

    most = max(batch.samples, key=lambda t: float(t.weight))
    most_x = bitstring_to_x(bitstring=most.bitstring, n=batch.n, bit_order=batch.bit_order)
    most_obj = float(evaluate_qubo(qubo=qubo, x=np.asarray(most_x, dtype=int)))

    return {
        "pre_best_objective": float(best_obj),
        "pre_best_bitstring": best_bitstring,
        "pre_most_frequent_objective": float(most_obj),
        "pre_most_frequent_bitstring": str(most.bitstring),
        "pre_most_frequent_weight": float(most.weight),
    }


def format_eval_stats(meta: Mapping[str, Any]) -> str:
    nfev = meta.get("nfev", None)
    best_cost = meta.get("best_cost", None)
    best_iter = meta.get("best_iter", None)
    final_cost = meta.get("final_cost", None)
    if nfev is None and best_cost is None and best_iter is None and final_cost is None:
        return "eval_stats: <missing>"
    return (
        "eval_stats:\n"
        f"  nfev={nfev}\n"
        f"  best_cost={best_cost}\n"
        f"  best_iter={best_iter}\n"
        f"  final_cost={final_cost}"
    )

def format_raw_sample_stats(meta: Mapping[str, Any]) -> str:
    pre_best = meta.get("pre_best_objective", None)
    pre_best_bs = meta.get("pre_best_bitstring", None)
    pre_mf = meta.get("pre_most_frequent_objective", None)
    pre_mf_bs = meta.get("pre_most_frequent_bitstring", None)
    pre_mf_w = meta.get("pre_most_frequent_weight", None)
    if pre_best is None and pre_mf is None:
        return "raw_samples: <missing>"
    return (
        "raw_samples (no local postproc):\n"
        f"  best_objective={pre_best}  best_bitstring={pre_best_bs}\n"
        f"  most_frequent_objective={pre_mf}  most_frequent_bitstring={pre_mf_bs}  weight={pre_mf_w}"
    )


def format_postprocessed_stats(meta: Mapping[str, Any]) -> str:
    post_obj = meta.get("post_objective", None)
    post_bs = meta.get("post_bitstring", None)
    post_delta = meta.get("post_delta_vs_pre_best", None)
    if post_obj is None and post_delta is None:
        return "postproc: <missing>"
    return (
        "postproc (after local postprocessor):\n"
        f"  objective={post_obj}  bitstring={post_bs}\n"
        f"  delta_vs_pre_best={post_delta}"
    )

