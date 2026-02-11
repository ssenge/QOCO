from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Generic, TypeVar, cast

from rich.console import Console
from qoco.core.optimizer import Optimizer
from qoco.core.solution import OptimizerRun, ProblemSummary, Solution
from qoco.preproc import Collapser, CollapseMapping, Reducer

from .context import PipelineContext

P = TypeVar("P")
M = TypeVar("M")
R = TypeVar("R")
Map = TypeVar("Map", bound=CollapseMapping)


@dataclass
class OptimizerPipeline(Generic[P, M, R, Map]):
    init: Callable[[], P]
    reducer: Reducer[P] | list[Reducer[P]]
    collapsers: list[Collapser[P, Map]]
    preproc: Callable[[PipelineContext], PipelineContext]
    builder: Callable[[PipelineContext], PipelineContext]
    optimizer: Optimizer[M, M, Solution, OptimizerRun, ProblemSummary]
    postproc: Callable[[PipelineContext], PipelineContext]
    # Optional oracle optimizer run on the same model; if provided, result is stored in ctx["oracle_result"].
    # Placed after `postproc` so dataclass non-default fields remain first.
    oracle_optimizer: Optimizer[Any, Any, Solution, OptimizerRun, ProblemSummary] | None = None
    name: str = "unnamed"
    print_steps: bool = False
    # If enabled, print the non-empty strings returned by the format hooks below.
    # Note: format hooks are ALWAYS CALLED (NoOp defaults), regardless of this flag.
    print_stats: bool = False
    log_results: bool = False
    log_dir: str | Path = "logs/optimizer_results"

    # ---------------------------------------------------------------------
    # Stats formatting hooks (NoOp defaults; always called; never None)
    # ---------------------------------------------------------------------
    # These intentionally make no assumptions about the model type.
    _NOOP_STR: Callable[..., str] = lambda *args, **kwargs: ""  # type: ignore[assignment]

    format_after_init: Callable[[P], str] = field(
        default_factory=lambda: cast(Callable[[P], str], OptimizerPipeline._NOOP_STR)
    )
    format_after_reducer: Callable[[P, Reducer[P]], str] = field(
        default_factory=lambda: cast(Callable[[P, Reducer[P]], str], OptimizerPipeline._NOOP_STR)
    )
    format_after_collapser: Callable[[P, Collapser[P, Map], Map], str] = field(
        default_factory=lambda: cast(Callable[[P, Collapser[P, Map], Map], str], OptimizerPipeline._NOOP_STR)
    )
    format_after_preproc: Callable[[PipelineContext], str] = field(
        default_factory=lambda: cast(Callable[[PipelineContext], str], OptimizerPipeline._NOOP_STR)
    )
    format_after_builder: Callable[[PipelineContext], str] = field(
        default_factory=lambda: cast(Callable[[PipelineContext], str], OptimizerPipeline._NOOP_STR)
    )
    format_after_optimize: Callable[[PipelineContext], str] = field(
        default_factory=lambda: cast(Callable[[PipelineContext], str], OptimizerPipeline._NOOP_STR)
    )
    format_after_postproc: Callable[[PipelineContext], str] = field(
        default_factory=lambda: cast(Callable[[PipelineContext], str], OptimizerPipeline._NOOP_STR)
    )

    _console = Console()

    def _timestamp(self) -> str:
        return datetime.now().isoformat(sep=" ", timespec="seconds")

    def _print_step(self, message: str) -> None:
        if self.print_steps:
            self._console.print(message, style="blue")

    def _format_duration(self, seconds: float) -> str:
        # Simple, human-friendly duration formatting (no extra deps).
        s = float(seconds)
        if s < 0:
            s = 0.0
        if s < 1.0:
            return f"{s * 1000.0:.0f} ms"
        if s < 60.0:
            return f"{s:.1f} sec"
        if s < 3600.0:
            return f"{s / 60.0:.1f} min"
        if s < 24 * 3600.0:
            return f"{s / 3600.0:.1f} h"
        return f"{s / (24 * 3600.0):.1f} d"

    def _print_stats(self, seconds: float, text: str) -> None:
        # IMPORTANT: the format hook is ALWAYS called by the caller before passing `text` here.
        # This method only controls printing.
        if not self.print_stats:
            return
        t = (text or "").strip()
        if not t:
            return
        self._console.print(f"[{self.name}] Finished in {self._format_duration(seconds)}", style="cyan")
        self._console.print(t)

    def _result_log_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{self.name}_{self.optimizer.name}.jsonl"
        return Path(self.log_dir) / filename

    def _load(self, ctx: PipelineContext) -> None:
        ctx["instance"] = self.init()

    def run(self) -> PipelineContext:
        pipeline_t0 = time.perf_counter()
        ctx = PipelineContext()
        if self.print_steps:
            self._print_step(f"*** Starting pipeline {self.name}...")
            self._print_step(f" - Init at {self._timestamp()}")
        t0 = time.perf_counter()
        self._load(ctx)
        t_init = time.perf_counter() - t0
        # Always call (NoOp default); printing is controlled by `print_stats`.
        self._print_stats(t_init, self.format_after_init(ctx["instance"]))  # type: ignore[arg-type]

        reducers = self.reducer if isinstance(self.reducer, list) else [self.reducer]
        for reducer in reducers:
            self._print_step(f" - Reducer {type(reducer).__name__} at {self._timestamp()}")
            t0 = time.perf_counter()
            ctx["instance"] = reducer.convert(ctx["instance"])
            dt = time.perf_counter() - t0
            self._print_stats(dt, self.format_after_reducer(ctx["instance"], reducer))

        collapse_mapping = None
        for collapser in self.collapsers:
            self._print_step(f" - Collapser {type(collapser).__name__} at {self._timestamp()}")
            t0 = time.perf_counter()
            collapsed = collapser.convert(ctx["instance"])
            dt = time.perf_counter() - t0
            ctx["instance"] = collapsed.problem
            if collapse_mapping is None:
                collapse_mapping = collapsed.mapping
            else:
                collapse_mapping = collapse_mapping.compose(collapsed.mapping)
            # Always call (NoOp default); pass the composed mapping so far.
            self._print_stats(dt, self.format_after_collapser(ctx["instance"], collapser, collapse_mapping))  # type: ignore[arg-type]
        if collapse_mapping is not None:
            ctx["collapse_mapping"] = collapse_mapping

        self._print_step(f"*** Preproc at {self._timestamp()}")
        t0 = time.perf_counter()
        ctx = self.preproc(ctx)
        self._print_stats(time.perf_counter() - t0, self.format_after_preproc(ctx))
        self._print_step(f"*** Model Builder at {self._timestamp()}")
        t0 = time.perf_counter()
        ctx = self.builder(ctx)
        self._print_stats(time.perf_counter() - t0, self.format_after_builder(ctx))
        self._print_step(f"*** Starting Optimizer '{self.optimizer.name}' at {self._timestamp()}")
        t0 = time.perf_counter()
        ctx["result"] = self.optimizer.optimize(ctx["model"])
        t_opt = time.perf_counter() - t0
        # Override problem summary from instance (optimizer receives model, not Problem)
        problem_summary = ctx["instance"].summary()
        # Extract model stats. We assume models implement Pyomo-like `nvariables()`/`nconstraints()`.
        # (QUBO implements these too.)
        problem_summary = replace(
            problem_summary,
            num_vars=int(ctx["model"].nvariables()),
            num_constraints=int(ctx["model"].nconstraints()),
        )
        ctx["result"] = replace(ctx["result"], problem=problem_summary)

        # Optional oracle run (same model); store result or None in ctx.
        if self.oracle_optimizer is None:
            ctx["oracle_result"] = None
        else:
            oracle_res = self.oracle_optimizer.optimize(ctx["model"])
            # Mirror problem summary override for consistency.
            ctx["oracle_result"] = replace(oracle_res, problem=problem_summary)

        self._print_stats(t_opt, self.format_after_optimize(ctx))
        self._print_step(f"*** Postproc at {self._timestamp()}")
        t0 = time.perf_counter()
        ctx = self.postproc(ctx)
        self._print_stats(time.perf_counter() - t0, self.format_after_postproc(ctx))
        if self.log_results:
            ctx["result"].write(self._result_log_path())
        if self.print_steps:
            total_dt = time.perf_counter() - pipeline_t0
            self._print_step(f"*** Pipeline finished in {self._format_duration(total_dt)} at {self._timestamp()}")
        return ctx
