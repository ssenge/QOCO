from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Generic, TypeVar

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
    name: str = "unnamed"
    print_steps: bool = False
    log_results: bool = False
    log_dir: str | Path = "logs/optimizer_results"

    _console = Console()

    def _timestamp(self) -> str:
        return datetime.now().isoformat(sep=" ", timespec="seconds")

    def _print_step(self, message: str) -> None:
        if self.print_steps:
            self._console.print(message, style="blue")

    def _result_log_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{self.name}_{self.optimizer.name}.jsonl"
        return Path(self.log_dir) / filename

    def _load(self, ctx: PipelineContext) -> None:
        ctx["instance"] = self.init()

    def run(self) -> PipelineContext:
        ctx = PipelineContext()
        if self.print_steps:
            self._print_step(f"*** Starting pipeline {self.name}...")
            self._print_step(f" - Init at {self._timestamp()}")
        self._load(ctx)

        reducers = self.reducer if isinstance(self.reducer, list) else [self.reducer]
        for reducer in reducers:
            self._print_step(f" - Reducer {type(reducer).__name__} at {self._timestamp()}")
            ctx["instance"] = reducer.convert(ctx["instance"])

        collapse_mapping = None
        for collapser in self.collapsers:
            self._print_step(f" - Collapser {type(collapser).__name__} at {self._timestamp()}")
            collapsed = collapser.convert(ctx["instance"])
            ctx["instance"] = collapsed.problem
            if collapse_mapping is None:
                collapse_mapping = collapsed.mapping
            else:
                collapse_mapping = collapse_mapping.compose(collapsed.mapping)
        if collapse_mapping is not None:
            ctx["collapse_mapping"] = collapse_mapping

        self._print_step(f"*** Preproc at {self._timestamp()}")
        ctx = self.preproc(ctx)
        self._print_step(f"*** Model Builder at {self._timestamp()}")
        ctx = self.builder(ctx)
        self._print_step(f"*** Starting Optimizer at {self._timestamp()}")
        ctx["result"] = self.optimizer.optimize(ctx["model"])
        # Override problem summary from instance (optimizer receives model, not Problem)
        problem_summary = ctx["instance"].summary()
        # Extract model stats (num_vars, num_constraints) from the model
        if hasattr(ctx["model"], "nvariables") and hasattr(ctx["model"], "nconstraints"):
            problem_summary = replace(
                problem_summary,
                num_vars=ctx["model"].nvariables(),
                num_constraints=ctx["model"].nconstraints(),
            )
        ctx["result"] = replace(ctx["result"], problem=problem_summary)
        self._print_step(f"*** Postproc at {self._timestamp()}")
        ctx = self.postproc(ctx)
        if self.log_results:
            ctx["result"].write(self._result_log_path())
        return ctx
