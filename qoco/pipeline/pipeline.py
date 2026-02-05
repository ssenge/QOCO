from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Generic, TypeVar

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
    reducers: list[Reducer[P]]
    collapsers: list[Collapser[P, Map]]
    preproc: Callable[[PipelineContext], None]
    builder: Callable[[PipelineContext], M]
    optimizer: Optimizer[M, M, Solution, OptimizerRun, ProblemSummary]
    postproc: Callable[[PipelineContext], None]
    name: str = "unnamed"
    print_steps: bool = False
    log_results: bool = False
    log_dir: str | Path = "logs/optimizer_results"

    def _timestamp(self) -> str:
        return datetime.now().isoformat(sep=" ", timespec="seconds")

    def _print_step(self, message: str) -> None:
        if self.print_steps:
            print(message)

    def _result_log_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{self.name}_{self.optimizer.name}.jsonl"
        return Path(self.log_dir) / filename

    def _load(self, ctx: PipelineContext) -> None:
        ctx["instance"] = self.init()

    def run(self) -> PipelineContext:
        ctx = PipelineContext()
        if self.print_steps:
            print(f"Starting pipeline {self.name}...")
            print(f"\tInit at {self._timestamp()}")
        self._load(ctx)

        for reducer in self.reducers:
            self._print_step(f"\tReducer {type(reducer).__name__} at {self._timestamp()}")
            ctx["instance"] = reducer.convert(ctx["instance"])

        collapse_mapping = None
        for collapser in self.collapsers:
            self._print_step(f"\tCollapser {type(collapser).__name__} at {self._timestamp()}")
            collapsed = collapser.convert(ctx["instance"])
            ctx["instance"] = collapsed.problem
            if collapse_mapping is None:
                collapse_mapping = collapsed.mapping
            else:
                collapse_mapping = collapse_mapping.compose(collapsed.mapping)
        if collapse_mapping is not None:
            ctx["collapse_mapping"] = collapse_mapping

        self._print_step(f"\tPreproc at {self._timestamp()}")
        self.preproc(ctx)
        self._print_step(f"\tBuilder at {self._timestamp()}")
        ctx["model"] = self.builder(ctx)
        self._print_step(f"\tOptimizer at {self._timestamp()}")
        ctx["result"] = self.optimizer.optimize(ctx["model"])
        self._print_step(f"\tPostproc at {self._timestamp()}")
        self.postproc(ctx)
        if self.log_results:
            ctx["result"].write(self._result_log_path())
        return ctx
