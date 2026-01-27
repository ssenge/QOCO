from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from qoco.core.optimizer import Optimizer
from qoco.preproc import Collapser, CollapseMapping, Reducer

from .context import PipelineContext

P = TypeVar("P")
M = TypeVar("M")
R = TypeVar("R")
Map = TypeVar("Map", bound=CollapseMapping)


@dataclass
class OptimizerPipeline(Generic[P, M, R, Map]):
    loader: Callable[[], P]
    reducers: list[Reducer[P]]
    collapsers: list[Collapser[P, Map]]
    preproc: Callable[[PipelineContext], None]
    builder: Callable[[PipelineContext], M]
    optimizer: Optimizer[M, M, R]
    postproc: Callable[[PipelineContext], None]

    def _load(self, ctx: PipelineContext) -> None:
        ctx["instance"] = self.loader()

    def run(self) -> PipelineContext:
        ctx = PipelineContext()
        self._load(ctx)

        for reducer in self.reducers:
            ctx["instance"] = reducer.convert(ctx["instance"])

        collapse_mapping = None
        for collapser in self.collapsers:
            collapsed = collapser.convert(ctx["instance"])
            ctx["instance"] = collapsed.problem
            if collapse_mapping is None:
                collapse_mapping = collapsed.mapping
            else:
                collapse_mapping = collapse_mapping.compose(collapsed.mapping)
        if collapse_mapping is not None:
            ctx["collapse_mapping"] = collapse_mapping

        self.preproc(ctx)
        ctx["model"] = self.builder(ctx)
        ctx["solution"] = self.optimizer.optimize(ctx["model"])
        self.postproc(ctx)
        return ctx
