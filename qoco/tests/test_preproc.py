from __future__ import annotations

from dataclasses import dataclass

from qoco.preproc.base import CollapseMapping, Collapsed, Collapser, Reducer


@dataclass(frozen=True)
class DummyProblem:
    items: list[int]


@dataclass
class EvenReducer(Reducer[DummyProblem]):
    def convert(self, problem: DummyProblem) -> DummyProblem:
        return DummyProblem(items=[x for x in problem.items if x % 2 == 0])


@dataclass
class PairCollapser(Collapser[DummyProblem, CollapseMapping[DummyProblem]]):
    def convert(self, problem: DummyProblem) -> Collapsed[DummyProblem, CollapseMapping[DummyProblem]]:
        collapsed: list[int] = []
        for i in range(0, len(problem.items), 2):
            chunk = problem.items[i : i + 2]
            collapsed.append(chunk[0])

        def _expand(x: DummyProblem) -> DummyProblem:
            return x

        mapping = CollapseMapping(steps=[_expand])
        return Collapsed(problem=DummyProblem(items=collapsed), mapping=mapping)


def test_reducer_keeps_type() -> None:
    reduced = EvenReducer().convert(DummyProblem(items=[1, 2, 3, 4]))
    assert reduced.items == [2, 4]


def test_collapser_chainable_via_mapping_compose() -> None:
    base = DummyProblem(items=[0, 1, 2, 3])
    first = PairCollapser().convert(base)
    second = PairCollapser().convert(first.problem)
    composed = first.mapping.compose(second.mapping)

    assert first.problem.items == [0, 2]
    assert second.problem.items == [0]
    assert len(composed.steps) == 2
