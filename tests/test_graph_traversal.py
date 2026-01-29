from __future__ import annotations

from typing import Iterable

import networkx as nx

from qoco.utils.graph.traversal import GraphTraversal, MaxPathLengthFilter, PathFilter


class SumLimitFilter(PathFilter[int, int]):
    def __init__(self, limit: int) -> None:
        self.limit = int(limit)

    def state_init(self, start: int, state: int | None) -> int:
        return int(start)

    def state_update(self, state: int | None, current: int, nxt: int) -> int:
        base = int(state) if state is not None else int(current)
        return base + int(nxt)

    def filter(self, path: list[int], nxt: int, state: int | None) -> bool:
        next_state = self.state_update(state, path[-1], nxt)
        return next_state <= self.limit


def _build_graph(edges: Iterable[tuple[int, int]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def test_dfs_order_and_paths() -> None:
    graph = _build_graph([(1, 2), (1, 3), (2, 4), (3, 4)])
    traversal = GraphTraversal(graph)

    paths = list(traversal.dfs([1]))

    assert paths == [[1], [1, 3], [1, 3, 4], [1, 2], [1, 2, 4]]


def test_bfs_order_and_paths() -> None:
    graph = _build_graph([(1, 2), (1, 3), (2, 4), (3, 4)])
    traversal = GraphTraversal(graph)

    paths = list(traversal.bfs([1]))

    assert paths == [[1], [1, 2], [1, 3], [1, 2, 4], [1, 3, 4]]


def test_filters_apply_to_expansion() -> None:
    graph = _build_graph([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    traversal = GraphTraversal(graph, filters=[MaxPathLengthFilter(2), SumLimitFilter(7)])

    dfs_paths = list(traversal.dfs([1]))
    bfs_paths = list(traversal.bfs([1]))

    assert dfs_paths == [[1], [1, 3], [1, 2]]
    assert bfs_paths == [[1], [1, 2], [1, 3]]
