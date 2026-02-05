from __future__ import annotations

from typing import Iterable

import networkx as nx

from qoco.utils.graph.traversal import GraphTraversal, TernaryFilter


class CombinedLimitFilter(TernaryFilter[list[int], int | None, object | None]):
    def __init__(self, max_length: int, max_sum: int) -> None:
        self.max_length = int(max_length)
        self.max_sum = int(max_sum)

    def accept(self, path: list[int], nxt: int | None, state: object | None) -> bool:
        if nxt is None:
            return True
        return len(path) < self.max_length and (sum(path) + int(nxt)) <= self.max_sum


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
    traversal = GraphTraversal(graph)
    path_filter = CombinedLimitFilter(max_length=2, max_sum=7)

    dfs_paths = list(traversal.dfs([1], path_filter=path_filter))
    bfs_paths = list(traversal.bfs([1], path_filter=path_filter))

    assert dfs_paths == [[1], [1, 3], [1, 2]]
    assert bfs_paths == [[1], [1, 2], [1, 3]]
