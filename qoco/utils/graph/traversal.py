from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, TypeVar

import networkx as nx

NodeT = TypeVar("NodeT")
StateT = TypeVar("StateT")


class PathFilter(Generic[NodeT, StateT]):
    def state_init(self, start: NodeT, state: StateT | None) -> StateT | None:
        return state

    def state_update(self, state: StateT | None, current: NodeT, nxt: NodeT) -> StateT | None:
        return state

    def filter(self, path: list[NodeT], nxt: NodeT, state: StateT | None) -> bool:
        return True

    def accept(self, path: list[NodeT], state: StateT | None) -> bool:
        return True


@dataclass(frozen=True)
class MaxPathLengthFilter(PathFilter[NodeT, StateT]):
    max_length: int

    def filter(self, path: list[NodeT], nxt: NodeT, state: StateT | None) -> bool:
        return len(path) < int(self.max_length)


@dataclass(frozen=True)
class MinPathLengthFilter(PathFilter[NodeT, StateT]):
    min_length: int

    def accept(self, path: list[NodeT], state: StateT | None) -> bool:
        return len(path) >= int(self.min_length)


@dataclass(frozen=True)
class RequiredNodesFilter(PathFilter[NodeT, StateT]):
    required: set[NodeT]

    def accept(self, path: list[NodeT], state: StateT | None) -> bool:
        return self.required.issubset(path)


@dataclass(frozen=True)
class ForbiddenNodesFilter(PathFilter[NodeT, StateT]):
    forbidden: set[NodeT]

    def filter(self, path: list[NodeT], nxt: NodeT, state: StateT | None) -> bool:
        return nxt not in self.forbidden


@dataclass(frozen=True)
class RequiredEdgesFilter(PathFilter[NodeT, StateT]):
    required: set[tuple[NodeT, NodeT]]

    def accept(self, path: list[NodeT], state: StateT | None) -> bool:
        edges = set(zip(path, path[1:]))
        return self.required.issubset(edges)


@dataclass(frozen=True)
class ForbiddenEdgesFilter(PathFilter[NodeT, StateT]):
    forbidden: set[tuple[NodeT, NodeT]]

    def filter(self, path: list[NodeT], nxt: NodeT, state: StateT | None) -> bool:
        return (path[-1], nxt) not in self.forbidden


@dataclass(frozen=True)
class UniqueNodeAttrFilter(PathFilter[NodeT, StateT]):
    graph: nx.DiGraph
    attr: str

    def filter(self, path: list[NodeT], nxt: NodeT, state: StateT | None) -> bool:
        nxt_val = self.graph.nodes[nxt].get(self.attr)
        return all(self.graph.nodes[n].get(self.attr) != nxt_val for n in path)


@dataclass(frozen=True)
class MaxEdgeAttrSumFilter(PathFilter[NodeT, float]):
    graph: nx.DiGraph
    attr: str
    max_total: float
    default: float = 0.0

    def state_init(self, start: NodeT, state: float | None) -> float:
        return 0.0

    def state_update(self, state: float | None, current: NodeT, nxt: NodeT) -> float:
        base = float(state) if state is not None else 0.0
        edge_val = float(self.graph.edges[current, nxt].get(self.attr, self.default))
        return base + edge_val

    def filter(self, path: list[NodeT], nxt: NodeT, state: float | None) -> bool:
        return self.state_update(state, path[-1], nxt) <= float(self.max_total)


@dataclass(frozen=True)
class MinEdgeAttrSumFilter(PathFilter[NodeT, float]):
    graph: nx.DiGraph
    attr: str
    min_total: float
    default: float = 0.0

    def state_init(self, start: NodeT, state: float | None) -> float:
        return 0.0

    def state_update(self, state: float | None, current: NodeT, nxt: NodeT) -> float:
        base = float(state) if state is not None else 0.0
        edge_val = float(self.graph.edges[current, nxt].get(self.attr, self.default))
        return base + edge_val

    def accept(self, path: list[NodeT], state: float | None) -> bool:
        total = float(state) if state is not None else 0.0
        return total >= float(self.min_total)


@dataclass(frozen=True)
class MaxNodeAttrSumFilter(PathFilter[NodeT, float]):
    graph: nx.DiGraph
    attr: str
    max_total: float
    default: float = 0.0

    def state_init(self, start: NodeT, state: float | None) -> float:
        return float(self.graph.nodes[start].get(self.attr, self.default))

    def state_update(self, state: float | None, current: NodeT, nxt: NodeT) -> float:
        base = float(state) if state is not None else 0.0
        node_val = float(self.graph.nodes[nxt].get(self.attr, self.default))
        return base + node_val

    def filter(self, path: list[NodeT], nxt: NodeT, state: float | None) -> bool:
        return self.state_update(state, path[-1], nxt) <= float(self.max_total)


@dataclass(frozen=True)
class MinNodeAttrSumFilter(PathFilter[NodeT, float]):
    graph: nx.DiGraph
    attr: str
    min_total: float
    default: float = 0.0

    def state_init(self, start: NodeT, state: float | None) -> float:
        return float(self.graph.nodes[start].get(self.attr, self.default))

    def state_update(self, state: float | None, current: NodeT, nxt: NodeT) -> float:
        base = float(state) if state is not None else 0.0
        node_val = float(self.graph.nodes[nxt].get(self.attr, self.default))
        return base + node_val

    def accept(self, path: list[NodeT], state: float | None) -> bool:
        total = float(state) if state is not None else 0.0
        return total >= float(self.min_total)


@dataclass(frozen=True)
class PathPredicateFilter(PathFilter[NodeT, StateT]):
    allow_step: Callable[[list[NodeT], NodeT, StateT | None], bool]
    allow_path: Callable[[list[NodeT], StateT | None], bool] | None = None

    def filter(self, path: list[NodeT], nxt: NodeT, state: StateT | None) -> bool:
        return bool(self.allow_step(path, nxt, state))

    def accept(self, path: list[NodeT], state: StateT | None) -> bool:
        if self.allow_path is None:
            return True
        return bool(self.allow_path(path, state))


@dataclass(frozen=True)
class GraphTraversal:
    graph: nx.DiGraph
    filters: list[PathFilter] | None = None

    def dfs(self, start_nodes: Iterable[NodeT]) -> Iterator[list[NodeT]]:
        active_filters = self.filters or []
        for start in start_nodes:
            state: StateT | None = None
            for f in active_filters:
                state = f.state_init(start, state)
            stack: list[tuple[NodeT, list[NodeT], StateT | None]] = [(start, [start], state)]
            while stack:
                node, path, state = stack.pop()
                if all(f.accept(path, state) for f in active_filters):
                    yield path
                for nxt in self.graph.neighbors(node):
                    if nxt in path:
                        continue
                    if not all(f.filter(path, nxt, state) for f in active_filters):
                        continue
                    next_state = state
                    for f in active_filters:
                        next_state = f.state_update(next_state, node, nxt)
                    stack.append((nxt, [*path, nxt], next_state))

    def bfs(self, start_nodes: Iterable[NodeT]) -> Iterator[list[NodeT]]:
        active_filters = self.filters or []
        queue: list[tuple[NodeT, list[NodeT], StateT | None]] = []
        for start in start_nodes:
            state: StateT | None = None
            for f in active_filters:
                state = f.state_init(start, state)
            queue.append((start, [start], state))
        while queue:
            node, path, state = queue.pop(0)
            if all(f.accept(path, state) for f in active_filters):
                yield path
            for nxt in self.graph.neighbors(node):
                if nxt in path:
                    continue
                if not all(f.filter(path, nxt, state) for f in active_filters):
                    continue
                next_state = state
                for f in active_filters:
                    next_state = f.state_update(next_state, node, nxt)
                queue.append((nxt, [*path, nxt], next_state))


def prune_graph_by_paths(
    graph: nx.DiGraph,
    start_nodes: Iterable[NodeT],
    filters: list[PathFilter] | None = None,
) -> nx.DiGraph:
    used_edges: set[tuple[NodeT, NodeT]] = set()
    for path in GraphTraversal(graph, filters=filters).dfs(start_nodes):
        used_edges.update(zip(path, path[1:]))
    return graph.edge_subgraph(used_edges).copy()
