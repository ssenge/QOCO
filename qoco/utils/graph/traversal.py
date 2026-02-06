from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Generic, Iterable, Iterator, TypeVar, Self

import networkx as nx

NodeT = TypeVar("NodeT")
StateT = TypeVar("StateT")
ItemT = TypeVar("ItemT")
LeftT = TypeVar("LeftT")
RightT = TypeVar("RightT")
MidT = TypeVar("MidT")


class Filter(Generic[ItemT]):
    @classmethod
    def combine(cls, filters: list[Self] | None) -> Self:
        return reduce(lambda left, right: left.compose(right), filters or [], AcceptFilter())

    def accept(self, *args) -> bool:
        return True

    def __call__(self, *args) -> bool:
        return self.accept(*args)

    def compose(self, other: Self) -> Self: 
        return _AndFilter(self, other)  # This is sofar AND-only composition. TODO: add OR-composition.


class AcceptFilter(Filter[ItemT]):
    def accept(self, *args) -> bool:
        return True


class RejectFilter(Filter[ItemT]):
    def accept(self, *args) -> bool:
        return False


class TernaryFilter(Filter[tuple[LeftT, MidT, RightT]], ABC):
    @abstractmethod
    def accept(self, left: LeftT, mid: MidT, right: RightT) -> bool:
        raise NotImplementedError


class UnaryFilter(Filter[ItemT], ABC):
    @abstractmethod
    def accept(self, item: ItemT) -> bool:
        raise NotImplementedError


class BinaryFilter(Filter[tuple[LeftT, RightT]], ABC):
    @abstractmethod
    def accept(self, left: LeftT, right: RightT) -> bool:
        raise NotImplementedError


class PathStateTracker(Generic[NodeT, StateT], ABC):
    @abstractmethod
    def state_init(self, start: NodeT) -> StateT | None:
        raise NotImplementedError

    @abstractmethod
    def state_update(self, state: StateT | None, current: NodeT, nxt: NodeT) -> StateT | None:
        raise NotImplementedError


@dataclass
class MaxItemFilter(UnaryFilter[ItemT]):
    max_items: int
    _count: int = field(default=0, init=False)

    def accept(self, item: ItemT) -> bool:
        if self._count >= self.max_items:
            return False
        self._count += 1
        return True


@dataclass
class _AndFilter(Filter[ItemT]):
    left: Filter[ItemT]
    right: Filter[ItemT]

    def accept(self, *args) -> bool:
        return self.left.accept(*args) and self.right.accept(*args)




def apply_filter(filter_obj: Filter[ItemT], items: Iterable[ItemT]) -> list[ItemT]:
    return [item for item in items if filter_obj.accept(item)]


def prepare_filter(filter_obj: Filter, *args, **kwargs) -> None:
    if hasattr(filter_obj, "prepare"):
        filter_obj.prepare(*args, **kwargs)
        return
    if isinstance(filter_obj, _AndFilter):
        prepare_filter(filter_obj.left, *args, **kwargs)
        prepare_filter(filter_obj.right, *args, **kwargs)
        return


def apply_neighbor_filter(
    filter_obj: BinaryFilter[NodeT, NodeT],
    node: NodeT,
    neighbors: Iterable[NodeT],
) -> list[NodeT]:
    return [nxt for nxt in neighbors if filter_obj.accept(node, nxt)]


@dataclass(frozen=True)
class MaxPathLengthFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    max_length: int

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            return True
        return len(path) < self.max_length


@dataclass(frozen=True)
class MinPathLengthFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    min_length: int

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        return len(path) >= self.min_length


@dataclass(frozen=True)
class RequiredNodesFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    required: set[NodeT]

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        return self.required.issubset(path)


@dataclass(frozen=True)
class ForbiddenNodesFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    forbidden: set[NodeT]

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            return True
        return nxt not in self.forbidden


@dataclass(frozen=True)
class RequiredEdgesFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    required: set[tuple[NodeT, NodeT]]

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        edges = set(zip(path, path[1:]))
        return self.required.issubset(edges)


@dataclass(frozen=True)
class ForbiddenEdgesFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    forbidden: set[tuple[NodeT, NodeT]]

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            return True
        return (path[-1], nxt) not in self.forbidden


@dataclass(frozen=True)
class UniqueNodeAttrFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    graph: nx.DiGraph
    attr: str

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            return True
        nxt_val = self.graph.nodes[nxt].get(self.attr)
        return all(self.graph.nodes[n].get(self.attr) != nxt_val for n in path)


@dataclass(frozen=True)
class MaxEdgeAttrSumFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    graph: nx.DiGraph
    attr: str
    max_total: float
    default: float = 0.0

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            return True
        total = 0.0
        for u, v in zip(path, path[1:]):
            total += float(self.graph.edges[u, v].get(self.attr, self.default))
        total += float(self.graph.edges[path[-1], nxt].get(self.attr, self.default))
        return total <= float(self.max_total)


@dataclass(frozen=True)
class MinEdgeAttrSumFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    graph: nx.DiGraph
    attr: str
    min_total: float
    default: float = 0.0

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        total = 0.0
        for u, v in zip(path, path[1:]):
            total += float(self.graph.edges[u, v].get(self.attr, self.default))
        return total >= float(self.min_total)


@dataclass(frozen=True)
class MaxNodeAttrSumFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    graph: nx.DiGraph
    attr: str
    max_total: float
    default: float = 0.0

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            return True
        total = 0.0
        for node in path:
            total += float(self.graph.nodes[node].get(self.attr, self.default))
        total += float(self.graph.nodes[nxt].get(self.attr, self.default))
        return total <= float(self.max_total)


@dataclass(frozen=True)
class MinNodeAttrSumFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    graph: nx.DiGraph
    attr: str
    min_total: float
    default: float = 0.0

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        total = 0.0
        for node in path:
            total += float(self.graph.nodes[node].get(self.attr, self.default))
        return total >= float(self.min_total)


@dataclass(frozen=True)
class PathPredicateFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    allow_step: Callable[[list[NodeT], NodeT | None, StateT | None], bool]
    allow_path: Callable[[list[NodeT], StateT | None], bool] | None = None

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is None:
            if self.allow_path is None:
                return True
            return bool(self.allow_path(path, state))
        return bool(self.allow_step(path, nxt, state))


@dataclass(frozen=True)
class GraphTraversal:
    graph: nx.DiGraph

    def dfs_v2(
        self,
        start_nodes: Iterable[NodeT],
        neighbor_filter: BinaryFilter[NodeT, NodeT] = AcceptFilter(),
        path_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        state_tracker: PathStateTracker[NodeT, StateT] | None = None,
        max_only: bool = False,
    ) -> Iterator[list[NodeT]]:
        if not max_only:
            for start in start_nodes:
                state: StateT | None = state_tracker.state_init(start) if state_tracker else None
                stack: list[tuple[NodeT, list[NodeT], StateT | None]] = [(start, [start], state)]
                while stack:
                    node, path, state = stack.pop()
                    if path_filter.accept(path, None, state):
                        if not global_filter.accept(path):
                            return
                        yield path
                    for nxt in apply_neighbor_filter(
                        neighbor_filter, node, self.graph.neighbors(node)
                    ):
                        if nxt in path:
                            continue
                        if not path_filter.accept(path, nxt, state):
                            continue
                        next_state = (
                            state_tracker.state_update(state, node, nxt)
                            if state_tracker
                            else None
                        )
                        stack.append((nxt, [*path, nxt], next_state))
            return
        for start in start_nodes:
            state: StateT | None = state_tracker.state_init(start) if state_tracker else None
            node = start
            path = [start]
            while True:
                neighbors = apply_neighbor_filter(
                    neighbor_filter, node, self.graph.neighbors(node)
                )
                next_node: NodeT | None = None
                for nxt in neighbors:
                    if nxt in path:
                        continue
                    if not path_filter.accept(path, nxt, state):
                        continue
                    next_node = nxt
                    break
                if next_node is None:
                    if path_filter.accept(path, None, state):
                        if not global_filter.accept(path):
                            return
                        yield path
                    break
                next_state = (
                    state_tracker.state_update(state, node, next_node)
                    if state_tracker
                    else None
                )
                path = [*path, next_node]
                node = next_node
                state = next_state

    def dfs_iterative(
        self,
        start_nodes: Iterable[NodeT],
        neighbor_filter: BinaryFilter[NodeT, NodeT] = AcceptFilter(),
        path_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        state_tracker: PathStateTracker[NodeT, StateT] | None = None,
        N: int = 100,
        T: int = 1,
        shuffle: bool = True,
        max_only: bool = True,
    ) -> Iterator[list[NodeT]]:
        starts = list(start_nodes)
        uncovered = set(starts)
        if not starts or N <= 0 or T <= 0:
            return
        for _ in range(T):
            if not uncovered:
                break
            remaining = N
            while remaining > 0 and uncovered:
                candidates = list(uncovered)
                if shuffle:
                    import random

                    random.shuffle(candidates)
                start = candidates[0]
                remaining -= 1
                for path in self.dfs_v2(
                    start_nodes=[start],
                    neighbor_filter=neighbor_filter,
                    path_filter=path_filter,
                    global_filter=global_filter,
                    state_tracker=state_tracker,
                    max_only=max_only,
                ):
                    uncovered -= set(path)
                    yield path
                if not uncovered:
                    break
    def dfs_max_only(
        self,
        start_nodes: Iterable[NodeT],
        neighbor_filter: BinaryFilter[NodeT, NodeT] = AcceptFilter(),
        path_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        state_tracker: PathStateTracker[NodeT, StateT] | None = None,
    ) -> Iterator[list[NodeT]]:
        for start in start_nodes:
            state: StateT | None = state_tracker.state_init(start) if state_tracker else None
            node = start
            path = [start]
            while True:
                neighbors = apply_neighbor_filter(
                    neighbor_filter, node, self.graph.neighbors(node)
                )
                next_node: NodeT | None = None
                for nxt in neighbors:
                    if nxt in path:
                        continue
                    if not path_filter.accept(path, nxt, state):
                        continue
                    next_node = nxt
                    break
                if next_node is None:
                    if path_filter.accept(path, None, state):
                        if not global_filter.accept(path):
                            return
                        yield path
                    break
                next_state = (
                    state_tracker.state_update(state, node, next_node)
                    if state_tracker
                    else None
                )
                path = [*path, next_node]
                node = next_node
                state = next_state

    def dfs(
        self,
        start_nodes: Iterable[NodeT],
        neighbor_filter: BinaryFilter[NodeT, NodeT] = AcceptFilter(),
        path_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        state_tracker: PathStateTracker[NodeT, StateT] | None = None,
    ) -> Iterator[list[NodeT]]:
        for start in start_nodes:
            state: StateT | None = state_tracker.state_init(start) if state_tracker else None
            stack: list[tuple[NodeT, list[NodeT], StateT | None]] = [(start, [start], state)]
            while stack:
                node, path, state = stack.pop()
                if path_filter.accept(path, None, state):
                    if not global_filter.accept(path):
                        return
                    yield path
                for nxt in apply_neighbor_filter(neighbor_filter, node, self.graph.neighbors(node)):
                    if nxt in path:
                        continue
                    if not path_filter.accept(path, nxt, state):
                        continue
                    next_state = (
                        state_tracker.state_update(state, node, nxt) if state_tracker else None
                    )
                    stack.append((nxt, [*path, nxt], next_state))

    def bfs(
        self,
        start_nodes: Iterable[NodeT],
        neighbor_filter: BinaryFilter[NodeT, NodeT] = AcceptFilter(),
        path_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        state_tracker: PathStateTracker[NodeT, StateT] | None = None,
    ) -> Iterator[list[NodeT]]:
        queue: list[tuple[NodeT, list[NodeT], StateT | None]] = []
        for start in start_nodes:
            state: StateT | None = state_tracker.state_init(start) if state_tracker else None
            queue.append((start, [start], state))
        while queue:
            node, path, state = queue.pop(0)
            if path_filter.accept(path, None, state):
                if not global_filter.accept(path):
                    return
                yield path
            for nxt in apply_neighbor_filter(neighbor_filter, node, self.graph.neighbors(node)):
                if nxt in path:
                    continue
                if not path_filter.accept(path, nxt, state):
                    continue
                next_state = (
                    state_tracker.state_update(state, node, nxt) if state_tracker else None
                )
                queue.append((nxt, [*path, nxt], next_state))


def prune_graph_by_paths(
    graph: nx.DiGraph,
    start_nodes: Iterable[NodeT],
    path_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
    state_tracker: PathStateTracker[NodeT, StateT] | None = None,
) -> nx.DiGraph:
    used_edges: set[tuple[NodeT, NodeT]] = set()
    for path in GraphTraversal(graph).dfs(
        start_nodes, path_filter=path_filter, state_tracker=state_tracker
    ):
        used_edges.update(zip(path, path[1:]))
    return graph.edge_subgraph(used_edges).copy()
