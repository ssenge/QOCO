from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
import random
from typing import Callable, Generic, Iterable, Iterator, TypeVar, Self

from tailrec import tailrec

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

    def __mul__(self, other: Self) -> Self:
        return self.compose(other)

    def __or__(self, other: Self) -> Self:
        return _OrFilter(self, other)


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
class MinLengthFilter(UnaryFilter[list[ItemT]]):
    min_len: int

    def accept(self, item: list[ItemT]) -> bool:
        return len(item) >= self.min_len


@dataclass
class SkipFilter(UnaryFilter[list[ItemT]]):
    step: int
    _count: int = field(default=0, init=False)

    def accept(self, item: list[ItemT]) -> bool:
        self._count += 1
        return (self._count % self.step) == 1


@dataclass
class RandomSkipFilter(UnaryFilter[list[ItemT]]):
    probability: float
    rng: random.Random

    def accept(self, item: list[ItemT]) -> bool:
        return self.rng.random() < self.probability


class SubpathExpander:
    @staticmethod
    def expand(
        path: list[NodeT],
        subpath_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        start_only: bool = False,
    ) -> list[list[NodeT]]:
        subpaths: list[list[NodeT]] = []
        start_range = range(1) if start_only else range(len(path))
        for start in start_range:
            for end in range(start + 1, len(path) + 1):
                subpath = path[start:end]
                if subpath_filter.accept(subpath):
                    subpaths.append(subpath)
        return subpaths


@dataclass
class _AndFilter(Filter[ItemT]):
    left: Filter[ItemT]
    right: Filter[ItemT]

    def accept(self, *args) -> bool:
        return self.left.accept(*args) and self.right.accept(*args)

    def order(self, node, neighbors):
        ordered = neighbors
        if hasattr(self.left, "order"):
            ordered = self.left.order(node, ordered)
        if hasattr(self.right, "order"):
            ordered = self.right.order(node, ordered)
        return ordered




@dataclass
class _OrFilter(Filter[ItemT]):
    left: Filter[ItemT]
    right: Filter[ItemT]

    def accept(self, *args) -> bool:
        return self.left.accept(*args) or self.right.accept(*args)

    def order(self, node, neighbors):
        ordered = neighbors
        if hasattr(self.left, "order"):
            ordered = self.left.order(node, ordered)
        if hasattr(self.right, "order"):
            ordered = self.right.order(node, ordered)
        return ordered



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
    if isinstance(filter_obj, _OrFilter):
        prepare_filter(filter_obj.left, *args, **kwargs)
        prepare_filter(filter_obj.right, *args, **kwargs)
        return


def apply_neighbor_filter(
    filter_obj: Filter[tuple[NodeT, NodeT]],
    node: NodeT,
    neighbors: Iterable[NodeT],
) -> list[NodeT]:
    ordered = list(neighbors)
    if hasattr(filter_obj, "order"):
        ordered = filter_obj.order(node, ordered)
    return [nxt for nxt in ordered if filter_obj.accept(node, nxt)]


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


@dataclass
class MinPathLengthEmitFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    min_len: int

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        return len(path) >= self.min_len


@dataclass
class KeepEmitFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    step: int
    _count: int = field(default=0, init=False)

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        if nxt is not None:
            return True
        self._count += 1
        return (self._count % self.step) == 1


@dataclass(frozen=True)
class PathPredicateFilter(TernaryFilter[list[NodeT], NodeT | None, StateT | None]):
    allow_step: Callable[[list[NodeT], NodeT | None, StateT | None], bool] = lambda _path, _nxt, _state: True
    allow_path: Callable[[list[NodeT], StateT | None], bool] = lambda _path, _state: True

    def accept(self, path: list[NodeT], nxt: NodeT | None, state: StateT | None) -> bool:
        return self.allow_path(path, state) if nxt is None else self.allow_step(path, nxt, state)


@dataclass(frozen=True)
class GraphTraversal:
    graph: nx.DiGraph
    
    def dfs(
        self,
        start_nodes: Iterable[NodeT],
        state_tracker: PathStateTracker[NodeT, StateT],
        neighbor_filter: Filter[tuple[NodeT, NodeT]] = AcceptFilter(),
        path_stop_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        path_emit_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] | None = None,
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        greedy_only: bool = False,
    ) -> list[list[NodeT]]:

        results: list[list[NodeT]] = []
        emit_filter = path_emit_filter
        emit_filter_or_stop = path_emit_filter or path_stop_filter

        @tailrec
        def _rec(stack: list[tuple[NodeT, list[NodeT], StateT]]):
            if not stack:
                return results
            node, path, state = stack.pop()
            if greedy_only:
                if emit_filter is not None and emit_filter.accept(path, None, state):
                    if not global_filter.accept(path):
                        return results
                    results.append(path)
                next_node: NodeT | None = None
                for nxt in apply_neighbor_filter(neighbor_filter, node, self.graph.neighbors(node)):
                    if nxt in path or not path_stop_filter.accept(path, nxt, state):
                        continue
                    next_node = nxt
                    break
                if next_node is None:
                    if emit_filter is None:
                        if not path_stop_filter.accept(path, None, state):
                            return results
                    elif not emit_filter.accept(path, None, state):
                        return results
                    if not global_filter.accept(path):
                        return results
                    results.append(path)
                    return results
                next_state = state_tracker.state_update(state, node, next_node)
                stack.append((next_node, [*path, next_node], next_state))
                return _rec(stack)
            if emit_filter_or_stop.accept(path, None, state):
                if not global_filter.accept(path):
                    return results
                results.append(path)
            for nxt in apply_neighbor_filter(neighbor_filter, node, self.graph.neighbors(node)):
                if nxt in path or not path_stop_filter.accept(path, nxt, state):
                    continue
                next_state = state_tracker.state_update(state, node, nxt)
                stack.append((nxt, [*path, nxt], next_state))
            return _rec(stack)

        for start in start_nodes:
            _rec([(start, [start], state_tracker.state_init(start))])
        return results


    def dfs_iterative(
        self,
        start_nodes: list[NodeT],
        state_tracker: PathStateTracker[NodeT, StateT],
        neighbor_filter: Filter[tuple[NodeT, NodeT]] = AcceptFilter(),
        path_stop_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] = AcceptFilter(),
        path_emit_filter: TernaryFilter[list[NodeT], NodeT | None, StateT | None] | None = None,
        global_filter: UnaryFilter[list[NodeT]] = AcceptFilter(),
        N: int = 1,
        T: int = 1,
        shuffle: bool = True,
        cover_visited: bool = True,
        greedy_only: bool = True,
        rng: random.Random | None = None,
    ) -> Iterator[list[NodeT]]:
        unvisited = set(start_nodes)
        path_emit_filter = path_emit_filter or path_stop_filter
        rng = rng or random

        for _ in range(T):
            if not unvisited:
                break
            for _ in range(N):
                if not unvisited:
                    break
                candidates = list(unvisited)
                if shuffle:
                    rng.shuffle(candidates)
                start = candidates[0]
                for path in self.dfs(
                    start_nodes=[start],
                    state_tracker=state_tracker,
                    neighbor_filter=neighbor_filter,
                    path_stop_filter=path_stop_filter,
                    path_emit_filter=path_emit_filter,
                    global_filter=global_filter,
                    greedy_only=greedy_only,
                ):
                    if cover_visited:
                        unvisited -= set(path)
                    yield path