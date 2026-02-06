from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Callable, Dict, Iterable, Generic, Sequence, TypeVar

import pyomo.environ as pyo

from qoco.utils.pyomo.mp_items import MPItem

class MPItemContext:
    pass


Ctx = TypeVar("Ctx", bound=MPItemContext)


@dataclass(frozen=True)
class MPItemFlags:
    skip: set[str] = field(default_factory=set)

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().lower().replace("_", "-")

    def enabled(self, label: str) -> bool:
        return self._normalize(label) not in self.skip

    def disable(self, *labels: str) -> MPItemFlags:
        if not labels: return self
        skip = set(self.skip)
        skip.update(self._normalize(label) for label in labels)
        return MPItemFlags(skip=skip)

    def enable(self, *labels: str) -> MPItemFlags:
        if not labels: return MPItemFlags()
        skip = set(self.skip)
        skip.difference_update(self._normalize(label) for label in labels)
        return MPItemFlags(skip=skip)


@dataclass
class MPItemRegistry(Generic[Ctx]):
    flags: MPItemFlags = field(default_factory=MPItemFlags)
    ctx: Ctx | None = None
    model_factory: Callable[[Ctx], pyo.ConcreteModel] | None = None
    _items: Dict[str, MPItem] = field(default_factory=dict, init=False)

    def register(
        self,
        item: MPItem | type[MPItem] | str,
        add_fn: Callable[[Ctx, pyo.ConcreteModel], None] | None = None,
    ) -> MPItemRegistry[Ctx]:
        if isinstance(item, str):
            if add_fn is None: raise ValueError("add_fn is required when registering by label")
            item = _FunctionMPItem(label=item, add_fn=add_fn)
        elif isinstance(item, type) and issubclass(item, MPItem):
            item = item()
        if not isinstance(item, MPItem):
            raise TypeError("register() expects MPItem, MPItem subclass, or label string")
        key = MPItemFlags._normalize(item.label)
        if key in self._items:
            raise ValueError(f"MPItem '{key}' already registered")
        self._items[key] = item
        return self

    def register_all(self, items: Iterable[MPItem | type[MPItem]]) -> MPItemRegistry[Ctx]:
        for item in items: self.register(item)
        return self

    def get(self, label: str) -> MPItem:
        key = MPItemFlags._normalize(label)
        if key not in self._items:
            raise KeyError(f"MPItem '{label}' not registered")
        return self._items[key]

    def compose(self, other: MPItemRegistry[Ctx]) -> MPItemRegistry[Ctx]:
        composed = MPItemRegistry(flags=self.flags, ctx=self.ctx, model_factory=self.model_factory)
        composed._items = dict(self._items)
        for key, item in other._items.items():
            if key in composed._items:
                raise ValueError(f"MPItem '{key}' already registered")
            composed._items[key] = item
        return composed

    def add(self, label: str, ctx: Ctx, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        key = MPItemFlags._normalize(label)
        if not self.flags.enabled(key):
            return model
        if key not in self._items:
            raise KeyError(f"MPItem '{label}' not registered")
        self._items[key].add(ctx, model)
        return model

    def add_all(self, labels: Iterable[str], ctx: Ctx, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        if not hasattr(model, "_item_timings"):
            model._item_timings = {}
        for label in labels:
            t0 = time.perf_counter()
            model = self.add(label, ctx, model)
            elapsed = time.perf_counter() - t0
            key = MPItemFlags._normalize(label)
            model._item_timings[key] = model._item_timings.get(key, 0.0) + elapsed
        return model

    def create(
        self,
        obj_terms: Sequence[str | MPItem | type[MPItem]] | None = None,
        objective_attr_resolver: Callable[[str], str] | None = None,
    ) -> pyo.ConcreteModel:
        if self.ctx is None:
            raise ValueError("MPItemRegistry ctx is not set")
        if self.model_factory is None:
            raise ValueError("MPItemRegistry model_factory is not set")
        start_ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        t0 = time.perf_counter()
        # TODO: profiling only, remove after analysis.
        print(f"[MPItemRegistry.create] start at {start_ts}")
        model = self.model_factory(self.ctx)
        # TODO: profiling only, remove after analysis.
        print(f"[MPItemRegistry.create] add_all start at {datetime.now().isoformat(sep=' ', timespec='seconds')}")
        model = self.add_all(self._items.keys(), self.ctx, model)
        # TODO: profiling only, remove after analysis.
        print(f"[MPItemRegistry.create] add_all done in {time.perf_counter() - t0:.2f}s")
        _print_item_timings(model)
        for name in ("D", "S", "E", "K", "KS", "KE"):
            if hasattr(model, name):
                try:
                    size = len(getattr(model, name))
                except Exception:
                    size = "?"
                # TODO: profiling only, remove after analysis.
                print(f"[MPItemRegistry.create] set {name} size: {size}")
        print("Done: adding all items to model, point 0")
        if obj_terms is None:
            print("Done: adding all items to model, point 0.1 -> return model")
            return model

        print("Done: adding all items to model, point 1")
        labels: list[str] = []
        for term in obj_terms:
            if isinstance(term, str):
                labels.append(term)
                continue
            label = getattr(term, "label", None)
            if label is None:
                raise TypeError("obj_terms entries must be str, MPItem, or MPItem class")
            labels.append(label)
        print("Done: adding all items to model, point 2")
        for label in labels:
            if not self.flags.enabled(label):
                raise ValueError(f"Objective term '{label}' is disabled via flags")
            self.add(label, self.ctx, model)
        print("Done: adding all items to model, point 3")
        if hasattr(model, "obj"):
            model.del_component(model.obj)
        if hasattr(model, "objective"):
            model.del_component(model.objective)

        print("Done: adding objective to model, point 4")
        resolve_attr = objective_attr_resolver or (lambda label: f"{label.replace('-', '_')}_objective")
        total = 0
        for label in labels:
            attr = resolve_attr(label)
            if not hasattr(model, attr):
                raise ValueError(f"Objective term '{label}' not found on model")
            total += getattr(model, attr)

        print("Done: adding objective to model, point 5")
        model.obj = pyo.Objective(expr=total)
        print("Done: adding objective to model, point 6")
        # TODO: profiling only, remove after analysis.
        print(f"[MPItemRegistry.create] done at {datetime.now().isoformat(sep=' ', timespec='seconds')}")
        return model


def _print_item_timings(model: pyo.ConcreteModel, top_n: int = 20) -> None:
    if not hasattr(model, "_item_timings"):
        return
    timings = model._item_timings
    if not timings:
        return
    ranked = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    print("[MPItemRegistry.create] slowest items:")
    for label, seconds in ranked:
        print(f"  {label}: {seconds:.3f}s")


@dataclass(frozen=True)
class _FunctionMPItem(MPItem):
    add_fn: Callable[[Ctx, pyo.ConcreteModel], None]

    def add(self, ctx: Ctx, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        self.add_fn(ctx, model)
        return model
