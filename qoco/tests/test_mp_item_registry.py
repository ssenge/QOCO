from __future__ import annotations

import pyomo.environ as pyo

from qoco.utils.pyomo.mp_item_registry import MPItemContext, MPItemFlags, MPItemRegistry


class DummyContext(MPItemContext):
    def __init__(self, value: int) -> None:
        self.value = int(value)


def test_registry_add_and_add_all_with_ctx() -> None:
    registry = MPItemRegistry[DummyContext]()
    registry.register("hc-1", lambda ctx, model: model.applied.append(ctx.value))
    registry.register("sc-2", lambda ctx, model: model.applied.append(ctx.value + 1))

    ctx = DummyContext(10)
    model = pyo.ConcreteModel()
    model.applied = []
    registry.add("HC_1", ctx, model)
    registry.add_all(["sc-2"], ctx, model)

    assert model.applied == [10, 11]


def test_registry_add_and_add_all_with_stored_ctx() -> None:
    ctx = DummyContext(7)
    registry = MPItemRegistry[DummyContext](ctx=ctx)
    registry.register("hc-1", lambda ctx, model: model.applied.append(ctx.value))
    registry.register("sc-2", lambda ctx, model: model.applied.append(ctx.value + 1))

    model = pyo.ConcreteModel()
    model.applied = []
    registry.add("hc-1", ctx, model)
    registry.add_all(["sc-2"], ctx, model)

    assert model.applied == [7, 8]


def test_registry_skip_via_flags() -> None:
    flags = MPItemFlags().disable("hc-1")
    registry = MPItemRegistry[DummyContext](flags=flags)
    registry.register("hc-1", lambda ctx, model: model.applied.append(ctx.value))
    registry.register("hc-2", lambda ctx, model: model.applied.append(ctx.value + 2))

    ctx = DummyContext(5)
    model = pyo.ConcreteModel()
    model.applied = []
    registry.add("hc-1", ctx, model)
    registry.add("hc-2", ctx, model)

    assert model.applied == [7]


def test_registry_duplicate_register() -> None:
    registry = MPItemRegistry[DummyContext]()
    registry.register("hc-1", lambda ctx, model: model.applied.append(ctx.value))
    try:
        registry.register("HC_1", lambda ctx, model: model.applied.append(ctx.value + 1))
    except ValueError as exc:
        assert "hc-1" in str(exc).lower()
    else:
        assert False


def test_registry_missing_item() -> None:
    registry = MPItemRegistry[DummyContext]()
    ctx = DummyContext(1)
    model = pyo.ConcreteModel()
    model.applied = []
    try:
        registry.add("hc-99", ctx, model)
    except KeyError as exc:
        assert "hc-99" in str(exc).lower()
    else:
        assert False


def test_create_model_from_flags() -> None:
    ctx = DummyContext(3)

    def _factory(factory_ctx: DummyContext) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.applied = []
        model.applied.append(factory_ctx.value)
        return model

    registry = MPItemRegistry[DummyContext](ctx=ctx, model_factory=_factory)
    registry.register("hc-1", lambda add_ctx, model: model.applied.append(add_ctx.value + 1))
    registry.register("sc-2", lambda add_ctx, model: model.applied.append(add_ctx.value + 2))

    model = registry.create_model_from_flags()

    assert model.applied == [3, 4, 5]
