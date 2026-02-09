"""Generic Set Cover MIP pattern.

This module provides reusable Set Cover formulations:
- SetCoverRegistry: Basic set cover (items, collections, coverage)
- AssigneeSetCoverRegistry: Set cover with assignees (items, collections, assignees, assignment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pyomo.environ as pyo

from qoco.utils.pyomo.mp_item_registry import MPItemFlags, MPItemRegistry, MPItemContext
from qoco.utils.pyomo.mp_items import MPConstraint, MPObjective, MPParam, MPSet, MPVar


# ============================================================================
# Set Cover (Base)
# ============================================================================

@dataclass(frozen=True)
class SetCoverContext(MPItemContext):
    """Context for basic set cover problem.
    
    Attributes:
        items: List of items that must be covered
        collections: List of collections/sets that can cover items
        item_collection_map: Dict mapping item -> list of collections that cover it
        collection_item_map: Dict mapping collection -> list of items it covers
        requirements: Dict mapping item -> required coverage count (default: 1)
        costs: Dict mapping collection -> cost (optional, for cost minimization)
    """
    items: list[Any]
    collections: list[Any]
    item_collection_map: dict[Any, list[Any]]  # collections_covering[item]
    collection_item_map: dict[Any, list[Any]]  # items_in[collection]
    requirements: dict[Any, int] | None = field(default=None)  # requirement per item (default: 1)
    costs: dict[Any, float] | None = field(default=None)  # cost per collection


@dataclass(frozen=True)
class SetCover_Set_Items(MPSet):
    """Set of items to cover."""
    label: str = "setcover-items"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        model.I = pyo.Set(initialize=ctx.items)
        return model


@dataclass(frozen=True)
class SetCover_Set_Collections(MPSet):
    """Set of collections that can cover items."""
    label: str = "setcover-collections"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        model.C = pyo.Set(initialize=ctx.collections)
        return model


@dataclass(frozen=True)
class SetCover_Set_ItemCollection(MPSet):
    """Set of (item, collection) pairs indicating coverage."""
    label: str = "setcover-item-collection"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        pairs = [
            (item, collection)
            for item in ctx.items
            for collection in ctx.item_collection_map.get(item, [])
        ]
        model.IC = pyo.Set(initialize=pairs, dimen=2)
        return model


@dataclass(frozen=True)
class SetCover_Var_CollectionSelection(MPVar):
    """Binary variable: x[c] = 1 if collection c is selected."""
    label: str = "setcover-var-x"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        model.x = pyo.Var(model.C, domain=pyo.Binary)
        return model


@dataclass(frozen=True)
class SetCover_Param_Requirements(MPParam):
    """Parameter: requirement[i] = required coverage count for item i."""
    label: str = "setcover-param-requirements"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        requirements = ctx.requirements or {item: 1 for item in ctx.items}
        model.requirement = pyo.Param(model.I, initialize=requirements, default=1)
        return model


@dataclass(frozen=True)
class SetCover_Param_Costs(MPParam):
    """Parameter: cost[c] = cost of selecting collection c."""
    label: str = "setcover-param-costs"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        if ctx.costs is not None:
            model.cost = pyo.Param(model.C, initialize=ctx.costs, default=0.0)
        else:
            model.cost = pyo.Param(model.C, initialize={c: 1.0 for c in ctx.collections}, default=1.0)
        return model


@dataclass(frozen=True)
class SetCover_Constraint_Coverage(MPConstraint):
    """Coverage constraint: each item must be covered at least requirement[i] times."""
    label: str = "setcover-coverage"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        def _coverage_rule(m, i):
            collections_covering = ctx.item_collection_map.get(i, [])
            if not collections_covering:
                return pyo.Constraint.Infeasible
            return sum(m.x[c] for c in collections_covering) >= m.requirement[i]
        
        model.setcover_coverage = pyo.Constraint(model.I, rule=_coverage_rule)
        return model


@dataclass(frozen=True)
class SetCover_Objective_MinimizeCost(MPObjective):
    """Minimize total cost of selected collections."""
    label: str = "setcover-obj-cost"
    
    def add(self, ctx: SetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        def _objective_rule(m):
            return sum(m.cost[c] * m.x[c] for c in m.C)
        
        model.setcover_obj_cost = pyo.Objective(rule=_objective_rule, sense=pyo.minimize)
        return model


@dataclass
class SetCoverRegistry(MPItemRegistry[SetCoverContext]):
    """Registry for basic set cover MIP.
    
    Provides:
    - Sets: I (items), C (collections), IC (item-collection pairs)
    - Variables: x[c] (binary: select collection c)
    - Parameters: requirement[i], cost[c]
    - Constraints: Coverage (each item covered >= requirement)
    - Objectives: Minimize cost or count
    """
    ctx: SetCoverContext | None = None
    flags: MPItemFlags = field(default_factory=MPItemFlags)
    model_factory: Callable[[SetCoverContext], pyo.ConcreteModel] | None = field(
        default_factory=lambda: (lambda _: pyo.ConcreteModel())
    )
    
    def __post_init__(self) -> None:
        items = [
            SetCover_Set_Items,
            SetCover_Set_Collections,
            SetCover_Set_ItemCollection,
            SetCover_Param_Requirements,
            SetCover_Param_Costs,
            SetCover_Var_CollectionSelection,
            SetCover_Constraint_Coverage,
            SetCover_Objective_MinimizeCost
        ]
        self.register_all(items)


# ============================================================================
# Assignee Set Cover (Extended)
# ============================================================================

@dataclass(frozen=True)
class AssigneeSetCoverContext(SetCoverContext):
    """Context for set cover with assignees.
    
    Extends SetCoverContext with:
        assignees: List of assignees that can be assigned to collections
        capacities: Dict mapping assignee -> max collections they can handle (optional)
        assignment_costs: Dict mapping (assignee, collection) -> cost (optional)
    """
    assignees: list[Any] = field(default_factory=list)
    capacities: dict[Any, int] | None = field(default=None)  # capacity per assignee
    assignment_costs: dict[tuple[Any, Any], float] | None = field(default=None)  # cost[(a, c)]


@dataclass(frozen=True)
class AssigneeSetCover_Set_Assignees(MPSet):
    """Set of assignees."""
    label: str = "assigneesetcover-assignees"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        model.A = pyo.Set(initialize=ctx.assignees)
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Set_AssigneeCollection(MPSet):
    """Set of (assignee, collection) pairs for assignment."""
    label: str = "assigneesetcover-assignee-collection"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        pairs = [(a, c) for a in ctx.assignees for c in ctx.collections]
        model.AC = pyo.Set(initialize=pairs, dimen=2)
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Var_Assignment(MPVar):
    """Binary variable: x[a, c] = 1 if assignee a is assigned to collection c."""
    label: str = "assigneesetcover-var-x-ac"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        model.x_ac = pyo.Var(model.A, model.C, domain=pyo.Binary)
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Param_Capacities(MPParam):
    """Parameter: capacity[a] = max collections assignee a can handle."""
    label: str = "assigneesetcover-param-capacities"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        if ctx.capacities is not None:
            model.capacity = pyo.Param(model.A, initialize=ctx.capacities, default=len(ctx.collections))
        else:
            model.capacity = pyo.Param(model.A, initialize={a: len(ctx.collections) for a in ctx.assignees}, default=len(ctx.collections))
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Param_AssignmentCosts(MPParam):
    """Parameter: assignment_cost[(a, c)] = cost of assigning assignee a to collection c."""
    label: str = "assigneesetcover-param-assignment-costs"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        if ctx.assignment_costs is not None:
            model.assignment_cost = pyo.Param(model.A, model.C, initialize=ctx.assignment_costs, default=0.0)
        else:
            model.assignment_cost = pyo.Param(model.A, model.C, initialize={(a, c): 0.0 for a in ctx.assignees for c in ctx.collections}, default=0.0)
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Constraint_AssignmentLink(MPConstraint):
    """Link assignment variables to collection selection: sum(x[a, c] for a) == x[c]."""
    label: str = "assigneesetcover-assignment-link"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        def _assignment_link_rule(m, c):
            return sum(m.x_ac[a, c] for a in m.A) == m.x[c]
        
        model.assigneesetcover_assignment_link = pyo.Constraint(model.C, rule=_assignment_link_rule)
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Constraint_Capacity(MPConstraint):
    """Capacity constraint: each assignee can handle at most capacity[a] collections."""
    label: str = "assigneesetcover-capacity"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        def _capacity_rule(m, a):
            return sum(m.x_ac[a, c] for c in m.C) <= m.capacity[a]
        
        model.assigneesetcover_capacity = pyo.Constraint(model.A, rule=_capacity_rule)
        return model


@dataclass(frozen=True)
class AssigneeSetCover_Objective_MinimizeAssignmentCost(MPObjective):
    """Minimize total assignment cost."""
    label: str = "assigneesetcover-obj-assignment-cost"
    
    def add(self, ctx: AssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        def _objective_rule(m):
            return sum(m.assignment_cost[a, c] * m.x_ac[a, c] for a in m.A for c in m.C)
        
        model.assigneesetcover_obj_assignment_cost = pyo.Objective(rule=_objective_rule, sense=pyo.minimize)
        return model


@dataclass
class AssigneeSetCoverRegistry(MPItemRegistry[AssigneeSetCoverContext]):
    """Registry for set cover with assignees.
    
    Extends SetCoverRegistry with:
    - Sets: A (assignees), AC (assignee-collection pairs)
    - Variables: x_ac[a, c] (binary: assignee a assigned to collection c)
    - Parameters: capacity[a], assignment_cost[(a, c)]
    - Constraints: Assignment link (sum(x_ac) == x[c]), Capacity
    - Objectives: Minimize assignment cost
    
    Note: This registry includes all SetCoverRegistry items plus assignee-specific items.
    The base collection selection variable x[c] is linked to assignments via x_ac[a, c].
    """
    ctx: AssigneeSetCoverContext | None = None
    flags: MPItemFlags = field(default_factory=MPItemFlags)
    model_factory: Callable[[AssigneeSetCoverContext], pyo.ConcreteModel] | None = field(
        default_factory=lambda: (lambda _: pyo.ConcreteModel())
    )
    
    def __post_init__(self) -> None:
        # First register base SetCover items
        base_reg = SetCoverRegistry(ctx=self.ctx, flags=self.flags, model_factory=self.model_factory)
        
        # Then register assignee-specific items
        assignee_items = [
            AssigneeSetCover_Set_Assignees,
            AssigneeSetCover_Set_AssigneeCollection,
            AssigneeSetCover_Param_Capacities,
            AssigneeSetCover_Param_AssignmentCosts,
            AssigneeSetCover_Var_Assignment,
            AssigneeSetCover_Constraint_AssignmentLink,
            AssigneeSetCover_Constraint_Capacity,
            AssigneeSetCover_Objective_MinimizeAssignmentCost,
        ]
        assignee_reg = MPItemRegistry(ctx=self.ctx, flags=self.flags, model_factory=self.model_factory)
        assignee_reg.register_all(assignee_items)
        
        # Compose both registries
        self._items = base_reg.compose(assignee_reg)._items
