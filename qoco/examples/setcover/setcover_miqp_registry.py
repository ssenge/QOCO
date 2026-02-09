"""Generic Set Cover MIQP pattern for QUBO conversion.

This module provides a Set Cover formulation as MIQP (Mixed Integer Quadratic Programming)
that can be directly converted to QUBO without additional variables:
- SetCoverMIQPRegistry: Set cover with quadratic objective encoding exact coverage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pyomo.environ as pyo

from qoco.examples.setcover.setcover_registry import (
    AssigneeSetCoverContext,
    AssigneeSetCover_Param_AssignmentCosts,
    AssigneeSetCover_Param_Capacities,
    AssigneeSetCover_Set_AssigneeCollection,
    AssigneeSetCover_Set_Assignees,
    AssigneeSetCover_Var_Assignment,
    SetCoverContext,
    SetCover_Param_Costs,
    SetCover_Param_Requirements,
    SetCover_Set_Collections,
    SetCover_Set_ItemCollection,
    SetCover_Set_Items,
    SetCover_Var_CollectionSelection,
)
from qoco.utils.pyomo.mp_item_registry import MPItemFlags, MPItemRegistry, MPItemContext
from qoco.utils.pyomo.mp_items import MPObjective, MPParam


@dataclass(frozen=True)
class PenalizedSetCoverContext(SetCoverContext):
    """Context for Set Cover MIQP with per-item penalty weights.
    
    Extends SetCoverContext with:
        penalties: Dict mapping item -> penalty weight for coverage violation (optional)
                   If None or item missing, uses default penalty computed from costs
    """
    penalties: dict[Any, float] | None = field(default=None)  # penalty per item


@dataclass(frozen=True)
class SetCoverMIQP_Param_Penalties(MPParam):
    """Parameter: penalty weight per item for coverage violations."""
    label: str = "setcovermiqp-param-penalties"
    
    def add(self, ctx: PenalizedSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        # Compute default penalty from cost magnitude if available
        if ctx.costs is not None:
            max_cost = max(abs(c) for c in ctx.costs.values()) if ctx.costs else 1.0
            default_penalty = max(1.0, max_cost * 10.0)
        else:
            default_penalty = 100.0
        
        # Initialize penalties: use provided per-item penalties or default
        penalty_dict = {}
        for item in ctx.items:
            if ctx.penalties is not None and item in ctx.penalties:
                penalty_dict[item] = ctx.penalties[item]
            else:
                penalty_dict[item] = default_penalty
        
        model.setcovermiqp_penalty = pyo.Param(model.I, initialize=penalty_dict, mutable=True)
        return model


@dataclass(frozen=True)
class SetCoverMIQP_Objective_QuadraticCostAndCoverage(MPObjective):
    """Quadratic objective encoding cost minimization + exact coverage penalty.
    
    Objective: minimize cost + sum(i) penalty[i] * (requirement[i] - sum(c covering i) x[c])^2
    
    This expands to:
    - Linear cost terms: sum(c) cost[c] * x[c]
    - Constant: sum(i) penalty[i] * requirement[i]^2
    - Linear penalty terms: sum(i) penalty[i] * -2*requirement[i]*sum(c covering i) x[c]
    - Quadratic penalty terms: sum(i) penalty[i] * sum(c1, c2 covering i) x[c1]*x[c2]
    """
    label: str = "setcovermiqp-obj-quadratic"
    
    def add(self, ctx: PenalizedSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        # Cost term (linear)
        cost_expr = sum(model.cost[c] * model.x[c] for c in model.C)
        
        # Coverage penalty term (quadratic)
        # For each item i: penalty[i] * (requirement[i] - sum(c covering i) x[c])^2
        penalty_expr = 0.0
        
        for i in model.I:
            requirement = model.requirement[i]
            penalty_i = model.setcovermiqp_penalty[i]
            collections_covering = ctx.item_collection_map.get(i, [])
            
            if not collections_covering:
                # Item has no covering collections - skip (or could add large penalty)
                continue
            
            # Coverage sum: sum(c covering i) x[c]
            coverage_sum = sum(model.x[c] for c in collections_covering)
            
            # Penalty: penalty[i] * (requirement[i] - coverage_sum)^2
            # Expand: penalty[i] * (requirement[i]^2 - 2*requirement[i]*coverage_sum + coverage_sum^2)
            penalty_expr += penalty_i * (
                requirement * requirement  # constant term
                - 2 * requirement * coverage_sum  # linear term
                + coverage_sum * coverage_sum  # quadratic term
            )
        
        # Total objective: cost + penalty
        total_expr = cost_expr + penalty_expr
        
        model.setcovermiqp_obj_quadratic = pyo.Objective(expr=total_expr, sense=pyo.minimize)
        return model


@dataclass
class SetCoverMIQPRegistry(MPItemRegistry[PenalizedSetCoverContext]):
    """Registry for Set Cover MIQP (Mixed Integer Quadratic Programming).
    
    This formulation encodes exact coverage directly in a quadratic objective,
    allowing direct conversion to QUBO without additional variables or constraints.
    
    Provides:
    - Sets: I (items), C (collections), IC (item-collection pairs)
    - Variables: x[c] (binary: select collection c)
    - Parameters: requirement[i], cost[c], penalty[i] (per-item coverage penalty weight)
    - Objectives: Quadratic objective (cost + exact coverage penalty)
    - NO constraints (coverage is encoded in objective)
    
    Usage:
        ctx = PenalizedSetCoverContext(
            items=[1,2,3], 
            collections=[A,B,C], 
            ...,
            penalties={1: 100.0, 2: 200.0}  # optional per-item penalties
        )
        registry = SetCoverMIQPRegistry(ctx=ctx)
        model = registry.create()
        # Convert to QUBO using PyomoToQuboConverter (no constraints to convert)
    """
    ctx: PenalizedSetCoverContext | None = None
    flags: MPItemFlags = field(default_factory=MPItemFlags)
    model_factory: Callable[[PenalizedSetCoverContext], pyo.ConcreteModel] | None = field(
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
            SetCoverMIQP_Param_Penalties,
            SetCoverMIQP_Objective_QuadraticCostAndCoverage,
        ]
        self.register_all(items)


# ============================================================================
# Assignee Set Cover MIQP (Extended)
# ============================================================================

@dataclass(frozen=True)
class PenalizedAssigneeSetCoverContext(PenalizedSetCoverContext):
    """Context for Assignee Set Cover MIQP with per-item, per-collection, and per-assignee penalty weights.
    
    Extends PenalizedSetCoverContext with:
        assignees: List of assignees that can be assigned to collections
        capacities: Dict mapping assignee -> max collections they can handle (optional)
        assignment_costs: Dict mapping (assignee, collection) -> cost (optional)
        assignment_link_penalties: Dict mapping collection -> penalty weight for assignment link violation (optional)
        capacity_penalties: Dict mapping assignee -> penalty weight for capacity violation (optional)
    """
    assignees: list[Any] = field(default_factory=list)
    capacities: dict[Any, int] | None = field(default=None)  # capacity per assignee
    assignment_costs: dict[tuple[Any, Any], float] | None = field(default=None)  # cost[(a, c)]
    assignment_link_penalties: dict[Any, float] | None = field(default=None)  # penalty per collection for assignment link
    capacity_penalties: dict[Any, float] | None = field(default=None)  # penalty per assignee for capacity violation


@dataclass(frozen=True)
class AssigneeSetCoverMIQP_Param_AssignmentLinkPenalties(MPParam):
    """Parameter: penalty weight per collection for assignment link violations."""
    label: str = "assigneesetcovermiqp-param-assignment-link-penalties"
    
    def add(self, ctx: PenalizedAssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        # Compute default penalty from assignment cost magnitude if available
        if ctx.assignment_costs is not None:
            max_cost = max(abs(c) for c in ctx.assignment_costs.values()) if ctx.assignment_costs else 1.0
            default_penalty = max(1.0, max_cost * 10.0)
        elif ctx.costs is not None:
            max_cost = max(abs(c) for c in ctx.costs.values()) if ctx.costs else 1.0
            default_penalty = max(1.0, max_cost * 10.0)
        else:
            default_penalty = 100.0
        
        # Initialize penalties: use provided per-collection penalties or default
        penalty_dict = {}
        for collection in ctx.collections:
            if ctx.assignment_link_penalties is not None and collection in ctx.assignment_link_penalties:
                penalty_dict[collection] = ctx.assignment_link_penalties[collection]
            else:
                penalty_dict[collection] = default_penalty
        
        model.assigneesetcovermiqp_assignment_link_penalty = pyo.Param(model.C, initialize=penalty_dict, mutable=True)
        return model


@dataclass(frozen=True)
class AssigneeSetCoverMIQP_Param_CapacityPenalties(MPParam):
    """Parameter: penalty weight per assignee for capacity violations."""
    label: str = "assigneesetcovermiqp-param-capacity-penalties"
    
    def add(self, ctx: PenalizedAssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        # Compute default penalty from assignment cost magnitude if available
        if ctx.assignment_costs is not None:
            max_cost = max(abs(c) for c in ctx.assignment_costs.values()) if ctx.assignment_costs else 1.0
            default_penalty = max(1.0, max_cost * 10.0)
        elif ctx.costs is not None:
            max_cost = max(abs(c) for c in ctx.costs.values()) if ctx.costs else 1.0
            default_penalty = max(1.0, max_cost * 10.0)
        else:
            default_penalty = 100.0
        
        # Initialize penalties: use provided per-assignee penalties or default
        penalty_dict = {}
        for assignee in ctx.assignees:
            if ctx.capacity_penalties is not None and assignee in ctx.capacity_penalties:
                penalty_dict[assignee] = ctx.capacity_penalties[assignee]
            else:
                penalty_dict[assignee] = default_penalty
        
        model.assigneesetcovermiqp_capacity_penalty = pyo.Param(model.A, initialize=penalty_dict, mutable=True)
        return model


@dataclass(frozen=True)
class AssigneeSetCoverMIQP_Objective_Quadratic(MPObjective):
    """Quadratic objective encoding assignment cost + coverage + assignment link + capacity penalties.
    
    Objective: minimize 
        assignment_cost + coverage_penalty + assignment_link_penalty + capacity_penalty
    
    Where:
    - assignment_cost = sum(a, c) assignment_cost[a, c] * x_ac[a, c] (linear)
    - coverage_penalty = sum(i) penalty[i] * (requirement[i] - sum(c covering i) sum(a) x_ac[a, c])^2 (quadratic)
    - assignment_link_penalty = sum(c) link_penalty[c] * (1 - sum(a) x_ac[a, c])^2 (quadratic)
    - capacity_penalty = sum(a) capacity_penalty[a] * max(0, sum(c) x_ac[a, c] - capacity[a])^2 (quadratic)
    """
    label: str = "assigneesetcovermiqp-obj-quadratic"
    
    def add(self, ctx: PenalizedAssigneeSetCoverContext, model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        # Assignment cost term (linear)
        cost_expr = sum(model.assignment_cost[a, c] * model.x_ac[a, c] for a in model.A for c in model.C)
        
        # Coverage penalty term (quadratic)
        # For each item i: penalty[i] * (requirement[i] - sum(c covering i) sum(a) x_ac[a, c])^2
        coverage_penalty_expr = 0.0
        
        for i in model.I:
            requirement = model.requirement[i]
            penalty_i = model.setcovermiqp_penalty[i]
            collections_covering = ctx.item_collection_map.get(i, [])
            
            if not collections_covering:
                continue
            
            # Coverage sum: sum(c covering i) sum(a) x_ac[a, c]
            coverage_sum = sum(
                model.x_ac[a, c]
                for c in collections_covering
                for a in model.A
            )
            
            # Penalty: penalty[i] * (requirement[i] - coverage_sum)^2
            coverage_penalty_expr += penalty_i * (
                requirement * requirement  # constant term
                - 2 * requirement * coverage_sum  # linear term
                + coverage_sum * coverage_sum  # quadratic term
            )
        
        # Assignment link penalty term (quadratic)
        # For each collection c: link_penalty[c] * (1 - sum(a) x_ac[a, c])^2
        # Ensures exactly one assignee per collection
        assignment_link_penalty_expr = 0.0
        
        for c in model.C:
            link_penalty_c = model.assigneesetcovermiqp_assignment_link_penalty[c]
            assignment_sum = sum(model.x_ac[a, c] for a in model.A)
            
            # Penalty: link_penalty[c] * (1 - assignment_sum)^2
            assignment_link_penalty_expr += link_penalty_c * (
                1.0  # constant term
                - 2 * assignment_sum  # linear term
                + assignment_sum * assignment_sum  # quadratic term
            )
        
        # Capacity penalty term (quadratic)
        # For each assignee a: capacity_penalty[a] * max(0, sum(c) x_ac[a, c] - capacity[a])^2
        capacity_penalty_expr = 0.0
        
        for a in model.A:
            capacity_a = model.capacity[a]
            capacity_penalty_a = model.assigneesetcovermiqp_capacity_penalty[a]
            assignment_count = sum(model.x_ac[a, c] for c in model.C)
            
            # Violation: max(0, assignment_count - capacity_a)
            # Penalty: capacity_penalty[a] * violation^2
            # Since we're minimizing, we can use: capacity_penalty[a] * (assignment_count - capacity_a)^2
            # but only penalize if violation > 0. However, for QUBO we need smooth function.
            # Use: capacity_penalty[a] * max(0, assignment_count - capacity_a)^2
            # For QUBO compatibility, we'll use: capacity_penalty[a] * (assignment_count - capacity_a)^2
            # but this penalizes both under and over. For capacity, we only want to penalize over.
            # However, max(0, x)^2 is not quadratic. So we use a quadratic approximation:
            # If assignment_count <= capacity_a: penalty = 0 (via large negative offset)
            # Actually, simpler: use (assignment_count - capacity_a)^2 but only when positive
            # For QUBO, we need pure quadratic, so we'll use: (assignment_count - capacity_a)^2
            # This penalizes both under and over, but with proper penalty weight, over-violations
            # will dominate.
            
            # Use: capacity_penalty[a] * (assignment_count - capacity_a)^2
            # This is quadratic and penalizes violations
            capacity_penalty_expr += capacity_penalty_a * (
                capacity_a * capacity_a  # constant term
                - 2 * capacity_a * assignment_count  # linear term
                + assignment_count * assignment_count  # quadratic term
            )
        
        # Total objective: cost + all penalties
        total_expr = cost_expr + coverage_penalty_expr + assignment_link_penalty_expr + capacity_penalty_expr
        
        model.assigneesetcovermiqp_obj_quadratic = pyo.Objective(expr=total_expr, sense=pyo.minimize)
        return model


@dataclass
class AssigneeSetCoverMIQPRegistry(MPItemRegistry[PenalizedAssigneeSetCoverContext]):
    """Registry for Assignee Set Cover MIQP (Mixed Integer Quadratic Programming).
    
    This formulation encodes exact coverage, assignment link, and capacity directly in a quadratic objective,
    allowing direct conversion to QUBO without additional variables or constraints.
    
    Uses Option B1: exactly one assignee per collection (no x[c] variable, only x_ac[a, c]).
    
    Provides:
    - Sets: I (items), C (collections), IC (item-collection pairs), A (assignees), AC (assignee-collection pairs)
    - Variables: x_ac[a, c] (binary: assignee a assigned to collection c)
    - Parameters: requirement[i], assignment_cost[(a, c)], capacity[a], 
                  penalty[i] (coverage), assignment_link_penalty[c], capacity_penalty[a]
    - Objectives: Quadratic objective (cost + coverage + assignment link + capacity penalties)
    - NO constraints (all encoded in objective)
    
    Usage:
        ctx = PenalizedAssigneeSetCoverContext(
            items=[1,2,3], 
            collections=[A,B,C],
            assignees=[D1, D2],
            ...,
            penalties={1: 100.0, 2: 200.0},  # optional per-item coverage penalties
            assignment_link_penalties={A: 150.0, B: 150.0},  # optional per-collection penalties
            capacity_penalties={D1: 200.0, D2: 200.0}  # optional per-assignee penalties
        )
        registry = AssigneeSetCoverMIQPRegistry(ctx=ctx)
        model = registry.create()
        # Convert to QUBO using PyomoToQuboConverter (no constraints to convert)
    """
    ctx: PenalizedAssigneeSetCoverContext | None = None
    flags: MPItemFlags = field(default_factory=MPItemFlags)
    model_factory: Callable[[PenalizedAssigneeSetCoverContext], pyo.ConcreteModel] | None = field(
        default_factory=lambda: (lambda _: pyo.ConcreteModel())
    )
    
    def __post_init__(self) -> None:
        items = [
            # Sets
            SetCover_Set_Items,
            SetCover_Set_Collections,
            SetCover_Set_ItemCollection,
            AssigneeSetCover_Set_Assignees,
            AssigneeSetCover_Set_AssigneeCollection,
            # Parameters
            SetCover_Param_Requirements,
            AssigneeSetCover_Param_Capacities,
            AssigneeSetCover_Param_AssignmentCosts,
            SetCoverMIQP_Param_Penalties,  # coverage penalties
            AssigneeSetCoverMIQP_Param_AssignmentLinkPenalties,
            AssigneeSetCoverMIQP_Param_CapacityPenalties,
            # Variables
            AssigneeSetCover_Var_Assignment,  # x_ac only, no x[c]
            # Objectives
            AssigneeSetCoverMIQP_Objective_Quadratic,
        ]
        self.register_all(items)
