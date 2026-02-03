from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, List, Mapping

import pyomo.environ as pyo
import torch

from qoco.core.solution import Solution, Status
from qoco.optimizers.rl.shared.backend import EnvBackend
from qoco.optimizers.rl.training.milp.generic import GenericMILPAdapter, GenericMILPConfig
from qoco.optimizers.rl.training.milp.selectors import AllBinarySelector, CandidateSelector, PaddedSelector, SelectorFactory


@dataclass
class GenericMILPBackend(EnvBackend):
    """Generic MILP backend with optional bridge (selector/decoder).

    If `selector_factory` is None, we default to `AllBinarySelector` (may explode).
    """

    name: str = "generic_milp"
    build_model: Callable[[Any], pyo.ConcreteModel] | None = None
    config: GenericMILPConfig | None = None
    n_nodes: int | None = None
    selector: CandidateSelector | None = None
    selector_factory: SelectorFactory | None = None
    decode_solution: Callable[[pyo.ConcreteModel, Status, float, dict[str, Any]], Solution] | None = None

    # Optional: provide a sampler to derive sample_train_instances, config.max_steps, n_nodes, bootstrap.
    sampler: Any | None = field(default=None, repr=False)

    # required by training ProblemAdapter-style API
    load_instances: Callable[[Path, int], List[Any]] | None = field(default=None, repr=False)
    sample_train_instances: Callable[[int], List[Any]] | None = field(default=None, repr=False)
    optimal_cost_fn: Callable[[Any], float] | None = field(default=None, repr=False)
    optimal_time_s_fn: Callable[[Any], float] | None = field(default=None, repr=False)

    bootstrap_instance: Any | None = field(default=None, repr=False)

    _impl: GenericMILPAdapter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Derive training instance sampling from sampler (if provided).
        if self.sample_train_instances is None and self.sampler is not None and hasattr(self.sampler, "sample"):
            self.sample_train_instances = lambda bs: [self.sampler.sample() for _ in range(int(bs))]

        # Bootstrap instance (for selector/kernel inference).
        if self.bootstrap_instance is None and self.sampler is not None and hasattr(self.sampler, "sample"):
            self.bootstrap_instance = self.sampler.sample()

        # Default max_steps from sampler.max_n if available.
        if self.config is None:
            if self.sampler is not None and hasattr(self.sampler, "max_n"):
                self.config = GenericMILPConfig(max_steps=int(getattr(self.sampler, "max_n")))
            else:
                raise ValueError("GenericMILPBackend requires config or sampler.max_n")

        # Default n_nodes from sampler.max_m if available.
        if self.n_nodes is None:
            if self.sampler is not None and hasattr(self.sampler, "max_m"):
                self.n_nodes = int(getattr(self.sampler, "max_m"))

        # Default optimal_* from instance attributes (if present).
        if self.optimal_cost_fn is None:
            self.optimal_cost_fn = lambda inst: float(getattr(inst, "optimal_cost", float("nan")) or float("nan"))
        if self.optimal_time_s_fn is None:
            self.optimal_time_s_fn = lambda inst: float(getattr(inst, "optimal_time", float("nan")) or float("nan"))

        # load_instances is required for evaluation via load_test_instances().
        if self.load_instances is None:
            def _missing_loader(_p: Path, _l: int) -> List[Any]:
                raise ValueError("GenericMILPBackend.load_instances is required to load eval instances (or avoid load_test_instances).")
            self.load_instances = _missing_loader

        # selector_factory defaulting:
        # - if selector is provided, we wrap+pad it to n_nodes
        # - else default to all-binary padded to n_nodes (may explode)
        if self.selector_factory is None:
            if self.n_nodes is None or int(self.n_nodes) <= 0:
                raise ValueError("GenericMILPBackend requires n_nodes (or sampler.max_m) when selector_factory is not provided")

            inner = self.selector if self.selector is not None else None

            if inner is None:
                def make_selector(model, kernel):
                    return PaddedSelector(
                        inner=AllBinarySelector.from_model(model, kernel),
                        target_n_actions=int(self.n_nodes),
                    )
            else:
                def make_selector(model, kernel):
                    return PaddedSelector(
                        inner=inner,
                        target_n_actions=int(self.n_nodes),
                    )
            self.selector_factory = SelectorFactory(make=make_selector)

    def _bootstrap(self) -> Any:
        if self.bootstrap_instance is not None:
            return self.bootstrap_instance
        if self.sample_train_instances is None:
            raise ValueError("GenericMILPBackend needs bootstrap_instance, sampler, or sample_train_instances")
        insts = self.sample_train_instances(1)
        if not insts:
            raise ValueError("GenericMILPBackend needs bootstrap_instance or sample_train_instances to return at least 1 instance")
        return insts[0]

    def _ensure_impl(self, inst: Any) -> GenericMILPAdapter:
        if self._impl is not None:
            return self._impl

        def _build(x: Any) -> pyo.ConcreteModel:
            if self.build_model is None:
                if isinstance(x, pyo.ConcreteModel):
                    return x
                raise TypeError("GenericMILPBackend.build_model is None but instance is not a ConcreteModel")
            return self.build_model(x)

        # If no selector_factory given, build it from the first model and freeze n_nodes.
        selector_factory = self.selector_factory
        n_nodes = int(self.n_nodes) if self.n_nodes is not None else 0
        if selector_factory is None:
            raise ValueError("GenericMILPBackend.__post_init__ should have set selector_factory")
        if n_nodes <= 0:
            raise ValueError("GenericMILPBackend requires n_nodes (or sampler.max_m)")

        self._impl = GenericMILPAdapter(
            name=self.name,
            build_model=_build,
            config=self.config,
            selector_factory=selector_factory,
            n_nodes=int(n_nodes),
            load_instances=self.load_instances,
            sample_train_instances=self.sample_train_instances,
            optimal_cost_fn=self.optimal_cost_fn,
            optimal_time_s_fn=self.optimal_time_s_fn,
            decode_solution=self.decode_solution,
        )
        return self._impl

    # Delegate EnvBackend surface to the underlying GenericMILPAdapter.
    def load_test_instances(self, path, limit: int):
        return self.load_instances(path, int(limit))

    def optimal_cost(self, inst: Any) -> float:
        return float(self.optimal_cost_fn(inst))

    def optimal_time_s(self, inst: Any) -> float:
        return float(self.optimal_time_s_fn(inst))

    def train_batch(self, batch_size: int, device: str):
        impl = self._ensure_impl(self._bootstrap())
        return impl.train_batch(batch_size, device)

    def make_eval_batch(self, inst: Any, device: str):
        impl = self._ensure_impl(inst)
        return impl.make_eval_batch(inst, device)

    def reset(self, batch):
        impl = self._ensure_impl(batch["inst"][0])
        return impl.reset(batch)

    def step(self, batch, action):
        impl = self._ensure_impl(batch["inst"][0])
        return impl.step(batch, action)

    def is_done(self, batch):
        impl = self._ensure_impl(batch["inst"][0])
        return impl.is_done(batch)

    def action_mask(self, batch):
        impl = self._ensure_impl(batch["inst"][0])
        return impl.action_mask(batch)

    def reward(self, batch):
        impl = self._ensure_impl(batch["inst"][0])
        return impl.reward(batch)

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        # Match GenericMILPAdapter reward convention:
        # - feasible: reward = -objective
        # - infeasible: reward = config.infeasible_reward (large negative)
        r = float(reward.reshape(-1)[0].item())
        impl = self._ensure_impl(self._bootstrap())
        thr = float(getattr(impl.config, "infeasible_reward", -1e6))
        if r <= thr + 1e-9:
            return Status.INFEASIBLE, float("inf")
        return Status.FEASIBLE, float(-r)

    def observe(self, batch) -> Mapping[str, torch.Tensor]:
        # GenericMILPAdapter already materializes these as tensors in the tensordict.
        return {
            "node_features": batch["node_features"].float(),
            "step_features": batch["step_features"].float(),
            "action_mask": self.action_mask(batch),
            "done": self.is_done(batch),
        }

    def score_eval_batch(self, batch) -> tuple[Status, float]:
        impl = self._ensure_impl(batch["inst"][0])
        return impl.score_eval_batch(batch)

    def to_solution(self, batch) -> Solution:
        impl = self._ensure_impl(batch["inst"][0])
        return impl.to_solution(batch)

