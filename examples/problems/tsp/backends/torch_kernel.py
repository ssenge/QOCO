from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np
import torch
from tensordict import TensorDict

from qoco.core.sampler import Sampler
from qoco.core.solution import InfoSolution, Status
from qoco.examples.problems.tsp.problem import TSP
from qoco.examples.problems.tsp.sampler import RandomTSPSampler
from qoco.examples.problems.tsp.torch_kernel import TSPTorchKernel
from qoco.optimizers.rl.shared.backend import EnvBackend


def _fixed_n_nodes(sampler: Sampler[TSP]) -> int:
    sizes = getattr(sampler, "sizes", None)
    if sizes:
        uniq = sorted({int(n) for n in sizes})
        if len(uniq) != 1:
            raise ValueError("TSP torch kernel backend requires a fixed n_nodes; pass sizes with a single value.")
        return int(uniq[0])
    inst = sampler.sample()
    return int(len(inst.dist))


def _batch_from_instances(insts: List[TSP], *, n_nodes: int, device: str) -> TensorDict:
    dists = [np.asarray(inst.dist, dtype=np.float32) for inst in insts]
    for dist in dists:
        if dist.shape != (n_nodes, n_nodes):
            raise ValueError(f"Expected dist shape {(n_nodes, n_nodes)}, got {dist.shape}")
    dist_np = np.stack(dists)
    dist = torch.from_numpy(dist_np).to(device)
    n = torch.full((len(insts), 1), int(n_nodes), dtype=torch.long, device=device)
    edge_masks = []
    for inst, dist_arr in zip(insts, dists):
        mask = getattr(inst, "edge_mask", None)
        if mask is None:
            mask = np.isfinite(dist_arr)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_nodes, n_nodes):
            raise ValueError(f"Expected edge_mask shape {(n_nodes, n_nodes)}, got {mask.shape}")
        np.fill_diagonal(mask, False)
        edge_masks.append(mask)
    edge_mask = torch.from_numpy(np.stack(edge_masks)).to(device)
    return TensorDict({"dist": dist, "n_nodes": n, "edge_mask": edge_mask}, batch_size=[len(insts)]).to(device)


@dataclass
class TSPTorchKernelBackend(EnvBackend):
    name: str = "tsp_torch_kernel"
    sampler: Sampler[TSP] = field(default_factory=RandomTSPSampler)
    edge_topk: int | None = None
    reachability_prune: bool = False
    return_to_start_prune: bool = False

    kernel: TSPTorchKernel = field(init=False, repr=False)
    n_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_nodes = _fixed_n_nodes(self.sampler)
        if self.edge_topk is None:
            self.edge_topk = int(getattr(self.sampler, "edge_topk", 0) or 0)
        if not self.reachability_prune:
            self.reachability_prune = bool(getattr(self.sampler, "reachability_prune", False))
        if not self.return_to_start_prune:
            self.return_to_start_prune = bool(getattr(self.sampler, "return_to_start_prune", False))
        self.kernel = TSPTorchKernel(
            n_nodes=int(self.n_nodes),
            edge_topk=self.edge_topk or None,
            reachability_prune=self.reachability_prune,
            return_to_start_prune=self.return_to_start_prune,
        )

    def load_test_instances(self, path: Path, limit: int) -> List[TSP]:
        raise ValueError("TSPTorchKernelBackend.load_test_instances is not implemented.")

    def optimal_cost(self, inst: Any) -> float:
        return float(getattr(inst, "optimal_cost", float("nan")) or float("nan"))

    def optimal_time_s(self, inst: Any) -> float:
        return float(getattr(inst, "optimal_time", float("nan")) or float("nan"))

    def train_batch(self, batch_size: int, device: str):
        insts = [self.sampler.sample() for _ in range(int(batch_size))]
        td = _batch_from_instances(insts, n_nodes=int(self.n_nodes), device=device)
        return self.kernel.reset(td, init_action_mask=True)

    def make_eval_batch(self, inst: Any, device: str):
        if not isinstance(inst, TSP):
            raise TypeError("TSPTorchKernelBackend expects a TSP instance for evaluation")
        td = _batch_from_instances([inst], n_nodes=int(self.n_nodes), device=device)
        return self.kernel.reset(td, init_action_mask=True)

    def reset(self, batch: TensorDict) -> TensorDict:
        return self.kernel.reset(batch, init_action_mask=True)

    def step(self, batch: TensorDict, action: torch.Tensor) -> TensorDict:
        return self.kernel.step(batch, action)

    def is_done(self, batch: TensorDict) -> torch.Tensor:
        return batch["done"]

    def action_mask(self, batch: TensorDict) -> torch.Tensor:
        return batch["action_mask"]

    def reward(self, batch: TensorDict) -> torch.Tensor:
        return batch["reward"]

    def observe(self, batch: TensorDict) -> Mapping[str, torch.Tensor]:
        dist = batch["dist"].float()
        cur = batch["current"].long()
        bs = int(batch.batch_size[0])
        device = batch.device
        cur = cur.clamp(0, int(self.n_nodes) - 1)
        row = dist[torch.arange(bs, device=device), cur]
        return {
            "node_features": dist,
            "step_features": row,
            "action_mask": self.action_mask(batch),
            "done": self.is_done(batch),
        }

    def score_eval_batch(self, batch: TensorDict) -> tuple[Status, float]:
        done = bool(batch["done"].reshape(-1)[0].item())
        if not done:
            return Status.INFEASIBLE, float("inf")
        r = float(batch["reward"].reshape(-1)[0].item())
        return Status.FEASIBLE, float(-r)

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        r = float(reward.reshape(-1)[0].item())
        return Status.FEASIBLE, float(-r)

    def to_solution(self, batch: TensorDict) -> InfoSolution:
        status, obj = self.score_eval_batch(batch)
        tour = batch["tour"][0].detach().cpu().numpy().tolist()
        info = {"tour": tour}
        return InfoSolution(status=status, objective=float(obj), var_values={}, info=info)


@dataclass
class TSPNativeBackend(TSPTorchKernelBackend):
    name: str = "tsp_native"
