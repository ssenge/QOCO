from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from qoco.core.solution import Solution, Status
from qoco.optimizers.rl.inference.rl4co.adapter import RL4COAdapter
from qoco.optimizers.rl.training.envs.adapter_env import AdapterEnv
from qoco.optimizers.rl.training.milp.generic import GenericMILPAdapter


class GenericMILPEnv(AdapterEnv):
    name = "generic_milp"

    def __init__(self, adapter: "GenericMILPRL4COAdapter", **kwargs):
        super().__init__(adapter=adapter, **kwargs)
        # RL4CO dataset() expects env.generator (callable returning TensorDict).
        self.generator = lambda batch_size: adapter.train_batch(
            int(batch_size[0] if isinstance(batch_size, list) else batch_size), device=str(self.device)
        )


class InitEmbeddingMILP(nn.Module):
    def __init__(self, embed_dim: int, node_feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(node_feat_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, td: TensorDict) -> torch.Tensor:
        x = td["node_features"].float()
        return self.net(x)


class ContextEmbeddingMILP(nn.Module):
    def __init__(self, embed_dim: int, step_feat_dim: int):
        super().__init__()
        self.step_proj = nn.Linear(step_feat_dim, embed_dim)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, embeddings: torch.Tensor, td: TensorDict) -> torch.Tensor:
        graph = embeddings.mean(dim=1)
        step = td["step_features"].float()
        if step.dim() == 1:
            step = step.unsqueeze(0)
        step_emb = self.step_proj(step)
        return self.proj(torch.cat([graph, step_emb], dim=-1))


class DynamicEmbeddingZeros(nn.Module):
    def __init__(self, embed_dim: int, n_nodes: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_nodes = int(n_nodes)

    def forward(self, td: TensorDict):
        b = int(td.batch_size[0])
        device = td.device
        z = torch.zeros(b, self.n_nodes, self.embed_dim, device=device)
        return z, z, z


class GenericMILPRL4COAdapter(GenericMILPAdapter, RL4COAdapter):
    """Generic MILP adapter that plugs into RL4CO minlib training/inference."""

    def build_rl4co_env(self):
        return GenericMILPEnv(self)

    def build_rl4co_policy(self, embed_dim: int, num_heads: int, num_encoder_layers: int):
        from rl4co.models import AttentionModelPolicy

        init = InitEmbeddingMILP(embed_dim=int(embed_dim), node_feat_dim=int(self.node_feat_dim))
        ctx = ContextEmbeddingMILP(embed_dim=int(embed_dim), step_feat_dim=int(self.step_feat_dim))
        dyn = DynamicEmbeddingZeros(embed_dim=int(embed_dim), n_nodes=int(self.n_nodes))
        return AttentionModelPolicy(
            env_name="generic_milp",
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_encoder_layers=int(num_encoder_layers),
            init_embedding=init,
            context_embedding=ctx,
            dynamic_embedding=dyn,
        )

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        r = float(reward.reshape(-1)[0].item())
        if abs(r - float(self.config.infeasible_reward)) < 1e-6:
            return Status.INFEASIBLE, float("inf")
        return Status.FEASIBLE, float(-r)

    def to_solution_from_actions(self, inst: Any, actions, status: Status, cost: float) -> Solution:
        # Robust decoding: replay the action sequence in our env core and return its Solution.
        td = self.make_eval_batch(inst, device="cpu")
        td = self.reset(td)
        a = actions
        if isinstance(a, torch.Tensor) and a.dim() == 3 and a.shape[-1] == 1:
            a = a.squeeze(-1)
        if isinstance(a, torch.Tensor) and a.dim() == 2:
            a = a[0]
        if isinstance(a, torch.Tensor):
            seq = a.detach().cpu().reshape(-1).tolist()
        else:
            seq = list(a)

        for step_action in seq:
            if bool(td["done"].reshape(-1)[0].item()):
                break
            td = self.step(td, torch.tensor([int(step_action)], dtype=torch.long))
        return self.to_solution(td)

