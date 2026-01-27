from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from qoco.core.solution import Status
from qoco.optimizers.rl.inference.relex.adapter import RelexAdapter
from qoco.optimizers.rl.inference.rl4co.adapter import RL4COAdapter
from qoco.optimizers.rl.training.envs.adapter_env import AdapterEnv
from qoco.optimizers.rl.shared.backend_wrapped_adapter import BackendWrappedAdapter


@dataclass
class BackendRelexAdapter(BackendWrappedAdapter, RelexAdapter):
    """Backend-agnostic adapter for the Relex runner."""

    _specs: dict | None = field(default=None, init=False, repr=False)

    def clone_feature_specs(self) -> dict:
        if self._specs is not None:
            return dict(self._specs)
        td = self.backend.reset(self.backend.train_batch(1, device="cpu"))
        obs = self.backend.observe(td)
        node = obs["node_features"]
        step = obs["step_features"]
        self._specs = {
            "n_nodes": int(node.shape[1]),
            "node_feat_dim": int(node.shape[-1]),
            "step_feat_dim": int(step.shape[-1]) if step.dim() > 1 else int(step.numel()),
        }
        return dict(self._specs)

    def clone_node_features(self, batch) -> torch.Tensor:
        return self.backend.observe(batch)["node_features"].float()

    def clone_step_features(self, batch) -> torch.Tensor:
        return self.backend.observe(batch)["step_features"].float()


class _GenericEnv(AdapterEnv):
    name = "backend_wrapped"

    def __init__(self, adapter: "BackendRL4COAdapter", **kwargs):
        super().__init__(adapter=adapter, **kwargs)


@dataclass
class BackendRL4COAdapter(BackendWrappedAdapter, RL4COAdapter):
    """Backend-agnostic adapter for RL4CO policies."""

    _specs: dict | None = field(default=None, init=False, repr=False)

    def _obs_specs(self) -> dict:
        if self._specs is not None:
            return dict(self._specs)
        td = self.backend.reset(self.backend.train_batch(1, device="cpu"))
        obs = self.backend.observe(td)
        node = obs["node_features"]
        step = obs["step_features"]
        self._specs = {
            "n_nodes": int(node.shape[1]),
            "node_feat_dim": int(node.shape[-1]),
            "step_feat_dim": int(step.shape[-1]) if step.dim() > 1 else int(step.numel()),
        }
        return dict(self._specs)

    def build_rl4co_env(self):
        return _GenericEnv(self)

    def build_rl4co_policy(self, embed_dim: int, num_heads: int, num_encoder_layers: int):
        from rl4co.models import AttentionModelPolicy

        s = self._obs_specs()
        init = _InitEmbeddingNodeFeatures(embed_dim=int(embed_dim), node_feat_dim=int(s["node_feat_dim"]))
        ctx = _ContextEmbeddingStepFeatures(embed_dim=int(embed_dim), step_feat_dim=int(s["step_feat_dim"]))
        dyn = _DynamicEmbeddingZeros(embed_dim=int(embed_dim), n_nodes=int(s["n_nodes"]))
        return AttentionModelPolicy(
            env_name=_GenericEnv.name,
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_encoder_layers=int(num_encoder_layers),
            init_embedding=init,
            context_embedding=ctx,
            dynamic_embedding=dyn,
        )

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        return self.backend.score_from_reward(reward)

    def to_solution_from_actions(self, inst: Any, actions, status: Status, cost: float):
        td = self.make_eval_batch(inst, device="cpu")
        td = self.reset(td)
        a = actions
        if isinstance(a, torch.Tensor) and a.dim() == 3 and a.shape[-1] == 1:
            a = a.squeeze(-1)
        if isinstance(a, torch.Tensor) and a.dim() == 2:
            a = a[0]
        seq = a.detach().cpu().reshape(-1).tolist() if isinstance(a, torch.Tensor) else list(a)
        for step_action in seq:
            if bool(td["done"].reshape(-1)[0].item()):
                break
            td = self.step(td, torch.tensor([int(step_action)], dtype=torch.long))
        return self.to_solution(td)


class _InitEmbeddingNodeFeatures(nn.Module):
    def __init__(self, embed_dim: int, node_feat_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(int(node_feat_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )

    def forward(self, td: TensorDict) -> torch.Tensor:
        x = td["node_features"].float()
        return self.embedding(x)


class _ContextEmbeddingStepFeatures(nn.Module):
    def __init__(self, embed_dim: int, step_feat_dim: int):
        super().__init__()
        self.step_proj = nn.Linear(int(step_feat_dim), int(embed_dim))
        self.proj = nn.Linear(int(embed_dim) * 2, int(embed_dim))

    def forward(self, embeddings: torch.Tensor, td: TensorDict) -> torch.Tensor:
        graph_mean = embeddings.mean(dim=1)
        step = td["step_features"].float()
        if step.dim() == 1:
            step = step.unsqueeze(0)
        step_emb = self.step_proj(step)
        return self.proj(torch.cat([graph_mean, step_emb], dim=-1))


class _DynamicEmbeddingZeros(nn.Module):
    def __init__(self, embed_dim: int, n_nodes: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_nodes = int(n_nodes)

    def forward(self, td: TensorDict):
        b = int(td.batch_size[0])
        device = td.device
        zeros = torch.zeros(b, self.n_nodes, self.embed_dim, device=device)
        return zeros, zeros, zeros
