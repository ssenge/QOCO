"""Relex model with a global bias residual."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from qoco.optimizers.rl.training.methods.relex.model_standard import (
    AMConfig,
    AttentionModelDecoder,
    GraphAttentionNetwork,
)


@dataclass(eq=False)
class GlobalBiasAttentionEncoder(nn.Module):
    cfg: AMConfig

    def __post_init__(self) -> None:
        cfg = self.cfg
        super().__init__()
        self.init_embedding = nn.Sequential(
            nn.Linear(int(cfg.node_feat_dim), int(cfg.embed_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim)),
        )
        self.net = GraphAttentionNetwork(int(cfg.embed_dim), int(cfg.num_heads), int(cfg.num_layers), int(cfg.ff_hidden))
        self.bias = nn.Parameter(torch.zeros(int(cfg.embed_dim)))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        init_h = self.init_embedding(node_features)
        h = self.net(init_h)
        return h + self.bias.view(1, 1, -1)


@dataclass(eq=False)
class GlobalBiasAttentionModel(nn.Module):
    cfg: AMConfig
    device: str = "cpu"
    model_name: str = "global_bias"

    def __post_init__(self) -> None:
        super().__init__()
        self.encoder = GlobalBiasAttentionEncoder(self.cfg)
        self.decoder = AttentionModelDecoder(self.cfg)
        self.to(torch.device(self.device))

    def forward(self, node_features: torch.Tensor, step_features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(node_features)
        cache = self.decoder.precompute_cache(h)
        return self.decoder(step_features, action_mask, cache)
