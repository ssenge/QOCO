"""Standard Relex attention model."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qoco.optimizers.rl.training.quantum.pqc_pool import PQCConfig


@dataclass(eq=False)
class SkipConnection(nn.Module):
    module: InitVar[nn.Module]

    def __post_init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self._module(x, **kwargs)


@dataclass(eq=False)
class BatchNorm1dSeq(nn.Module):
    """BatchNorm over feature dim for (B, N, E) tensors."""

    embed_dim: int

    def __post_init__(self) -> None:
        e = int(self.embed_dim)
        super().__init__()
        self.bn = nn.BatchNorm1d(e, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


@dataclass(eq=False)
class MLP(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(self.input_dim), int(self.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim), int(self.output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(eq=False)
class MultiHeadAttention(nn.Module):
    """RL4CO-style MHA wrapper using PyTorch SDPA."""

    embed_dim: int
    num_heads: int
    bias: bool = True

    def __post_init__(self) -> None:
        e = int(self.embed_dim)
        h = int(self.num_heads)
        super().__init__()
        self.head_dim = e // h
        self.Wqkv = nn.Linear(e, 3 * e, bias=bool(self.bias))
        self.out_proj = nn.Linear(e, e, bias=bool(self.bias))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.Wqkv(x).view(b, n, 3, int(self.num_heads), int(self.head_dim)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, int(self.embed_dim))
        return self.out_proj(out)


@dataclass(eq=False)
class MultiHeadAttentionLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_hidden: int

    def __post_init__(self) -> None:
        super().__init__()
        e = int(self.embed_dim)
        self.mha = SkipConnection(MultiHeadAttention(e, int(self.num_heads), bias=True))
        self.bn1 = BatchNorm1dSeq(e)
        self.ffn = SkipConnection(MLP(e, e, hidden_dim=int(self.ff_hidden)))
        self.bn2 = BatchNorm1dSeq(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.mha(x))
        x = self.bn2(self.ffn(x))
        return x


@dataclass(eq=False)
class GraphAttentionNetwork(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_hidden: int

    def __post_init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    embed_dim=int(self.embed_dim),
                    num_heads=int(self.num_heads),
                    ff_hidden=int(self.ff_hidden),
                )
                for _ in range(int(self.num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass(eq=False)
class PointerAttention(nn.Module):
    embed_dim: int
    num_heads: int
    out_bias: bool = False

    def __post_init__(self) -> None:
        e = int(self.embed_dim)
        h = int(self.num_heads)
        super().__init__()
        self.project_out = nn.Linear(e, e, bias=bool(self.out_bias))

    def _make_heads(self, x: torch.Tensor) -> torch.Tensor:
        *prefix, n, e = x.shape
        d = e // int(self.num_heads)
        return x.view(*prefix, n, int(self.num_heads), d).transpose(-3, -2)

    def forward(
        self,
        query: torch.Tensor,      # (B, 1, E)
        key: torch.Tensor,        # (B, N, E)
        value: torch.Tensor,      # (B, N, E)
        logit_key: torch.Tensor,  # (B, N, E)
        attn_mask: torch.Tensor,  # (B, N) with True=allowed
    ) -> torch.Tensor:
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)

        mask = attn_mask.unsqueeze(1).unsqueeze(2)
        heads = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

        glimpse = heads.transpose(1, 2).contiguous().view(query.shape[0], 1, int(self.embed_dim))
        glimpse = self.project_out(glimpse)

        logits = torch.bmm(glimpse, logit_key.transpose(1, 2)).squeeze(1)
        logits = logits / (int(self.embed_dim) ** 0.5)
        logits = logits.masked_fill(~attn_mask, float("-inf"))
        return logits


@dataclass(frozen=True)
class AMConfig:
    n_nodes: int
    node_feat_dim: int
    step_feat_dim: int
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_hidden: int = 512
    lr: float = 1e-4
    use_pqc: bool = False
    pqc: PQCConfig = PQCConfig()


@dataclass
class PrecomputedCache:
    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


@dataclass(eq=False)
class AttentionModelEncoder(nn.Module):
    cfg: AMConfig
    pqc_block: InitVar[nn.Module]

    def __post_init__(self, pqc_block: nn.Module) -> None:
        cfg = self.cfg
        super().__init__()
        self.init_embedding = nn.Sequential(
            nn.Linear(int(cfg.node_feat_dim), int(cfg.embed_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim)),
        )
        self.net = GraphAttentionNetwork(int(cfg.embed_dim), int(cfg.num_heads), int(cfg.num_layers), int(cfg.ff_hidden))
        self._pqc = pqc_block

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        init_h = self.init_embedding(node_features)
        h = self.net(init_h)
        h = self._pqc(h)
        return h


@dataclass(eq=False)
class AttentionModelDecoder(nn.Module):
    cfg: AMConfig

    def __post_init__(self) -> None:
        cfg = self.cfg
        super().__init__()
        self.step_embedding = nn.Sequential(
            nn.Linear(int(cfg.step_feat_dim), int(cfg.embed_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim)),
        )
        self.dynamic_embedding = None
        self.pointer = PointerAttention(int(cfg.embed_dim), int(cfg.num_heads), out_bias=False)
        self.project_node_embeddings = nn.Linear(int(cfg.embed_dim), 3 * int(cfg.embed_dim), bias=False)
        self.project_fixed_context = nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim), bias=False)
        self.use_graph_context = True

    def precompute_cache(self, embeddings: torch.Tensor) -> PrecomputedCache:
        gk, gv, lk = self.project_node_embeddings(embeddings).chunk(3, dim=-1)
        graph_context = self.project_fixed_context(embeddings.mean(1)) if self.use_graph_context else 0.0
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=gk,
            glimpse_val=gv,
            logit_key=lk,
        )

    def forward(self, step_features: torch.Tensor, action_mask: torch.Tensor, cache: PrecomputedCache) -> torch.Tensor:
        step_context = self.step_embedding(step_features)
        graph_context = cache.graph_context
        if isinstance(graph_context, torch.Tensor):
            glimpse_q = step_context + graph_context
        else:
            glimpse_q = step_context
        glimpse_q = glimpse_q.unsqueeze(1)

        glimpse_k = cache.glimpse_key
        glimpse_v = cache.glimpse_val
        logit_k = cache.logit_key

        return self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, action_mask)


@dataclass(eq=False)
class AttentionModel(nn.Module):
    cfg: AMConfig
    pqc_block: InitVar[nn.Module]
    model_name: str = "standard"
    device: str = "cpu"

    def __post_init__(self, pqc_block: nn.Module) -> None:
        super().__init__()
        self.encoder = AttentionModelEncoder(self.cfg, pqc_block)
        self.decoder = AttentionModelDecoder(self.cfg)
        self.to(torch.device(self.device))

    @property
    def device_t(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, node_features: torch.Tensor, step_features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(node_features)
        cache = self.decoder.precompute_cache(h)
        return self.decoder(step_features, action_mask, cache)
