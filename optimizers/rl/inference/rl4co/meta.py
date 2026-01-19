from dataclasses import dataclass

from qoco.optimizers.rl.shared.base import PolicyMeta


@dataclass(frozen=True)
class RL4COPolicyMeta(PolicyMeta):
    embed_dim: int
    num_heads: int = 8
    num_encoder_layers: int = 3

