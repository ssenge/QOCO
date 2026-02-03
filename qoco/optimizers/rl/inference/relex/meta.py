from dataclasses import dataclass

from qoco.optimizers.rl.shared.base import PolicyMeta
from qoco.optimizers.rl.training.quantum.pqc_pool import PQCConfig


@dataclass(frozen=True)
class RelexAMConfig:
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
    model_name: str = "standard"


@dataclass(frozen=True)
class RelexPolicyMeta(PolicyMeta):
    cfg: RelexAMConfig

