from __future__ import annotations

import torch
from tensordict import TensorDict

from qoco.optimizers.rl.inference.relex.adapter import RelexAdapter
from qoco.optimizers.rl.training.milp.generic import GenericMILPAdapter


class GenericMILPRelexAdapter(GenericMILPAdapter, RelexAdapter):
    """Generic MILP adapter that plugs into the clone/relex method and runner."""

    def clone_feature_specs(self) -> dict:
        return {
            "n_nodes": int(self.n_nodes),
            "node_feat_dim": int(self.node_feat_dim),
            "step_feat_dim": int(self.step_feat_dim),
        }

    def clone_node_features(self, batch: TensorDict) -> torch.Tensor:
        return batch["node_features"].float()

    def clone_step_features(self, batch: TensorDict) -> torch.Tensor:
        return batch["step_features"].float()

