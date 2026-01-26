from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from qoco.optimizers.rl.shared.torch_kernel import BaseTorchKernel


def _pack_to_tensordict(*, dist: np.ndarray) -> TensorDict:
    n = int(dist.shape[0])
    return TensorDict(
        {
            "dist": torch.from_numpy(dist.astype(np.float32)),
            "n_nodes": torch.tensor([n], dtype=torch.long),
        },
        batch_size=[1],
    )


@dataclass
class TSPTorchKernel(BaseTorchKernel):
    n_nodes: int

    def reset(self, td: TensorDict, *, init_action_mask: bool = True) -> TensorDict:
        bs = td.batch_size[0]
        device = td.device
        td = td.clone()
        td["current"] = torch.zeros((bs,), dtype=torch.long, device=device)
        td["step"] = torch.zeros((bs,), dtype=torch.long, device=device)
        td["visited"] = torch.zeros((bs, self.n_nodes), dtype=torch.bool, device=device)
        td["visited"][:, 0] = True
        td["tour"] = torch.full((bs, self.n_nodes + 1), -1, dtype=torch.long, device=device)
        td["tour"][:, 0] = 0
        td["done"] = torch.zeros((bs,), dtype=torch.bool, device=device)
        td["reward"] = torch.zeros((bs,), dtype=torch.float32, device=device)
        if init_action_mask:
            td["action_mask"] = self.action_mask(td)
        return td

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        td = td.clone()
        bs = td.batch_size[0]
        device = td.device

        step = td["step"]
        cur = td["current"]
        action = action.to(device)

        batch_idx = torch.arange(bs, device=device)
        td["tour"][batch_idx, step + 1] = action
        td["visited"][batch_idx, action] = True
        td["current"] = action
        td["step"] = step + 1

        done = td["step"] >= self.n_nodes
        td["done"] = done
        td["reward"] = self.reward(td)
        td["action_mask"] = self.action_mask(td)
        return td

    def action_mask(self, td: TensorDict) -> torch.Tensor:
        visited = td["visited"]
        step = td["step"]
        done = td["done"]
        # Allow returning to start on final step.
        mask = ~visited
        mask = torch.where(step.unsqueeze(-1) >= self.n_nodes - 1, torch.zeros_like(mask), mask)
        mask[:, 0] = step >= self.n_nodes - 1
        mask = torch.where(done.unsqueeze(-1), torch.zeros_like(mask), mask)
        return mask

    def reward(self, td: TensorDict) -> torch.Tensor:
        if not td["done"].any():
            return torch.zeros_like(td["reward"])
        dist = td["dist"].float()
        tour = td["tour"]
        bs = tour.shape[0]
        out = torch.zeros((bs,), dtype=torch.float32, device=td.device)
        for b in range(bs):
            if not bool(td["done"][b].item()):
                continue
            seq = tour[b, : self.n_nodes + 1]
            total = 0.0
            for i in range(self.n_nodes):
                a = int(seq[i].item())
                bnode = int(seq[i + 1].item())
                total += float(dist[a, bnode])
            out[b] = -total
        return out


def tsp_to_tensordict(dist: np.ndarray) -> TensorDict:
    return _pack_to_tensordict(dist=dist)
