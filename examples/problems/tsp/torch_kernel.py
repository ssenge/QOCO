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
    edge_topk: int | None = None
    reachability_prune: bool = False
    return_to_start_prune: bool = False

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
        cur = td["current"].long()
        bs = int(td.batch_size[0])
        device = td.device

        if "edge_mask" in td.keys():
            edge_mask = td["edge_mask"].bool()
        else:
            edge_mask = torch.isfinite(td["dist"]).bool()
            edge_mask = edge_mask.clone()
            edge_mask[..., torch.arange(self.n_nodes, device=device), torch.arange(self.n_nodes, device=device)] = False

        batch_idx = torch.arange(bs, device=device)
        allowed = edge_mask[batch_idx, cur]
        mask = allowed & ~visited

        final = step >= self.n_nodes - 1
        if final.any():
            only_start = torch.zeros_like(mask)
            only_start[:, 0] = allowed[:, 0]
            mask = torch.where(final.unsqueeze(-1), only_start, mask)

        mask = torch.where(done.unsqueeze(-1), torch.zeros_like(mask), mask)

        k = int(self.edge_topk) if self.edge_topk is not None else 0
        if k > 0:
            k = min(k, int(self.n_nodes))
            base_mask = mask
            dist_row = td["dist"][batch_idx, cur].float().clone()
            dist_row = torch.where(base_mask, dist_row, torch.full_like(dist_row, float("inf")))
            _, idx = torch.topk(dist_row, k, largest=False)
            topk_mask = torch.zeros_like(base_mask)
            topk_mask.scatter_(1, idx, True)
            mask = base_mask & topk_mask
            no_valid = ~mask.any(dim=-1, keepdim=True)
            mask = torch.where(no_valid, base_mask, mask)

        if self.reachability_prune:
            mask = self._prune_dead_ends(td, mask, edge_mask)

        return mask

    def _prune_dead_ends(self, td: TensorDict, mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        visited = td["visited"]
        bs, n = visited.shape
        out = mask.clone()
        remaining = ~visited
        for b in range(bs):
            if not bool(mask[b].any()):
                continue
            for a in range(n):
                if not bool(mask[b, a]):
                    continue
                rem = remaining[b].clone()
                rem[a] = False
                if not bool(rem.any()):
                    continue
                allowed_nodes = rem.clone()
                allowed_nodes[0] = True
                outgoing_ok = edge_mask[b][rem][:, allowed_nodes].any(dim=1)
                incoming_ok = edge_mask[b][allowed_nodes][:, rem].any(dim=0)
                if not bool(outgoing_ok.all() and incoming_ok.all()):
                    out[b, a] = False
                    continue
                if self.return_to_start_prune and not self._all_reachable_to_start(edge_mask[b], allowed_nodes):
                    out[b, a] = False
        no_valid = ~out.any(dim=-1, keepdim=True)
        return torch.where(no_valid, mask, out)

    def _all_reachable_to_start(self, edge_mask: torch.Tensor, allowed: torch.Tensor) -> bool:
        start = 0
        if not bool(allowed[start].item()):
            return False
        reachable_from = self._bfs(edge_mask, start, allowed)
        reachable_to = self._bfs(edge_mask.transpose(0, 1), start, allowed)
        required = allowed.clone()
        return bool((reachable_from[required].all() and reachable_to[required].all()).item())

    def _bfs(self, edge_mask: torch.Tensor, start: int, allowed: torch.Tensor) -> torch.Tensor:
        n = edge_mask.shape[0]
        visited = torch.zeros((n,), dtype=torch.bool, device=edge_mask.device)
        frontier = torch.zeros_like(visited)
        if not bool(allowed[start].item()):
            return visited
        visited[start] = True
        frontier[start] = True
        while bool(frontier.any().item()):
            neighbors = edge_mask[frontier].any(dim=0)
            neighbors = neighbors & allowed & ~visited
            if not bool(neighbors.any().item()):
                break
            visited |= neighbors
            frontier = neighbors
        return visited

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
                total += float(dist[b, a, bnode].item())
            out[b] = -total
        return out


def tsp_to_tensordict(dist: np.ndarray) -> TensorDict:
    return _pack_to_tensordict(dist=dist)
