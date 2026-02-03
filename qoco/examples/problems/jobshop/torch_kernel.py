from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from qoco.optimizers.rl.shared.torch_kernel import BaseTorchKernel


def jobshop_to_tensordict(*, machines: np.ndarray, durations: np.ndarray) -> TensorDict:
    n_jobs, n_ops = durations.shape
    return TensorDict(
        {
            "machines": torch.from_numpy(machines.astype(np.int64)),
            "durations": torch.from_numpy(durations.astype(np.float32)),
            "n_jobs": torch.tensor([n_jobs], dtype=torch.long),
            "n_ops": torch.tensor([n_ops], dtype=torch.long),
        },
        batch_size=[1],
    )


@dataclass
class JobShopTorchKernel(BaseTorchKernel):
    n_jobs: int
    n_ops: int
    n_machines: int

    def reset(self, td: TensorDict, *, init_action_mask: bool = True) -> TensorDict:
        bs = td.batch_size[0]
        device = td.device
        td = td.clone()
        td["job_next"] = torch.zeros((bs, self.n_jobs), dtype=torch.long, device=device)
        td["job_time"] = torch.zeros((bs, self.n_jobs), dtype=torch.float32, device=device)
        td["machine_time"] = torch.zeros((bs, self.n_machines), dtype=torch.float32, device=device)
        td["done"] = torch.zeros((bs,), dtype=torch.bool, device=device)
        td["reward"] = torch.zeros((bs,), dtype=torch.float32, device=device)
        if init_action_mask:
            td["action_mask"] = self.action_mask(td)
        return td

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        td = td.clone()
        device = td.device
        bs = td.batch_size[0]
        action = action.to(device)

        jobs = td["job_next"]
        machines = td["machines"]
        durations = td["durations"]
        job_time = td["job_time"]
        machine_time = td["machine_time"]

        batch_idx = torch.arange(bs, device=device)
        job = action
        op = jobs[batch_idx, job]
        valid = op < self.n_ops

        m = machines[batch_idx[valid], job[valid], op[valid]]
        p = durations[batch_idx[valid], job[valid], op[valid]]
        start = torch.maximum(job_time[batch_idx[valid], job[valid]], machine_time[batch_idx[valid], m])
        finish = start + p
        job_time[batch_idx[valid], job[valid]] = finish
        machine_time[batch_idx[valid], m] = finish
        jobs[batch_idx[valid], job[valid]] = op[valid] + 1

        td["job_next"] = jobs
        td["job_time"] = job_time
        td["machine_time"] = machine_time

        done = (jobs >= self.n_ops).all(dim=-1)
        td["done"] = done
        td["reward"] = self.reward(td)
        td["action_mask"] = self.action_mask(td)
        return td

    def action_mask(self, td: TensorDict) -> torch.Tensor:
        jobs = td["job_next"]
        mask = jobs < self.n_ops
        mask = torch.where(td["done"].unsqueeze(-1), torch.zeros_like(mask), mask)
        return mask

    def reward(self, td: TensorDict) -> torch.Tensor:
        if not td["done"].any():
            return torch.zeros_like(td["reward"])
        makespan = td["machine_time"].max(dim=-1).values
        return torch.where(td["done"], -makespan, torch.zeros_like(makespan))
