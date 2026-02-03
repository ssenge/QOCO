from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np
import torch
from tensordict import TensorDict

from qoco.core.sampler import Sampler
from qoco.core.solution import Solution, Status
from qoco.examples.problems.jobshop.problem import JobShopScheduling
from qoco.examples.problems.jobshop.sampler import RandomJobShopSampler
from qoco.examples.problems.jobshop.torch_kernel import JobShopTorchKernel
from qoco.optimizers.rl.shared.backend import EnvBackend


def _fixed_jobshop_shape(sampler: Sampler[JobShopScheduling]) -> tuple[int, int, int]:
    sizes = getattr(sampler, "sizes", None)
    if sizes:
        uniq = sorted({tuple(int(v) for v in s) for s in sizes})
        if len(uniq) != 1:
            raise ValueError("JobShop torch kernel backend requires fixed (n_jobs, n_machines, n_ops); pass a single size.")
        return uniq[0]
    inst = sampler.sample()
    n_jobs = int(len(inst.jobs))
    n_ops = int(len(inst.jobs[0])) if inst.jobs else 0
    n_machines = max((op.machine for job in inst.jobs for op in job), default=-1) + 1
    return n_jobs, n_machines, n_ops


def _matrices_from_instance(inst: JobShopScheduling, *, n_jobs: int, n_ops: int) -> tuple[np.ndarray, np.ndarray]:
    if len(inst.jobs) != n_jobs:
        raise ValueError(f"Expected {n_jobs} jobs, got {len(inst.jobs)}")
    if any(len(job) != n_ops for job in inst.jobs):
        raise ValueError("JobShopTorchKernelBackend requires a fixed n_ops per job.")

    machines = np.zeros((n_jobs, n_ops), dtype=np.int64)
    durations = np.zeros((n_jobs, n_ops), dtype=np.float32)
    for j, job in enumerate(inst.jobs):
        for o, op in enumerate(job):
            machines[j, o] = int(op.machine)
            durations[j, o] = float(op.duration)
    return machines, durations


def _batch_from_instances(insts: List[JobShopScheduling], *, n_jobs: int, n_ops: int, device: str) -> TensorDict:
    mats = [_matrices_from_instance(inst, n_jobs=n_jobs, n_ops=n_ops) for inst in insts]
    machines = torch.from_numpy(np.stack([m for (m, _) in mats])).to(device)
    durations = torch.from_numpy(np.stack([d for (_, d) in mats])).to(device)
    n_jobs_t = torch.full((len(insts), 1), int(n_jobs), dtype=torch.long, device=device)
    n_ops_t = torch.full((len(insts), 1), int(n_ops), dtype=torch.long, device=device)
    return TensorDict(
        {"machines": machines, "durations": durations, "n_jobs": n_jobs_t, "n_ops": n_ops_t},
        batch_size=[len(insts)],
    ).to(device)


@dataclass
class JobShopTorchKernelBackend(EnvBackend):
    name: str = "jobshop_torch_kernel"
    sampler: Sampler[JobShopScheduling] = field(default_factory=RandomJobShopSampler)

    kernel: JobShopTorchKernel = field(init=False, repr=False)
    n_jobs: int = field(init=False)
    n_ops: int = field(init=False)
    n_machines: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_jobs, self.n_machines, self.n_ops = _fixed_jobshop_shape(self.sampler)
        self.kernel = JobShopTorchKernel(
            n_jobs=int(self.n_jobs),
            n_ops=int(self.n_ops),
            n_machines=int(self.n_machines),
        )

    def load_test_instances(self, path: Path, limit: int) -> List[JobShopScheduling]:
        raise ValueError("JobShopTorchKernelBackend.load_test_instances is not implemented.")

    def optimal_cost(self, inst: Any) -> float:
        return float(getattr(inst, "optimal_cost", float("nan")) or float("nan"))

    def optimal_time_s(self, inst: Any) -> float:
        return float(getattr(inst, "optimal_time", float("nan")) or float("nan"))

    def train_batch(self, batch_size: int, device: str):
        insts = [self.sampler.sample() for _ in range(int(batch_size))]
        td = _batch_from_instances(insts, n_jobs=int(self.n_jobs), n_ops=int(self.n_ops), device=device)
        return self.kernel.reset(td, init_action_mask=True)

    def make_eval_batch(self, inst: Any, device: str):
        if not isinstance(inst, JobShopScheduling):
            raise TypeError("JobShopTorchKernelBackend expects a JobShopScheduling instance for evaluation")
        td = _batch_from_instances([inst], n_jobs=int(self.n_jobs), n_ops=int(self.n_ops), device=device)
        return self.kernel.reset(td, init_action_mask=True)

    def reset(self, batch: TensorDict) -> TensorDict:
        return self.kernel.reset(batch, init_action_mask=True)

    def step(self, batch: TensorDict, action: torch.Tensor) -> TensorDict:
        return self.kernel.step(batch, action)

    def is_done(self, batch: TensorDict) -> torch.Tensor:
        return batch["done"]

    def action_mask(self, batch: TensorDict) -> torch.Tensor:
        return batch["action_mask"]

    def reward(self, batch: TensorDict) -> torch.Tensor:
        return batch["reward"]

    def observe(self, batch: TensorDict) -> Mapping[str, torch.Tensor]:
        jobs = batch["job_next"]
        machines = batch["machines"]
        durations = batch["durations"]
        job_time = batch["job_time"]

        n_ops = int(self.n_ops)
        n_machines = max(1, int(self.n_machines) - 1)
        max_dur = durations.max().clamp(min=1.0)

        next_idx = jobs.clamp(0, n_ops - 1).unsqueeze(-1)
        next_machine = torch.gather(machines, 2, next_idx).squeeze(-1)
        next_duration = torch.gather(durations, 2, next_idx).squeeze(-1)
        active = (jobs < n_ops).float()
        next_machine = next_machine.float() * active
        next_duration = next_duration.float() * active

        node_features = torch.stack(
            [
                jobs.float() / max(1.0, float(n_ops)),
                job_time.float(),
                next_machine / float(n_machines),
                next_duration / max_dur,
            ],
            dim=-1,
        )

        makespan = batch["machine_time"].max(dim=-1).values
        progress = jobs.float().mean(dim=-1) / max(1.0, float(n_ops))
        step_features = torch.stack([makespan, progress], dim=-1)

        return {
            "node_features": node_features,
            "step_features": step_features,
            "action_mask": self.action_mask(batch),
            "done": self.is_done(batch),
        }

    def score_eval_batch(self, batch: TensorDict) -> tuple[Status, float]:
        done = bool(batch["done"].reshape(-1)[0].item())
        if not done:
            return Status.INFEASIBLE, float("inf")
        makespan = float(batch["machine_time"].max().item())
        return Status.FEASIBLE, makespan

    def score_from_reward(self, reward: torch.Tensor) -> tuple[Status, float]:
        r = float(reward.reshape(-1)[0].item())
        return Status.FEASIBLE, float(-r)

    def to_solution(self, batch: TensorDict) -> Solution:
        status, obj = self.score_eval_batch(batch)
        return Solution(status=status, objective=float(obj), var_values={})


@dataclass
class JobShopNativeBackend(JobShopTorchKernelBackend):
    name: str = "jobshop_native"
