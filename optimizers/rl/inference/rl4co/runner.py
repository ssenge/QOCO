from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from qoco.core.solution import Solution
from qoco.optimizers.rl.shared.base import PolicyRunner, load_policy_checkpoint
from qoco.optimizers.rl.inference.rl4co.adapter import RL4COAdapter
from qoco.optimizers.rl.inference.rl4co.meta import RL4COPolicyMeta


@dataclass
class RL4CORunner(PolicyRunner[RL4COAdapter]):
    policy: torch.nn.Module
    env: object
    embed_dim: int

    @classmethod
    def load(cls, checkpoint_path: Path, *, device: str, adapter: RL4COAdapter) -> "RL4CORunner":
        ckpt = load_policy_checkpoint(checkpoint_path, map_location=device)
        meta = ckpt.meta
        if not isinstance(meta, RL4COPolicyMeta):
            raise TypeError(f"Expected RL4COPolicyMeta, got {type(meta)}")
        embed_dim = int(meta.embed_dim)

        policy = adapter.build_rl4co_policy(
            embed_dim=embed_dim,
            num_heads=int(meta.num_heads),
            num_encoder_layers=int(meta.num_encoder_layers),
        )
        policy.load_state_dict(ckpt.state)
        policy.to(device)
        policy.eval()
        env = adapter.build_rl4co_env()
        return cls(policy=policy, env=env, embed_dim=embed_dim)

    def run(self, *, adapter: RL4COAdapter, problem, device: str) -> Solution:
        td = adapter.make_eval_batch(problem, device=device)
        td0 = adapter.reset(td)
        with torch.no_grad():
            out = self.policy(td0, env=self.env, phase="test", decode_type="greedy", return_actions=True)
        status, cost = adapter.score_from_reward(out["reward"])
        return adapter.to_solution_from_actions(inst=problem, actions=out["actions"], status=status, cost=float(cost))

