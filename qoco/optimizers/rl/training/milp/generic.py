from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import pyomo.environ as pyo
import torch
from tensordict import TensorDict

from qoco.core.solution import Solution, Status
from qoco.optimizers.rl.training.milp.pyomo_eval import (
    CompiledNumericMILPKernel,
    apply_delta_numeric_inplace,
    compile_numeric_milp_kernel,
    feasible_all_numeric,
    init_lhs_numeric,
)
from qoco.optimizers.rl.training.problem_adapter import ProblemAdapter
from qoco.optimizers.rl.training.milp.selectors import CandidateSelector, SelectorFactory


@dataclass(frozen=True)
class GenericMILPConfig:
    max_steps: int
    infeasible_reward: float = -1000.0
    ignore_lower_bounds_during_construction: bool = True
    tol: float = 1e-6


@dataclass
class _EpisodeCache:
    model: pyo.ConcreteModel
    kernel: CompiledNumericMILPKernel
    selector: CandidateSelector
    lhs: np.ndarray
    selected: np.ndarray  # (n_vars,) bool
    obj_value: float
    idx_to_name: list[str]
    step: int = 0


def _objective_from_kernel(kernel: CompiledNumericMILPKernel, selected: np.ndarray) -> float:
    v = float(kernel.obj_constant)
    for vidx, coef in kernel.obj_var_coef.items():
        if bool(selected[int(vidx)]):
            v += float(coef)
    return float(v)


def _feasible_candidate_mask(
    *,
    kernel: CompiledNumericMILPKernel,
    lhs: np.ndarray,
    selected: np.ndarray,
    var_indices: np.ndarray,
    delta: float,
    ignore_lower: bool,
    tol: float,
) -> np.ndarray:
    """Feasibility for setting candidates to 1, allowing partial construction.

    If ignore_lower=True, we only enforce upper bounds during construction.
    This supports common "assignment-style" equalities where the lower bound
    is only satisfiable after enough actions have been taken.
    """
    var_indices = np.asarray(var_indices, dtype=np.int64)
    out = np.zeros((int(var_indices.size),), dtype=bool)
    for k in range(int(var_indices.size)):
        vidx = int(var_indices[k])
        if vidx < 0:
            continue
        if bool(selected[vidx]):
            continue
        # fixed values (hard)
        fv = kernel.fixed_idx_value.get(vidx)
        if fv is not None and abs((0.0 + float(delta)) - float(fv)) > 1e-9:
            continue
        ok = True
        for cidx, coef in kernel.var_to_cons[vidx]:
            body = float(lhs[cidx] + float(coef) * float(delta))
            if body > float(kernel.upper[cidx] + tol):
                ok = False
                break
            if not ignore_lower and body < float(kernel.lower[cidx] - tol):
                ok = False
                break
        out[k] = ok
    return out


@dataclass
class GenericMILPAdapter(ProblemAdapter):
    """Generic MILP adapter for incremental construction.

    This adapter is intentionally model-centric:
    - Input is any object that can be converted to a Pyomo `ConcreteModel`
    - Actions set binary variables to 1 (monotone construction)
    - Feasibility is evaluated via a compiled numeric kernel (no solver)

    Bridge points (optional):
    - `selector_factory`: controls action space (default: all binary vars)
    - `to_solution`: override decoding
    """

    name: str
    build_model: Callable[[Any], pyo.ConcreteModel]
    config: GenericMILPConfig
    selector_factory: SelectorFactory
    # Bridge points (required for training/eval plumbing in our stack)
    n_nodes: int
    load_instances: Callable[[Path, int], List[Any]]
    sample_train_instances: Callable[[int], List[Any]]
    optimal_cost_fn: Callable[[Any], float]
    optimal_time_s_fn: Callable[[Any], float]
    node_feat_dim: int = 4
    step_feat_dim: int = 2
    decode_solution: Callable[[pyo.ConcreteModel, Status, float, dict[str, Any]], Solution] | None = None
    advance_step: Callable[[str], bool] = lambda _name: True

    _episodes: dict[int, _EpisodeCache] = field(default_factory=dict, init=False, repr=False)
    _next_episode_id: int = field(default=1, init=False, repr=False)

    def load_test_instances(self, path: Path, limit: int) -> List[Any]:
        return self.load_instances(path, int(limit))

    def optimal_cost(self, inst: Any) -> float:
        return float(self.optimal_cost_fn(inst))

    def optimal_time_s(self, inst: Any) -> float:
        return float(self.optimal_time_s_fn(inst))

    def train_batch(self, batch_size: int, device: str) -> Any:
        bs = int(batch_size)
        insts = self.sample_train_instances(bs)
        if len(insts) != bs:
            raise ValueError(f"sample_train_instances({bs}) returned {len(insts)} instances")
        return TensorDict({"inst": insts}, batch_size=[bs], device=torch.device(device))

    def make_eval_batch(self, inst: Any, device: str) -> Any:
        return self.reset(TensorDict({"inst": [inst]}, batch_size=[1], device=torch.device(device)))

    def reward(self, batch: Any) -> Any:
        return batch["reward"]

    def _create_episode(self, inst: Any) -> int:
        model = self.build_model(inst)
        kernel = compile_numeric_milp_kernel(model=model)
        if kernel.has_nonlinear:
            raise ValueError("GenericMILPAdapter requires linear constraints/objective")

        idx_to_name: list[str] = [""] * int(kernel.n_vars)
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
            vid = id(v)
            vidx = kernel.var_id_to_idx.get(vid)
            if vidx is not None:
                idx_to_name[int(vidx)] = str(v.name)

        selector = self.selector_factory.build(model, kernel).prepare(model, kernel)
        if int(selector.n_actions) != int(self.n_nodes):
            raise ValueError(f"selector.n_actions={int(selector.n_actions)} != n_nodes={int(self.n_nodes)}")

        x0 = np.zeros((kernel.n_vars,), dtype=np.float64)
        lhs = init_lhs_numeric(kernel=kernel, x=x0)
        selected = np.zeros((kernel.n_vars,), dtype=bool)
        obj_value = _objective_from_kernel(kernel, selected)

        eid = int(self._next_episode_id)
        self._next_episode_id += 1
        self._episodes[eid] = _EpisodeCache(
            model=model,
            kernel=kernel,
            selector=selector,
            lhs=lhs,
            selected=selected,
            obj_value=obj_value,
            idx_to_name=idx_to_name,
            step=0,
        )
        return eid

    def _candidate_mask(self, ep: _EpisodeCache) -> tuple[np.ndarray, np.ndarray]:
        var_idxs = ep.selector.var_indices(int(ep.step))
        mask = _feasible_candidate_mask(
            kernel=ep.kernel,
            lhs=ep.lhs,
            selected=ep.selected,
            var_indices=var_idxs,
            delta=1.0,
            ignore_lower=bool(self.config.ignore_lower_bounds_during_construction),
            tol=float(self.config.tol),
        )
        return var_idxs.astype(np.int64, copy=False), mask.astype(bool, copy=False)

    def _build_features(self, ep: _EpisodeCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        var_idxs, mask_real = self._candidate_mask(ep)
        mask = mask_real.copy()
        if not bool(mask.any()) and int(mask.size) > 0:
            neg = np.flatnonzero(var_idxs < 0)
            k = int(neg[0]) if int(neg.size) > 0 else 0
            mask[k] = True

        node = np.zeros((int(self.n_nodes), int(self.node_feat_dim)), dtype=np.float32)
        for a in range(int(self.n_nodes)):
            vidx = int(var_idxs[a])
            if vidx < 0:
                continue
            obj_c = float(ep.kernel.obj_var_coef.get(vidx, 0.0))
            deg = float(len(ep.kernel.var_to_cons[vidx]))
            sel = 1.0 if bool(ep.selected[vidx]) else 0.0
            fix = 1.0 if vidx in ep.kernel.fixed_idx_value else 0.0
            node[a, 0] = obj_c
            node[a, 1] = deg
            node[a, 2] = sel
            node[a, 3] = fix

        step = np.zeros((int(self.step_feat_dim),), dtype=np.float32)
        s = float(ep.step)
        ms = float(self.config.max_steps) if int(self.config.max_steps) > 0 else 1.0
        step[0] = s / ms
        step[1] = float(ep.selected.sum()) / ms
        return var_idxs.astype(np.int64, copy=False), mask.astype(bool, copy=False), node, step

    def reset(self, td: TensorDict) -> TensorDict:
        bs = int(td.batch_size[0]) if td.batch_size else 1
        insts = td["inst"]
        dev = td.device

        eids: list[int] = []
        steps: list[int] = []
        dones: list[bool] = []
        rewards: list[float] = []
        masks: list[np.ndarray] = []
        cand: list[np.ndarray] = []
        node_feats: list[np.ndarray] = []
        step_feats: list[np.ndarray] = []

        for b in range(bs):
            inst = insts[b] if bs > 1 else insts[0]
            eid = self._create_episode(inst)
            ep = self._episodes[eid]
            var_idxs, mask, node, step = self._build_features(ep)
            eids.append(eid)
            steps.append(0)
            dones.append(not bool(mask.any()))
            rewards.append(0.0)
            masks.append(mask)
            cand.append(var_idxs)
            node_feats.append(node)
            step_feats.append(step)

        td_out = TensorDict({}, batch_size=[bs], device=dev)
        td_out["inst"] = insts
        td_out["episode_id"] = torch.tensor(eids, dtype=torch.long, device=dev)
        td_out["step"] = torch.tensor(steps, dtype=torch.long, device=dev)
        td_out["done"] = torch.tensor(dones, dtype=torch.bool, device=dev)
        td_out["reward"] = torch.tensor(rewards, dtype=torch.float32, device=dev)
        td_out["candidate_var_idx"] = torch.tensor(np.stack(cand, axis=0), dtype=torch.long, device=dev)
        td_out["action_mask"] = torch.tensor(np.stack(masks, axis=0), dtype=torch.bool, device=dev)
        td_out["node_features"] = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float32, device=dev)
        td_out["step_features"] = torch.tensor(np.stack(step_feats, axis=0), dtype=torch.float32, device=dev)
        return td_out

    def is_done(self, td: TensorDict) -> torch.Tensor:
        eids = td["episode_id"].reshape(-1).tolist()
        out: list[bool] = []
        for eid in eids:
            ep = self._episodes[int(eid)]
            if bool(ep.step >= int(self.config.max_steps)):
                out.append(True)
                continue
            vi, mask_real = self._candidate_mask(ep)
            out.append(bool(vi.size == 0 or (vi < 0).all() or (not bool(mask_real.any()))))
        return torch.tensor(out, dtype=torch.bool, device=td.device)

    def action_mask(self, td: TensorDict) -> torch.Tensor:
        eids = td["episode_id"].reshape(-1).tolist()
        masks: list[np.ndarray] = []
        for eid in eids:
            ep = self._episodes[int(eid)]
            _, mask, _, _ = self._build_features(ep)
            masks.append(mask)
        return torch.tensor(np.stack(masks, axis=0), dtype=torch.bool, device=td.device)

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        eids = td["episode_id"].reshape(-1).tolist()
        acts = action.reshape(-1).tolist()
        bs = int(len(eids))

        steps: list[int] = []
        dones: list[bool] = []
        rewards: list[float] = []
        masks: list[np.ndarray] = []
        cand: list[np.ndarray] = []
        node_feats: list[np.ndarray] = []
        step_feats: list[np.ndarray] = []

        for b in range(bs):
            eid = int(eids[b])
            ep = self._episodes[eid]
            a = int(acts[b])
            var_idxs, mask_real = self._candidate_mask(ep)
            if not bool(mask_real.any()):
                var_idxs2, mask2, node2, step2 = self._build_features(ep)
                steps.append(int(ep.step))
                dones.append(True)
                rewards.append(float(self.config.infeasible_reward))
                cand.append(var_idxs2)
                masks.append(mask2)
                node_feats.append(node2)
                step_feats.append(step2)
                continue
            if a < 0 or a >= int(var_idxs.size):
                raise IndexError(f"action {a} out of range for n_actions={int(var_idxs.size)}")
            vidx = int(var_idxs[a])
            if vidx < 0 or not bool(mask_real[a]):
                # No valid action to apply -> terminate episode as infeasible.
                var_idxs2, mask2, node2, step2 = self._build_features(ep)
                steps.append(int(ep.step))
                dones.append(True)
                rewards.append(float(self.config.infeasible_reward))
                cand.append(var_idxs2)
                masks.append(mask2)
                node_feats.append(node2)
                step_feats.append(step2)
                continue

            apply_delta_numeric_inplace(kernel=ep.kernel, lhs=ep.lhs, var_idx=vidx, delta=1.0)
            ep.selected[vidx] = True
            ep.obj_value = _objective_from_kernel(ep.kernel, ep.selected)
            if bool(self.advance_step(ep.idx_to_name[vidx])):
                ep.step += 1

            vi2 = ep.selector.var_indices(int(ep.step))
            no_candidates = bool(vi2.size == 0 or (vi2 < 0).all())
            _, mask2, _, _ = self._build_features(ep)
            no_feasible = not bool(mask2.any())
            done = bool(ep.step >= int(self.config.max_steps) or no_candidates or no_feasible)
            reward = 0.0
            if done:
                feasible = feasible_all_numeric(kernel=ep.kernel, lhs=ep.lhs, tol=float(self.config.tol))
                reward = float(-ep.obj_value) if feasible else float(self.config.infeasible_reward)

            var_idxs2, mask2, node2, step2 = self._build_features(ep)
            steps.append(int(ep.step))
            dones.append(done)
            rewards.append(float(reward))
            cand.append(var_idxs2)
            masks.append(mask2)
            node_feats.append(node2)
            step_feats.append(step2)

        td_out = td.clone()
        td_out["step"] = torch.tensor(steps, dtype=torch.long, device=td.device)
        td_out["done"] = torch.tensor(dones, dtype=torch.bool, device=td.device)
        td_out["reward"] = torch.tensor(rewards, dtype=torch.float32, device=td.device)
        td_out["candidate_var_idx"] = torch.tensor(np.stack(cand, axis=0), dtype=torch.long, device=td.device)
        td_out["action_mask"] = torch.tensor(np.stack(masks, axis=0), dtype=torch.bool, device=td.device)
        td_out["node_features"] = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float32, device=td.device)
        td_out["step_features"] = torch.tensor(np.stack(step_feats, axis=0), dtype=torch.float32, device=td.device)
        return td_out

    def score_eval_batch(self, batch: TensorDict) -> tuple[Status, float]:
        eid = int(batch["episode_id"].reshape(-1)[0].item())
        ep = self._episodes[eid]
        feasible = feasible_all_numeric(kernel=ep.kernel, lhs=ep.lhs, tol=float(self.config.tol))
        if feasible:
            return Status.FEASIBLE, float(ep.obj_value)
        return Status.INFEASIBLE, float("inf")

    def to_solution(self, td: TensorDict) -> Solution:
        eid = int(td["episode_id"].reshape(-1)[0].item())
        ep = self._episodes[eid]
        status, cost = self.score_eval_batch(td)
        if self.decode_solution is not None:
            return self.decode_solution(ep.model, status, float(cost), {})

        # default: report selected var names
        chosen = np.flatnonzero(ep.selected).tolist()
        names = [ep.idx_to_name[int(i)] for i in chosen]
        var_values = {str(n): 1.0 for n in names if n}
        return Solution(status=status, objective=float(cost), var_values=var_values)

