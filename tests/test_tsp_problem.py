from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qoco.core.solution import Status
from qoco.examples.problems.tsp.problem import TSP
from qoco.optimizers.highs import HiGHSOptimizer
from qoco.optimizers.rl.inference.ml_policy import MLPolicyOptimizer
from qoco.optimizers.rl.shared.base import save_policy_checkpoint
from qoco.optimizers.rl.training.milp.generic import GenericMILPConfig
from qoco.optimizers.rl.training.milp.pyomo_eval import compile_numeric_milp_kernel
from qoco.optimizers.rl.training.milp.selectors import ComponentIndexGatedSelector, SelectorFactory


def _tsp_instance() -> TSP:
    dist = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ]
    return TSP(name="tsp_3", dist=dist)


def test_tsp_highs() -> None:
    problem = _tsp_instance()
    solver = HiGHSOptimizer(converter=TSP.MILPConverter())
    sol = solver.optimize(problem, log=False)
    assert sol.status in (Status.OPTIMAL, Status.FEASIBLE)
    assert abs(sol.objective - 6.0) < 1e-6


def _make_adapter(problem: TSP):
    build_model = TSP.MILPConverter().convert
    model = build_model(problem)
    kernel = compile_numeric_milp_kernel(model=model)
    selector_factory = SelectorFactory(
        make=lambda m, k: ComponentIndexGatedSelector(component="x", index_pos=0)
    )
    selector = selector_factory.build(model, kernel).prepare(model, kernel)
    n_nodes = int(selector.n_actions)
    config = GenericMILPConfig(max_steps=len(problem.dist))

    common = dict(
        name="tsp_milp",
        build_model=build_model,
        config=config,
        selector_factory=selector_factory,
        n_nodes=n_nodes,
        load_instances=lambda _path, limit: [problem] * int(limit),
        sample_train_instances=lambda n: [problem] * int(n),
        optimal_cost_fn=lambda _inst: 0.0,
        optimal_time_s_fn=lambda _inst: 0.0,
    )
    return common


def test_tsp_relex_rl4co() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rl4co")

    try:
        from qoco.optimizers.rl.inference.relex.meta import RelexAMConfig, RelexPolicyMeta
        from qoco.optimizers.rl.inference.relex.runner import RelexRunner
        from qoco.optimizers.rl.inference.rl4co.meta import RL4COPolicyMeta
        from qoco.optimizers.rl.inference.rl4co.runner import RL4CORunner
        from qoco.optimizers.rl.training.milp.relex import GenericMILPRelexAdapter
        from qoco.optimizers.rl.training.milp.rl4co import GenericMILPRL4COAdapter
        from qoco.optimizers.rl.training.methods.relex.model_standard import AMConfig, AttentionModel
        from qoco.optimizers.rl.training.quantum.pqc_pool import NoOpPQCBlock
    except Exception as exc:  # pragma: no cover - optional deps
        pytest.skip(f"Missing RL dependencies or incompatible env: {exc}")

    problem = _tsp_instance()
    common = _make_adapter(problem)

    adapter_rl4co = GenericMILPRL4COAdapter(**common)
    adapter_relex = GenericMILPRelexAdapter(**common)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # RL4CO checkpoint
        embed_dim = 16
        rl4co_meta = RL4COPolicyMeta(embed_dim=embed_dim, num_heads=2, num_encoder_layers=1)
        rl4co_policy = adapter_rl4co.build_rl4co_policy(
            embed_dim=embed_dim,
            num_heads=2,
            num_encoder_layers=1,
        )
        rl4co_ckpt = tmp_path / "rl4co.pt"
        save_policy_checkpoint(rl4co_ckpt, meta=rl4co_meta, state=rl4co_policy.state_dict())

        # Relex checkpoint
        relex_cfg = RelexAMConfig(
            n_nodes=int(common["n_nodes"]),
            node_feat_dim=int(adapter_relex.node_feat_dim),
            step_feat_dim=int(adapter_relex.step_feat_dim),
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_hidden=64,
            lr=1e-4,
            use_pqc=False,
            model_name="standard",
        )
        relex_meta = RelexPolicyMeta(cfg=relex_cfg)
        relex_model = AttentionModel(
            AMConfig(
                n_nodes=relex_cfg.n_nodes,
                node_feat_dim=relex_cfg.node_feat_dim,
                step_feat_dim=relex_cfg.step_feat_dim,
                embed_dim=relex_cfg.embed_dim,
                num_heads=relex_cfg.num_heads,
                num_layers=relex_cfg.num_layers,
                ff_hidden=relex_cfg.ff_hidden,
                lr=relex_cfg.lr,
                use_pqc=relex_cfg.use_pqc,
                pqc=relex_cfg.pqc,
            ),
            pqc_block=NoOpPQCBlock(),
        )
        relex_ckpt = tmp_path / "relex.pt"
        save_policy_checkpoint(relex_ckpt, meta=relex_meta, state=relex_model.state_dict())

        opt_rl4co = MLPolicyOptimizer(
            adapter=adapter_rl4co,
            checkpoint_path=rl4co_ckpt,
            runner_cls=RL4CORunner,
        )
        sol_rl4co = opt_rl4co.optimize(problem, log=False)
        assert sol_rl4co.status in (Status.FEASIBLE, Status.INFEASIBLE)

        opt_relex = MLPolicyOptimizer(
            adapter=adapter_relex,
            checkpoint_path=relex_ckpt,
            runner_cls=RelexRunner,
        )
        sol_relex = opt_relex.optimize(problem, log=False)
        assert sol_relex.status in (Status.FEASIBLE, Status.INFEASIBLE)
