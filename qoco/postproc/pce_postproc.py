from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import multiprocessing as mp

from qoco.core.qubo import QUBO
from qoco.postproc.qubo_postproc import (
    PostProcResult,
    PostProcessor,
    SampleBatch,
    TakeBestPostProcessor,
    bitstring_to_x,
    evaluate_qubo,
    greedy_1flip_local_search,
)

PCETrainingMode = Literal["on_the_fly", "online"]


@dataclass(frozen=True)
class PCESchema:
    n: int
    edges: list[tuple[int, int]]  # i<j ordering


def _x_to_z(x: np.ndarray) -> np.ndarray:
    # x in {0,1} -> z in {+1,-1} with z = 1 - 2x
    xx = np.asarray(x, dtype=int).reshape(-1)
    return np.asarray(1 - 2 * xx, dtype=int)


def _schema_from_qubo(*, qubo: QUBO, tol: float = 0.0, top_k_edges: int | None = None) -> PCESchema:
    Q = np.asarray(qubo.Q, dtype=float)
    n = int(Q.shape[0])
    pairs: list[tuple[tuple[int, int], float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(float(Q[i, j])) + abs(float(Q[j, i]))
            if w > float(tol):
                pairs.append(((int(i), int(j)), float(w)))
    pairs.sort(key=lambda t: float(t[1]), reverse=True)
    if top_k_edges is not None:
        pairs = pairs[: int(top_k_edges)]
    return PCESchema(n=n, edges=[p for p, _w in pairs])


def _pce_features(*, batch: SampleBatch, schema: PCESchema) -> np.ndarray:
    if int(batch.n) != int(schema.n):
        raise ValueError("PCE schema n mismatch")

    if not batch.samples:
        raise ValueError("empty sample batch")

    n = int(schema.n)
    total_w = float(sum(float(s.weight) for s in batch.samples))
    if total_w <= 0.0:
        raise ValueError("non-positive total sample weight")

    m = np.zeros((n,), dtype=float)
    c = np.zeros((len(schema.edges),), dtype=float)

    for s in batch.samples:
        x = bitstring_to_x(bitstring=s.bitstring, n=n, bit_order=batch.bit_order)
        z = _x_to_z(x).astype(float)
        w = float(s.weight) / total_w
        m += w * z
        for k, (i, j) in enumerate(schema.edges):
            c[k] += w * float(z[int(i)] * z[int(j)])

    return np.concatenate([m, c], axis=0)


def _bootstrap_batches(*, batch: SampleBatch, num_examples: int, rng: np.random.Generator) -> list[SampleBatch]:
    # Create multiple training examples by resampling samples with replacement.
    if not batch.samples:
        raise ValueError("empty sample batch")
    weights = np.asarray([max(0.0, float(s.weight)) for s in batch.samples], dtype=float)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights, dtype=float)
    probs = weights / float(np.sum(weights))

    samples = batch.samples
    out: list[SampleBatch] = []
    for _ in range(int(num_examples)):
        idx = rng.choice(len(samples), size=len(samples), replace=True, p=probs)
        # keep the original weights of the selected samples (proxy for counts)
        boot = [samples[int(i)] for i in idx]
        out.append(SampleBatch(n=int(batch.n), samples=boot, bit_order=batch.bit_order))
    return out


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("PCE requires torch to be installed") from exc
    return torch


def _build_mlp(*, in_dim: int, out_dim: int, hidden: list[int]) -> Any:
    torch = _require_torch()
    nn = torch.nn

    layers: list[Any] = []
    d = int(in_dim)
    for h in list(hidden):
        layers.append(nn.Linear(d, int(h)))
        layers.append(nn.ReLU())
        d = int(h)
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


@dataclass
class PCEPostProcessor(PostProcessor):
    """Trainable PCE post-processor supporting on-the-fly and online learning.

    This is intentionally isolated and does not integrate into optimizers yet.
    """

    mode: PCETrainingMode = "on_the_fly"
    checkpoint_path: str | Path | None = None

    # Feature schema selection
    edge_tol: float = 0.0
    top_k_edges: int | None = 200

    # NN architecture
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])

    # Training
    num_bootstrap_examples: int = 25
    epochs: int = 50
    lr: float = 1e-2
    seed: int = 0

    # Label strategy: use best sample, optionally refined by local search
    label_refine_local_search: bool = True
    label_local_search_max_iters: int = 200

    # Output refinement
    refine_local_search: bool = True
    refine_local_search_max_iters: int = 200

    # Internal state
    _schema: PCESchema | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        path = Path(self.checkpoint_path)
        if not path.exists():
            return

        torch = _require_torch()
        payload = torch.load(str(path), map_location="cpu")
        schema = payload.get("schema", None)
        if not isinstance(schema, dict) or "n" not in schema or "edges" not in schema:
            raise ValueError("invalid PCE checkpoint schema")
        self._schema = PCESchema(n=int(schema["n"]), edges=[tuple(map(int, e)) for e in schema["edges"]])

        model_state = payload.get("model_state", None)
        if model_state is None:
            raise ValueError("invalid PCE checkpoint model_state")

        self._model = _build_mlp(
            in_dim=int(self._schema.n) + int(len(self._schema.edges)),
            out_dim=int(self._schema.n),
            hidden=list(self.hidden_sizes),
        )
        self._model.load_state_dict(model_state)

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        if self._schema is None or self._model is None:
            return
        torch = _require_torch()
        payload = {
            "schema": {"n": int(self._schema.n), "edges": [list(map(int, e)) for e in self._schema.edges]},
            "model_state": self._model.state_dict(),
        }
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    def _ensure_model(self, *, qubo: QUBO) -> None:
        if self.mode == "online":
            if self._model is None:
                self._load_checkpoint()

        if self._schema is None:
            self._schema = _schema_from_qubo(qubo=qubo, tol=float(self.edge_tol), top_k_edges=self.top_k_edges)
        if self._model is None:
            self._model = _build_mlp(
                in_dim=int(self._schema.n) + int(len(self._schema.edges)),
                out_dim=int(self._schema.n),
                hidden=list(self.hidden_sizes),
            )

    def _label_x(self, *, qubo: QUBO, batch: SampleBatch) -> np.ndarray:
        best = TakeBestPostProcessor().run(qubo=qubo, batch=batch)
        x = np.asarray(best.x, dtype=int)
        if not self.label_refine_local_search:
            return x
        x2, _obj2, _it = greedy_1flip_local_search(
            qubo=qubo,
            x0=x,
            max_iters=int(self.label_local_search_max_iters),
            first_improvement=True,
        )
        return np.asarray(x2, dtype=int)

    def _train(self, *, qubo: QUBO, batch: SampleBatch) -> None:
        torch = _require_torch()
        rng = np.random.default_rng(int(self.seed))

        assert self._schema is not None and self._model is not None

        y = self._label_x(qubo=qubo, batch=batch).astype(float)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(1, -1)

        boots = _bootstrap_batches(batch=batch, num_examples=int(self.num_bootstrap_examples), rng=rng)
        X = np.stack([_pce_features(batch=b, schema=self._schema) for b in boots], axis=0).astype(float)
        X_t = torch.tensor(X, dtype=torch.float32)

        # Repeat the same label per bootstrap example (per-run training target).
        Y_t = y_t.repeat(int(X_t.shape[0]), 1)

        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=float(self.lr))
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(int(self.epochs)):
            opt.zero_grad()
            logits = self._model(X_t)
            loss = loss_fn(logits, Y_t)
            loss.backward()
            opt.step()

    def _predict_x(self, *, batch: SampleBatch) -> np.ndarray:
        torch = _require_torch()
        assert self._schema is not None and self._model is not None

        feats = _pce_features(batch=batch, schema=self._schema).astype(float)
        X_t = torch.tensor(feats, dtype=torch.float32).reshape(1, -1)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t).reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.asarray((probs >= 0.5).astype(int), dtype=int)

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        self._ensure_model(qubo=qubo)
        if self.mode in ("on_the_fly", "online"):
            self._train(qubo=qubo, batch=batch)
            if self.mode == "online":
                self._save_checkpoint()

        x = self._predict_x(batch=batch)
        obj = evaluate_qubo(qubo=qubo, x=x)

        x_ref = x
        obj_ref = obj
        it_ref = 0
        if self.refine_local_search:
            x2, obj2, it2 = greedy_1flip_local_search(
                qubo=qubo,
                x0=x,
                max_iters=int(self.refine_local_search_max_iters),
                first_improvement=True,
            )
            x_ref = np.asarray(x2, dtype=int)
            obj_ref = float(obj2)
            it_ref = int(it2)

        return PostProcResult(
            x=x_ref,
            objective=float(obj_ref),
            bitstring=None,
            info={
                "method": "pce_nn",
                "mode": str(self.mode),
                "n": int(batch.n),
                "edges": int(len(self._schema.edges)) if self._schema is not None else None,
                "trained": True,
                "refined": bool(self.refine_local_search),
                "refine_iters": int(it_ref),
            },
        )


def _safe_pce_worker(inner: PCEPostProcessor, qubo: QUBO, batch: SampleBatch, conn) -> None:
    try:
        res = inner.run(qubo=qubo, batch=batch)
        conn.send(res)
    except BaseException as exc:
        conn.send(exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class SafePCEPostProcessor(PostProcessor):
    """Run PCE in a separate process and fall back on failure.

    Reason: on some environments, importing torch can terminate the process (OpenMP runtime conflicts).
    Spawning isolates this failure mode so the caller can still make progress.
    """

    inner: PCEPostProcessor = field(default_factory=PCEPostProcessor)
    fallback: PostProcessor = field(default_factory=TakeBestPostProcessor)
    timeout_s: float = 60.0

    def run(self, *, qubo: QUBO, batch: SampleBatch) -> PostProcResult:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_safe_pce_worker, args=(self.inner, qubo, batch, child))
        proc.start()
        try:
            if parent.poll(timeout=float(self.timeout_s)):
                msg = parent.recv()
                if isinstance(msg, PostProcResult):
                    return msg
        except Exception:
            pass
        finally:
            try:
                proc.join(timeout=0.1)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.join(timeout=0.5)
                except Exception:
                    pass
        return self.fallback.run(qubo=qubo, batch=batch)

