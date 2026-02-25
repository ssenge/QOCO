from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitDiskEmbeddingResult:
    positions_xy: np.ndarray  # shape: (n_vertices, 2)
    blockade_radius: float

    n_vertices: int
    n_edges_target: int
    n_edges_realized: int

    max_edge_distance: float | None
    min_nonedge_distance: float | None
    gap: float | None  # min_nonedge - max_edge

    n_missing_edges: int
    n_extra_edges: int

    is_unit_disk_exact: bool
    best_loss: float


def _pairs_upper(n: int) -> tuple[np.ndarray, np.ndarray]:
    # i<j indices
    return np.triu_indices(n, k=1)


def _pairwise_distances(positions_xy: np.ndarray) -> np.ndarray:
    # Returns a dense upper-triangular distance matrix in a full (n,n) array.
    # For n<=256 this is fine and simplifies diagnostics.
    diff = positions_xy[:, None, :] - positions_xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _softplus(x: np.ndarray) -> np.ndarray:
    # Stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _loss_unit_disk(
    flat_xy: np.ndarray,
    *,
    n: int,
    is_edge_upper: np.ndarray,
    margin: float,
    target_radius: float,
    repel_strength: float,
) -> float:
    positions_xy = flat_xy.reshape((n, 2))
    dist = _pairwise_distances(positions_xy)
    iu, ju = _pairs_upper(n)
    du = dist[iu, ju]

    edge_mask = is_edge_upper
    nonedge_mask = ~is_edge_upper

    # Edges: want dist <= R - margin
    edge_violation = _softplus(du[edge_mask] - (target_radius - margin))
    # Non-edges: want dist >= R + margin
    nonedge_violation = _softplus((target_radius + margin) - du[nonedge_mask])

    # Keep solutions from collapsing into the origin.
    # Also encourages spreading out non-edges if feasible.
    repel = 1.0 / (du + 1e-6)

    loss = float(np.sum(edge_violation * edge_violation))
    loss += float(np.sum(nonedge_violation * nonedge_violation))
    loss += float(repel_strength * np.sum(repel))
    loss += float(1e-4 * np.sum(positions_xy * positions_xy))
    return float(loss)


def _diagnose(
    *,
    positions_xy: np.ndarray,
    edges: frozenset[tuple[int, int]],
) -> tuple[float, float, float, int, int, int, float | None, float | None, float | None]:
    n = int(positions_xy.shape[0])
    dist = _pairwise_distances(positions_xy)

    # Compute max edge distance / min non-edge distance.
    if edges:
        edge_dists = [float(dist[u, v]) for (u, v) in edges]
        max_edge = float(np.max(edge_dists))
    else:
        max_edge = 0.0

    nonedge_dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edges:
                continue
            nonedge_dists.append(float(dist[i, j]))
    min_nonedge = float(np.min(nonedge_dists)) if nonedge_dists else float("inf")

    gap = float(min_nonedge - max_edge) if np.isfinite(min_nonedge) else None

    # Choose a separating threshold if possible, otherwise fall back to max_edge.
    if np.isfinite(min_nonedge) and min_nonedge > max_edge:
        threshold = float(0.5 * (max_edge + min_nonedge))
    else:
        threshold = float(max_edge)

    realized_edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if float(dist[i, j]) <= threshold:
                realized_edges.add((i, j))

    missing = int(len(edges - realized_edges))
    extra = int(len(realized_edges - edges))

    return (
        float(threshold),
        float(max_edge),
        float(min_nonedge) if np.isfinite(min_nonedge) else float("inf"),
        int(len(realized_edges)),
        missing,
        extra,
        float(gap) if gap is not None else None,
        float(max_edge) if edges else None,
        float(min_nonedge) if nonedge_dists else None,
    )


def embed_wmis_unit_disk(
    *,
    vertex_count: int,
    edges: frozenset[tuple[int, int]],
    seed: int = 0,
    num_restarts: int = 32,
    maxiter: int = 2000,
    margin: float = 1e-3,
    target_radius: float = 1.0,
    repel_strength: float = 1e-3,
) -> UnitDiskEmbeddingResult:
    """Try to embed an arbitrary graph as a unit-disk graph in 2D.

    This is a best-effort heuristic intended for small graphs. If the graph is not a UDG,
    the optimizer will usually converge to a geometry with either missing edges or extra
    edges. In that case we still return a useful diagnostic geometry.
    """
    n = int(vertex_count)
    if n <= 0:
        raise ValueError("vertex_count must be >= 1")

    iu, ju = _pairs_upper(n)
    edges_set = set(edges)
    is_edge_upper = np.asarray([(int(i), int(j)) in edges_set for i, j in zip(iu, ju)], dtype=bool)

    best_xy: np.ndarray | None = None
    best_loss = float("inf")

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for unit-disk embedding (install scipy).") from exc

    rng = np.random.default_rng(int(seed))
    for _ in range(int(num_restarts)):
        init_xy = rng.normal(size=(n, 2)).astype(float) * 0.5
        init_xy -= np.mean(init_xy, axis=0, keepdims=True)
        x0 = init_xy.reshape(-1)

        res = minimize(
            lambda x: _loss_unit_disk(
                x,
                n=n,
                is_edge_upper=is_edge_upper,
                margin=float(margin),
                target_radius=float(target_radius),
                repel_strength=float(repel_strength),
            ),
            x0,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter)},
        )

        loss = float(res.fun)
        if loss < best_loss:
            best_loss = loss
            best_xy = np.asarray(res.x, dtype=float).reshape((n, 2))

    if best_xy is None:  # pragma: no cover
        raise RuntimeError("Failed to produce an embedding candidate.")

    threshold, max_edge, min_nonedge, realized_edges, missing, extra, gap, max_edge_opt, min_nonedge_opt = _diagnose(
        positions_xy=best_xy,
        edges=edges,
    )

    # Exact means: no missing/extra edges. If both edges and non-edges exist, we also
    # expect a strict separation gap. For complete/empty graphs, any threshold works.
    has_edges = bool(len(edges) > 0)
    has_nonedges = bool(int(n * (n - 1) // 2) > int(len(edges)))
    is_exact = bool((missing == 0) and (extra == 0) and ((not has_edges) or (not has_nonedges) or ((gap is not None) and (gap > 0.0))))

    return UnitDiskEmbeddingResult(
        positions_xy=best_xy,
        blockade_radius=float(threshold),
        n_vertices=int(n),
        n_edges_target=int(len(edges)),
        n_edges_realized=int(realized_edges),
        max_edge_distance=float(max_edge_opt) if max_edge_opt is not None else None,
        min_nonedge_distance=float(min_nonedge_opt) if min_nonedge_opt is not None else None,
        gap=float(gap) if gap is not None else None,
        n_missing_edges=int(missing),
        n_extra_edges=int(extra),
        is_unit_disk_exact=bool(is_exact),
        best_loss=float(best_loss),
    )

