from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qoco.core.qubo import QUBO

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return None
    try:
        return float(np.percentile(v, float(q)))
    except Exception:
        return None


def _labels_from_var_map(*, var_map: dict[str, int], n: int, label_mode: str) -> list[str]:
    if str(label_mode) == "index":
        return [str(i) for i in range(int(n))]
    labels = [""] * int(n)
    for key, index in var_map.items():
        idx = int(index)
        if not (0 <= idx < int(n)):
            raise ValueError(f"var_map index out of bounds: {key} -> {idx}")
        if labels[idx]:
            raise ValueError(f"var_map has duplicate index: {idx}")
        labels[idx] = str(key)
    if any(not label for label in labels):
        raise ValueError("var_map does not cover all variable indices 0..n-1")
    return labels


@dataclass(frozen=True)
class QuboHeatmap:
    """Standalone QUBO heatmap plotter (Matplotlib only)."""

    qubo: QUBO
    zero_tol: float = 0.0
    label_mode: str = "key"  # "key" | "index"

    def __post_init__(self) -> None:
        Q = np.asarray(self.qubo.Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square; got shape={Q.shape}")
        lower_triangle_max_abs = float(np.max(np.abs(np.tril(Q, -1)))) if Q.size else 0.0
        if lower_triangle_max_abs > float(self.zero_tol):
            raise ValueError(
                "QUBO convention violation: lower triangle must be zero "
                f"(max_abs={lower_triangle_max_abs}, zero_tol={self.zero_tol})."
            )

    @property
    def n_vars(self) -> int:
        return int(self.qubo.n_vars)

    def plot(
        self,
        *,
        matrix: Literal["pair", "objective"] = "pair",
        cmap: str = "RdBu_r",
        center: float = 0.0,
        clip_percentile: tuple[int, int] | None = (1, 99),
        symmetric: bool = True,
        show_labels: bool = False,
        max_ticks: int = 40,
        figsize: tuple[float, float] | None = None,
        dpi: int = 150,
    ) -> tuple[Figure, Axes]:
        """Create a Matplotlib heatmap for the QUBO matrix."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import TwoSlopeNorm
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Matplotlib is required for QuboHeatmap.plot().") from exc

        Q = np.asarray(self.qubo.Q, dtype=float)
        n = int(self.n_vars)
        if matrix == "pair":
            A = Q + Q.T - np.diag(np.diag(Q))
            title = "QUBO heatmap (pair coefficients; mirrored)"
        else:
            A = (Q + Q.T) / 2.0
            title = "QUBO heatmap (objective-equivalent Q_sym)"

        tol = float(self.zero_tol)
        values = A[np.abs(A) > tol]
        vmax = float(np.max(np.abs(values))) if values.size else 0.0
        if clip_percentile is not None and values.size:
            hi = _safe_percentile(np.abs(values), float(clip_percentile[1]))
            if hi is not None and hi > 0:
                vmax = min(vmax, float(hi))
        if symmetric:
            vmin = -vmax
        else:
            vmin = float(np.min(values)) if values.size else 0.0

        if figsize is None:
            side = 6.5 if n <= 80 else 9.5
            figsize = (side, side)

        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi), constrained_layout=True)
        norm = TwoSlopeNorm(vcenter=float(center), vmin=float(vmin), vmax=float(vmax)) if vmax > 0 else None
        im = ax.imshow(A, cmap=str(cmap), norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.set_ylabel("coefficient", rotation=90)
        ax.set_title(f"{title} — n={n}")

        if show_labels:
            labels = _labels_from_var_map(var_map=dict(self.qubo.var_map), n=n, label_mode=str(self.label_mode))
        else:
            labels = [str(i) for i in range(n)]

        tick_step = max(1, int(np.ceil(n / max(1, int(max_ticks)))))
        ticks = list(range(0, n, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)

        ax.set_xlabel("j")
        ax.set_ylabel("i")
        return fig, ax


def plot_qubo_heatmap(qubo: QUBO, **kwargs) -> tuple[Figure, Axes]:
    """Functional wrapper around `QuboHeatmap(qubo).plot(...)`."""
    return QuboHeatmap(qubo=qubo).plot(**kwargs)

