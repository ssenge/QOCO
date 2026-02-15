from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


def _format_float(x: float | None) -> str:
    if x is None:
        return "-"
    v = float(x)
    if v == 0.0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.3g}"
    return f"{v:.6g}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = str(hex_color).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)


def _blend_hex(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    tt = float(np.clip(float(t), 0.0, 1.0))
    r = int(round((1.0 - tt) * a[0] + tt * b[0]))
    g = int(round((1.0 - tt) * a[1] + tt * b[1]))
    b2 = int(round((1.0 - tt) * a[2] + tt * b[2]))
    return f"#{r:02x}{g:02x}{b2:02x}"


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
    """Standalone QUBO heatmap renderer (table + Matplotlib plot).

    This module intentionally does not depend on `QuboAnalyser` so it can be committed/used independently.
    """

    qubo: QUBO
    zero_tol: float = 0.0
    clip_percentile: tuple[int, int] = (1, 99)
    max_print_n: int = 30
    label_mode: str = "key"  # "key" | "index"
    console_width: int = 200
    force_color: bool = True
    console_color_system: str = "auto"  # "auto" | "256" | "truecolor"

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

    def _default_console(self) -> Console:
        return Console(
            force_terminal=bool(self.force_color),
            color_system=str(self.console_color_system),
            no_color=False,
            width=int(self.console_width),
        )

    def _pair_coef(self, i: int, j: int) -> float:
        if int(i) == int(j):
            return float(np.asarray(self.qubo.Q, dtype=float)[int(i), int(i)])
        a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        return float(np.asarray(self.qubo.Q, dtype=float)[a, b])

    def _style_for_value(self, value: float, vmax: float, *, color_system: str) -> tuple[str, str]:
        if abs(float(value)) <= float(self.zero_tol) or float(vmax) <= 0:
            return "grey11", "grey70"
        t = min(1.0, abs(float(value)) / float(vmax))
        if str(color_system).lower() in {"truecolor", "24bit"}:
            white = (250, 250, 250)
            red = (220, 60, 60)
            blue = (60, 120, 220)
            bg = _blend_hex(white, red if float(value) > 0 else blue, t)
            rgb = _hex_to_rgb(bg)
            fg = "#000000" if _luminance(rgb) > 140 else "#ffffff"
            return bg, fg

        red_scale = [52, 88, 124, 160, 196]  # dark -> bright
        blue_scale = [17, 18, 19, 20, 21, 27, 33, 39, 45]
        if float(value) > 0:
            idx = min(len(red_scale) - 1, int(round(t * (len(red_scale) - 1))))
            bg = f"color({red_scale[idx]})"
            fg = "white" if red_scale[idx] <= 124 else "black"
            return bg, fg
        idx = min(len(blue_scale) - 1, int(round(t * (len(blue_scale) - 1))))
        bg = f"color({blue_scale[idx]})"
        fg = "white"
        return bg, fg

    def table(self) -> Panel:
        Q = np.asarray(self.qubo.Q, dtype=float)
        n = int(self.n_vars)
        upper = np.triu(Q, 0)
        values = upper[np.abs(upper) > float(self.zero_tol)]
        abs_values = np.abs(values)

        if abs_values.size:
            hi = _safe_percentile(abs_values, float(self.clip_percentile[1]))
            vmax = float(hi) if hi is not None and hi > 0 else float(np.max(abs_values))
        else:
            vmax = 0.0

        labels = _labels_from_var_map(var_map=dict(self.qubo.var_map), n=n, label_mode=str(self.label_mode))
        indices = list(range(n))
        truncated = False
        if n > int(self.max_print_n):
            truncated = True
            k = max(2, int(self.max_print_n) // 2)
            indices = list(range(k)) + list(range(n - k, n))

        table = Table(
            title=f"QUBO heatmap (pair coefficients; mirrored) — n={n}, vmax≈{_format_float(vmax)}",
            show_header=True,
            header_style="bold",
            pad_edge=False,
            expand=True,
        )
        table.add_column("", justify="right", no_wrap=True)
        for j in indices:
            table.add_column(labels[j], justify="right", no_wrap=True)
        if truncated:
            table.add_column("…", justify="center", no_wrap=True)

        for i in indices:
            row = [Text(labels[i], style="bold")]
            for j in indices:
                v = self._pair_coef(i, j)
                bg, fg = self._style_for_value(v, vmax, color_system=str(self.console_color_system))
                row.append(Text(f"{v:+.3g}", style=f"{fg} on {bg}"))
            if truncated:
                row.append(Text("…", style="dim"))
            table.add_row(*row)
        if truncated:
            table.add_row(*([Text("…", style="dim")] * (len(indices) + 2)))

        return Panel(table, border_style="dim")

    def print_table(self, *, console: Console | None = None) -> None:
        (console or self._default_console()).print(self.table())

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
        """Create a Matplotlib heatmap for the QUBO matrix.

        matrix:
        - "pair": mirrored pair coefficients C (easy to interpret).
        - "objective": objective-equivalent symmetric Q_sym=(Q+Q.T)/2.
        """
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

