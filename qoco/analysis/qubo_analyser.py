from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy as np
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qoco.core.qubo import QUBO

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class QuboCouplingEntry:
    i: int
    j: int
    coef: float
    label_i: str
    label_j: str

    @property
    def abs_coef(self) -> float:
        return float(abs(self.coef))


@dataclass(frozen=True)
class QuboVariableStrengthEntry:
    i: int
    strength: float
    label: str


@dataclass(frozen=True)
class QuboCorrelationSummary:
    mode: str
    n_pairs: int
    mean: float | None
    max: float | None
    argmax: tuple[int, int] | None


@dataclass(frozen=True)
class QuboSpectralSummary:
    min_eig: float
    max_eig: float

    @property
    def spread(self) -> float:
        return float(self.max_eig - self.min_eig)

    @property
    def max_abs(self) -> float:
        return float(max(abs(self.min_eig), abs(self.max_eig)))


@dataclass(frozen=True)
class QuboGraphSummary:
    n_nodes: int
    n_edges: int
    max_degree: int
    avg_degree: float
    clustering_avg: float
    transitivity: float | None
    assortativity: float | None
    n_communities: int | None
    modularity: float | None


@dataclass(frozen=True)
class QuboFrustrationSummary:
    negative_edge_fraction: float | None
    triangle_samples: int
    triangles_found: int
    frustrated_triangle_fraction: float | None


@dataclass(frozen=True)
class QuboTreewidthProxySummary:
    width_estimate: int
    fill_edges_added: int


@dataclass(frozen=True)
class QuboStats:
    n_vars: int
    offset: float
    lower_triangle_max_abs: float
    n_nonzero_diag: int
    n_nonzero_offdiag: int
    density_upper_including_diag: float
    diag_abs_sum: float
    offdiag_abs_sum: float
    diag_offdiag_abs_ratio: float | None
    max_abs_coef: float | None
    dynamic_range_abs: float | None
    offdiag_negative_fraction: float | None
    coupling_strength: Sequence[float]
    top_variables: Sequence[QuboVariableStrengthEntry]
    top_couplings: Sequence[QuboCouplingEntry]
    cosine_similarity: QuboCorrelationSummary | None
    pearson_correlation: QuboCorrelationSummary | None
    spectral: QuboSpectralSummary | None
    graph: QuboGraphSummary | None
    frustration: QuboFrustrationSummary | None
    treewidth_proxy: QuboTreewidthProxySummary | None


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _stats_basic(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "min": None, "p50": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p50": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _blend_hex(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> str:
    t = float(max(0.0, min(1.0, t)))
    r = int(round(c0[0] + (c1[0] - c0[0]) * t))
    g = int(round(c0[1] + (c1[1] - c0[1]) * t))
    b = int(round(c0[2] + (c1[2] - c0[2]) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _table_kv(title: str, rows: Sequence[tuple[str, str]]) -> Panel:
    table = Table(title=title, show_header=False, box=None, pad_edge=False, expand=True)
    table.add_column("k", style="bold")
    table.add_column("v")
    for key, value in rows:
        table.add_row(key, value)
    return Panel(table, border_style="dim")


def _format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.6g}"


class _WeightedGraph:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.neighbors: list[dict[int, float]] = [dict() for _ in range(self.n)]
        self.sign: list[dict[int, int]] = [dict() for _ in range(self.n)]
        self._n_edges = 0
        self._weight_sum = 0.0

    def add_edge(self, i: int, j: int, coef: float) -> None:
        if i == j:
            return
        if j in self.neighbors[i]:
            return
        w = float(abs(coef))
        self.neighbors[i][j] = w
        self.neighbors[j][i] = w
        s = 1 if coef > 0 else (-1 if coef < 0 else 0)
        self.sign[i][j] = s
        self.sign[j][i] = s
        self._n_edges += 1
        self._weight_sum += w

    @property
    def n_edges(self) -> int:
        return int(self._n_edges)

    @property
    def total_weight(self) -> float:
        return float(self._weight_sum)

    def degree(self) -> list[int]:
        return [len(self.neighbors[i]) for i in range(self.n)]

    def weighted_degree(self) -> list[float]:
        return [float(sum(self.neighbors[i].values())) for i in range(self.n)]

    def edges(self) -> Iterable[tuple[int, int]]:
        for i in range(self.n):
            for j in self.neighbors[i]:
                if i < j:
                    yield i, j

    def has_edge(self, i: int, j: int) -> bool:
        return j in self.neighbors[i]

    def edge_sign(self, i: int, j: int) -> int:
        return int(self.sign[i].get(j, 0))


def _label_propagation_communities(graph: _WeightedGraph, max_iters: int = 50) -> list[list[int]]:
    labels = list(range(graph.n))
    order = np.arange(graph.n)
    for _ in range(max_iters):
        changed = 0
        np.random.shuffle(order)
        for v in order.tolist():
            if not graph.neighbors[v]:
                continue
            counts: dict[int, int] = {}
            for u in graph.neighbors[v]:
                counts[labels[u]] = counts.get(labels[u], 0) + 1
            best_count = max(counts.values())
            best_labels = [lab for lab, cnt in counts.items() if cnt == best_count]
            best_label = min(best_labels)
            if best_label != labels[v]:
                labels[v] = best_label
                changed += 1
        if changed == 0:
            break
    communities: dict[int, list[int]] = {}
    for v, lab in enumerate(labels):
        communities.setdefault(lab, []).append(v)
    return list(communities.values())


def _modularity(graph: _WeightedGraph, communities: Sequence[Sequence[int]]) -> float | None:
    if graph.n_edges == 0:
        return None
    m2 = 2.0 * graph.total_weight
    if m2 <= 0:
        return None
    k = graph.weighted_degree()
    comm_of = [-1] * graph.n
    for ci, comm in enumerate(communities):
        for v in comm:
            comm_of[v] = ci

    sum_in = [0.0 for _ in range(len(communities))]
    sum_tot = [0.0 for _ in range(len(communities))]
    for v in range(graph.n):
        ci = comm_of[v]
        if ci >= 0:
            sum_tot[ci] += k[v]
    for i, j in graph.edges():
        ci = comm_of[i]
        cj = comm_of[j]
        if ci >= 0 and ci == cj:
            sum_in[ci] += graph.neighbors[i][j]

    q = 0.0
    for ci in range(len(communities)):
        q += sum_in[ci] / m2 - (sum_tot[ci] / m2) ** 2
    return float(q)


def _graph_clustering_and_transitivity(graph: _WeightedGraph) -> tuple[float, float | None]:
    deg = graph.degree()
    if graph.n == 0:
        return 0.0, None
    clustering_sum = 0.0
    triples = 0
    triangles_times3 = 0
    neighbor_sets = [set(graph.neighbors[i].keys()) for i in range(graph.n)]
    for v in range(graph.n):
        dv = deg[v]
        if dv < 2:
            continue
        triples += dv * (dv - 1) // 2
        tri2 = 0
        nv = neighbor_sets[v]
        for u in nv:
            tri2 += len(nv.intersection(neighbor_sets[u]))
        triangles_v = tri2 // 2
        triangles_times3 += triangles_v
        clustering_sum += (2.0 * triangles_v) / (dv * (dv - 1))
    clustering_avg = clustering_sum / float(graph.n)
    if triples == 0:
        return float(clustering_avg), None
    triangles = triangles_times3 / 3.0
    transitivity = 3.0 * triangles / float(triples)
    return float(clustering_avg), float(transitivity)


def _assortativity_degree(graph: _WeightedGraph) -> float | None:
    if graph.n_edges == 0:
        return None
    deg = graph.degree()
    xs = []
    ys = []
    for i, j in graph.edges():
        xs.append(deg[i])
        ys.append(deg[j])
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return None
    return float(np.sum(x * y) / denom)


def _treewidth_min_fill_proxy(graph: _WeightedGraph, max_n: int = 3000) -> QuboTreewidthProxySummary | None:
    if graph.n > max_n:
        return None
    neighbor_sets = [set(graph.neighbors[i].keys()) for i in range(graph.n)]
    alive = set(range(graph.n))
    width = 0
    fill_edges_added = 0
    while alive:
        best_v = None
        best_fill = None
        best_degree = None
        for v in alive:
            nv = neighbor_sets[v].intersection(alive)
            dv = len(nv)
            if dv <= 1:
                fill = 0
            else:
                missing = 0
                nv_list = list(nv)
                for a_idx in range(len(nv_list)):
                    a = nv_list[a_idx]
                    na = neighbor_sets[a].intersection(alive)
                    for b in nv_list[a_idx + 1 :]:
                        if b not in na:
                            missing += 1
                fill = missing
            candidate = (fill, dv)
            if best_fill is None or candidate < (best_fill, best_degree or 0):
                best_v = v
                best_fill = fill
                best_degree = dv
        assert best_v is not None and best_degree is not None and best_fill is not None
        width = max(width, best_degree)
        nv = list(neighbor_sets[best_v].intersection(alive))
        for a_idx in range(len(nv)):
            a = nv[a_idx]
            for b in nv[a_idx + 1 :]:
                if b not in neighbor_sets[a]:
                    neighbor_sets[a].add(b)
                    neighbor_sets[b].add(a)
                    fill_edges_added += 1
        for u in neighbor_sets[best_v]:
            neighbor_sets[u].discard(best_v)
        neighbor_sets[best_v].clear()
        alive.remove(best_v)
    return QuboTreewidthProxySummary(width_estimate=int(width), fill_edges_added=int(fill_edges_added))


def _triangle_frustration(
    graph: _WeightedGraph, triangle_samples: int = 2000, max_tries_factor: int = 50
) -> QuboFrustrationSummary:
    if graph.n_edges == 0:
        return QuboFrustrationSummary(
            negative_edge_fraction=None,
            triangle_samples=int(triangle_samples),
            triangles_found=0,
            frustrated_triangle_fraction=None,
        )

    neg_edges = 0
    for i, j in graph.edges():
        if graph.edge_sign(i, j) < 0:
            neg_edges += 1
    negative_edge_fraction = float(neg_edges / graph.n_edges)

    triangles_found = 0
    frustrated = 0
    tries = 0
    max_tries = int(triangle_samples * max_tries_factor)
    nodes = np.arange(graph.n)
    while triangles_found < triangle_samples and tries < max_tries:
        tries += 1
        v = int(np.random.choice(nodes))
        neigh = list(graph.neighbors[v].keys())
        if len(neigh) < 2:
            continue
        u, w = np.random.choice(neigh, size=2, replace=False).tolist()
        if not graph.has_edge(u, w):
            continue
        triangles_found += 1
        s = graph.edge_sign(v, u) * graph.edge_sign(v, w) * graph.edge_sign(u, w)
        if s < 0:
            frustrated += 1

    frustrated_triangle_fraction = None
    if triangles_found > 0:
        frustrated_triangle_fraction = float(frustrated / triangles_found)

    return QuboFrustrationSummary(
        negative_edge_fraction=negative_edge_fraction,
        triangle_samples=int(triangle_samples),
        triangles_found=int(triangles_found),
        frustrated_triangle_fraction=frustrated_triangle_fraction,
    )


def _correlation_summaries(
    W: np.ndarray, max_full_n: int, sample_pairs: int, rng: np.random.Generator
) -> tuple[QuboCorrelationSummary | None, QuboCorrelationSummary | None]:
    n = int(W.shape[0])
    if n < 2:
        return None, None

    norms = np.linalg.norm(W, axis=1)
    nonzero_rows = norms > 0
    if not np.any(nonzero_rows):
        return None, None

    def cosine_for_pair(i: int, j: int) -> float | None:
        denom = float(norms[i] * norms[j])
        if denom <= 0:
            return None
        return float(np.dot(W[i], W[j]) / denom)

    def pearson_for_pair(i: int, j: int) -> float | None:
        xi = W[i]
        xj = W[j]
        xi = xi - float(np.mean(xi))
        xj = xj - float(np.mean(xj))
        denom = float(np.linalg.norm(xi) * np.linalg.norm(xj))
        if denom <= 0:
            return None
        return float(np.dot(xi, xj) / denom)

    if n <= max_full_n:
        dot = W @ W.T
        denom = np.outer(norms, norms)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = dot / denom
        np.fill_diagonal(cosine, np.nan)
        cosine_vals = cosine[np.isfinite(cosine)]
        cosine_mean = float(np.mean(cosine_vals)) if cosine_vals.size else None
        cosine_max = float(np.nanmax(cosine)) if np.any(np.isfinite(cosine)) else None
        cosine_argmax = None
        if cosine_max is not None:
            idx = int(np.nanargmax(cosine))
            i, j = divmod(idx, n)
            cosine_argmax = (int(i), int(j))

        Wc = W - np.mean(W, axis=1, keepdims=True)
        norms_c = np.linalg.norm(Wc, axis=1)
        dotc = Wc @ Wc.T
        denomc = np.outer(norms_c, norms_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            pearson = dotc / denomc
        np.fill_diagonal(pearson, np.nan)
        pearson_vals = pearson[np.isfinite(pearson)]
        pearson_mean = float(np.mean(pearson_vals)) if pearson_vals.size else None
        pearson_max = float(np.nanmax(pearson)) if np.any(np.isfinite(pearson)) else None
        pearson_argmax = None
        if pearson_max is not None:
            idx = int(np.nanargmax(pearson))
            i, j = divmod(idx, n)
            pearson_argmax = (int(i), int(j))

        n_pairs = int(n * (n - 1) / 2)
        return (
            QuboCorrelationSummary(
                mode="full",
                n_pairs=n_pairs,
                mean=cosine_mean,
                max=cosine_max,
                argmax=cosine_argmax,
            ),
            QuboCorrelationSummary(
                mode="full",
                n_pairs=n_pairs,
                mean=pearson_mean,
                max=pearson_max,
                argmax=pearson_argmax,
            ),
        )

    pairs_i = rng.integers(0, n, size=sample_pairs, endpoint=False)
    pairs_j = rng.integers(0, n, size=sample_pairs, endpoint=False)
    mask = pairs_i != pairs_j
    pairs_i = pairs_i[mask]
    pairs_j = pairs_j[mask]
    if pairs_i.size == 0:
        return None, None

    cos_vals = []
    pear_vals = []
    cos_max = None
    pear_max = None
    cos_argmax = None
    pear_argmax = None
    for i, j in zip(pairs_i.tolist(), pairs_j.tolist(), strict=False):
        c = cosine_for_pair(int(i), int(j))
        p = pearson_for_pair(int(i), int(j))
        if c is not None:
            cos_vals.append(c)
            if cos_max is None or c > cos_max:
                cos_max = c
                cos_argmax = (int(i), int(j))
        if p is not None:
            pear_vals.append(p)
            if pear_max is None or p > pear_max:
                pear_max = p
                pear_argmax = (int(i), int(j))

    cos_mean = float(np.mean(np.asarray(cos_vals))) if cos_vals else None
    pear_mean = float(np.mean(np.asarray(pear_vals))) if pear_vals else None
    n_pairs = int(pairs_i.size)
    return (
        QuboCorrelationSummary(
            mode="sample",
            n_pairs=n_pairs,
            mean=cos_mean,
            max=cos_max,
            argmax=cos_argmax,
        ),
        QuboCorrelationSummary(
            mode="sample",
            n_pairs=n_pairs,
            mean=pear_mean,
            max=pear_max,
            argmax=pear_argmax,
        ),
    )


@dataclass
class QuboAnalyser:
    qubo: QUBO
    zero_tol: float = 0.0
    max_print_n: int = 30
    clip_percentile: tuple[int, int] = (1, 99)
    top_k: int = 10
    label_mode: str = "key"  # "key" or "index"
    max_corr_n: int = 200
    corr_sample_pairs: int = 20_000
    max_eig_n: int = 512
    triangle_samples: int = 2_000
    max_treewidth_n: int = 3_000
    console_width: int = 200
    force_color: bool = True
    console_color_system: str = "auto"  # "auto", "256", "truecolor"

    def __post_init__(self) -> None:
        Q = np.asarray(self.qubo.Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square; got shape={Q.shape}")
        lower = np.tril(Q, -1)
        lower_triangle_max_abs = float(np.max(np.abs(lower))) if lower.size else 0.0
        if lower_triangle_max_abs > float(self.zero_tol):
            raise ValueError(
                "QUBO convention violation: lower triangle must be zero "
                f"(max_abs={lower_triangle_max_abs}, zero_tol={self.zero_tol})."
            )

    @property
    def n_vars(self) -> int:
        return int(self.qubo.n_vars)

    def labels(self) -> list[str]:
        if self.label_mode == "index":
            return [str(i) for i in range(self.n_vars)]
        labels = [""] * self.n_vars
        for key, index in self.qubo.var_map.items():
            if not (0 <= index < self.n_vars):
                raise ValueError(f"var_map index out of bounds: {key} -> {index}")
            if labels[index]:
                raise ValueError(f"var_map has duplicate index: {index}")
            labels[index] = key
        if any(not label for label in labels):
            raise ValueError("var_map does not cover all variable indices 0..n-1")
        return labels

    def _pair_coef(self, i: int, j: int) -> float:
        if i == j:
            return float(self.qubo.Q[i, i])
        a, b = (i, j) if i < j else (j, i)
        return float(self.qubo.Q[a, b])

    def _style_for_value(self, value: float, vmax: float, *, color_system: str) -> tuple[str, str]:
        if abs(value) <= float(self.zero_tol) or vmax <= 0:
            return "grey11", "grey70"
        t = min(1.0, abs(value) / vmax)
        if str(color_system).lower() in {"truecolor", "24bit"}:
            white = (250, 250, 250)
            red = (220, 60, 60)
            blue = (60, 120, 220)
            bg = _blend_hex(white, red if value > 0 else blue, t)
            rgb = _hex_to_rgb(bg)
            fg = "#000000" if _luminance(rgb) > 140 else "#ffffff"
            return bg, fg

        red_scale = [52, 88, 124, 160, 196]  # dark -> bright
        blue_scale = [17, 18, 19, 20, 21, 27, 33, 39, 45]
        if value > 0:
            idx = min(len(red_scale) - 1, int(round(t * (len(red_scale) - 1))))
            bg = f"color({red_scale[idx]})"
            fg = "white" if red_scale[idx] <= 124 else "black"
            return bg, fg
        idx = min(len(blue_scale) - 1, int(round(t * (len(blue_scale) - 1))))
        bg = f"color({blue_scale[idx]})"
        fg = "white"
        return bg, fg

    def _default_console(self) -> Console:
        return Console(
            force_terminal=bool(self.force_color),
            color_system=str(self.console_color_system),
            no_color=False,
            width=int(self.console_width),
        )

    def color_test(self, console: Console | None = None) -> None:
        console = console or self._default_console()
        table = Table(title="Color test (background swatches)", pad_edge=False, expand=True)
        table.add_column("label", style="bold")
        table.add_column("swatch", justify="center")
        table.add_row("positive (256)", Text("  +  ", style="black on color(196)"))
        table.add_row("negative (256)", Text("  -  ", style="white on color(27)"))
        table.add_row("zero (256)", Text("  0  ", style="grey70 on grey11"))
        table.add_row("positive (truecolor)", Text("  +  ", style="black on #dc3c3c"))
        table.add_row("negative (truecolor)", Text("  -  ", style="white on #3c78dc"))
        console.print(Panel(table, border_style="dim"))

    def _heatmap_table(self) -> Panel:
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

        labels = self.labels()
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

    def plot_heatmap(
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
    ) -> tuple["Figure", "Axes"]:
        """Create a Matplotlib heatmap for the QUBO matrix.

        matrix:
        - "pair": mirrored pair coefficients C (easy to interpret).
        - "objective": objective-equivalent symmetric Q_sym=(Q+Q.T)/2.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import TwoSlopeNorm
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Matplotlib is required for plot_heatmap().") from exc

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
            labels = self.labels()
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

    def stats(self) -> QuboStats:
        Q = np.asarray(self.qubo.Q, dtype=float)
        n = int(self.n_vars)
        tol = float(self.zero_tol)

        diag = np.diag(Q)
        diag_nz = diag[np.abs(diag) > tol]
        iu = np.triu_indices(n, 1)
        offdiag = Q[iu]
        offdiag_nz = offdiag[np.abs(offdiag) > tol]

        lower_triangle_max_abs = float(np.max(np.abs(np.tril(Q, -1)))) if n else 0.0

        total_terms = n * (n + 1) // 2
        n_nonzero_diag = int(diag_nz.size)
        n_nonzero_offdiag = int(offdiag_nz.size)
        density = float((n_nonzero_diag + n_nonzero_offdiag) / total_terms) if total_terms else 0.0

        diag_abs_sum = float(np.sum(np.abs(diag_nz))) if diag_nz.size else 0.0
        offdiag_abs_sum = float(np.sum(np.abs(offdiag_nz))) if offdiag_nz.size else 0.0
        diag_offdiag_abs_ratio = None
        if offdiag_abs_sum > 0:
            diag_offdiag_abs_ratio = float(diag_abs_sum / offdiag_abs_sum)

        max_abs_coef = None
        all_upper = np.triu(Q, 0)
        all_upper_nz = all_upper[np.abs(all_upper) > tol]
        if all_upper_nz.size:
            max_abs_coef = float(np.max(np.abs(all_upper_nz)))
        dynamic_range_abs = None
        if all_upper_nz.size:
            median_abs = float(np.median(np.abs(all_upper_nz)))
            if median_abs > 0 and max_abs_coef is not None:
                dynamic_range_abs = float(max_abs_coef / median_abs)

        offdiag_negative_fraction = None
        if offdiag_nz.size:
            offdiag_negative_fraction = float(np.mean(offdiag_nz < 0))

        abs_upper_offdiag = np.abs(np.triu(Q, 1))
        coupling_strength = abs_upper_offdiag.sum(axis=0) + abs_upper_offdiag.sum(axis=1)
        coupling_strength = coupling_strength.astype(float, copy=False)

        labels = self.labels()
        top_k = int(self.top_k)
        top_var_idx = np.argsort(-coupling_strength)[:top_k]
        top_variables = [
            QuboVariableStrengthEntry(i=int(i), strength=float(coupling_strength[int(i)]), label=labels[int(i)])
            for i in top_var_idx.tolist()
        ]

        top_couplings: list[QuboCouplingEntry] = []
        if offdiag_nz.size and top_k > 0:
            abs_vals = np.abs(offdiag)
            mask = abs_vals > tol
            idxs = np.flatnonzero(mask)
            if idxs.size:
                k = min(top_k, int(idxs.size))
                part = np.argpartition(abs_vals[mask], -k)[-k:]
                chosen = idxs[part]
                chosen = chosen[np.argsort(-abs_vals[chosen])]
                for flat_index in chosen.tolist():
                    i = int(iu[0][flat_index])
                    j = int(iu[1][flat_index])
                    coef = float(Q[i, j])
                    top_couplings.append(
                        QuboCouplingEntry(i=i, j=j, coef=coef, label_i=labels[i], label_j=labels[j])
                    )

        W = np.triu(Q, 1)
        W = W + W.T
        rng = np.random.default_rng(0)
        cosine_summary, pearson_summary = _correlation_summaries(
            W=W, max_full_n=int(self.max_corr_n), sample_pairs=int(self.corr_sample_pairs), rng=rng
        )

        spectral = None
        if n <= int(self.max_eig_n) and n > 0:
            Qsym = (Q + Q.T) / 2.0
            eigs = np.linalg.eigvalsh(Qsym)
            spectral = QuboSpectralSummary(min_eig=float(eigs[0]), max_eig=float(eigs[-1]))

        graph = _WeightedGraph(n=n)
        for coef, i, j in zip(offdiag, iu[0], iu[1], strict=False):
            if abs(float(coef)) > tol:
                graph.add_edge(int(i), int(j), float(coef))

        graph_summary = None
        frustration = None
        treewidth_proxy = None
        if graph.n_edges > 0:
            deg = graph.degree()
            clustering_avg, transitivity = _graph_clustering_and_transitivity(graph)
            assortativity = _assortativity_degree(graph)
            communities = _label_propagation_communities(graph)
            modularity = _modularity(graph, communities)
            graph_summary = QuboGraphSummary(
                n_nodes=int(n),
                n_edges=int(graph.n_edges),
                max_degree=int(max(deg) if deg else 0),
                avg_degree=float(np.mean(np.asarray(deg, dtype=float))) if deg else 0.0,
                clustering_avg=float(clustering_avg),
                transitivity=transitivity,
                assortativity=assortativity,
                n_communities=int(len(communities)),
                modularity=modularity,
            )
            frustration = _triangle_frustration(graph, triangle_samples=int(self.triangle_samples))
            treewidth_proxy = _treewidth_min_fill_proxy(graph, max_n=int(self.max_treewidth_n))

        return QuboStats(
            n_vars=int(n),
            offset=float(self.qubo.offset),
            lower_triangle_max_abs=lower_triangle_max_abs,
            n_nonzero_diag=n_nonzero_diag,
            n_nonzero_offdiag=n_nonzero_offdiag,
            density_upper_including_diag=float(density),
            diag_abs_sum=diag_abs_sum,
            offdiag_abs_sum=offdiag_abs_sum,
            diag_offdiag_abs_ratio=diag_offdiag_abs_ratio,
            max_abs_coef=max_abs_coef,
            dynamic_range_abs=dynamic_range_abs,
            offdiag_negative_fraction=offdiag_negative_fraction,
            coupling_strength=coupling_strength.tolist(),
            top_variables=top_variables,
            top_couplings=top_couplings,
            cosine_similarity=cosine_summary,
            pearson_correlation=pearson_summary,
            spectral=spectral,
            graph=graph_summary,
            frustration=frustration,
            treewidth_proxy=treewidth_proxy,
        )

    def summary(self, console: Console | None = None) -> None:
        console = console or self._default_console()
        stats = self.stats()

        rows = [
            ("n_vars", str(stats.n_vars)),
            ("offset", _format_float(stats.offset)),
            ("lower_triangle_max_abs", _format_float(stats.lower_triangle_max_abs)),
            ("nnz(diag)", str(stats.n_nonzero_diag)),
            ("nnz(offdiag)", str(stats.n_nonzero_offdiag)),
            ("density(upper incl diag)", _format_float(stats.density_upper_including_diag)),
            ("sum|diag|", _format_float(stats.diag_abs_sum)),
            ("sum|offdiag|", _format_float(stats.offdiag_abs_sum)),
            ("|diag|/|offdiag|", _format_float(stats.diag_offdiag_abs_ratio)),
            ("max|coef|", _format_float(stats.max_abs_coef)),
            ("dynamic_range_abs", _format_float(stats.dynamic_range_abs)),
            ("offdiag_negative_fraction", _format_float(stats.offdiag_negative_fraction)),
        ]
        console.print(_table_kv("QUBO summary", rows))

        if stats.spectral is not None:
            console.print(
                _table_kv(
                    "Spectral (Q_sym=(Q+Q.T)/2)",
                    [
                        ("min_eig", _format_float(stats.spectral.min_eig)),
                        ("max_eig", _format_float(stats.spectral.max_eig)),
                        ("spread", _format_float(stats.spectral.spread)),
                        ("max_abs", _format_float(stats.spectral.max_abs)),
                    ],
                )
            )

        if stats.graph is not None:
            console.print(
                _table_kv(
                    "Graph (edges: nonzero offdiag, weight=|coef|)",
                    [
                        ("n_nodes", str(stats.graph.n_nodes)),
                        ("n_edges", str(stats.graph.n_edges)),
                        ("max_degree", str(stats.graph.max_degree)),
                        ("avg_degree", _format_float(stats.graph.avg_degree)),
                        ("clustering_avg", _format_float(stats.graph.clustering_avg)),
                        ("transitivity", _format_float(stats.graph.transitivity)),
                        ("assortativity", _format_float(stats.graph.assortativity)),
                        ("n_communities", str(stats.graph.n_communities) if stats.graph.n_communities else "-"),
                        ("modularity", _format_float(stats.graph.modularity)),
                    ],
                )
            )

        if stats.treewidth_proxy is not None:
            console.print(
                _table_kv(
                    "Treewidth proxy (min-fill heuristic)",
                    [
                        ("width_estimate", str(stats.treewidth_proxy.width_estimate)),
                        ("fill_edges_added", str(stats.treewidth_proxy.fill_edges_added)),
                    ],
                )
            )

        if stats.frustration is not None:
            console.print(
                _table_kv(
                    "Frustration proxies",
                    [
                        ("negative_edge_fraction", _format_float(stats.frustration.negative_edge_fraction)),
                        ("triangle_samples", str(stats.frustration.triangle_samples)),
                        ("triangles_found", str(stats.frustration.triangles_found)),
                        ("frustrated_triangle_fraction", _format_float(stats.frustration.frustrated_triangle_fraction)),
                    ],
                )
            )

        console.print(self._top_variables_panel(stats.top_variables))
        console.print(self._top_couplings_panel(stats.top_couplings))
        console.print(self._correlation_panel(stats))

    def report(self, console: Console | None = None) -> None:
        console = console or self._default_console()
        console.print(self._heatmap_table())
        self.summary(console=console)

    def _top_variables_panel(self, entries: Sequence[QuboVariableStrengthEntry]) -> Panel:
        table = Table(
            title=f"Top variables by coupling strength (sum_j!=i |coef_ij|), k={len(entries)}",
            pad_edge=False,
            expand=True,
        )
        table.add_column("i", justify="right")
        table.add_column("label")
        table.add_column("strength", justify="right")
        for entry in entries:
            table.add_row(str(entry.i), entry.label, _format_float(entry.strength))
        return Panel(table, border_style="dim")

    def _top_couplings_panel(self, entries: Sequence[QuboCouplingEntry]) -> Panel:
        table = Table(title=f"Top couplings by |coef| (upper triangle), k={len(entries)}", pad_edge=False, expand=True)
        table.add_column("i", justify="right")
        table.add_column("j", justify="right")
        table.add_column("label_i")
        table.add_column("label_j")
        table.add_column("coef", justify="right")
        table.add_column("|coef|", justify="right")
        for entry in entries:
            table.add_row(
                str(entry.i),
                str(entry.j),
                entry.label_i,
                entry.label_j,
                _format_float(entry.coef),
                _format_float(entry.abs_coef),
            )
        return Panel(table, border_style="dim")

    def _correlation_panel(self, stats: QuboStats) -> Panel:
        table = Table(title="Coupling-vector correlation (offdiag-only, mirrored)", pad_edge=False, expand=True)
        table.add_column("metric", style="bold")
        table.add_column("mode")
        table.add_column("n_pairs", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("max", justify="right")
        table.add_column("argmax", justify="right")

        def add_row(name: str, summary: QuboCorrelationSummary | None) -> None:
            if summary is None:
                table.add_row(name, "-", "-", "-", "-", "-")
                return
            arg = "-"
            if summary.argmax is not None:
                i, j = summary.argmax
                arg = f"({i},{j})"
            table.add_row(
                name,
                summary.mode,
                str(summary.n_pairs),
                _format_float(summary.mean),
                _format_float(summary.max),
                arg,
            )

        add_row("cosine", stats.cosine_similarity)
        add_row("pearson", stats.pearson_correlation)
        return Panel(table, border_style="dim")

