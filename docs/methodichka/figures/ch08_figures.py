"""Рисунки главы 8 (персистентные гомологии). Запуск:
.venv/bin/python ch08_figures.py
Все диаграммы и счётчики считаются функциями пакета morse_lidar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from morse_lidar.compare import bottleneck
from morse_lidar.critical import analyze_morse
from morse_lidar.mesh import grid_mesh, periodic_grid_mesh
from morse_lidar.persistence import (
    annotate_persistence,
    critical_pairs,
    persistence_diagram,
    persistent_counts,
)

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
MIN_C, SADDLE_C, MAX_C = NAVY, "#2A9D8F", "#E4572E"
DIM_COLORS = {0: MIN_C, 1: SADDLE_C, 2: MAX_C}
OUT = Path(__file__).resolve().parent


def torus_field(rows: int = 40, cols: int = 48) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    return np.sin(2 * np.pi * x / cols) + 0.4 * np.cos(2 * np.pi * y / rows)


def fig_pairs_1d() -> None:
    f = np.array([3, 1, 4, 2, 6, 0, 5], dtype=float)
    # Полоса из трёх одинаковых строк: SoS (значение, затем индекс) выбирает
    # вершины первой строки, и пары совпадают с парами функции на отрезке.
    pairs = critical_pairs(grid_mesh(np.tile(f, (3, 1))))
    cols = len(f)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.plot(range(cols), f, color=INK, lw=2, zorder=1)
    ax.scatter(range(cols), f, color=SOFT, s=25, zorder=2)
    for i, v in enumerate(f):
        ax.annotate(f"$x_{i}$", (i, v), textcoords="offset points", xytext=(6, -12) if i in (1, 3, 5) else (-16, 6), fontsize=9, color=INK)
    for pair in pairs:
        b = pair.birth_vertex % cols
        ax.scatter([b], [pair.birth], color=MIN_C, s=70, zorder=3)
        top = pair.death if np.isfinite(pair.death) else 7.0
        ax.annotate("", xy=(b, top), xytext=(b, pair.birth),
                    arrowprops=dict(arrowstyle="->", color=ACCENT if np.isfinite(pair.death) else NAVY, lw=1.6))
        if np.isfinite(pair.death):
            d = pair.death_vertex % cols
            ax.scatter([d], [pair.death], color=SADDLE_C, s=70, marker="s", zorder=3)
            ax.plot([b, d], [pair.death, pair.death], ls=":", color=ACCENT, lw=1)
            ax.text(b + 0.08, (pair.birth + pair.death) / 2, f"({pair.birth:g}, {pair.death:g})", color=ACCENT, fontsize=9)
        else:
            ax.text(b + 0.08, 6.6, r"$(0, \infty)$", color=NAVY, fontsize=9)
    ax.set_ylim(-0.6, 7.3)
    ax.set_xlabel("вершина")
    ax.set_ylabel("$f$")
    ax.set_title("Пары (рождение, смерть) функции на отрезке")
    finite = [(p.birth, p.death) for p in pairs if np.isfinite(p.death)]
    bx.plot([-0.5, 7], [-0.5, 7], color="0.6", lw=1)
    bx.scatter(*zip(*finite), color=MIN_C, s=50, label="конечные классы, dim 0")
    ess = [p.birth for p in pairs if not np.isfinite(p.death)]
    bx.scatter(ess, [6.8] * len(ess), color=MIN_C, marker="^", s=70, label=r"смерть $=\infty$")
    bx.axhline(6.8, color="0.8", ls="--", lw=0.8)
    bx.set_xlim(-0.5, 7)
    bx.set_ylim(-0.5, 7.2)
    bx.set_xlabel("рождение")
    bx.set_ylabel("смерть")
    bx.set_title("Диаграмма персистентности")
    bx.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ch08_pairs1d.png", dpi=200)
    plt.close(fig)


def _plot_diagram(ax, diagram: dict[int, list[tuple[float, float]]], title: str, lo: float, hi: float) -> None:
    inf_level = hi + 0.25
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1)
    ax.axhline(inf_level, color="0.8", ls="--", lw=0.8)
    for dim, points in diagram.items():
        finite = [(b, d) for b, d in points if np.isfinite(d)]
        ess = [b for b, d in points if not np.isfinite(d)]
        if finite:
            ax.scatter(*zip(*finite), s=12, color=DIM_COLORS[dim], alpha=0.75, label=f"dim {dim}: {len(finite)} конечных")
        if ess:
            ax.scatter(ess, [inf_level] * len(ess), marker="^", s=60, color=DIM_COLORS[dim], edgecolor="k", lw=0.5,
                       label=f"dim {dim}: {len(ess)} существенных")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, inf_level + 0.2)
    ax.set_xlabel("рождение")
    ax.set_ylabel(r"смерть (верхняя линия: $\infty$)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")


def fig_diagrams() -> None:
    rows, cols = 40, 48
    y, x = np.mgrid[0:rows, 0:cols]
    smooth = np.sin(4 * np.pi * x / cols) + 0.4 * np.cos(2 * np.pi * y / rows) + 0.2 * np.sin(2 * np.pi * x / cols)
    noisy = smooth + 0.05 * np.random.default_rng(3).normal(size=smooth.shape)
    d_smooth = persistence_diagram(critical_pairs(periodic_grid_mesh(smooth)))
    d_noisy = persistence_diagram(critical_pairs(periodic_grid_mesh(noisy)))
    lo, hi = -2.0, 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    _plot_diagram(axes[0], d_smooth, "Гладкая функция на торе", lo, hi)
    _plot_diagram(axes[1], d_noisy, "Та же функция + шум N(0, 0.05²)", lo, hi)
    fig.tight_layout()
    fig.savefig(OUT / "ch08_diagrams.png", dpi=200)
    plt.close(fig)


def fig_counts_threshold() -> None:
    thresholds = np.linspace(0, 0.6, 61)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, noise in zip(axes, (0.05, 0.1)):
        field = torus_field(128, 128) + noise * np.random.default_rng(1).normal(size=(128, 128))
        mesh = periodic_grid_mesh(field)
        pairs = critical_pairs(mesh)
        points = annotate_persistence(analyze_morse(mesh), pairs)
        counts = [persistent_counts(points, pairs, float(t)) for t in thresholds]
        for kind, color, label in (("min", MIN_C, "минимумы"), ("saddle", SADDLE_C, "сёдла (с кратностью)"), ("max", MAX_C, "максимумы")):
            ax.step(thresholds, [c[kind] for c in counts], where="post", color=color, lw=1.8, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("порог персистентности")
        ax.set_title(f"Тор 128×128, шум σ = {noise}: {counts[0]['min']}/{counts[0]['saddle']}/{counts[0]['max']} → "
                     f"{counts[-1]['min']}/{counts[-1]['saddle']}/{counts[-1]['max']}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("число критических точек")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ch08_counts_threshold.png", dpi=200)
    plt.close(fig)


def fig_stability() -> None:
    f = torus_field()
    reference = persistence_diagram(critical_pairs(periodic_grid_mesh(f)))
    rng = np.random.default_rng(0)
    eps_values = np.linspace(0.01, 0.5, 15)
    sup, dist = [], {0: [], 1: [], 2: []}
    for eps in eps_values:
        for _ in range(4):
            noise = rng.uniform(-eps, eps, size=f.shape)
            diagram = persistence_diagram(critical_pairs(periodic_grid_mesh(f + noise)))
            sup.append(np.abs(noise).max())
            for dim in dist:
                dist[dim].append(bottleneck(reference.get(dim, []), diagram.get(dim, [])))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([0, 0.52], [0, 0.52], color=ACCENT, lw=2, label=r"$y = \|f-g\|_\infty$ (граница теоремы)")
    for dim, values in dist.items():
        ax.scatter(sup, values, s=14, color=DIM_COLORS[dim], alpha=0.8, label=f"$d_B$, dim {dim}")
    ax.set_xlabel(r"$\|f-g\|_\infty$ (амплитуда равномерного шума)")
    ax.set_ylabel("расстояние bottleneck")
    ax.set_title("Устойчивость: все точки ниже прямой")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ch08_stability.png", dpi=200)
    plt.close(fig)
    worst = max(max(v / s for v, s in zip(values, sup)) for values in dist.values())
    print(f"stability: max d_B / sup-norm = {worst:.3f}")


if __name__ == "__main__":
    fig_pairs_1d()
    fig_diagrams()
    fig_counts_threshold()
    fig_stability()
