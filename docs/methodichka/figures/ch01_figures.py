"""Рисунки к главе 1. Запуск: .venv/bin/python figures/ch01_figures.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

from morse_lidar import PointCloud, align_pca

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
OUT = Path(__file__).resolve().parent


def pca_figure() -> None:
    rng = np.random.default_rng(2)
    local = rng.normal(size=(1500, 3)) * [3.0, 1.0, 0.1]
    angle = np.deg2rad(30)
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    cloud = local @ rotation.T + [5.0, 2.0, 1.0]
    centre = cloud.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((cloud - centre).T))
    aligned = align_pca(PointCloud(cloud)).points

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    shifted = cloud - centre
    ax.scatter(shifted[:, 1], shifted[:, 2], s=2, color=SOFT, alpha=0.35)
    colors = {1: TEAL, 0: ACCENT}  # eigh returns eigenvalues in ascending order
    names = {1: r"$\mathbf{e}_2 \to Y$", 0: r"$\mathbf{e}_3 \to Z$"}
    for k in (1, 0):
        direction = vectors[1:, k] / np.linalg.norm(vectors[1:, k]) * 1.8
        ax.add_patch(FancyArrowPatch((0, 0), direction, arrowstyle="-|>",
                                     mutation_scale=16, color=colors[k], linewidth=2.2))
        ax.text(direction[0] + 0.1, direction[1] + (0.1 if k == 0 else -0.55), names[k], color=colors[k], fontsize=9,
                ha="center" if k == 0 else "left")
    ax.set_title("До: сечение y–z, облако наклонено на 30°\n$\\mathbf{e}_1$ (наибольшая дисперсия) идёт вдоль x", fontsize=10)
    ax.set_xlabel("y − ȳ"), ax.set_ylabel("z − z̄")
    ax2.scatter(aligned[:, 1], aligned[:, 2], s=2, color=SOFT, alpha=0.4)
    ax2.set_xlabel("Y после align_pca"), ax2.set_ylabel("Z после align_pca")
    var = aligned.var(axis=0)
    ax2.set_title(f"После align_pca: сечение Y–Z\nдисперсии по X, Y, Z: {var[0]:.2f}, {var[1]:.2f}, {var[2]:.3f}", fontsize=10)
    for axis in (ax, ax2):
        axis.set_xlim(-3.5, 3.5), axis.set_ylim(-2.6, 2.6)
        axis.set_aspect("equal")
        axis.axhline(0, color=INK, linewidth=0.5), axis.axvline(0, color=INK, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "ch01_pca.png", dpi=200)
    plt.close(fig)


def triangle_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    a, b, c = np.array([0.0, 0.0]), np.array([3.0, 0.4]), np.array([1.2, 2.4])
    for ax, order, title, sign in (
        (axes[0], (a, b, c), "Обход $a\\to b\\to c$ против часовой:\n$\\mathbf{n}$ смотрит на нас ($+z$)", 1),
        (axes[1], (a, c, b), "Обход $a\\to c\\to b$ по часовой:\n$\\mathbf{n}$ смотрит от нас ($-z$)", -1),
    ):
        ax.add_patch(Polygon([a, b, c], closed=True, facecolor=LIGHT, edgecolor=INK, linewidth=1.5))
        p0, p1, p2 = order
        for start, end in ((p0, p1), (p1, p2), (p2, p0)):
            s = start + 0.18 * (end - start)
            e = start + 0.82 * (end - start)
            ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=16, color=NAVY, linewidth=1.8))
        for point, name in ((a, "a"), (b, "b"), (c, "c")):
            offset = (point - (a + b + c) / 3) * 0.18
            ax.text(*(point + offset), name, fontsize=14, ha="center", va="center", color=INK)
        centre = (a + b + c) / 3
        if sign > 0:
            ax.plot(*centre, "o", markersize=13, markerfacecolor="white", markeredgecolor=ACCENT, markeredgewidth=2)
            ax.plot(*centre, "o", markersize=4, color=ACCENT)
        else:
            ax.plot(*centre, "o", markersize=13, markerfacecolor="white", markeredgecolor=ACCENT, markeredgewidth=2)
            ax.plot(*centre, "x", markersize=9, color=ACCENT, markeredgewidth=2)
        ax.text(centre[0] + 0.25, centre[1] - 0.05, r"$\mathbf{n}$", color=ACCENT, fontsize=14)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, 3.5), ax.set_ylim(-0.5, 2.9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, 0.02, r"$\mathbf{n}=(\mathbf{b}-\mathbf{a})\times(\mathbf{c}-\mathbf{a})$, "
             r"$|\mathbf{n}| = 2S$; точка — вектор на нас, крест — от нас", ha="center", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT / "ch01_triangle.png", dpi=200)
    plt.close(fig)


def gradient_figure() -> None:
    p = np.array([[0.0, 0.0], [3.0, 0.4], [1.2, 2.4]])
    f = np.array([1.0, 2.5, 4.0])
    grid = np.linspace(0, 1, 300)
    u, v = np.meshgrid(grid, grid)
    inside = u + v <= 1
    w = 1 - u - v
    x = w * p[0, 0] + u * p[1, 0] + v * p[2, 0]
    y = w * p[0, 1] + u * p[1, 1] + v * p[2, 1]
    value = w * f[0] + u * f[1] + v * f[2]
    value = np.where(inside, value, np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = ax.pcolormesh(x, y, value, cmap="Blues", shading="auto")
    ax.contour(x, y, value, levels=8, colors=INK, linewidths=0.6)
    ax.add_patch(Polygon(p, closed=True, fill=False, edgecolor=INK, linewidth=1.5))
    e1, e2 = p[1] - p[0], p[2] - p[0]
    gram = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
    coeff = np.linalg.solve(gram, [f[1] - f[0], f[2] - f[0]])
    gradient = coeff[0] * e1 + coeff[1] * e2
    centre = p.mean(axis=0)
    ax.add_patch(FancyArrowPatch(centre, centre + 0.45 * gradient / np.linalg.norm(gradient) * 2,
                                 arrowstyle="-|>", mutation_scale=18, color=ACCENT, linewidth=2.2))
    ax.text(*(centre + 0.95 * gradient / np.linalg.norm(gradient)), r"$\nabla f$", color=ACCENT, fontsize=14)
    for point, val, name in zip(p, f, ("a", "b", "c")):
        ax.text(point[0], point[1] - 0.22 if name != "c" else point[1] + 0.12, f"{name}: f={val:g}",
                ha="center", fontsize=10, color=INK)
    fig.colorbar(mesh, ax=ax, label="f (линейная интерполяция)")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Линейная функция на треугольнике и её градиент", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ch01_gradient.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    pca_figure()
    triangle_figure()
    gradient_figure()
