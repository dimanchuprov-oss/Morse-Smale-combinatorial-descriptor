"""Рисунки главы 9. Запуск: morse-lidar/.venv/bin/python ch09_figures.py"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Wedge

from morse_lidar.critical import analyze_morse
from morse_lidar.intrinsic import mean_curvature, shape_index
from morse_lidar.mesh import TriMesh, grid_mesh

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
SADDLE = "#2A9D8F"
OUT = Path(__file__).resolve().parent
DIVERGING = LinearSegmentedColormap.from_list("ml_div", [NAVY, SOFT, "#F4F4F4", "#F2A48F", ACCENT])
KIND_STYLE = {"min": (NAVY, "o"), "saddle": (SADDLE, "X"), "max": (ACCENT, "^")}


def relief(n: int = 81, size: float = 6.0, background: float = 0.03) -> tuple[np.ndarray, float, np.ndarray]:
    xs = np.linspace(-size / 2, size / 2, n)
    x, y = np.meshgrid(xs, xs)
    z = (
        0.8 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) / 0.5)
        + 0.6 * np.exp(-((x + 1.1) ** 2 + (y + 0.4) ** 2) / 0.6)
        - 0.5 * np.exp(-((x - 0.2) ** 2 + (y + 1.4) ** 2) / 0.4)
        + background * (x**2 - 0.5 * y**2)
    )
    return z, float(xs[1] - xs[0]), xs


def figure_shape_index() -> None:
    z, h, xs = relief()
    mesh = grid_mesh(z, spacing=(h, h))
    H = mean_curvature(mesh).reshape(z.shape)
    S = shape_index(mesh).reshape(z.shape)
    extent = (0, h * (z.shape[1] - 1), 0, h * (z.shape[0] - 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (
        (z, "Высота $z$", "viridis", None),
        (H, "Средняя кривизна $H$ (1/длина)", DIVERGING, float(np.percentile(np.abs(H), 99))),
        (S, "Shape index $S$", DIVERGING, 1.0),
    )
    for ax, (field, title, cmap, limit) in zip(axes, panels):
        kwargs = {"vmin": -limit, "vmax": limit} if limit else {}
        image = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, **kwargs)
        ax.contour(np.linspace(0, extent[1], z.shape[1]), np.linspace(0, extent[3], z.shape[0]), z, 10,
                   colors="k", linewidths=0.4, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(image, ax=ax, shrink=0.85)
    axes[0].set_ylabel("y")
    cbar = axes[2].images[0].colorbar
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["−1 чаша", "−0.5", "0 седло", "0.5 цилиндр", "+1 купол"])
    fig.tight_layout()
    fig.savefig(OUT / "ch09_shape_index.png", dpi=200)
    plt.close(fig)


def figure_angle_defect() -> None:
    # Вершина-«пирамида» с пятью треугольниками: апекс поднят над правильным пятиугольником.
    count, radius, apex_height = 5, 1.0, 0.55
    phi = 2 * np.pi * np.arange(count) / count
    ring = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.zeros(count)))
    apex = np.array([0.0, 0.0, apex_height])
    angles = []
    for i in range(count):
        a, b = ring[i] - apex, ring[(i + 1) % count] - apex
        angles.append(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))
    angles = np.array(angles)
    defect = 2 * np.pi - angles.sum()
    edge = np.linalg.norm(ring[0] - apex)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6))
    # Слева: вид сбоку-сверху (аксонометрия) звезды вершины.
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, 0.45, 0.9]])
    proj = lambda p: tilt @ p  # noqa: E731
    for i in range(count):
        tri = np.array([proj(apex), proj(ring[i]), proj(ring[(i + 1) % count])])
        left.add_patch(Polygon(tri, closed=True, facecolor=LIGHT if i % 2 else "#DDF8FF", edgecolor=INK, lw=1.2))
    centre = proj(apex)
    left.plot(*centre, "o", color=ACCENT, ms=8, zorder=5)
    left.annotate("вершина $v$", centre, xytext=(centre[0] + 0.35, centre[1] + 0.45),
                  arrowprops={"arrowstyle": "->", "color": INK}, color=INK)
    left.set_title("Звезда вершины в пространстве")
    left.set_aspect("equal")
    left.axis("off")
    left.set_xlim(-1.3, 1.3)
    left.set_ylim(-0.8, 1.2)

    # Справа: та же звезда, развёрнутая на плоскость. Не хватает угла delta.
    start = np.pi / 2
    for i, angle in enumerate(angles):
        theta0 = start + angles[:i].sum()
        p0 = edge * np.array([np.cos(theta0), np.sin(theta0)])
        p1 = edge * np.array([np.cos(theta0 + angle), np.sin(theta0 + angle)])
        right.add_patch(Polygon([[0, 0], p0, p1], closed=True, facecolor=LIGHT if i % 2 else "#DDF8FF",
                                edgecolor=INK, lw=1.2))
        mid = theta0 + angle / 2
        right.text(0.32 * np.cos(mid), 0.32 * np.sin(mid), rf"$\theta_{i + 1}$", ha="center", va="center",
                   color=INK, fontsize=10)
    gap_start = np.degrees(start + angles.sum())
    right.add_patch(Wedge((0, 0), edge, gap_start, gap_start + np.degrees(defect), facecolor=ACCENT, alpha=0.25,
                          edgecolor=ACCENT, lw=1.5, ls="--"))
    mid = np.radians(gap_start + np.degrees(defect) / 2)
    right.text(0.7 * edge * np.cos(mid), 0.7 * edge * np.sin(mid),
               "$\\delta = 2\\pi - \\sum\\theta_i$\n$\\approx %.0f^\\circ$" % np.degrees(defect),
               color=ACCENT, ha="center", va="center", fontsize=11)
    right.plot(0, 0, "o", color=ACCENT, ms=8)
    right.set_title("Развёртка на плоскость: зазор = угловой дефект")
    right.set_aspect("equal")
    right.axis("off")
    right.set_xlim(-1.35, 1.35)
    right.set_ylim(-1.35, 1.35)
    fig.tight_layout()
    fig.savefig(OUT / "ch09_angle_defect.png", dpi=200)
    plt.close(fig)


def _points(mesh: TriMesh) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {"min": [], "saddle": [], "max": []}
    for point in analyze_morse(mesh):
        if point.kind.value in result:
            result[point.kind.value].append(point.vertex)
    return {kind: np.array(items, dtype=int) for kind, items in result.items()}


def figure_rotation() -> None:
    z, h, _ = relief(n=61, background=0.0)
    mesh = grid_mesh(z, spacing=(h, h))
    t = np.radians(20.0)
    rotation = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    rotated_vertices = mesh.vertices @ rotation.T
    rotated = TriMesh(rotated_vertices, mesh.faces, rotated_vertices[:, 2])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, current, title in ((axes[0], mesh, "Поза 1: высота $z$"),
                               (axes[1], rotated, "Поза 2: скан повёрнут на 20° вокруг оси y")):
        values = current.values.reshape(z.shape)
        ax.imshow(values, origin="lower", extent=(0, h * 60, 0, h * 60), cmap="viridis")
        ax.contour(np.linspace(0, h * 60, 61), np.linspace(0, h * 60, 61), values, 14, colors="k", linewidths=0.4,
                   alpha=0.6)
        for kind, vertices in _points(current).items():
            colour, marker = KIND_STYLE[kind]
            if len(vertices):
                xy = mesh.vertices[vertices, :2]
                ax.scatter(xy[:, 0], xy[:, 1], c=colour, marker=marker, s=80, edgecolors="white", linewidths=1.2,
                           label=f"{kind}: {len(vertices)}", zorder=5)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlabel("x (исходные координаты)")
    fig.tight_layout()
    fig.savefig(OUT / "ch09_rotation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figure_shape_index()
    figure_angle_defect()
    figure_rotation()
    print("ok")
