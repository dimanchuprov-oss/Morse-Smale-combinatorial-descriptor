"""Рисунки к главе 2. Запуск: .venv/bin/python figures/ch02_figures.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from morse_lidar import PointCloud, grid_mesh, reconstruct_delaunay

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
OUT = Path(__file__).resolve().parent


def _draw_edges(ax, mesh, color="#B8C4D6", width=0.8):
    for left, right in mesh.edge_counts:
        p, q = mesh.vertices[left], mesh.vertices[right]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=width, zorder=1)


def star_link_figure() -> None:
    mesh = grid_mesh(np.zeros((5, 5)))
    neighbors, link_edges = mesh.neighbors_and_link()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, vertex, title in (
        (axes[0], 12, "Внутренняя вершина 12:\nлинк — цикл из 6 рёбер"),
        (axes[1], 2, "Граничная вершина 2:\nлинк — путь из 3 рёбер"),
    ):
        for face in mesh.faces:
            if vertex in face:
                ax.add_patch(Polygon(mesh.vertices[face][:, :2], closed=True, facecolor=LIGHT, edgecolor="none", zorder=0))
        _draw_edges(ax, mesh)
        for other in neighbors[vertex]:
            p, q = mesh.vertices[vertex], mesh.vertices[other]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=SOFT, linewidth=2.2, zorder=2)
        for left, right in link_edges[vertex]:
            p, q = mesh.vertices[left], mesh.vertices[right]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=ACCENT, linewidth=3.2, zorder=3)
        link_vertices = sorted(neighbors[vertex])
        ax.scatter(*mesh.vertices[link_vertices][:, :2].T, s=60, color=ACCENT, zorder=4)
        ax.scatter(*mesh.vertices[vertex][:2], s=110, color=NAVY, zorder=5)
        for index, point in enumerate(mesh.vertices):
            ax.text(point[0] + 0.07, point[1] + 0.07, str(index), fontsize=8, color=INK, zorder=6)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-0.4, 4.4), ax.set_ylim(-0.4, 4.4)
        ax.axis("off")
    fig.text(0.5, 0.02, "голубая заливка и синие рёбра — звезда (замкнутая), красное — линк",
             ha="center", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "ch02_star_link.png", dpi=200)
    plt.close(fig)


def grid_figure() -> None:
    mesh = grid_mesh(np.zeros((4, 4)))
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for left, right in mesh.edge_counts:
        p, q = mesh.vertices[left], mesh.vertices[right]
        boundary = (left, right) in mesh.boundary_edges
        ax.plot([p[0], q[0]], [p[1], q[1]], color=ACCENT if boundary else NAVY,
                linewidth=2.6 if boundary else 1.3, zorder=1)
    for index, face in enumerate(mesh.faces):
        centre = mesh.vertices[face][:, :2].mean(axis=0)
        ax.text(*centre, f"f{index}", fontsize=7, color=SOFT, ha="center", va="center")
    for index, point in enumerate(mesh.vertices):
        ax.scatter(point[0], point[1], s=40, color=ACCENT if index in mesh.boundary_vertices else NAVY, zorder=3)
        ax.text(point[0] - 0.1, point[1] + 0.08, str(index), fontsize=9, color=INK, ha="right")
    for name, (x, y) in {"a": (0, 0), "b": (1, 0), "c": (0, 1), "d": (1, 1)}.items():
        ax.text(x + (0.15 if x == 0 else -0.15), y + (0.12 if y == 0 else -0.18), name, fontsize=11,
                color=TEAL, ha="center", fontweight="bold")
    v, e, f = len(mesh.vertices), len(mesh.edge_counts), len(mesh.faces)
    ax.set_title(f"grid_mesh 4×4: V={v}, E={e}, F={f}, χ={mesh.euler_characteristic}\n"
                 f"граничных рёбер {len(mesh.boundary_edges)} (красные), граничных вершин {len(mesh.boundary_vertices)}",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 3.3), ax.set_ylim(-0.3, 3.3)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "ch02_grid.png", dpi=200)
    plt.close(fig)


def overhang_figure() -> None:
    rng = np.random.default_rng(3)
    ground = np.column_stack((rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500), np.zeros(500)))
    shelf = np.column_stack((rng.uniform(-0.4, 0.4, 150), rng.uniform(-1, 1, 150), np.full(150, 0.6)))
    points = np.vstack((ground, shelf))
    mesh = reconstruct_delaunay(PointCloud(points))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    band = np.abs(points[:, 1]) < 0.12
    axes[0].scatter(points[band & (points[:, 2] == 0), 0], points[band & (points[:, 2] == 0), 2], s=8, color=NAVY,
                    label="пол")
    axes[0].scatter(points[band & (points[:, 2] > 0), 0], points[band & (points[:, 2] > 0), 2], s=8, color=ACCENT,
                    label="полка (нависание)")
    axes[0].set_title("Облако, срез |y| < 0.12: под полкой есть пол", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    for face in mesh.faces:
        corners = mesh.vertices[face]
        if np.all(np.abs(corners[:, 1]) < 0.12):
            closed = np.vstack((corners, corners[:1]))
            axes[1].plot(closed[:, 0], closed[:, 2], color=SOFT, linewidth=0.7)
    axes[1].set_title("reconstruct_delaunay: рёбра треугольников в срезе\n«пила» между полом и полкой", fontsize=10)
    for ax in axes:
        ax.set_xlabel("x"), ax.set_ylabel("z")
        ax.set_ylim(-0.15, 0.85), ax.set_xlim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "ch02_overhang.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    star_link_figure()
    grid_figure()
    overhang_figure()
