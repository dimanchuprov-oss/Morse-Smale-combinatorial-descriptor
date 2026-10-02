"""Рисунки главы 10. Запуск: morse-lidar/.venv/bin/python ch10_figures.py"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from morse_lidar.critical import analyze_morse
from morse_lidar.descriptor import build_descriptor
from morse_lidar.intrinsic import _corner_geometry, _raw_gaussian_curvature, _raw_mean_curvature, mean_curvature
from morse_lidar.mesh import grid_mesh

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
SADDLE = "#2A9D8F"
OUT = Path(__file__).resolve().parent
KIND_STYLE = {"min": (NAVY, "o"), "saddle": (SADDLE, "X"), "max": (ACCENT, "^")}
DIVERGING = LinearSegmentedColormap.from_list("ml_div", [NAVY, SOFT, "#F4F4F4", "#F2A48F", ACCENT])


def tilted_relief(n: int = 41, size: float = 6.0, tilt: float = 0.15) -> tuple[np.ndarray, float]:
    """Тот же рельеф, что в гл. 9, плюс наклон tilt*x и лёгкая рябь вдоль y."""
    xs = np.linspace(-size / 2, size / 2, n)
    x, y = np.meshgrid(xs, xs)
    z = (
        0.8 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) / 0.5)
        + 0.6 * np.exp(-((x + 1.1) ** 2 + (y + 0.4) ** 2) / 0.6)
        - 0.5 * np.exp(-((x - 0.2) ** 2 + (y + 1.4) ** 2) / 0.4)
    )
    return z + tilt * x + 0.05 * np.sin(3 * y), float(xs[1] - xs[0])


def _background(ax, mesh, shape, values=None, cmap="viridis", **kwargs):
    field = (mesh.values if values is None else values).reshape(shape)
    extent = (0, mesh.vertices[:, 0].max(), 0, mesh.vertices[:, 1].max())
    image = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, **kwargs)
    if values is None:
        ax.contour(np.linspace(0, extent[1], shape[1]), np.linspace(0, extent[3], shape[0]), field, 14, colors="k",
                   linewidths=0.4, alpha=0.5)
    ax.set_aspect("equal")
    return image


def _scatter(ax, mesh, points, title):
    for kind, (colour, marker) in KIND_STYLE.items():
        vertices = [p.vertex for p in points if p.kind.value == kind]
        if vertices:
            xy = mesh.vertices[vertices, :2]
            rim = sum(v in mesh.boundary_vertices for v in vertices)
            ax.scatter(xy[:, 0], xy[:, 1], c=colour, marker=marker, s=70, edgecolors="white", linewidths=1.1,
                       zorder=5, clip_on=False, label=f"{kind}: {len(vertices)} (на краю {rim})")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def figure_boundary_extrema() -> None:
    z, h = tilted_relief()
    mesh = grid_mesh(z, spacing=(h, h))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    for ax, include, title in ((axes[0], True, "include_boundary=True"),
                               (axes[1], False, "include_boundary=False (по умолчанию)")):
        _background(ax, mesh, z.shape)
        _scatter(ax, mesh, analyze_morse(mesh, include_boundary=include), title)
    fig.tight_layout()
    fig.savefig(OUT / "ch10_boundary_extrema.png", dpi=200)
    plt.close(fig)


def figure_separatrices() -> None:
    z, h = tilted_relief()
    mesh = grid_mesh(z, spacing=(h, h))
    descriptor = build_descriptor(mesh)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    _background(ax, mesh, z.shape)
    drawn = set()
    for arc in descriptor.separatrices:
        xy = mesh.vertices[list(arc.vertices), :2]
        if arc.ends_on_boundary:
            style = {"color": CYAN, "lw": 2.6, "ls": "--"}
            label = "ends_on_boundary=True"
        else:
            style = {"color": "white", "lw": 2.2, "ls": "-"}
            label = "дуга между внутренними точками"
        ax.plot(xy[:, 0], xy[:, 1], label=None if label in drawn else label, zorder=4, **style)
        drawn.add(label)
        if arc.ends_on_boundary:
            ax.plot(*xy[-1], "s", color=CYAN, ms=7, mec="white", zorder=6, clip_on=False)
    _scatter(ax, mesh, descriptor.critical_points, "")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)
    count = sum(arc.ends_on_boundary for arc in descriptor.separatrices)
    ax.set_title(f"Сепаратрисы: {len(descriptor.separatrices)}, из них в край: {count}")
    fig.tight_layout()
    fig.savefig(OUT / "ch10_separatrices.png", dpi=200)
    plt.close(fig)


def figure_rim_curvature() -> None:
    z, h = tilted_relief()
    mesh = grid_mesh(z, spacing=(h, h))
    corners, angles, cotangents = _corner_geometry(mesh)
    raw_k = _raw_gaussian_curvature(mesh, angles)
    raw_h = _raw_mean_curvature(mesh, corners, cotangents)
    ext_h = mean_curvature(mesh)
    rows, cols = z.shape
    corner_ids = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
    interior = np.setdiff1d(np.arange(len(raw_k)), sorted(mesh.boundary_vertices))
    k_limit = float(np.abs(raw_k[interior]).max())
    h_limit = float(np.abs(ext_h).max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = (
        (raw_k, k_limit, "Сырая $K$: углы сетки", "1/длина²"),
        (raw_h, h_limit, "Сырая $H$: на краю $\\approx 0$", "1/длина"),
        (ext_h, h_limit, "mean_curvature(): край продолжен слоями", "1/длина"),
    )
    for ax, (values, limit, title, unit) in zip(axes, panels):
        image = _background(ax, mesh, z.shape, values=values, cmap=DIVERGING, vmin=-limit, vmax=limit)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85, extend="both", label=unit)
    for vertex in corner_ids:
        x, y = mesh.vertices[vertex, :2]
        axes[0].annotate(f"{raw_k[vertex]:.0f}", (x, y), xytext=(x + (-1.2 if x > 3 else 0.3), y + (-0.5 if y > 3 else 0.3)),
                         color=ACCENT, fontsize=10, fontweight="bold", arrowprops={"arrowstyle": "->", "color": ACCENT})
    fig.tight_layout()
    fig.savefig(OUT / "ch10_rim_curvature.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figure_boundary_extrema()
    figure_separatrices()
    figure_rim_curvature()
    print("ok")
