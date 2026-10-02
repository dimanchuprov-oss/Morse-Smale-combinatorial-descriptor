"""Рисунки главы 4 «Гладкая теория Морса».

Запуск: .venv/bin/python ch04_figures.py
Критические точки на торе считаются самой библиотекой morse_lidar (analyze_morse).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from morse_lidar import TriMesh, analyze_morse
from morse_lidar.mesh import _grid_faces

OUT = Path(__file__).resolve().parent
NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
C_MIN, C_SAD, C_MAX = NAVY, "#2A9D8F", ACCENT
CMAP = LinearSegmentedColormap.from_list("ml", [INK, SOFT, TEAL, LIGHT])
KIND_COLOR = {"min": C_MIN, "saddle": C_SAD, "max": C_MAX}


def fig_models() -> None:
    models = [
        ("$x^2+y^2$\nминимум, индекс 0", lambda x, y: x**2 + y**2, "min"),
        ("$x^2-y^2$\nседло, индекс 1", lambda x, y: x**2 - y**2, "saddle"),
        ("$-x^2-y^2$\nмаксимум, индекс 2", lambda x, y: -(x**2) - y**2, "max"),
        ("$x^3-3xy^2$\nобезьянье седло (вырожд.)", lambda x, y: x**3 - 3 * x * y**2, "saddle"),
    ]
    s = np.linspace(-1, 1, 81)
    x, y = np.meshgrid(s, s)
    fig = plt.figure(figsize=(13, 6.4))
    for col, (title, func, kind) in enumerate(models):
        z = func(x, y)
        ax = fig.add_subplot(2, 4, col + 1, projection="3d")
        ax.plot_surface(x, y, z, cmap=CMAP, linewidth=0, antialiased=True, rstride=2, cstride=2, alpha=0.95)
        ax.scatter([0], [0], [0], s=60, color=KIND_COLOR[kind], edgecolor="k", depthshade=False, zorder=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
        ax.view_init(elev=28, azim=-55)
        ax2 = fig.add_subplot(2, 4, col + 5)
        ax2.contourf(x, y, z, levels=14, cmap=CMAP)
        ax2.contour(x, y, z, levels=14, colors="k", linewidths=0.4)
        ax2.contour(x, y, z, levels=[0], colors=[ACCENT if kind != "min" else "none"], linewidths=1.6)
        ax2.plot(0, 0, "o", ms=9, color=KIND_COLOR[kind], mec="k")
        ax2.set_aspect("equal")
        ax2.set_xticks([]), ax2.set_yticks([])
        if col == 1:
            ax2.text(0.62, 0.05, "$f>0$", fontsize=10, color="w")
            ax2.text(-0.12, 0.75, "$f<0$", fontsize=10, color="w")
    fig.text(0.5, 0.01, "Нижний ряд: линии уровня; красным — уровень $f=0$, проходящий через критическую точку.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "ch04_models.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def relief(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x**2 - 1) ** 2 + y**2 + 0.3 * x


def fig_sublevel() -> None:
    xs, ys = np.linspace(-2, 2, 400), np.linspace(-1.5, 1.5, 300)
    x, y = np.meshgrid(xs, ys)
    z = relief(x, y)
    crit = [(-1.0356, 0.0, "min", -0.305), (0.9601, 0.0, "min", 0.294), (0.0754, 0.0, "saddle", 1.011)]
    levels = [-0.1, 0.6, 0.98, 1.4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, a in zip(axes, levels):
        ax.contour(x, y, z, levels=np.linspace(-0.3, 4, 18), colors="0.6", linewidths=0.5)
        ax.contourf(x, y, z, levels=[z.min() - 1, a], colors=[TEAL], alpha=0.55)
        ax.contour(x, y, z, levels=[a], colors=[NAVY], linewidths=1.8)
        for cx, cy, kind, val in crit:
            filled = val <= a
            ax.plot(cx, cy, "o", ms=9, color=KIND_COLOR[kind] if filled else "white", mec=KIND_COLOR[kind], mew=2)
        from scipy import ndimage

        ncomp = ndimage.label(z <= a)[1]
        ax.set_title(f"$a={a}$:  компонент $M_a$ — {ncomp}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Подуровневые множества $M_a=\\{f\\leq a\\}$ для $f=(x^2-1)^2+y^2+0.3x$; "
                 "минимумы $f=-0.31$ и $0.29$, седло $f=1.01$", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ch04_sublevel.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def torus_mesh(nu: int = 160, nv: int = 64, big: float = 2.0, small: float = 0.8) -> TriMesh:
    """Тор, стоящий «на ребре»; функция — высота z. Сдвиг сетки убирает точные симметрии."""
    u = 2 * np.pi * (np.arange(nu) + 0.3) / nu
    v = 2 * np.pi * (np.arange(nv) + 0.2) / nv
    vv, uu = np.meshgrid(v, u)
    xx = (big + small * np.cos(vv)) * np.cos(uu)
    yy = small * np.sin(vv)
    zz = (big + small * np.cos(vv)) * np.sin(uu)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    return TriMesh(vertices, _grid_faces(nu, nv, periodic=True), zz.ravel())


def project(points: np.ndarray, elev: float = 22.0) -> tuple[np.ndarray, np.ndarray]:
    """Поворот вокруг оси x и ортогональная проекция на плоскость (x, z); вторая величина — глубина."""
    t = np.radians(elev)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    depth = y * np.cos(t) + z * np.sin(t)
    screen_z = -y * np.sin(t) + z * np.cos(t)
    return np.stack((x, screen_z), axis=-1), depth


def draw_torus(ax, mesh: TriMesh, level: float | None, points) -> None:
    from matplotlib.collections import PolyCollection

    tri = mesh.vertices[mesh.faces]
    screen, depth = project(tri)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(normals @ light)
    face_max = mesh.values[mesh.faces].max(axis=1)
    inside = np.ones(len(mesh.faces), bool) if level is None else face_max <= level
    colors = np.zeros((len(mesh.faces), 4))
    colors[:] = matplotlib.colors.to_rgba(TEAL)
    colors[:, :3] = 1 - (1 - colors[:, :3]) * shade[:, None]
    colors[:, :3] *= (0.55 + 0.45 * shade)[:, None]
    colors[~inside] = (0.9, 0.9, 0.9, 1.0)
    order = np.argsort(-depth.mean(axis=1))  # дальние грани рисуются первыми
    ax.add_collection(PolyCollection(screen[order], facecolors=colors[order], edgecolors=colors[order], linewidths=0.2))
    for point in points:
        if level is not None and mesh.values[point.vertex] > level:
            continue
        (px, pz), _ = project(mesh.vertices[point.vertex])
        ax.plot(px, pz, "o", ms=9, color=KIND_COLOR[point.kind.value], mec="k", zorder=5)
    ax.set_xlim(-3, 3), ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def fig_torus() -> None:
    mesh = torus_mesh()
    points = [p for p in analyze_morse(mesh) if p.kind.value != "regular"]
    fig = plt.figure(figsize=(14, 4.4))
    ax = fig.add_subplot(1, 5, 1)
    draw_torus(ax, mesh, None, points)
    ax.set_title("Тор, $f=z$: 1 мин., 2 седла, 1 макс.", fontsize=10)
    for point in points:
        (px, pz), _ = project(mesh.vertices[point.vertex])
        label = {"min": "индекс 0", "saddle": "индекс 1", "max": "индекс 2"}[point.kind.value]
        ax.text(px + 0.35, pz - 0.1, label, fontsize=8, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    stages = [(-2.0, "диск"), (0.0, "цилиндр"), (2.0, "тор без диска"), (3.0, "весь тор")]
    for k, (a, name) in enumerate(stages):
        ax = fig.add_subplot(1, 5, k + 2)
        draw_torus(ax, mesh, a, points)
        ax.set_title(f"$M_a$, $a={a}$: {name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ch04_torus.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    fig_models()
    fig_sublevel()
    fig_torus()
