"""Рисунки главы 5 «Критические точки на триангулированных поверхностях».

Запуск: .venv/bin/python ch05_figures.py
Все классификации на рисунках получены вызовом morse_lidar.analyze_morse / lower_upper_links.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from morse_lidar import TriMesh, analyze_morse, grid_mesh
from morse_lidar.critical import lower_upper_links
from morse_lidar.persistence import annotate_persistence, critical_pairs, persistent_counts

OUT = Path(__file__).resolve().parent
NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
C_MIN, C_SAD, C_MAX = NAVY, "#2A9D8F", ACCENT
KIND_COLOR = {"min": C_MIN, "saddle": C_SAD, "max": C_MAX, "regular": "0.75"}
KIND_RU = {"min": "минимум", "saddle": "седло", "max": "максимум", "regular": "регулярная"}
CMAP = LinearSegmentedColormap.from_list("ml", [INK, SOFT, TEAL, LIGHT])
LOWER, UPPER = CYAN, ACCENT

# Порядок соседей внутренней вершины сетки grid_mesh по кругу: В, СВ, С, З, ЮЗ, Ю.
HEX_DIRS = np.array([(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)], float)
HEX_NAMES = ["В", "СВ", "С", "З", "ЮЗ", "Ю"]


def fan_mesh(center: float, ring: list[float]) -> TriMesh:
    """Веер из шести треугольников: вершина 0 в центре, 1..6 — её линк."""
    vertices = np.vstack(([0.0, 0.0, 0.0], np.column_stack((HEX_DIRS, np.zeros(6)))))
    faces = np.array([[0, 1 + i, 1 + (i + 1) % 6] for i in range(6)])
    return TriMesh(vertices, faces, np.array([center, *ring], float))


def draw_link(ax, mesh: TriMesh, title: str) -> None:
    point = analyze_morse(mesh)[0]  # граничные вершины 1..6 замаскированы, остаётся центр
    lower, upper = lower_upper_links(mesh, 0)
    xy = mesh.vertices[:, :2]
    for i in range(6):
        ax.plot(*np.array([xy[0], xy[1 + i]]).T, color="0.8", lw=1, zorder=1)
    _, link_edges = mesh.neighbors_and_link()
    for left, right in link_edges[0]:
        same = next((c for c in lower + upper if left in c and right in c), None)
        color = "0.7" if same is None else (LOWER if same in lower else UPPER)
        width = 1.0 if same is None else 4.0
        ax.plot(*xy[[left, right]].T, color=color, lw=width, zorder=2, solid_capstyle="round")
    for vertex in range(1, 7):
        is_lower = any(vertex in c for c in lower)
        ax.plot(*xy[vertex], "o", ms=22, color=LOWER if is_lower else UPPER, mec="k", zorder=3)
        ax.text(*xy[vertex], f"{mesh.values[vertex]:g}", ha="center", va="center", fontsize=11,
                color="white", weight="bold", zorder=4)
    kind = point.kind.value
    ax.plot(*xy[0], "s", ms=24, color=KIND_COLOR[kind], mec="k", zorder=3)
    ax.text(*xy[0], f"{mesh.values[0]:g}", ha="center", va="center", fontsize=11, color="white",
            weight="bold", zorder=4)
    ax.set_title(f"{title}\nнижних {point.lower_components}, верхних {point.upper_components}"
                 f" → {KIND_RU[kind]}" + (f", кратн. {point.saddle_multiplicity}" if kind == "saddle" else ""),
                 fontsize=10.5)
    ax.set_xlim(-1.6, 1.6), ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


LINK_CASES = [
    ("а) регулярная", 5, [7, 8, 6, 4, 3, 2]),
    ("б) минимум", 5, [6, 9, 7, 8, 6, 7]),
    ("в) простое седло", 5, [7, 6, 3, 8, 2, 4]),
    ("г) обезьянье седло", 5, [7, 3, 8, 2, 6, 4]),
]


def fig_links() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    for ax, (title, center, ring) in zip(axes, LINK_CASES):
        draw_link(ax, fan_mesh(center, ring), title)
    fig.text(0.5, 0.02, "Голубые вершины и рёбра — нижний линк (значение меньше, чем в центре), "
             "оранжевые — верхний. Рёбра между нижней и верхней вершиной (серые) не входят ни в один.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "ch05_links.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def draw_grid(ax, mesh: TriMesh, title: str) -> None:
    xy = mesh.vertices[:, :2]
    for face in mesh.faces:
        ax.fill(*xy[face].T, facecolor="none", edgecolor="0.8", lw=0.8)
    kinds = {p.vertex: p for p in analyze_morse(mesh)}
    for vertex in range(len(xy)):
        point = kinds.get(vertex)
        color = "white" if point is None else KIND_COLOR[point.kind.value]
        ax.plot(*xy[vertex], "o", ms=24, color=color, mec="k", zorder=3)
        ax.text(xy[vertex][0], xy[vertex][1], f"{mesh.values[vertex]:g}", ha="center", va="center",
                fontsize=9.5, color="k" if color in ("white", "0.75") else "white", weight="bold", zorder=4)
        ax.text(xy[vertex][0] + 0.17, xy[vertex][1] - 0.22, f"#{vertex}", fontsize=7, color="0.35", zorder=4)
    sad = [p for p in kinds.values() if p.kind.value == "saddle"]
    summary = ", ".join(f"#{p.vertex}: {p.lower_components}/{p.upper_components}" for p in sad)
    ax.set_title(f"{title}\nсёдла (нижн./верхн.): {summary}", fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def monkey_meshes() -> tuple[TriMesh, TriMesh]:
    y, x = np.mgrid[-2:3, -2:3].astype(float)
    mesh = grid_mesh(x**3 - 3 * x * y**2)
    n = len(mesh.vertices)
    reverse = np.arange(n)[::-1]  # та же геометрия и значения, номера вершин в обратном порядке
    flipped = TriMesh(mesh.vertices[reverse], n - 1 - mesh.faces, mesh.values[reverse])
    return mesh, flipped


def fig_sos() -> None:
    mesh, flipped = monkey_meshes()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    draw_grid(axes[0], mesh, "а) обычная нумерация")
    draw_grid(axes[1], flipped, "б) нумерация в обратном порядке")
    fig.text(0.5, 0.02, "$f=x^3-3xy^2$ на сетке 5×5 из grid_mesh. Граничные вершины (белые) замаскированы. "
             "Нули рядом с центром разводятся по номеру вершины.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "ch05_sos.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def relief(rows: int = 60, cols: int = 80) -> np.ndarray:
    y, x = np.meshgrid(np.linspace(-1.5, 1.5, rows), np.linspace(-2, 2, cols), indexing="ij")
    return (x**2 - 1) ** 2 + y**2 + 0.3 * x


def fig_critical_map() -> None:
    height = relief()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, sigma in zip(axes, [0.0, 0.01, 0.03]):
        field = height + sigma * np.random.default_rng(0).normal(size=height.shape)
        mesh = grid_mesh(field)
        points = [p for p in analyze_morse(mesh) if p.kind.value != "regular"]
        ax.imshow(field, origin="lower", cmap=CMAP, vmin=-0.4, vmax=3.0)
        ax.contour(field, levels=np.linspace(-0.3, 3, 12), colors="white", linewidths=0.4, alpha=0.6)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), field.shape[1], field.shape[0], fill=False, ec="0.4", lw=6,
                                   alpha=0.5))
        counts = Counter(p.kind.value for p in points)
        for p in points:
            x, y = mesh.vertices[p.vertex][:2]
            marker = "*" if p.saddle_multiplicity >= 2 else "o"
            ax.plot(x, y, marker, ms=11 if marker == "*" else 7, color=KIND_COLOR[p.kind.value], mec="k", mew=0.7)
        ax.set_title(f"σ = {sigma}: мин {counts['min']}, сёдел {counts['saddle']}, макс {counts['max']}",
                     fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=KIND_COLOR[k], mec="k", label=KIND_RU[k])
               for k in ("min", "saddle", "max")]
    handles.append(plt.Line2D([], [], marker="*", ls="", ms=11, color=C_SAD, mec="k", label="мульти-седло"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "ch05_critical_map.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def noise_table(sigmas=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1), seeds=range(5)):
    height = relief()
    rows = []
    for sigma in sigmas:
        stats = []
        for seed in seeds:
            mesh = grid_mesh(height + sigma * np.random.default_rng(seed).normal(size=height.shape))
            points = analyze_morse(mesh)
            pairs = critical_pairs(mesh)
            kept = persistent_counts(annotate_persistence(points, pairs), pairs, 0.3)
            c = Counter(p.kind.value for p in points)
            multi = sum(p.kind.value == "saddle" and p.saddle_multiplicity >= 2 for p in points)
            stats.append((c["min"], c["saddle"], c["max"], multi, kept["min"], kept["saddle"], kept["max"]))
        rows.append((sigma, np.mean(stats, axis=0)))
    return rows


def fig_noise(rows) -> None:
    sig = np.array([r[0] for r in rows[1:]])
    data = np.array([r[1] for r in rows[1:]])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, kind in ((0, "min"), (1, "saddle"), (2, "max")):
        ax.loglog(sig, np.where(data[:, col] > 0, data[:, col], np.nan), "o-", color=KIND_COLOR[kind], label=KIND_RU[kind])
    ax.loglog(sig, data[:, 4] + data[:, 5] + data[:, 6], "s--", color="0.4",
              label="все типы после фильтра персистентности 0.3")
    ax.set_xlabel("σ шума (в единицах высоты)")
    ax.set_ylabel("число критических точек (среднее по 5 seed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ch05_noise.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    fig_links()
    fig_sos()
    fig_critical_map()
    table = noise_table()
    for sigma, mean in table:
        print(sigma, np.round(mean, 1))
    fig_noise(table)
