"""Рисунки главы 7. Запуск: .venv/bin/python figures/ch07_figures.py (из папки методички)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from morse_lidar.critical import CriticalPoint, CriticalType
from morse_lidar.descriptor import validate_colored_graph
from morse_lidar.mesh import grid_mesh
from morse_lidar.separatrices import Separatrix
from morse_lidar.ttk_backend import TTKQuadrangulation, _colored_dual

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
SADDLE = "#2A9D8F"
COLOR = {"u": NAVY, "s": ACCENT, "t": TEAL}
OUT = Path(__file__).resolve().parent


def dual_graph_figure() -> None:
    """Двойственный граф маленькой открытой триангуляции grid_mesh (4 x 4 вершины)."""
    mesh = grid_mesh(np.zeros((4, 4)))
    xy = mesh.vertices[:, :2]
    faces = mesh.faces
    centroids = xy[faces].mean(axis=1)
    by_edge: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            by_edge.setdefault((min(a, b), max(a, b)), []).append(index)

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for (a, b), adjacent in by_edge.items():
        boundary = len(adjacent) == 1
        ax.plot(*xy[[a, b]].T, color=INK if boundary else SOFT, linewidth=2.2 if boundary else 1.0, zorder=1)
        if len(adjacent) == 2:
            ax.plot(*centroids[adjacent].T, color=ACCENT, linewidth=1.8, zorder=2)
    degree = np.zeros(len(faces), dtype=int)
    for adjacent in by_edge.values():
        if len(adjacent) == 2:
            degree[adjacent] += 1
    ax.scatter(*xy.T, s=25, c=INK, zorder=3)
    ax.scatter(*centroids.T, s=150, c=LIGHT, edgecolor=ACCENT, linewidth=1.5, zorder=4)
    for index, (cx, cy) in enumerate(centroids):
        ax.text(cx, cy, str(degree[index]), ha="center", va="center", fontsize=8, color=INK, zorder=5)
    ax.plot([], [], color=SOFT, linewidth=1.0, label="внутреннее ребро (2 треугольника)")
    ax.plot([], [], color=INK, linewidth=2.2, label="граничное ребро (1 треугольник)")
    ax.plot([], [], color=ACCENT, linewidth=1.8, label="ребро двойственного графа")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("grid_mesh 4×4: 18 треугольников; в кружке — степень узла", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ch07_dual_graph.png", dpi=200)
    plt.close(fig)


# ---------- Тор f = sin x + 0.4 cos y: 1 min, 2 седла, 1 max ----------
# id критических точек: 0 = m (min), 1 = S1, 2 = M (max), 3 = S2
NAMES = {"u1+": 10, "u1-": 11, "s1+": 12, "s1-": 13, "s2+": 14, "s2-": 15, "u2+": 16, "u2-": 17}
ENDS = {"u1+": (1, 0), "u1-": (1, 0), "s1+": (1, 2), "s1-": (1, 2),
        "s2+": (3, 2), "s2-": (3, 2), "u2+": (3, 0), "u2-": (3, 0)}
# ячейка: углы по циклу и стороны (сторона j соединяет угол j и угол j+1), координаты углов в единицах pi
CELLS = {
    "A": ((2, 3, 0, 1), ("s2-", "u2+", "u1+", "s1-"), ((0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1))),
    "B": ((3, 2, 1, 0), ("s2+", "s1-", "u1-", "u2+"), ((1.5, 0), (2.5, 0), (2.5, 1), (1.5, 1))),
    "C": ((1, 0, 3, 2), ("u1+", "u2-", "s2-", "s1+"), ((0.5, 1), (1.5, 1), (1.5, 2), (0.5, 2))),
    "D": ((0, 1, 2, 3), ("u1-", "s1+", "s2+", "u2-"), ((1.5, 1), (2.5, 1), (2.5, 2), (1.5, 2))),
}


def torus_example():
    critical = (
        CriticalPoint(0, CriticalType.MINIMUM, 0, 1, identifier=0),
        CriticalPoint(1, CriticalType.SADDLE, 2, 2, 1, identifier=1),
        CriticalPoint(2, CriticalType.MAXIMUM, 1, 0, identifier=2),
        CriticalPoint(3, CriticalType.SADDLE, 2, 2, 1, identifier=3),
    )
    separatrices = tuple(Separatrix(ENDS[k][0], ENDS[k][1], k[0], (), identifier=v) for k, v in NAMES.items())
    points, types, ids = [], [], []

    def add(kind: int, identifier: int) -> int:
        points.append((0.0, 0.0, 0.0))
        types.append(kind)
        ids.append(identifier)
        return len(points) - 1

    corner = {c: add(0, c) for c in range(4)}
    middle = {k: add(1, v) for k, v in NAMES.items()}
    quads = []
    for index, (corners, sides, _) in enumerate(CELLS.values()):
        center = add(2, 100 + index)
        for j in range(4):
            quads.append((corner[corners[j]], middle[sides[j]], center, middle[sides[j - 1]]))
    quadrangulation = TTKQuadrangulation(tuple(points), tuple(quads), tuple(types), tuple(ids))
    cells, graph = _colored_dual(quadrangulation, critical, separatrices)
    validate_colored_graph(graph)
    return cells, graph


def colored_graph_figure() -> None:
    cells, graph = torus_example()
    inverse = {v: k for k, v in NAMES.items()}
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    marker = {0: ("o", NAVY), 1: ("X", SADDLE), 2: ("^", ACCENT), 3: ("X", SADDLE)}
    label = {0: "m", 1: "$S_1$", 2: "M", 3: "$S_2$"}
    node_xy: dict[int, np.ndarray] = {}
    arc_mid: dict[tuple[int, tuple[str, int]], np.ndarray] = {}
    for cell_index, (name, (corners, sides, coords)) in enumerate(CELLS.items()):
        pos = {c: np.array(p, dtype=float) for c, p in zip(corners, coords)}
        ax.fill(*np.array(coords).T, color=LIGHT, alpha=0.35, zorder=0)
        for j in range(4):
            a, b = corners[j], corners[(j + 1) % 4]
            kind = sides[j][0]
            ax.plot(*np.array([pos[a], pos[b]]).T, color=COLOR[kind], linewidth=3, alpha=0.45, zorder=1)
        ax.plot(*np.array([pos[0], pos[2]]).T, color=TEAL, linestyle="--", linewidth=1.2, zorder=1)
        center = np.mean(np.array(coords), axis=0)
        ax.text(*(pos[2] + 0.17 * np.sign(center - pos[2])), name, fontsize=14,
                color=INK, alpha=0.6, ha="center", va="center")
        for c, p in pos.items():
            m, col = marker[c]
            ax.scatter(*p, s=120, marker=m, c=col, edgecolor="white", zorder=6)
        # узлы графа: два треугольника ячейки (min, saddle_a, max) и (min, max, saddle_b)
        for local in range(2):
            node = 2 * cell_index + local
            tri = graph.triangles[node]
            node_xy[node] = np.mean([pos[c] for c in tri], axis=0)
            for arc in graph.primal_arcs[node]:
                if arc[0] == "t":
                    arc_mid[(node, arc)] = (pos[0] + pos[2]) / 2
                else:
                    a, b = ENDS[inverse[arc[1]]]
                    arc_mid[(node, arc)] = (pos[a] + pos[b]) / 2
    for edge in graph.edges:
        for node in (edge.left, edge.right):
            ax.plot(*np.array([node_xy[node], arc_mid[(node, edge.primal_edge)]]).T,
                    color=COLOR[edge.color], linewidth=2.2, zorder=4)
        if edge.primal_edge[0] == "separatrix":
            ends = [arc_mid[(n, edge.primal_edge)] for n in (edge.left, edge.right)]
            for p in ends:
                ax.text(*(p + np.array([0.05, 0.05])), inverse[edge.primal_edge[1]], fontsize=7,
                        color=COLOR[edge.color])
    for node, p in node_xy.items():
        ax.scatter(*p, s=170, c="white", edgecolor=INK, linewidth=1.4, zorder=7)
        ax.text(*p, str(node), ha="center", va="center", fontsize=8, color=INK, zorder=8)
    ax.plot([], [], color=NAVY, linewidth=2.2, label="u (седло–min)")
    ax.plot([], [], color=ACCENT, linewidth=2.2, label="s (седло–max)")
    ax.plot([], [], color=TEAL, linewidth=2.2, label="t (диагональ min–max)")
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(-0.1, 2.1)
    ax.set_xticks([0.5, 1.5, 2.5], ["π/2", "3π/2", "5π/2"])
    ax.set_yticks([0, 1, 2], ["0", "π", "2π"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Тор, f = sin x + 0.4 cos y: 4 ячейки, 8 узлов, 12 рёбер (по 4 каждого цвета)", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ch07_colored_graph.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    dual_graph_figure()
    colored_graph_figure()
