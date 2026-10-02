"""Рисунки главы 6. Запуск: .venv/bin/python figures/ch06_figures.py (из папки методички)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from morse_lidar.critical import CriticalType
from morse_lidar.descriptor import build_descriptor
from morse_lidar.mesh import periodic_grid_mesh

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
SADDLE = "#2A9D8F"
OUT = Path(__file__).resolve().parent


def gaussian_field(n: int = 64) -> np.ndarray:
    """Сумма периодических гауссиан на торе [0,1)^2 (три холма и две ямы)."""
    y, x = np.mgrid[0:n, 0:n] / n

    def bump(cx: float, cy: float, s: float, a: float) -> np.ndarray:
        dx = np.minimum(abs(x - cx), 1 - abs(x - cx))
        dy = np.minimum(abs(y - cy), 1 - abs(y - cy))
        return a * np.exp(-(dx**2 + dy**2) / (2 * s * s))

    return (
        bump(0.30, 0.30, 0.12, 1.0)
        + bump(0.72, 0.35, 0.10, 0.8)
        + bump(0.50, 0.75, 0.13, 0.9)
        - bump(0.15, 0.75, 0.08, 0.5)
        - bump(0.80, 0.80, 0.09, 0.6)
    )


def plot_polyline(ax, xy: np.ndarray, **style) -> None:
    """Рисует ломаную на торе, разрывая её там, где она переходит через край квадрата."""
    start = 0
    for i in range(1, len(xy)):
        if np.abs(xy[i] - xy[i - 1]).max() > 0.5:
            ax.plot(xy[start:i, 0], xy[start:i, 1], **style)
            start = i
    ax.plot(xy[start:, 0], xy[start:, 1], **style)


def separatrices_figure() -> None:
    n = 64
    field = gaussian_field(n)
    mesh = periodic_grid_mesh(field, (1 / n, 1 / n))
    descriptor = build_descriptor(mesh, euler_characteristic=0)
    xy = mesh.vertices[:, :2]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    gx, gy = np.meshgrid(np.arange(n) / n, np.arange(n) / n)
    ax.contourf(gx, gy, field, levels=24, cmap="Blues", alpha=0.35)
    ax.contour(gx, gy, field, levels=24, colors=SOFT, linewidths=0.5)
    for arc in descriptor.separatrices:
        color = NAVY if arc.kind == "u" else ACCENT
        plot_polyline(ax, xy[list(arc.vertices)], color=color, linewidth=1.8)
    style = {
        CriticalType.MINIMUM: dict(color=NAVY, marker="o", label="минимум"),
        CriticalType.SADDLE: dict(color=SADDLE, marker="X", label="седло"),
        CriticalType.MAXIMUM: dict(color=ACCENT, marker="^", label="максимум"),
    }
    for kind, params in style.items():
        pts = np.array([xy[p.vertex] for p in descriptor.critical_points if p.kind == kind])
        ax.scatter(pts[:, 0], pts[:, 1], s=90, edgecolor="white", linewidth=1.2, zorder=5,
                   c=params["color"], marker=params["marker"], label=params["label"])
    ax.plot([], [], color=NAVY, linewidth=1.8, label="сепаратриса u (вниз)")
    ax.plot([], [], color=ACCENT, linewidth=1.8, label="сепаратриса s (вверх)")
    ax.set_xlim(0, 1 - 1 / n)
    ax.set_ylim(0, 1 - 1 / n)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = descriptor.counts
    ax.set_title(f"trace_separatrices: {c['min']} min, {c['saddle']} saddle, {c['max']} max, "
                 f"{len(descriptor.separatrices)} дуги", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ch06_separatrices.png", dpi=200)
    plt.close(fig)


def quad_cell_figure() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    pts = {
        "min": np.array([0.0, 0.0]),
        "saddle_a": np.array([1.0, 1.0]),
        "max": np.array([0.0, 2.0]),
        "saddle_b": np.array([-1.0, 1.0]),
    }
    curve = 0.12

    def arc(saddle: str, extremum: str, color: str, label: str, side: float) -> None:
        """Дуга от седла к экстремуму; стрелка в сторону трассировки (_trace)."""
        p, q = pts[saddle], pts[extremum]
        mid = (p + q) / 2
        normal = np.array([-(q - p)[1], (q - p)[0]])
        normal = normal / np.linalg.norm(normal)
        if np.dot(normal, mid - np.array([0.0, 1.0])) < 0:
            normal = -normal  # выгибаем наружу ячейки
        t = np.linspace(0, 1, 60)[:, None]
        ctrl = mid + side * curve * normal
        path = (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t**2 * q
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5)
        k = len(path) // 2
        ax.annotate("", xy=path[k + 2], xytext=path[k - 2],
                    arrowprops=dict(arrowstyle="-|>", color=color, mutation_scale=18))
        ax.text(*(mid + 0.32 * normal), label, color=color, fontsize=13, ha="center", va="center")

    arc("saddle_a", "min", NAVY, "$u_a$", 1)
    arc("saddle_a", "max", ACCENT, "$s_a$", 1)
    arc("saddle_b", "max", ACCENT, "$s_b$", 1)
    arc("saddle_b", "min", NAVY, "$u_b$", 1)
    ax.plot([0, 0], [0, 2], color=TEAL, linestyle="--", linewidth=1.5)
    ax.text(0.08, 1.0, "диагональ $t$\n(гл. 7)", color=TEAL, fontsize=9, va="center")
    ax.fill([0, 1, 0, -1], [0, 1, 2, 1], color=LIGHT, alpha=0.5, zorder=0)
    labels = {"min": ("min", NAVY, "o", (0, -0.22)), "saddle_a": ("saddle_a", SADDLE, "X", (0.42, 0)),
              "max": ("max", ACCENT, "^", (0, 0.22)), "saddle_b": ("saddle_b", SADDLE, "X", (-0.42, 0))}
    for key, (text, color, marker, offset) in labels.items():
        ax.scatter(*pts[key], s=160, c=color, marker=marker, edgecolor="white", zorder=5)
        ax.text(*(pts[key] + np.array(offset)), text, ha="center", va="center", fontsize=11, color=INK)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-0.45, 2.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Ячейка комплекса Морса–Смейла: min – saddle_a – max – saddle_b", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ch06_quad_cell.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    separatrices_figure()
    quad_cell_figure()
