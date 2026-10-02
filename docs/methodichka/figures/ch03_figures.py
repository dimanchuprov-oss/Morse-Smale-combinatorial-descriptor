"""Рисунки к главе 3. Запуск: .venv/bin/python figures/ch03_figures.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from morse_lidar import CriticalType, analyze_morse, critical_pairs, periodic_grid_mesh, persistent_counts

NAVY, CYAN, TEAL, LIGHT, INK, SOFT, ACCENT = "#355DA6", "#0498BB", "#00A0CB", "#C0F6FF", "#2B4C8C", "#498BCD", "#E4572E"
SADDLE = "#2A9D8F"
KIND_STYLE = {
    CriticalType.MINIMUM: (NAVY, "o", "минимум"),
    CriticalType.SADDLE: (SADDLE, "X", "седло"),
    CriticalType.MAXIMUM: (ACCENT, "^", "максимум"),
}
OUT = Path(__file__).resolve().parent
N = 32


def torus_field(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    field = np.cos(X) + np.cos(Y)
    if noise:
        field = field + noise * np.random.default_rng(seed).standard_normal(field.shape)
    return field


def torus_figure() -> None:
    step = 2 * np.pi / N
    fields = (torus_field(), torus_field(0.05))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, field, title in zip(axes, fields, ("cos x + cos y", "cos x + cos y + шум σ=0.05")):
        mesh = periodic_grid_mesh(field, spacing=(step, step))
        image = ax.imshow(field, origin="lower", extent=(-step / 2, 2 * np.pi - step / 2) * 2, cmap="Blues")
        points = analyze_morse(mesh)
        counts = {kind: 0 for kind in KIND_STYLE}
        for kind, (color, marker, label) in KIND_STYLE.items():
            chosen = [p for p in points if p.kind == kind]
            counts[kind] = len(chosen)
            if chosen:
                xy = mesh.vertices[[p.vertex for p in chosen]][:, :2]
                ax.scatter(*xy.T, s=70, color=color, marker=marker, edgecolor="white", linewidth=0.8, label=label,
                           zorder=3)
        ax.set_title(f"{title}\nmin={counts[CriticalType.MINIMUM]}, saddle={counts[CriticalType.SADDLE]}, "
                     f"max={counts[CriticalType.MAXIMUM]}, χ={mesh.euler_characteristic}", fontsize=10)
        ax.set_xlabel("x"), ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.85)
    axes[0].legend(fontsize=8, loc="upper center", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "ch03_torus.png", dpi=200)
    plt.close(fig)


def persistent_figure() -> None:
    step = 2 * np.pi / N
    mesh = periodic_grid_mesh(torus_field(0.05), spacing=(step, step))
    points = analyze_morse(mesh)
    pairs = critical_pairs(mesh)
    thresholds = np.linspace(0, 0.4, 81)
    rows = [persistent_counts(points, pairs, float(t)) for t in thresholds]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.step(thresholds, [r["min"] for r in rows], where="post", color=NAVY, label="min")
    ax.step(thresholds, [r["saddle"] for r in rows], where="post", color=SADDLE, label="saddle")
    ax.step(thresholds, [r["max"] for r in rows], where="post", color=ACCENT, label="max")
    ax.step(thresholds, [r["min"] - r["saddle"] + r["max"] for r in rows], where="post", color=INK,
            linestyle="--", label="min − saddle + max")
    ax.set_xlabel("порог персистентности")
    ax.set_ylabel("число точек")
    ax.set_title("persistent_counts на зашумлённом торе: баланс остаётся равным χ = 0", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ch03_persistent.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    torus_figure()
    persistent_figure()
