"""Experimental M4: find a nose and forehead from an unlabelled face cloud.

Input is a cropped, upright head in millimetres (as returned by
``face.extract_head``). Only XYZ is used: no mesh indices, neutral template,
identity, camera pose or Multiface labels. This first version targets frontal
snapshots with both sides of the nose visible, not arbitrary full heads.

A horizontal PCA estimates the face plane. Its sign follows the convexity
of the face, and a locally smoothed prominence finds the nose. A robust plane
through the surrounding skin corrects small yaw/pitch errors. Conservative
metric windows above that landmark select the nose/bridge and forehead.
Missing support raises an error; there is no silent full-face fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .face import _kd_tree

REGION_VERSION = 2


@dataclass(frozen=True)
class FaceRegions:
    nose_tip: np.ndarray
    # Columns: lateral, up, forward; coordinates = (points - nose_tip) @ axes.
    axes: np.ndarray
    nose: np.ndarray
    forehead: np.ndarray
    prominence_mm: float

    @property
    def mask(self) -> np.ndarray:
        return self.nose | self.forehead

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.nose_tip) @ self.axes

    def as_dict(self) -> dict[str, Any]:
        return {"version": REGION_VERSION, "nose_tip_mm": self.nose_tip.tolist(),
                "axes": self.axes.tolist(), "prominence_mm": self.prominence_mm,
                "nose_points": int(self.nose.sum()), "forehead_points": int(self.forehead.sum()),
                "selected_points": int(self.mask.sum()), "input_points": len(self.mask)}


def _frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initial face axes; infer front/back from bilateral convexity, not axis signs."""
    top = np.quantile(points[:, 2], 0.99)
    band = points[(points[:, 2] > top - 140.0) & (points[:, 2] < top - 20.0)]
    if len(band) < 100:
        raise ValueError("M4: insufficient face surface to estimate orientation")
    # A mean commutes with rotations; a coordinate-wise median does not.
    centre = band.mean(axis=0)
    horizontal = band[:, :2] - centre[:2]
    _, vectors = np.linalg.eigh(horizontal.T @ horizontal / len(band))
    forward = np.append(vectors[:, 0], 0.0)
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, forward)
    across, depth = np.round((band - centre) @ lateral, 9), (band - centre) @ forward
    middle = np.abs(across) < 15.0
    sides = (np.abs(across) > 35.0) & (np.abs(across) < 65.0)
    if middle.sum() < 30 or sides.sum() < 30:
        raise ValueError("M4: both sides of a frontal face must be visible")
    convexity = float(np.median(depth[middle]) - np.median(depth[sides]))
    if abs(convexity) < 3.0:
        raise ValueError("M4: no clear convex face surface")
    if convexity < 0:
        forward, lateral = -forward, -lateral
    return centre, np.column_stack((lateral, up, forward))


def _skin_plane(local: np.ndarray) -> np.ndarray:
    """Robust depth plane around the nose; exclude its protruding central ridge."""
    # Stabilise strict metric window edges under rigid transforms. Nanometre
    # rounding is far below scan precision but prevents whole grid rows from
    # entering/leaving the fit due to floating-point noise.
    local = np.round(local, 9)
    x, y, _ = local.T
    surrounding = (np.abs(x) > 20.0) & (np.abs(x) < 50.0) & (y > -10.0) & (y < 60.0)
    if surrounding.sum() < 60:
        raise ValueError("M4: insufficient skin around the nose")
    skin = local[surrounding]
    design = np.column_stack((skin[:, :2], np.ones(len(skin))))
    weights = np.ones(len(skin))
    for _ in range(5):
        root = np.sqrt(weights)
        coefficient, *_ = np.linalg.lstsq(design * root[:, None], skin[:, 2] * root, rcond=None)
        residual = skin[:, 2] - design @ coefficient
        scale = max(1.0, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
    # A large correction means the frontal-snapshot assumption did not hold.
    if np.linalg.norm(coefficient[:2]) > np.tan(np.radians(30.0)):
        raise ValueError("M4: face tilt exceeds the supported frontal range")
    return coefficient


def locate_regions(points: np.ndarray) -> FaceRegions:
    """Find rigid regions on an upright, cropped face, sampled at about 2 mm.

    The returned masks index the input array. The input is never modified.
    Nose/forehead sizes are fixed anatomical priors, not fitted to identities
    or matching scores. The detector must still be validated on real scans.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 150 or not np.isfinite(points).all():
        raise ValueError("M4: need at least 150 finite XYZ points in millimetres")
    extent = np.quantile(points[:, 2], 0.99) - np.quantile(points[:, 2], 0.01)
    if not 100.0 <= extent <= 300.0:
        raise ValueError("M4: expected a cropped upright head in millimetres")
    centre, axes = _frame(points)
    projected = np.round((points - centre) @ axes, 9)
    x, y, depth = projected.T
    # Median smoothing in a bounded neighbourhood suppresses isolated depth spikes.
    tree = _kd_tree(projected[:, :2])
    distance, index = tree.query(projected[:, :2], k=9)
    smooth = np.median(depth[index], axis=1)
    top = np.quantile(y, 0.99)
    candidates = np.flatnonzero((np.abs(x) < 25.0) & (y > top - 145.0) & (y < top - 30.0)
                                & (distance[:, -1] < 9.0))
    best: tuple[float, int] | None = None
    for i in candidates:
        neighbours = np.asarray(tree.query_ball_point(projected[i, :2], 35.0), dtype=int)
        dx, dy = (projected[neighbours, :2] - projected[i, :2]).T
        left = (dx < -14.0) & (dx > -32.0) & (np.abs(dy) < 10.0)
        right = (dx > 14.0) & (dx < 32.0) & (np.abs(dy) < 10.0)
        if min(left.sum(), right.sum()) < 8:
            continue
        # Both sides must recede. This rejects a steep cheek or a scan boundary.
        prominence = float(smooth[i] - max(np.median(smooth[neighbours[left]]),
                                           np.median(smooth[neighbours[right]])))
        if best is None or prominence > best[0]:
            best = prominence, int(i)
    if best is None or best[0] < 5.0:
        raise ValueError("M4: no supported nose prominence found")
    prominence, tip_index = best
    # Use a small cap centroid for sub-sample stability, without following outliers.
    cap = (np.linalg.norm(projected[:, :2] - projected[tip_index, :2], axis=1) < 4.0 - 1e-8)
    cap &= np.abs(depth - smooth[tip_index]) < 4.0
    if not cap.any():
        raise ValueError("M4: nose tip has no consistent local surface")
    tip = points[cap].mean(axis=0)
    local = (points - tip) @ axes
    plane = _skin_plane(local)
    forward = axes @ np.array([-plane[0], -plane[1], 1.0])
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    up -= forward * np.dot(up, forward)
    up /= np.linalg.norm(up)
    lateral = np.cross(up, forward)
    axes = np.column_stack((lateral, up, forward))
    x, y, depth = np.round((points - tip) @ axes, 9).T
    nose = (np.abs(x) < 18.0) & (y > -8.0) & (y < 55.0) & (depth > -40.0) & (depth < 8.0)
    # Stay out of the eyebrows, retaining the narrow bridge between them.
    nose &= (y < 35.0) | (np.abs(x) < 6.0)
    forehead = (np.abs(x) < 44.0) & (y > 58.0) & (y < 95.0) & (depth > -75.0) & (depth < 8.0)
    if nose.sum() < 50:
        raise ValueError("M4: insufficient visible nose surface")
    if forehead.sum() < 50:
        raise ValueError("M4: insufficient visible forehead surface")
    return FaceRegions(tip, axes, nose, forehead, prominence)


def plot_regions(entries: dict[str, tuple[np.ndarray, FaceRegions | str]], out: Path) -> list[str]:
    """Paginated diagnostics; failed detections remain visible in the report."""
    from .face_cli import _pyplot

    plt = _pyplot()
    names = list(entries)
    written = []
    for start in range(0, len(names), 20):
        page = names[start:start + 20]
        fig, axes = plt.subplots(int(np.ceil(len(page) / 5)), 5, figsize=(15, 3.5 * np.ceil(len(page) / 5)),
                                 squeeze=False)
        for axis, name in zip(axes.ravel(), page):
            points, region = entries[name]
            if isinstance(region, str):
                if len(points):
                    try:
                        centre, frame = _frame(points)
                        local = (points - centre) @ frame
                        axis.scatter(local[:, 0], local[:, 1], s=1, color="#d5dbe6", rasterized=True)
                        axis.set_aspect("equal")
                    except ValueError:
                        pass
                axis.text(0.02, -0.04, region.replace("ValueError: M4: ", ""), transform=axis.transAxes,
                          fontsize=7, wrap=True)
                axis.set_title(name, fontsize=8, color="#b32929")
            else:
                local = region.coordinates(points)
                axis.scatter(local[:, 0], local[:, 1], s=1, color="#d5dbe6", rasterized=True)
                for mask, colour in ((region.nose, "#355DA6"), (region.forehead, "#2A9D8F")):
                    axis.scatter(local[mask, 0], local[mask, 1], s=2, color=colour, rasterized=True)
                axis.scatter([0], [0], marker="+", s=35, color="#E4572E")
                axis.set_title(name, fontsize=8)
                axis.set_aspect("equal")
            axis.axis("off")
        for axis in axes.ravel()[len(page):]:
            axis.axis("off")
        fig.suptitle("M4: nose / bridge (blue), forehead (green), tip (+)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        name = f"auto_regions_{start // 20 + 1:02d}.png"
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        written.append(name)
    return written
