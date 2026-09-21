"""Intrinsic scalar fields computed directly on triangular surfaces."""

from __future__ import annotations

import numpy as np

from .mesh import TriMesh


def _corner_angle(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> float:
    left = first - center
    right = last - center
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    cosine = np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
    return float(np.arccos(cosine))


def gaussian_curvature(mesh: TriMesh) -> np.ndarray:
    """Estimate Gaussian curvature by angle deficit at each vertex."""
    angle_sum = np.zeros(len(mesh.vertices), dtype=float)
    for face in mesh.faces:
        first, center, last = (mesh.vertices[int(index)] for index in face)
        angle_sum[int(face[0])] += _corner_angle(center, first, last)
        angle_sum[int(face[1])] += _corner_angle(last, center, first)
        angle_sum[int(face[2])] += _corner_angle(first, last, center)
    target = np.full(len(mesh.vertices), 2.0 * np.pi)
    target[list(mesh.boundary_vertices)] = np.pi
    return target - angle_sum


def mean_curvature(mesh: TriMesh) -> np.ndarray:
    """Estimate mean-curvature magnitude with a cotangent Laplacian."""
    laplacian = np.zeros_like(mesh.vertices)
    mixed_area = np.zeros(len(mesh.vertices), dtype=float)
    for face in mesh.faces:
        indices = [int(index) for index in face]
        points = mesh.vertices[indices]
        edge_a, edge_b, edge_c = points[1] - points[0], points[2] - points[1], points[0] - points[2]
        area = 0.5 * np.linalg.norm(np.cross(edge_a, -edge_c))
        if area == 0:
            continue
        cotangents = (
            np.dot(points[1] - points[0], points[2] - points[0]) / np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])),
            np.dot(points[0] - points[1], points[2] - points[1]) / np.linalg.norm(np.cross(points[0] - points[1], points[2] - points[1])),
            np.dot(points[0] - points[2], points[1] - points[2]) / np.linalg.norm(np.cross(points[0] - points[2], points[1] - points[2])),
        )
        laplacian[indices[0]] += cotangents[1] * (points[2] - points[0]) + cotangents[2] * (points[1] - points[0])
        laplacian[indices[1]] += cotangents[2] * (points[0] - points[1]) + cotangents[0] * (points[2] - points[1])
        laplacian[indices[2]] += cotangents[0] * (points[1] - points[2]) + cotangents[1] * (points[0] - points[2])
        mixed_area[indices] += area / 3.0
    safe_area = np.maximum(mixed_area, np.finfo(float).eps)
    return 0.5 * np.linalg.norm(laplacian / safe_area[:, None], axis=1)


def shape_index(mesh: TriMesh) -> np.ndarray:
    """Estimate Koenderink shape index from mean and Gaussian curvature."""
    mean = mean_curvature(mesh)
    gaussian = gaussian_curvature(mesh)
    discriminant = np.sqrt(np.maximum(mean * mean - gaussian, 0.0))
    denominator = np.maximum(2.0 * discriminant, np.finfo(float).eps)
    return -(2.0 / np.pi) * np.arctan2(2.0 * mean, denominator)


def intrinsic_scalar(mesh: TriMesh, kind: str) -> np.ndarray:
    if kind == "height":
        return mesh.values.copy()
    if kind == "gaussian-curvature":
        return gaussian_curvature(mesh)
    if kind == "mean-curvature":
        return mean_curvature(mesh)
    if kind == "shape-index":
        return shape_index(mesh)
    raise ValueError(f"unknown scalar field: {kind}")
