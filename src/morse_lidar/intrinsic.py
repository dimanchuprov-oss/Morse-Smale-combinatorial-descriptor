"""Pointwise curvature scalar fields computed directly on triangular surfaces.

All fields are per unit area (Gaussian curvature in 1/length^2, mean curvature
in 1/length), so values do not depend on mesh density. Mean curvature is signed
with respect to the face orientation: a dome whose normals point away from its
centre of curvature has ``H > 0``; for height fields built by this package the
normals point towards +Z.
"""

from __future__ import annotations

import numpy as np

from .mesh import TriMesh


def _corner_geometry(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return corner points, corner angles and corner cotangents per face."""
    corners = mesh.vertices[mesh.faces]
    angles = np.empty(mesh.faces.shape, dtype=float)
    cotangents = np.empty(mesh.faces.shape, dtype=float)
    for corner in range(3):
        first = corners[:, (corner + 1) % 3] - corners[:, corner]
        second = corners[:, (corner + 2) % 3] - corners[:, corner]
        sine = np.linalg.norm(np.cross(first, second), axis=1)
        cosine = np.einsum("ij,ij->i", first, second)
        angles[:, corner] = np.arctan2(sine, cosine)
        cotangents[:, corner] = cosine / np.maximum(sine, np.finfo(float).eps)
    return corners, angles, cotangents


def _safe_area(mesh: TriMesh) -> np.ndarray:
    return np.maximum(mesh.vertex_areas, np.finfo(float).eps)


def _extend_to_boundary(mesh: TriMesh, values: np.ndarray) -> np.ndarray:
    """Replace boundary estimates by the mean of their interior neighbours.

    Discrete curvature operators are not defined on the rim of an open scan and
    otherwise produce large artificial extrema there.
    """
    boundary = mesh.boundary_vertices
    if not boundary:
        return values
    neighbors, _ = mesh.neighbors_and_link()
    result = values.copy()
    for vertex in boundary:
        interior = [item for item in neighbors[vertex] if item not in boundary]
        if interior:
            result[vertex] = float(np.mean(values[interior]))
    return result


def _raw_gaussian_curvature(mesh: TriMesh, angles: np.ndarray) -> np.ndarray:
    angle_sum = np.zeros(len(mesh.vertices), dtype=float)
    np.add.at(angle_sum, mesh.faces.ravel(), angles.ravel())
    target = np.full(len(mesh.vertices), 2.0 * np.pi)
    target[list(mesh.boundary_vertices)] = np.pi
    return (target - angle_sum) / _safe_area(mesh)


def _raw_mean_curvature(mesh: TriMesh, corners: np.ndarray, cotangents: np.ndarray) -> np.ndarray:
    laplacian = np.zeros_like(mesh.vertices)
    for corner in range(3):
        following, previous = (corner + 1) % 3, (corner + 2) % 3
        # Edge (corner, following) is opposite ``previous`` and vice versa.
        contribution = cotangents[:, previous, None] * (corners[:, following] - corners[:, corner]) + cotangents[
            :, following, None
        ] * (corners[:, previous] - corners[:, corner])
        np.add.at(laplacian, mesh.faces[:, corner], contribution)
    # Laplace-Beltrami of the embedding: laplacian / (2A) = -2 H n.
    return -np.einsum("ij,ij->i", laplacian, mesh.vertex_normals) / (4.0 * _safe_area(mesh))


def gaussian_curvature(mesh: TriMesh) -> np.ndarray:
    """Pointwise Gaussian curvature: angle deficit divided by barycentric area."""
    _, angles, _ = _corner_geometry(mesh)
    return _extend_to_boundary(mesh, _raw_gaussian_curvature(mesh, angles))


def mean_curvature(mesh: TriMesh) -> np.ndarray:
    """Signed pointwise mean curvature from the cotangent Laplacian."""
    corners, _, cotangents = _corner_geometry(mesh)
    return _extend_to_boundary(mesh, _raw_mean_curvature(mesh, corners, cotangents))


def shape_index(mesh: TriMesh) -> np.ndarray:
    """Koenderink shape index in [-1, 1]: +1 dome (cap), 0 saddle, -1 bowl (cup).

    Planar vertices (H = K = 0) get 0.
    """
    corners, angles, cotangents = _corner_geometry(mesh)
    mean = _raw_mean_curvature(mesh, corners, cotangents)
    gaussian = _raw_gaussian_curvature(mesh, angles)
    # k1,2 = H +- sqrt(H^2 - K); S = (2/pi) arctan((k1 + k2) / (k1 - k2)).
    discriminant = np.sqrt(np.maximum(mean * mean - gaussian, 0.0))
    return _extend_to_boundary(mesh, (2.0 / np.pi) * np.arctan2(mean, discriminant))


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
