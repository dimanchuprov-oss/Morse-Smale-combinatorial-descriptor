"""Small, explicit triangular mesh representation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TriMesh:
    """An oriented triangular mesh (open or closed) with one scalar per vertex."""

    vertices: np.ndarray
    faces: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if values.shape != (len(vertices),):
            raise ValueError("values must have one scalar per vertex")
        if not np.isfinite(vertices).all() or not np.isfinite(values).all():
            raise ValueError("vertices and values must be finite")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face index is outside vertices")
        # Own immutable buffers: frozen dataclasses alone do not freeze arrays.
        vertices = np.frombuffer(vertices.tobytes(), dtype=vertices.dtype).reshape(vertices.shape)
        faces = np.frombuffer(faces.tobytes(), dtype=faces.dtype).reshape(faces.shape)
        values = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "values", values)

    # Topology queries are cached: the arrays are fixed after construction and
    # these properties are read inside per-vertex loops.
    @cached_property
    def edge_counts(self) -> Mapping[tuple[int, int], int]:
        edges = np.sort(self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        return MappingProxyType({(int(left), int(right)): int(count) for (left, right), count in zip(unique, counts)})

    @cached_property
    def closed(self) -> bool:
        return bool(self.edge_counts) and all(count == 2 for count in self.edge_counts.values())

    @cached_property
    def boundary_edges(self) -> frozenset[tuple[int, int]]:
        return frozenset(edge for edge, count in self.edge_counts.items() if count == 1)

    @cached_property
    def boundary_vertices(self) -> frozenset[int]:
        return frozenset(vertex for edge in self.boundary_edges for vertex in edge)

    @cached_property
    def euler_characteristic(self) -> int:
        return len(self.vertices) - len(self.edge_counts) + len(self.faces)

    @cached_property
    def vertex_areas(self) -> np.ndarray:
        """Barycentric area (one third of each incident triangle) per vertex."""
        corners = self.vertices[self.faces]
        areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        result = np.zeros(len(self.vertices), dtype=float)
        np.add.at(result, self.faces.ravel(), np.repeat(areas / 3.0, 3))
        return result

    @cached_property
    def vertex_normals(self) -> np.ndarray:
        """Area-weighted unit normals following the face orientation."""
        corners = self.vertices[self.faces]
        face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        result = np.zeros_like(self.vertices)
        for corner in range(3):
            np.add.at(result, self.faces[:, corner], face_normals)
        lengths = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(lengths, np.finfo(float).eps)

    @cached_property
    def order(self) -> np.ndarray:
        """Rank of every vertex under simulation of simplicity (value, then index)."""
        ranks = np.empty(len(self.values), dtype=np.int64)
        ranks[np.lexsort((np.arange(len(self.values)), self.values))] = np.arange(len(self.values))
        return ranks

    def neighbors_and_link(
        self,
    ) -> tuple[tuple[frozenset[int], ...], tuple[frozenset[tuple[int, int]], ...]]:
        """Per-vertex neighbours and link edges (cached, immutable)."""
        return self._neighbors_and_link

    @cached_property
    def _neighbors_and_link(
        self,
    ) -> tuple[tuple[frozenset[int], ...], tuple[frozenset[tuple[int, int]], ...]]:
        neighbors = [set() for _ in self.vertices]
        link_edges = [set() for _ in self.vertices]
        for face in self.faces:
            for vertex in face:
                others = [int(item) for item in face if item != vertex]
                if len(others) != 2:
                    raise ValueError("faces must not repeat a vertex")
                first, second = others
                neighbors[int(vertex)].update((first, second))
                link_edges[int(vertex)].add(tuple(sorted((first, second))))
        return tuple(map(frozenset, neighbors)), tuple(map(frozenset, link_edges))


def _grid_vertices(field: np.ndarray, spacing: tuple[float, float], origin: tuple[float, float]) -> np.ndarray:
    if len(spacing) != 2 or not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("spacing must contain two positive finite values")
    rows, cols = field.shape
    y, x = np.mgrid[0:rows, 0:cols]
    return np.column_stack(
        (origin[0] + spacing[0] * x.ravel(), origin[1] + spacing[1] * y.ravel(), field.ravel())
    ).astype(float)


def _grid_faces(rows: int, cols: int, periodic: bool) -> np.ndarray:
    y_count, x_count = (rows, cols) if periodic else (rows - 1, cols - 1)
    y, x = np.mgrid[0:y_count, 0:x_count]
    y, x = y.ravel(), x.ravel()

    def index(row: np.ndarray, col: np.ndarray) -> np.ndarray:
        return (row % rows) * cols + (col % cols)

    a, b, c, d = index(y, x), index(y, x + 1), index(y + 1, x), index(y + 1, x + 1)
    faces = np.empty((2 * len(a), 3), dtype=np.int64)
    faces[0::2] = np.column_stack((a, b, d))
    faces[1::2] = np.column_stack((a, d, c))
    return faces


def _validated_field(height: np.ndarray) -> np.ndarray:
    field = np.asarray(height, dtype=float)
    if field.ndim != 2 or min(field.shape) < 3:
        raise ValueError("height must be a 2D array with at least 3 rows/columns")
    return field


def periodic_grid_mesh(
    height: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
) -> TriMesh:
    """Create a closed toroidal triangulation from a 2D periodic height map.

    ``spacing`` is the (x, y) cell size in the same unit as the heights, so
    that curvature is computed on a metrically correct surface.
    """
    field = _validated_field(height)
    rows, cols = field.shape
    return TriMesh(_grid_vertices(field, spacing, origin), _grid_faces(rows, cols, periodic=True), field.ravel())


def grid_mesh(
    height: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
) -> TriMesh:
    """Create a triangulated rectangular height surface with an open boundary."""
    field = _validated_field(height)
    rows, cols = field.shape
    return TriMesh(_grid_vertices(field, spacing, origin), _grid_faces(rows, cols, periodic=False), field.ravel())
