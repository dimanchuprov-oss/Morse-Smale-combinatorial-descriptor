"""Small, explicit triangular mesh representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriMesh:
    """A closed oriented triangular mesh with one scalar value per vertex."""

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
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "values", values)

    @property
    def closed(self) -> bool:
        edges: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for left, right in zip(face, np.roll(face, -1)):
                edge = tuple(sorted((int(left), int(right))))
                edges[edge] = edges.get(edge, 0) + 1
        return bool(edges) and all(count == 2 for count in edges.values())

    @property
    def boundary_edges(self) -> set[tuple[int, int]]:
        edges: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for left, right in zip(face, np.roll(face, -1)):
                edge = tuple(sorted((int(left), int(right))))
                edges[edge] = edges.get(edge, 0) + 1
        return {edge for edge, count in edges.items() if count == 1}

    @property
    def boundary_vertices(self) -> set[int]:
        return {vertex for edge in self.boundary_edges for vertex in edge}

    @property
    def euler_characteristic(self) -> int:
        edges = {
            tuple(sorted((int(left), int(right))))
            for face in self.faces
            for left, right in zip(face, np.roll(face, -1))
        }
        return len(self.vertices) - len(edges) + len(self.faces)

    def neighbors_and_link(self) -> tuple[list[set[int]], list[set[tuple[int, int]]]]:
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
        return neighbors, link_edges


def periodic_grid_mesh(height: np.ndarray) -> TriMesh:
    """Create a closed toroidal triangulation from a 2D periodic height map."""
    field = np.asarray(height, dtype=float)
    if field.ndim != 2 or min(field.shape) < 3:
        raise ValueError("height must be a 2D array with at least 3 rows/columns")
    rows, cols = field.shape
    vertices = np.array(
        [[float(x), float(y), float(field[y, x])] for y in range(rows) for x in range(cols)]
    )

    def index(y: int, x: int) -> int:
        return (y % rows) * cols + (x % cols)

    faces: list[tuple[int, int, int]] = []
    for y in range(rows):
        for x in range(cols):
            a, b, c, d = index(y, x), index(y, x + 1), index(y + 1, x), index(y + 1, x + 1)
            faces.extend(((a, b, d), (a, d, c)))
    return TriMesh(vertices, np.asarray(faces, dtype=np.int64), field.ravel())


def grid_mesh(height: np.ndarray) -> TriMesh:
    """Create a triangulated rectangular height surface with an open boundary."""
    field = np.asarray(height, dtype=float)
    if field.ndim != 2 or min(field.shape) < 3:
        raise ValueError("height must be a 2D array with at least 3 rows/columns")
    rows, cols = field.shape
    vertices = np.array(
        [[float(x), float(y), float(field[y, x])] for y in range(rows) for x in range(cols)]
    )

    def index(y: int, x: int) -> int:
        return y * cols + x

    faces: list[tuple[int, int, int]] = []
    for y in range(rows - 1):
        for x in range(cols - 1):
            a, b, c, d = index(y, x), index(y, x + 1), index(y + 1, x), index(y + 1, x + 1)
            faces.extend(((a, b, d), (a, d, c)))
    return TriMesh(vertices, np.asarray(faces, dtype=np.int64), field.ravel())
