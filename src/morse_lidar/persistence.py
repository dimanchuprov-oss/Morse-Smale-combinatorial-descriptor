"""GUDHI-backed cubical persistence for regular LiDAR depth maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from .mesh import TriMesh


@dataclass(frozen=True)
class PersistenceInterval:
    dimension: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return inf if np.isinf(self.death) else self.death - self.birth


@dataclass(frozen=True)
class PersistenceReport:
    backend: str
    intervals: tuple[PersistenceInterval, ...]

    def significant(self, threshold: float, dimension: int | None = None) -> tuple[PersistenceInterval, ...]:
        if threshold < 0:
            raise ValueError("persistence threshold must be non-negative")
        return tuple(
            interval
            for interval in self.intervals
            if (dimension is None or interval.dimension == dimension)
            and interval.persistence >= threshold
        )


def _require_gudhi():
    try:
        import gudhi
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional topology extra: python -m pip install '.[topology]'") from error
    return gudhi


def cubical_persistence(depth: np.ndarray) -> PersistenceReport:
    """Compute sublevel-set persistence intervals with GUDHI.

    The returned intervals describe the scalar field on the regular cubical
    complex. Minima are represented by dimension-0 intervals. Running this
    function on ``-depth`` gives the corresponding maxima filtration.
    """
    field = np.asarray(depth, dtype=float)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError("depth must be a finite 2D array")
    gudhi = _require_gudhi()
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=field.tolist())
    intervals = []
    for dimension, pairs in complex_.persistence():
        birth, death = pairs
        intervals.append(PersistenceInterval(int(dimension), float(birth), float(death)))
    return PersistenceReport("gudhi-cubical", tuple(intervals))


def depth_persistence(depth: np.ndarray) -> dict[str, PersistenceReport]:
    """Compute both minimum and maximum persistence reports."""
    field = np.asarray(depth, dtype=float)
    return {"minima": cubical_persistence(field), "maxima": cubical_persistence(-field)}


def simplicial_persistence(mesh: TriMesh) -> PersistenceReport:
    """Compute lower-star persistence for a triangular mesh scalar field."""
    gudhi = _require_gudhi()
    if not np.isfinite(mesh.values).all():
        raise ValueError("mesh scalar values must be finite")
    complex_ = gudhi.SimplexTree()
    for vertex, value in enumerate(mesh.values):
        complex_.insert([int(vertex)], filtration=float(value))
    for face in mesh.faces:
        vertices = [int(vertex) for vertex in face]
        complex_.insert(vertices, filtration=float(np.max(mesh.values[vertices])))
    complex_.make_filtration_non_decreasing()
    intervals = [
        PersistenceInterval(int(dimension), float(pair[0]), float(pair[1]))
        for dimension, pair in complex_.persistence()
    ]
    return PersistenceReport("gudhi-simplicial", tuple(intervals))


def mesh_persistence(mesh: TriMesh) -> dict[str, PersistenceReport]:
    """Compute persistence for minima and maxima of a mesh scalar field."""
    inverted = TriMesh(mesh.vertices, mesh.faces, -mesh.values)
    return {"minima": simplicial_persistence(mesh), "maxima": simplicial_persistence(inverted)}
