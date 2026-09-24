"""GUDHI-backed persistence for LiDAR depth maps and triangular meshes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf

import numpy as np

from .critical import CriticalPoint, CriticalType
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


def _lower_star_tree(mesh: TriMesh, filtration: np.ndarray):
    """Lower-star simplex tree: every simplex gets the max of its vertex values.

    Edges must be inserted before triangles; otherwise GUDHI creates them
    implicitly with the triangle's (larger) filtration value.
    """
    gudhi = _require_gudhi()
    tree = gudhi.SimplexTree()
    for vertex, value in enumerate(filtration):
        tree.insert([int(vertex)], filtration=float(value))
    for left, right in mesh.edge_counts:
        tree.insert([left, right], filtration=float(max(filtration[left], filtration[right])))
    for face in mesh.faces:
        vertices = [int(vertex) for vertex in face]
        tree.insert(vertices, filtration=float(np.max(filtration[vertices])))
    return tree


def simplicial_persistence(mesh: TriMesh) -> PersistenceReport:
    """Compute lower-star persistence for a triangular mesh scalar field."""
    if not np.isfinite(mesh.values).all():
        raise ValueError("mesh scalar values must be finite")
    tree = _lower_star_tree(mesh, mesh.values)
    intervals = [
        PersistenceInterval(int(dimension), float(pair[0]), float(pair[1]))
        for dimension, pair in tree.persistence(persistence_dim_max=True)
    ]
    return PersistenceReport("gudhi-simplicial", tuple(intervals))


def mesh_persistence(mesh: TriMesh) -> dict[str, PersistenceReport]:
    """Compute persistence for minima and maxima of a mesh scalar field."""
    inverted = TriMesh(mesh.vertices, mesh.faces, -mesh.values)
    return {"minima": simplicial_persistence(mesh), "maxima": simplicial_persistence(inverted)}


@dataclass(frozen=True)
class CriticalPair:
    """A persistence pair expressed by the critical vertices that create it.

    In a lower-star filtration of a surface, dimension 0 pairs a minimum with
    a saddle, dimension 1 pairs a saddle with a maximum, and essential classes
    (``death_vertex is None``) are the global minimum, the genus loops and, on
    a closed surface, the global maximum.
    """

    dimension: int
    birth_vertex: int
    death_vertex: int | None
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return inf if self.death_vertex is None else self.death - self.birth


def critical_pairs(mesh: TriMesh) -> tuple[CriticalPair, ...]:
    """Persistence pairs whose vertices match :func:`analyze_morse`.

    The filtration uses the mesh's simulation-of-simplicity rank instead of
    raw values, so ties on plateaus are broken exactly as the critical point
    classifier breaks them; persistence is then reported in scalar units.
    """
    order = mesh.order
    tree = _lower_star_tree(mesh, order.astype(float))
    tree.persistence(homology_coeff_field=2, min_persistence=0, persistence_dim_max=True)
    pairs: list[CriticalPair] = []
    for birth_simplex, death_simplex in tree.persistence_pairs():
        birth_vertex = max(birth_simplex, key=lambda vertex: order[vertex])
        if death_simplex:
            death_vertex = int(max(death_simplex, key=lambda vertex: order[vertex]))
            if death_vertex == birth_vertex:
                continue
            death = float(mesh.values[death_vertex])
        else:
            death_vertex, death = None, inf
        pairs.append(
            CriticalPair(len(birth_simplex) - 1, int(birth_vertex), death_vertex, float(mesh.values[birth_vertex]), death)
        )
    return tuple(pairs)


def persistence_diagram(pairs: tuple[CriticalPair, ...]) -> dict[int, list[tuple[float, float]]]:
    """Group pairs into (birth, death) diagrams by homological dimension."""
    diagram: dict[int, list[tuple[float, float]]] = {}
    for pair in pairs:
        diagram.setdefault(pair.dimension, []).append((pair.birth, pair.death))
    return {dimension: sorted(points) for dimension, points in sorted(diagram.items())}


def annotate_persistence(
    points: tuple[CriticalPoint, ...] | list[CriticalPoint],
    pairs: tuple[CriticalPair, ...],
) -> tuple[CriticalPoint, ...]:
    """Attach to each critical point the largest persistence of its pairs.

    Regular vertices are returned unchanged. A critical vertex found in no
    pair gets persistence 0 (it only pairs with itself up to ties).
    """
    best: dict[int, float] = {}
    for pair in pairs:
        for vertex in (pair.birth_vertex, pair.death_vertex):
            if vertex is not None:
                best[vertex] = max(best.get(vertex, 0.0), pair.persistence)
    return tuple(
        point
        if point.kind == CriticalType.REGULAR or point.vertex < 0
        else replace(point, persistence=best.get(point.vertex, 0.0))
        for point in points
    )


def persistent_counts(
    points: tuple[CriticalPoint, ...] | list[CriticalPoint],
    pairs: tuple[CriticalPair, ...],
    threshold: float,
) -> dict[str, int]:
    """Count critical points that survive a persistence threshold.

    Every critical vertex belongs to as many pairs as its multiplicity (a
    k-fold saddle to k pairs). Saddles are weighted by their significant
    pairs, extrema are counted once. On a closed surface the result therefore
    keeps ``min - saddle + max`` equal to the Euler characteristic.

    This removes critical points of low-persistence pairs but does not reroute
    separatrices, i.e. it is not a full Morse-Smale cancellation.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("persistence threshold must be a finite non-negative number")
    significant: dict[int, int] = {}
    for pair in pairs:
        if pair.persistence >= threshold:
            for vertex in (pair.birth_vertex, pair.death_vertex):
                if vertex is not None:
                    significant[vertex] = significant.get(vertex, 0) + 1
    counts = {kind.value: 0 for kind in CriticalType if kind != CriticalType.REGULAR}
    for point in points:
        if point.kind == CriticalType.REGULAR:
            continue
        hits = significant.get(point.vertex, 0)
        counts[point.kind.value] += hits if point.kind == CriticalType.SADDLE else min(hits, 1)
    return counts
