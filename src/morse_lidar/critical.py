"""Critical point classification using lower/upper link components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .mesh import TriMesh


class CriticalType(str, Enum):
    MINIMUM = "min"
    REGULAR = "regular"
    SADDLE = "saddle"
    MAXIMUM = "max"


@dataclass(frozen=True)
class CriticalPoint:
    vertex: int
    kind: CriticalType
    lower_components: int
    upper_components: int
    saddle_multiplicity: int = 0
    identifier: int | None = None
    position: tuple[float, float, float] | None = None
    cell_id: int | None = None
    cell_dimension: int | None = None
    scalar: float | None = None
    is_on_boundary: bool = False
    backend: str = "lower-upper-link"


def _components(vertices: set[int], link_edges: set[tuple[int, int]]) -> int:
    if not vertices:
        return 0
    graph = {vertex: set() for vertex in vertices}
    for left, right in link_edges:
        if left in vertices and right in vertices:
            graph[left].add(right)
            graph[right].add(left)
    count = 0
    unseen = set(vertices)
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in graph[current] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
    return count


def analyze_morse(mesh: TriMesh, include_boundary: bool = False) -> list[CriticalPoint]:
    """Classify vertices, masking open-surface boundary vertices by default."""
    neighbors, link_edges = mesh.neighbors_and_link()
    order = np.empty(len(mesh.values), dtype=np.int64)
    sorted_vertices = np.lexsort((np.arange(len(mesh.values)), mesh.values))
    order[sorted_vertices] = np.arange(len(mesh.values))
    points: list[CriticalPoint] = []
    for vertex, adjacent in enumerate(neighbors):
        if not include_boundary and vertex in mesh.boundary_vertices:
            continue
        lower = {item for item in adjacent if order[item] < order[vertex]}
        upper = adjacent - lower
        lower_count = _components(lower, link_edges[vertex])
        upper_count = _components(upper, link_edges[vertex])
        if lower_count == 0:
            kind = CriticalType.MINIMUM
        elif upper_count == 0:
            kind = CriticalType.MAXIMUM
        elif lower_count >= 2 or upper_count >= 2:
            kind = CriticalType.SADDLE
        else:
            kind = CriticalType.REGULAR
        points.append(CriticalPoint(vertex, kind, lower_count, upper_count, max(lower_count, upper_count) - 1))
    return points
