"""Critical point classification using lower/upper link components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    # Largest persistence of a pair this vertex belongs to; None if not computed.
    persistence: float | None = None


def link_components(vertices: set[int], link_edges: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> list[set[int]]:
    """Split a subset of a vertex link into connected components."""
    graph = {vertex: set() for vertex in vertices}
    for left, right in link_edges:
        if left in vertices and right in vertices:
            graph[left].add(right)
            graph[right].add(left)
    components: list[set[int]] = []
    unseen = set(vertices)
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    return components


def lower_upper_links(mesh: TriMesh, vertex: int) -> tuple[list[set[int]], list[set[int]]]:
    """Lower and upper link components of ``vertex`` under the mesh SoS order."""
    neighbors, link_edges = mesh.neighbors_and_link()
    order = mesh.order
    lower = {item for item in neighbors[vertex] if order[item] < order[vertex]}
    upper = neighbors[vertex] - lower
    return link_components(lower, link_edges[vertex]), link_components(upper, link_edges[vertex])


def analyze_morse(mesh: TriMesh, include_boundary: bool = False) -> list[CriticalPoint]:
    """Classify vertices, masking open-surface boundary vertices by default."""
    boundary = mesh.boundary_vertices
    points: list[CriticalPoint] = []
    for vertex in range(len(mesh.vertices)):
        if not include_boundary and vertex in boundary:
            continue
        lower, upper = lower_upper_links(mesh, vertex)
        lower_count, upper_count = len(lower), len(upper)
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
