"""A deterministic baseline for tracing monotone separatrix arcs."""

from __future__ import annotations

from dataclasses import dataclass

from .critical import CriticalPoint, CriticalType, lower_upper_links
from .mesh import TriMesh


@dataclass(frozen=True)
class Separatrix:
    source: int
    target: int
    kind: str
    vertices: tuple[int, ...]
    identifier: int | None = None
    points: tuple[tuple[float, float, float], ...] = ()
    source_cell_id: int | None = None
    destination_cell_id: int | None = None
    backend: str = "steepest-monotone-baseline"
    # True when the arc leaves the interior and stops at a masked boundary
    # vertex instead of an interior extremum (open surfaces only).
    ends_on_boundary: bool = False


def _trace(mesh: TriMesh, start: int, direction: int, neighbors: tuple[frozenset[int], ...]) -> tuple[int, ...]:
    """Follow the steepest neighbour in the SoS order until an extremum.

    The walk is strictly monotone in ``mesh.order``, so it cannot revisit a
    vertex and always terminates at a (possibly boundary) local extremum.
    """
    order = mesh.order
    path = [start]
    current = start
    while True:
        candidates = [item for item in neighbors[current] if direction * (order[item] - order[current]) > 0]
        if not candidates:
            return tuple(path)
        current = min(candidates, key=lambda item: direction * order[item])
        path.append(current)


def trace_separatrices(mesh: TriMesh, points: list[CriticalPoint]) -> list[Separatrix]:
    """Trace one steepest monotone arc per lower/upper link component of a saddle.

    This is intentionally not presented as a substitute for TTK's MSC. It is
    useful for fixtures and for checking the data contract before integrating
    a production discrete-gradient implementation. A simple saddle yields
    exactly two descending (``u``) and two ascending (``s``) arcs.
    """
    neighbors, _ = mesh.neighbors_and_link()
    order = mesh.order
    boundary = mesh.boundary_vertices
    by_vertex = {point.vertex: point for point in points}
    arcs: list[Separatrix] = []
    for point in points:
        if point.kind != CriticalType.SADDLE:
            continue
        lower, upper = lower_upper_links(mesh, point.vertex)
        for direction, kind, components, expected in (
            (-1, "u", lower, CriticalType.MINIMUM),
            (1, "s", upper, CriticalType.MAXIMUM),
        ):
            for component in sorted(components, key=min):
                start = min(component, key=lambda item: direction * order[item])
                path = _trace(mesh, start, direction, neighbors)
                endpoint = path[-1]
                endpoint_point = by_vertex.get(endpoint)
                on_boundary = endpoint in boundary and endpoint_point is None
                if not on_boundary and (endpoint_point is None or endpoint_point.kind != expected):
                    raise RuntimeError(f"separatrix from saddle {point.vertex} ended at non-extremum {endpoint}")
                arcs.append(
                    Separatrix(point.vertex, endpoint, kind, (point.vertex, *path), ends_on_boundary=on_boundary)
                )
    return arcs
