"""A deterministic baseline for tracing monotone separatrix arcs."""

from __future__ import annotations

from dataclasses import dataclass

from .critical import CriticalPoint, CriticalType
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


def _trace(mesh: TriMesh, start: int, direction: int, neighbors: list[set[int]], limit: int) -> tuple[int, ...]:
    path = [start]
    current = start
    for _ in range(limit):
        candidates = [item for item in neighbors[current] if direction * (mesh.values[item] - mesh.values[current]) > 0]
        if not candidates:
            break
        next_vertex = min(candidates, key=lambda item: (direction * mesh.values[item], item))
        if next_vertex in path:
            break
        path.append(next_vertex)
        current = next_vertex
    return tuple(path)


def trace_separatrices(mesh: TriMesh, points: list[CriticalPoint]) -> list[Separatrix]:
    """Trace steepest monotone arcs as a baseline until a critical endpoint.

    This is intentionally not presented as a substitute for TTK's MSC. It is
    useful for fixtures and for checking the data contract before integrating
    a production discrete-gradient implementation.
    """
    neighbors, _ = mesh.neighbors_and_link()
    by_vertex = {point.vertex: point for point in points}
    arcs: list[Separatrix] = []
    for point in points:
        if point.kind != CriticalType.SADDLE:
            continue
        for direction, kind in ((-1, "u"), (1, "s")):
            candidates = [item for item in neighbors[point.vertex] if direction * (mesh.values[item] - mesh.values[point.vertex]) > 0]
            for candidate in sorted(candidates):
                path = _trace(mesh, candidate, direction, neighbors, len(mesh.vertices))
                endpoint = path[-1]
                endpoint_point = by_vertex.get(endpoint)
                if endpoint_point is None or endpoint_point.kind not in (CriticalType.MINIMUM, CriticalType.MAXIMUM):
                    continue
                arcs.append(Separatrix(point.vertex, endpoint, kind, (point.vertex, *path)))
    return arcs
