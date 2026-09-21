"""Topology descriptors and fail-fast invariant checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .critical import CriticalPoint, CriticalType, analyze_morse
from .mesh import TriMesh
from .separatrices import Separatrix, trace_separatrices

if TYPE_CHECKING:
    from .ttk_backend import TTKResult


@dataclass(frozen=True)
class ColoredEdge:
    left: int
    right: int
    color: str
    # Exact primal quadrangulation edge shared by the two dual faces.  Keeping
    # it makes it impossible to collapse the graph to critical-point pairs.
    primal_edge: tuple[str, int]


@dataclass(frozen=True)
class ColoredGraph:
    triangles: tuple[tuple[int, int, int], ...]
    primal_arcs: tuple[tuple[tuple[str, int], tuple[str, int], tuple[str, int]], ...]
    edges: tuple[ColoredEdge, ...]

    @property
    def node_count(self) -> int:
        return len(self.triangles)


@dataclass(frozen=True)
class MorseSmaleQuadrilateral:
    identifier: int
    critical_points: tuple[int, int, int, int]
    separatrices: tuple[int, int, int, int]


@dataclass(frozen=True)
class Descriptor:
    critical_points: tuple[CriticalPoint, ...]
    separatrices: tuple[Separatrix, ...]
    euler_characteristic: int | None
    quadrilaterals: tuple[MorseSmaleQuadrilateral, ...] = ()
    colored_graph: ColoredGraph | None = None
    backend: str = "heuristic"

    @property
    def counts(self) -> dict[str, int]:
        return {kind.value: sum(point.kind == kind for point in self.critical_points) for kind in CriticalType}


def validate_euler(descriptor: Descriptor) -> None:
    counts = descriptor.counts
    actual = counts["max"] - counts["saddle"] + counts["min"]
    if actual != descriptor.euler_characteristic:
        raise ValueError(f"Morse Euler invariant failed: {actual} != {descriptor.euler_characteristic}")


def validate_colored_trivalent_graph(nodes: int, edges: list[tuple[int, int, str]]) -> None:
    """Validate that every dual node has one incident edge of each color."""
    if nodes <= 0:
        raise ValueError("colored dual graph has no nodes")
    degree = [0] * nodes
    colors: list[list[str]] = [[] for _ in range(nodes)]
    seen: set[tuple[int, int, str]] = set()
    for left, right, color in edges:
        if color not in {"s", "u", "t"}:
            raise ValueError(f"unknown edge color: {color}")
        if not (0 <= left < nodes and 0 <= right < nodes):
            raise ValueError("colored graph edge references an unknown node")
        if left == right:
            raise ValueError("colored graph contains a self-loop")
        key = (min(left, right), max(left, right), color)
        if key in seen:
            raise ValueError("colored graph contains a duplicate edge")
        seen.add(key)
        degree[left] += 1
        degree[right] += 1
        colors[left].append(color)
        colors[right].append(color)
    if any(value != 3 for value in degree):
        raise ValueError("dual graph is not trivalent")
    if any(sorted(value) != ["s", "t", "u"] for value in colors):
        raise ValueError("each dual node must have exactly one s, u, and t edge")


def validate_colored_graph(graph: ColoredGraph) -> None:
    """Also prove that every dual edge comes from one shared primal arc."""
    if len(graph.primal_arcs) != graph.node_count:
        raise ValueError("each dual node must retain its three primal arcs")
    incidence: dict[tuple[str, int], list[int]] = {}
    for triangle_id, arcs in enumerate(graph.primal_arcs):
        if len(set(arcs)) != 3:
            raise ValueError("a triangle does not have three distinct primal arcs")
        for arc in arcs:
            incidence.setdefault(arc, []).append(triangle_id)
    expected: dict[tuple[str, int], tuple[int, int]] = {}
    for primal_edge, adjacent in incidence.items():
        if len(adjacent) != 2:
            raise ValueError(f"primal arc {primal_edge} is shared by {len(adjacent)} triangles, not two")
        expected[primal_edge] = tuple(sorted(adjacent))
    actual: dict[tuple[str, int], tuple[int, int]] = {}
    for edge in graph.edges:
        if edge.primal_edge in actual:
            raise ValueError("multiple dual edges reference the same primal arc")
        actual[edge.primal_edge] = tuple(sorted((edge.left, edge.right)))
    if actual != expected:
        raise ValueError("dual graph does not correspond exactly to shared primal arcs")
    validate_colored_trivalent_graph(
        graph.node_count,
        [(edge.left, edge.right, edge.color) for edge in graph.edges],
    )


def build_descriptor(
    mesh: TriMesh,
    euler_characteristic: int | None = None,
    include_boundary: bool = False,
) -> Descriptor:
    points = tuple(analyze_morse(mesh, include_boundary=include_boundary))
    descriptor = Descriptor(points, tuple(trace_separatrices(mesh, list(points))), euler_characteristic)
    if euler_characteristic is not None:
        validate_euler(descriptor)
    return descriptor


def build_ttk_descriptor(result: TTKResult, euler_characteristic: int) -> Descriptor:
    """Build a descriptor exclusively from TTK outputs and validate it."""
    descriptor = Descriptor(
        critical_points=result.critical_points,
        separatrices=result.separatrices,
        euler_characteristic=euler_characteristic,
        quadrilaterals=result.morse_smale_cells,
        colored_graph=result.colored_graph,
        backend=result.backend,
    )
    validate_euler(descriptor)
    validate_colored_graph(result.colored_graph)
    return descriptor
