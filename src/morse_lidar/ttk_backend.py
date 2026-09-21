"""Bridge from :class:`TriMesh` to a ParaView/TTK Morse-Smale complex."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .critical import CriticalPoint, CriticalType
from .descriptor import ColoredEdge, ColoredGraph, MorseSmaleQuadrilateral
from .mesh import TriMesh
from .separatrices import Separatrix


@dataclass(frozen=True)
class TTKStatus:
    available: bool
    module_path: str | None
    message: str


@dataclass(frozen=True)
class TTKQuadrangulation:
    points: tuple[tuple[float, float, float], ...]
    cells: tuple[tuple[int, int, int, int], ...]
    vertex_types: tuple[int, ...]
    cell_ids: tuple[int, ...]


@dataclass(frozen=True)
class TTKResult:
    critical_points: tuple[CriticalPoint, ...]
    separatrices: tuple[Separatrix, ...]
    quadrangulation: TTKQuadrangulation
    morse_smale_cells: tuple[MorseSmaleQuadrilateral, ...]
    colored_graph: ColoredGraph
    arrays: dict[str, dict[str, list[str]]]
    backend: str = "ttk-morse-smale-complex"


# Backward-compatible public name; TTK and heuristic arcs now share one model.
TTKSeparatrix = Separatrix


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundled_pvpython() -> Path:
    return _project_root() / ".tools" / "ttk-runtime" / "ttk-paraview" / "bin" / "pvpython"


def _find_pvpython(explicit: str | None = None) -> str | None:
    candidates = (explicit, os.environ.get("MORSE_LIDAR_PVPYTHON"), shutil.which("pvpython"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    bundled = _bundled_pvpython()
    return str(bundled) if bundled.is_file() else None


def status() -> TTKStatus:
    """Report whether a ParaView ``pvpython`` runtime is available."""
    executable = _find_pvpython()
    if executable:
        return TTKStatus(True, executable, "ParaView pvpython is available.")
    return TTKStatus(
        False,
        None,
        "TTK is unavailable; install ParaView with TTK or set MORSE_LIDAR_PVPYTHON.",
    )


def require() -> TTKStatus:
    current = status()
    if not current.available:
        raise RuntimeError(current.message)
    return current


def export_legacy_vtk(mesh: TriMesh, path: str | Path, scalar_name: str = "ScalarField") -> None:
    """Write a triangle mesh readable by VTK's legacy polydata reader."""
    target = Path(path)
    with target.open("w", encoding="ascii") as handle:
        handle.write("# vtk DataFile Version 3.0\nMorse-LiDAR mesh\nASCII\nDATASET POLYDATA\n")
        handle.write(f"POINTS {len(mesh.vertices)} double\n")
        for point in mesh.vertices:
            handle.write(f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")
        handle.write(f"POLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
        for face in mesh.faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
        handle.write(f"POINT_DATA {len(mesh.values)}\nSCALARS {scalar_name} double 1\nLOOKUP_TABLE default\n")
        for value in mesh.values:
            handle.write(f"{float(value):.17g}\n")


def _runtime_environment(executable: str) -> dict[str, str]:
    environment = os.environ.copy()
    root = _project_root() / ".tools"
    if Path(executable).resolve() != _bundled_pvpython().resolve():
        return environment
    python_home = root / "python313-package" / "Python_Framework.pkg" / "Payload" / "Versions" / "3.13"
    ttk_root = root / "ttk-runtime" / "ttk"
    paraview_root = root / "ttk-runtime" / "ttk-paraview"
    ttk_python = ttk_root / "lib" / "python3.13" / "site-packages" / "topologytoolkit"
    libomp = root / "libomp" / "libomp" / "23.1.1" / "lib"
    required = (python_home / "Python", ttk_python / "ttkMorseSmaleComplex.so", libomp / "libomp.dylib")
    if not all(path.exists() for path in required):
        raise RuntimeError("The project-local pvpython exists but its Python/TTK/libomp runtime is incomplete")
    environment["PYTHONHOME"] = str(python_home)
    environment["TTK_PYTHON_DIR"] = str(ttk_python)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ttk_python), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        [str(libomp), str(ttk_root / "lib"), str(paraview_root / "lib"), environment.get("DYLD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return environment


def _critical_point(item: dict[str, object]) -> CriticalPoint:
    code = int(item["critical_type"])
    kinds = {0: CriticalType.MINIMUM, 1: CriticalType.SADDLE, 2: CriticalType.MAXIMUM}
    if code not in kinds:
        raise RuntimeError(f"unsupported TTK critical type {code} for a 2-manifold")
    kind = kinds[code]
    return CriticalPoint(
        vertex=int(item["vertex_id"]),
        kind=kind,
        lower_components=-1,
        upper_components=-1,
        saddle_multiplicity=1 if kind == CriticalType.SADDLE else 0,
        identifier=int(item["id"]),
        position=tuple(float(value) for value in item["position"]),
        cell_id=int(item["cell_id"]),
        cell_dimension=int(item["cell_dimension"]),
        scalar=float(item["scalar"]),
        is_on_boundary=bool(item["is_on_boundary"]),
        backend="ttk",
    )


def _separatrix(item: dict[str, object]) -> Separatrix:
    return Separatrix(
        source=int(item["source_id"]),
        target=int(item["destination_id"]),
        kind=str(item["kind"]),
        vertices=(),
        identifier=int(item["id"]),
        points=tuple(tuple(float(value) for value in point) for point in item["points"]),
        source_cell_id=int(item["source_cell_id"]),
        destination_cell_id=int(item["destination_cell_id"]),
        backend="ttk",
    )


def _parse_quadrangulation(payload: dict[str, object]) -> TTKQuadrangulation:
    arrays = payload["point_arrays"]
    if "QuadVertType" not in arrays or "QuadCellId" not in arrays:
        raise RuntimeError("TTK quadrangulation lacks QuadVertType or QuadCellId")
    cells = tuple(tuple(int(vertex) for vertex in cell) for cell in payload["cells"])
    if any(len(cell) != 4 for cell in cells):
        raise RuntimeError("TTK Morse-Smale quadrangulation emitted a non-quadrilateral cell")
    return TTKQuadrangulation(
        points=tuple(tuple(float(value) for value in point) for point in payload["points"]),
        cells=cells,
        vertex_types=tuple(int(value) for value in arrays["QuadVertType"]),
        cell_ids=tuple(int(value) for value in arrays["QuadCellId"]),
    )


def _colored_dual(
    quadrangulation: TTKQuadrangulation,
    critical_points: tuple[CriticalPoint, ...],
    separatrices: tuple[Separatrix, ...],
) -> tuple[tuple[MorseSmaleQuadrilateral, ...], ColoredGraph]:
    """Recover MSC cells and cut each min-saddle-max-saddle cell by a t arc.

    TTK subdivides every Morse-Smale cell into four output quadrilaterals.
    Their vertex classes are critical point, separatrix midpoint, and cell
    barycenter. Grouping by barycenter recovers the original cell without ever
    identifying arcs merely by their endpoint critical-point pair.
    """
    critical_kind = {point.identifier: point.kind for point in critical_points}
    separatrix_kind = {arc.identifier: arc.kind for arc in separatrices}
    cells_by_center: dict[int, list[tuple[int, int, int]]] = {}
    for cell in quadrangulation.cells:
        types = [quadrangulation.vertex_types[vertex] for vertex in cell]
        try:
            zero = types.index(0)
        except ValueError as error:
            raise RuntimeError("quadrilateral has no QuadVertType=0 vertex") from error
        ordered = cell[zero:] + cell[:zero]
        ordered_types = [quadrangulation.vertex_types[vertex] for vertex in ordered]
        if ordered_types != [0, 1, 2, 1]:
            raise RuntimeError(f"unexpected TTK quadrilateral vertex types: {ordered_types}")
        cells_by_center.setdefault(ordered[2], []).append((ordered[0], ordered[1], ordered[3]))

    morse_smale_cells: list[MorseSmaleQuadrilateral] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_arcs: list[tuple[tuple[str, int], tuple[str, int], tuple[str, int]]] = []
    for center, pieces in sorted(cells_by_center.items()):
        if len(pieces) != 4:
            raise RuntimeError(f"TTK Morse-Smale cell {center} contains {len(pieces)} pieces, not four")
        midpoint_critical: dict[int, list[int]] = {}
        for critical, left_midpoint, right_midpoint in pieces:
            midpoint_critical.setdefault(left_midpoint, []).append(critical)
            midpoint_critical.setdefault(right_midpoint, []).append(critical)
        if any(len(vertices) != 2 for vertices in midpoint_critical.values()):
            raise RuntimeError("TTK cell pieces do not join pairwise along separatrix midpoints")
        adjacency: dict[int, list[tuple[int, int]]] = {}
        for midpoint, (left, right) in midpoint_critical.items():
            separatrix_id = quadrangulation.cell_ids[midpoint]
            adjacency.setdefault(left, []).append((right, separatrix_id))
            adjacency.setdefault(right, []).append((left, separatrix_id))

        def point_identifier(point: int) -> int:
            return quadrangulation.cell_ids[point]

        minima = [point for point in adjacency if critical_kind.get(point_identifier(point)) == CriticalType.MINIMUM]
        maxima = [point for point in adjacency if critical_kind.get(point_identifier(point)) == CriticalType.MAXIMUM]
        if len(minima) != 1 or len(maxima) != 1:
            raise RuntimeError("Morse-Smale quadrilateral must contain one minimum and one maximum")
        minimum, maximum = minima[0], maxima[0]
        saddle_paths: list[tuple[int, int, int]] = []
        for saddle, first_sep in adjacency[minimum]:
            if critical_kind.get(point_identifier(saddle)) != CriticalType.SADDLE:
                raise RuntimeError("minimum is not adjacent to a saddle in TTK quadrangulation")
            continuation = [(neighbor, sep) for neighbor, sep in adjacency[saddle] if neighbor != minimum]
            if len(continuation) != 1 or continuation[0][0] != maximum:
                raise RuntimeError("saddle does not connect the cell minimum to its maximum")
            saddle_paths.append((saddle, first_sep, continuation[0][1]))
        if len(saddle_paths) != 2:
            raise RuntimeError("Morse-Smale quadrilateral must contain two saddle paths")
        saddle_paths.sort(key=lambda item: (item[1], point_identifier(item[0])))
        (saddle_a, u_a, s_a), (saddle_b, u_b, s_b) = saddle_paths
        ids = (
            point_identifier(minimum),
            point_identifier(saddle_a),
            point_identifier(maximum),
            point_identifier(saddle_b),
        )
        for separatrix_id, expected_kind in ((u_a, "u"), (u_b, "u"), (s_a, "s"), (s_b, "s")):
            if separatrix_kind.get(separatrix_id) != expected_kind:
                raise RuntimeError(
                    f"separatrix {separatrix_id} has kind {separatrix_kind.get(separatrix_id)!r}, "
                    f"expected {expected_kind!r}"
                )
        morse_smale_cells.append(
            MorseSmaleQuadrilateral(center, ids, (u_a, s_a, s_b, u_b))
        )
        t_arc = ("t", center)
        triangles.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
        triangle_arcs.extend(
            (
                (("separatrix", u_a), ("separatrix", s_a), t_arc),
                (("separatrix", s_b), ("separatrix", u_b), t_arc),
            )
        )

    incidence: dict[tuple[str, int], list[int]] = {}
    for triangle_id, arcs in enumerate(triangle_arcs):
        for arc in arcs:
            incidence.setdefault(arc, []).append(triangle_id)
    edges: list[ColoredEdge] = []
    for primal_edge, adjacent in sorted(incidence.items()):
        if len(adjacent) != 2:
            raise RuntimeError(
                f"TTK quadrangulation is not closed: primal arc {primal_edge} has {len(adjacent)} incident triangles"
            )
        color = "t" if primal_edge[0] == "t" else separatrix_kind[primal_edge[1]]
        edges.append(ColoredEdge(adjacent[0], adjacent[1], color, primal_edge))
    return tuple(morse_smale_cells), ColoredGraph(tuple(triangles), tuple(triangle_arcs), tuple(edges))


def run_ttk_msc(mesh: TriMesh, pvpython: str | None = None, timeout: float = 300.0) -> TTKResult:
    """Run TTK and return critical points, polylines, quadrangles and dual graph."""
    executable = _find_pvpython(pvpython)
    if executable is None:
        raise RuntimeError(status().message)
    with tempfile.TemporaryDirectory(prefix="morse_lidar_ttk-") as directory:
        root = Path(directory)
        input_path = root / "input.vtk"
        output_path = root / "msc.json"
        export_legacy_vtk(mesh, input_path)
        completed = subprocess.run(
            [executable, str(Path(__file__).with_name("ttk_pipeline.py")), str(input_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_runtime_environment(executable),
        )
        if completed.returncode:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"TTK pipeline failed ({completed.returncode}): {details}")
        if not output_path.exists():
            raise RuntimeError("TTK pipeline completed without producing its JSON result")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    critical_points = tuple(_critical_point(item) for item in payload["critical_points"])
    separatrices = tuple(_separatrix(item) for item in payload["separatrices"])
    quadrangulation = _parse_quadrangulation(payload["quadrangulation"])
    morse_smale_cells, colored_graph = _colored_dual(quadrangulation, critical_points, separatrices)
    return TTKResult(
        critical_points,
        separatrices,
        quadrangulation,
        morse_smale_cells,
        colored_graph,
        payload["arrays"],
    )
