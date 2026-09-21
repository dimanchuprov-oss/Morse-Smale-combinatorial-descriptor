import numpy as np
import pytest

from morse_lidar.descriptor import build_ttk_descriptor, validate_colored_graph
from morse_lidar.mesh import TriMesh, grid_mesh
from morse_lidar.ttk_backend import export_legacy_vtk, run_ttk_msc, status


def test_ttk_status_and_vtk_export_are_available_without_paraview(tmp_path):
    mesh = grid_mesh(np.arange(20, dtype=float).reshape(4, 5))
    target = tmp_path / "mesh.vtk"
    export_legacy_vtk(mesh, target)
    text = target.read_text()
    assert "DATASET POLYDATA" in text
    assert "SCALARS ScalarField" in text
    assert isinstance(status().available, bool)


@pytest.mark.skipif(not status().available, reason="ParaView/TTK runtime is not installed")
def test_real_ttk_pipeline_builds_trivalent_graph_from_shared_arcs():
    rows = cols = 12
    vertices = []
    values = []
    for y in range(rows):
        v = 2 * np.pi * y / rows
        for x in range(cols):
            u = 2 * np.pi * x / cols
            vertices.append(
                ((2 + 0.6 * np.cos(v)) * np.cos(u), (2 + 0.6 * np.cos(v)) * np.sin(u), 0.6 * np.sin(v))
            )
            values.append(np.cos(u) + 0.45 * np.cos(v) + 1e-6 * (y * cols + x))

    def index(y, x):
        return (y % rows) * cols + (x % cols)

    faces = []
    for y in range(rows):
        for x in range(cols):
            a, b, c, d = index(y, x), index(y, x + 1), index(y + 1, x), index(y + 1, x + 1)
            faces.extend(((a, b, d), (a, d, c)))
    mesh = TriMesh(np.asarray(vertices), np.asarray(faces), np.asarray(values))

    result = run_ttk_msc(mesh)
    descriptor = build_ttk_descriptor(result, mesh.euler_characteristic)

    assert descriptor.backend == "ttk-morse-smale-complex"
    assert descriptor.counts == {"min": 1, "regular": 0, "saddle": 2, "max": 1}
    assert len(descriptor.separatrices) == 8
    assert all(arc.backend == "ttk" and len(arc.points) >= 2 for arc in descriptor.separatrices)
    assert "SourceId" in result.arrays["separatrices"]["cell_data"]
    assert "DestinationId" in result.arrays["separatrices"]["cell_data"]
    assert "SeparatrixType" in result.arrays["separatrices"]["cell_data"]
    # TTK 1.4 encodes 2-D critical type as CellDimension, not CriticalType.
    assert "CellDimension" in result.arrays["critical_points"]["point_data"]
    validate_colored_graph(result.colored_graph)
    assert len(result.morse_smale_cells) == 4
    assert result.colored_graph.node_count == 8
    assert len(result.colored_graph.edges) == 12
    assert {edge.primal_edge[0] for edge in result.colored_graph.edges} == {"separatrix", "t"}
    assert len(result.colored_graph.edges) * 2 == result.colored_graph.node_count * 3
    for node in range(result.colored_graph.node_count):
        colors = [edge.color for edge in result.colored_graph.edges if node in (edge.left, edge.right)]
        assert sorted(colors) == ["s", "t", "u"]
