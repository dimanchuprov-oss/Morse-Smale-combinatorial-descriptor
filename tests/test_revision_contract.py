import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, median_filter

from morse_lidar.cli import main
from morse_lidar.compare import compare_reports
from morse_lidar.experiment import run_experiment
from morse_lidar.intrinsic import gaussian_curvature, mean_curvature, shape_index
from morse_lidar.mesh import TriMesh, grid_mesh
from morse_lidar.preprocess import denoise_depth


@pytest.mark.parametrize("field", [gaussian_curvature, mean_curvature, shape_index])
def test_entire_flat_surface_has_zero_curvature(field):
    mesh = grid_mesh(np.zeros((5, 5)), spacing=(0.1, 0.1))
    assert np.allclose(field(mesh), 0, atol=1e-10)


def test_boundary_only_component_rejected():
    mesh = TriMesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]), np.zeros(3))
    with pytest.raises(ValueError, match="no interior"):
        gaussian_curvature(mesh)


def test_mesh_owns_immutable_data():
    depth = np.zeros((5, 5))
    mesh = grid_mesh(depth)
    order = mesh.order.copy()
    depth[2, 2] = 100
    assert mesh.values[12] == 0
    assert np.array_equal(mesh.order, order)
    for array in (mesh.vertices, mesh.faces, mesh.values):
        with pytest.raises(ValueError):
            array.flat[0] = 1
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_periodic_smoothing_wraps_both_filters():
    field = np.random.default_rng(5).normal(size=(9, 11))
    expected = gaussian_filter(median_filter(field, size=3, mode="wrap"), sigma=1, mode="wrap")
    assert np.allclose(denoise_depth(field, 3, 1, periodic=True), expected)
    assert not np.allclose(denoise_depth(field, 3, 1), expected)


@pytest.mark.parametrize("geometry", ["delaunay", "poisson"])
@pytest.mark.parametrize("option,value", [("--sigma", "1"), ("--median-size", "3")])
def test_nonraster_smoothing_rejected_before_loading(geometry, option, value):
    with pytest.raises(SystemExit, match="2"):
        main(["--input", "missing.xyz", "--output", "unused.json", "--geometry", geometry, option, value])


def test_json_references_and_parameter_comparison(tmp_path):
    path = tmp_path / "depth.npy"
    np.save(path, np.random.default_rng(4).normal(size=(10, 10)))
    output = tmp_path / "report.json"
    main(["--input", str(path), "--output", str(output), "--persistence-threshold", "0.1"])
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    ids = {p["id"] for p in report["critical_points"]}
    assert None not in ids and len(ids) == len(report["critical_points"])
    assert len({a["id"] for a in report["separatrices"]}) == len(report["separatrices"])
    for arc in report["separatrices"]:
        assert arc["source_id"] in ids
        if arc["ends_on_boundary"]:
            assert arc["destination_id"] is None
            assert arc["destination_vertex_id"] is not None
        else:
            assert arc["destination_id"] in ids
    changed = {**report, "parameters": {**report["parameters"], "sigma": 1}}
    with pytest.raises(ValueError, match="parameters"):
        compare_reports(report, changed)


def test_repeat_experiment_with_synthetic_fixture(tmp_path):
    scans = []
    for i in range(3):
        path = tmp_path / f"scan{i}.npy"
        np.save(path, np.random.default_rng(i).normal(size=(8, 8)))
        scans.append({"path": path.name, "object": "a" if i < 2 else "b"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"units": "m", "parameters": {"up_axis": "z", "geometry": "raster", "scalar": "height", "persistence_threshold": 0.02}, "scans": scans}))
    result = run_experiment(manifest, tmp_path / "run")
    assert len(result["pairs"]) == 3
    assert sum(p["same_object"] for p in result["pairs"]) == 1
    assert all(len(s["sha256"]) == 64 for s in result["inputs"])
    assert json.loads((tmp_path / "run/results.json").read_text()) == result
