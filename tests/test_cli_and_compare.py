import json
import struct

import numpy as np
import pytest

from morse_lidar.cli import main
from morse_lidar.compare import compare_reports
from morse_lidar.pointcloud import PointCloud, align_pca, load_point_cloud, rasterize_depth, remove_dominant_plane, set_up_axis


def _run(tmp_path, name, *args):
    output = tmp_path / f"{name}.json"
    main([*args, "--output", str(output)])
    return json.loads(output.read_text())


def _torus(tmp_path, name, noise, seed):
    y, x = np.mgrid[0:32, 0:40]
    field = np.sin(2 * np.pi * x / 40) + 0.4 * np.cos(2 * np.pi * y / 32)
    path = tmp_path / f"{name}.npy"
    np.save(path, field + noise * np.random.default_rng(seed).normal(size=field.shape))
    return path


def test_cli_reports_persistent_counts_and_compare_is_stable(tmp_path):
    reports = [
        _run(tmp_path, f"t{seed}", "--input", str(_torus(tmp_path, f"t{seed}", 0.03, seed)), "--surface", "periodic", "--persistence-threshold", "0.5")
        for seed in (1, 2)
    ]
    for report in reports:
        assert report["persistent_counts"] == {"min": 1, "saddle": 2, "max": 1}
        assert report["critical_points"] and report["separatrices"]
        assert all(point["kind"] != "regular" for point in report["critical_points"])
    result = compare_reports(*reports)
    assert result["persistent_counts_equal"]
    # Stability: bottleneck distance is bounded by the sup-norm of the difference.
    assert result["bottleneck_max"] <= 2 * 0.03 * 5


def test_periodic_alias_is_validated_like_surface_periodic(tmp_path):
    cloud = tmp_path / "cloud.xyz"
    np.savetxt(cloud, np.random.default_rng(0).uniform(size=(50, 3)))
    with pytest.raises(SystemExit):
        main(["--input", str(cloud), "--output", str(tmp_path / "x.json"), "--periodic", "--geometry", "delaunay"])


def test_binary_ply_loads_without_open3d(tmp_path):
    points = np.random.default_rng(0).normal(size=(20, 3)).astype("<f4")
    path = tmp_path / "binary.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 20\n"
        "property float x\nproperty float y\nproperty float z\nproperty uchar red\nend_header\n"
    )
    body = b"".join(struct.pack("<fffB", *point, 7) for point in points)
    path.write_bytes(header.encode("ascii") + body)
    assert np.allclose(load_point_cloud(path).points, points)


def test_raster_spacing_is_metric():
    y, x = np.mgrid[0:11, 0:21]
    cloud = PointCloud(np.column_stack((0.1 * x.ravel(), 0.05 * y.ravel(), np.zeros(x.size))))
    raster = rasterize_depth(cloud, rows=11, cols=21)
    assert raster.spacing == pytest.approx((0.1, 0.05))
    assert raster.coverage == 1.0


def test_up_axis_pca_and_plane_removal():
    cloud = PointCloud(np.array([[1.0, 2.0, 3.0]]))
    assert set_up_axis(cloud, "y").points[0].tolist() == [3.0, 1.0, 2.0]
    rng = np.random.default_rng(0)
    flat = np.column_stack((rng.uniform(-5, 5, 400), rng.uniform(-1, 1, 400), 0.01 * rng.normal(size=400)))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    aligned = align_pca(PointCloud(flat @ rotation.T)).points
    assert np.ptp(aligned[:, 2]) < 0.2 < np.ptp(aligned[:, 0])
    floor = np.column_stack((rng.uniform(-1, 1, (300, 2)), np.zeros(300)))
    box = rng.uniform(0.1, 0.3, (100, 3))
    remaining, removed = remove_dominant_plane(PointCloud(np.vstack((floor, box))), 0.01)
    assert removed == 300 and len(remaining.points) == 100


def test_pca_keeps_a_dome_a_dome():
    y, x = np.mgrid[-1:1:21j, -1:1:21j]
    dome = np.column_stack((x.ravel(), y.ravel(), 0.3 * (1 - x.ravel() ** 2 - y.ravel() ** 2)))
    aligned = align_pca(PointCloud(dome)).points
    centre = np.argmin(np.linalg.norm(dome[:, :2], axis=1))
    assert aligned[centre, 2] == pytest.approx(aligned[:, 2].max())
    assert np.linalg.det(np.linalg.lstsq(dome - dome.mean(axis=0), aligned, rcond=None)[0]) == pytest.approx(1.0)


def test_ttk_backend_still_gets_persistent_counts():
    from morse_lidar.cli import _persistence_json
    from morse_lidar.descriptor import Descriptor
    from morse_lidar.mesh import periodic_grid_mesh

    y, x = np.mgrid[0:32, 0:40]
    field = np.sin(2 * np.pi * x / 40) + 0.4 * np.cos(2 * np.pi * y / 32)
    mesh = periodic_grid_mesh(field + 0.03 * np.random.default_rng(5).normal(size=field.shape))
    ttk_like = Descriptor((), (), 0, backend="ttk-morse-smale-complex")
    report = {}
    assert _persistence_json(ttk_like, mesh, 0.5, report) is ttk_like
    assert report["persistent_counts"] == {"min": 1, "saddle": 2, "max": 1}
    assert report["persistent_minima"] == 1 and report["persistent_maxima"] == 1


def test_periodic_curvature_and_pixel_size_misuse_are_rejected(tmp_path):
    path = _torus(tmp_path, "p", 0.0, 0)
    with pytest.raises(SystemExit):
        main(["--input", str(path), "--output", str(tmp_path / "x.json"), "--surface", "periodic", "--scalar", "mean-curvature"])
    cloud = tmp_path / "c.xyz"
    np.savetxt(cloud, np.random.default_rng(0).uniform(size=(50, 3)))
    with pytest.raises(SystemExit):
        main(["--input", str(cloud), "--output", str(tmp_path / "x.json"), "--pixel-size", "0.1"])


def test_ascii_ply_tolerates_blank_lines(tmp_path):
    path = tmp_path / "blank.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n\n1 2 3\n\n4 5 6\n")
    assert load_point_cloud(path).points.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_compare_reports_missing_diagram_cleanly():
    with pytest.raises(ValueError, match="persistence_diagram"):
        compare_reports({"persistence_diagram": None}, {"persistence_diagram": {}})
