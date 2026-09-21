import numpy as np

from morse_lidar.pointcloud import PointCloud
from morse_lidar.surface import reconstruct_delaunay


def test_delaunay_reconstructs_open_surface_from_xyz():
    y, x = np.mgrid[0:8, 0:10]
    points = np.column_stack((x.ravel(), y.ravel(), (0.1 * x**2 + 0.2 * y).ravel()))
    mesh = reconstruct_delaunay(PointCloud(points))

    assert len(mesh.vertices) == 80
    assert len(mesh.faces) > 0
    assert not mesh.closed
    assert len(mesh.boundary_vertices) > 0
    assert np.isfinite(mesh.values).all()