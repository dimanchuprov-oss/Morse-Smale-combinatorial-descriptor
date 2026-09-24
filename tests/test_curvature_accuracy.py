import numpy as np
import pytest

from morse_lidar.intrinsic import gaussian_curvature, mean_curvature, shape_index
from morse_lidar.mesh import grid_mesh
from morse_lidar.pointcloud import PointCloud
from morse_lidar.surface import reconstruct_delaunay


def _cap(spacing: float, sign: float = 1.0):
    xs = np.arange(-3, 3 + 1e-9, spacing)
    x, y = np.meshgrid(xs, xs)
    return grid_mesh(sign * np.sqrt(100 - x**2 - y**2), spacing=(spacing, spacing)), (len(xs) ** 2) // 2


@pytest.mark.parametrize("spacing", [0.5, 0.25, 0.1])
def test_sphere_cap_curvature_is_pointwise_and_resolution_independent(spacing):
    mesh, centre = _cap(spacing)
    assert gaussian_curvature(mesh)[centre] == pytest.approx(0.01, rel=1e-2)
    assert mean_curvature(mesh)[centre] == pytest.approx(0.1, rel=1e-2)


def test_shape_index_distinguishes_dome_bowl_saddle_and_cylinder():
    mesh, centre = _cap(0.25)
    assert shape_index(mesh)[centre] == pytest.approx(1.0, abs=1e-2)
    bowl, _ = _cap(0.25, sign=-1.0)
    assert shape_index(bowl)[centre] == pytest.approx(-1.0, abs=1e-2)
    xs = np.arange(-3, 3 + 1e-9, 0.25)
    x, y = np.meshgrid(xs, xs)
    saddle = grid_mesh(0.05 * (x**2 - y**2), spacing=(0.25, 0.25))
    assert shape_index(saddle)[centre] == pytest.approx(0.0, abs=1e-2)
    cylinder = grid_mesh(np.sqrt(100 - x**2), spacing=(0.25, 0.25))
    assert shape_index(cylinder)[centre] == pytest.approx(0.5, abs=1e-2)


def test_delaunay_faces_are_oriented_upwards_so_curvature_sign_is_stable():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-3, 3, size=(600, 2))
    z = np.sqrt(100 - (xy**2).sum(axis=1))
    mesh = reconstruct_delaunay(PointCloud(np.column_stack((xy, z))))
    corners = mesh.vertices[mesh.faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert (normals[:, 2] > 0).all()
    interior = [v for v in range(len(mesh.vertices)) if v not in mesh.boundary_vertices]
    assert np.median(mean_curvature(mesh)[interior]) == pytest.approx(0.1, rel=0.1)
