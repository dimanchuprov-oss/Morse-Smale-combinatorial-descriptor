import numpy as np

from morse_lidar.intrinsic import gaussian_curvature, intrinsic_scalar, mean_curvature, shape_index
from morse_lidar.mesh import grid_mesh


def test_intrinsic_scalars_are_finite_on_open_surface():
    y, x = np.mgrid[0:10, 0:12]
    mesh = grid_mesh(0.03 * x**2 + 0.02 * y**2)
    for kind in ("gaussian-curvature", "mean-curvature", "shape-index"):
        values = intrinsic_scalar(mesh, kind)
        assert values.shape == (len(mesh.vertices),)
        assert np.isfinite(values).all()

    assert np.isfinite(gaussian_curvature(mesh)).all()
    assert np.isfinite(mean_curvature(mesh)).all()
    assert np.isfinite(shape_index(mesh)).all()