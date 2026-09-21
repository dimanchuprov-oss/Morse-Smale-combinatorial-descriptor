import numpy as np

from morse_lidar.mesh import grid_mesh
from morse_lidar.persistence import mesh_persistence, simplicial_persistence


def test_simplicial_persistence_accepts_mesh_scalar_field():
    y, x = np.mgrid[0:8, 0:10]
    mesh = grid_mesh((x - 4.5) ** 2 + (y - 3.5) ** 2)
    report = simplicial_persistence(mesh)
    assert report.backend == "gudhi-simplicial"
    assert report.intervals
    both = mesh_persistence(mesh)
    assert both["minima"].intervals
    assert both["maxima"].intervals