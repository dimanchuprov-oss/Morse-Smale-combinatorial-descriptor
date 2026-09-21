import numpy as np

from morse_lidar import CriticalType, analyze_morse, build_descriptor, grid_mesh, periodic_grid_mesh


def test_periodic_grid_is_closed_and_has_torus_euler_characteristic():
    y, x = np.mgrid[0:12, 0:16]
    height = np.sin(2 * np.pi * x / 16) + 0.4 * np.cos(2 * np.pi * y / 12)
    mesh = periodic_grid_mesh(height)
    points = analyze_morse(mesh)
    counts = {kind: sum(point.kind == kind for point in points) for kind in CriticalType}

    assert mesh.closed
    assert counts[CriticalType.MAXIMUM] - counts[CriticalType.SADDLE] + counts[CriticalType.MINIMUM] == 0
    assert counts[CriticalType.MINIMUM] > 0
    assert counts[CriticalType.MAXIMUM] > 0


def test_descriptor_keeps_euler_invariant_after_noise():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:10, 0:14]
    height = np.sin(2 * np.pi * x / 14) + 0.5 * np.cos(2 * np.pi * y / 10)
    descriptor = build_descriptor(periodic_grid_mesh(height + 0.01 * rng.normal(size=height.shape)), 0)
    assert descriptor.euler_characteristic == 0
    assert descriptor.counts["min"] - descriptor.counts["saddle"] + descriptor.counts["max"] == 0


def test_open_grid_masks_boundary_without_torus_wrapping():
    y, x = np.mgrid[0:12, 0:16]
    mesh = grid_mesh(np.sin(x / 4) + np.cos(y / 5))
    points = analyze_morse(mesh)
    descriptor = build_descriptor(mesh)

    assert not mesh.closed
    assert len(mesh.boundary_vertices) == 2 * 12 + 2 * (16 - 2)
    assert all(point.vertex not in mesh.boundary_vertices for point in points)
    assert descriptor.euler_characteristic is None
