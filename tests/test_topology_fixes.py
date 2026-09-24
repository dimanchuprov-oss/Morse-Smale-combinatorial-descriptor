import time

import numpy as np
import pytest

from morse_lidar import build_descriptor, grid_mesh, periodic_grid_mesh
from morse_lidar.critical import CriticalType
from morse_lidar.persistence import annotate_persistence, critical_pairs, persistence_diagram, persistent_counts


def _torus_field(rows=40, cols=48):
    y, x = np.mgrid[0:rows, 0:cols]
    return np.sin(2 * np.pi * x / cols) + 0.4 * np.cos(2 * np.pi * y / rows)


def test_boundary_queries_are_cached_and_classification_is_fast():
    y, x = np.mgrid[0:128, 0:128]
    mesh = grid_mesh(np.sin(x / 5) + np.cos(y / 4))
    assert mesh.boundary_vertices is mesh.boundary_vertices
    started = time.perf_counter()
    build_descriptor(mesh)
    assert time.perf_counter() - started < 10


def test_simple_saddles_emit_exactly_two_arcs_each_way():
    descriptor = build_descriptor(periodic_grid_mesh(_torus_field(12, 16)), 0)
    assert descriptor.counts["saddle"] == 2
    assert sorted(arc.kind for arc in descriptor.separatrices) == ["s"] * 4 + ["u"] * 4


def test_plateaus_are_handled_by_the_same_tie_break_everywhere():
    descriptor = build_descriptor(periodic_grid_mesh(np.zeros((8, 8))), 0)
    assert descriptor.saddle_index_sum == descriptor.counts["min"] + descriptor.counts["max"]
    saddles = [point for point in descriptor.critical_points if point.kind == CriticalType.SADDLE]
    assert len(descriptor.separatrices) == sum(point.lower_components + point.upper_components for point in saddles)


def test_open_surface_arcs_to_the_rim_are_flagged_not_dropped():
    y, x = np.mgrid[0:24, 0:24]
    descriptor = build_descriptor(grid_mesh(np.sin(x / 3) + np.cos(y / 3)))
    saddles = [point for point in descriptor.critical_points if point.kind == CriticalType.SADDLE]
    assert len(descriptor.separatrices) == sum(point.lower_components + point.upper_components for point in saddles)
    assert any(arc.ends_on_boundary for arc in descriptor.separatrices)


@pytest.mark.parametrize("noise", [0.0, 0.02, 0.05])
def test_multisaddles_keep_euler_and_persistence_recovers_clean_topology(noise):
    rng = np.random.default_rng(1)
    mesh = periodic_grid_mesh(_torus_field() + noise * rng.normal(size=(40, 48)))
    descriptor = build_descriptor(mesh, 0)  # raises if the Euler relation fails
    pairs = critical_pairs(mesh)
    points = annotate_persistence(descriptor.critical_points, pairs)
    assert persistent_counts(points, pairs, 0.5) == {"min": 1, "saddle": 2, "max": 1}
    raw = persistent_counts(points, pairs, 0.0)
    assert raw["min"] - raw["saddle"] + raw["max"] == 0
    assert {dimension: sum(np.isinf(d) for _, d in diagram) for dimension, diagram in persistence_diagram(pairs).items()} == {
        0: 1,
        1: 2,
        2: 1,
    }
