import numpy as np
import pytest

from morse_lidar.virtual_scan import Shot, camera_axes, face_normals, ideal, sample_surface, shoot


def sphere(radius: float = 80.0, rows: int = 60, columns: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Closed UV sphere mesh in mm, centred at the origin."""
    polar, azimuth = np.meshgrid(np.linspace(0.0, np.pi, rows), np.linspace(0.0, 2 * np.pi, columns, endpoint=False),
                                 indexing="ij")  # fmt: skip
    vertices = radius * np.column_stack((np.sin(polar.ravel()) * np.cos(azimuth.ravel()), np.cos(polar.ravel()),
                                         np.sin(polar.ravel()) * np.sin(azimuth.ravel())))  # fmt: skip
    faces = []
    for row in range(rows - 1):
        for column in range(columns):
            a, b = row * columns + column, row * columns + (column + 1) % columns
            c, d = a + columns, b + columns
            faces += [[a, c, b], [b, c, d]]
    return vertices, np.array(faces)


def plane(size: float = 100.0, step: float = 5.0, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Square facing +Z, ``size`` mm across."""
    count = int(size / step) + 1
    x, y = np.meshgrid(np.linspace(-size / 2, size / 2, count), np.linspace(-size / 2, size / 2, count))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
    index = np.arange(count * count).reshape(count, count)
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel()
    return vertices, np.concatenate((np.column_stack((a, b, c)), np.column_stack((b, d, c))))


def test_surface_samples_have_the_requested_density():
    vertices, faces = plane()
    points, chosen, weights = sample_surface(vertices, faces, 1.0, np.random.default_rng(0))
    assert len(points) == 10_000
    assert np.allclose(points[:, 2], 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.abs(points[:, :2]).max() <= 50.0
    assert len(ideal(vertices, faces, 2.0).points) == 2_500


def test_normals_point_outwards():
    vertices, faces = sphere()
    normals = face_normals(vertices, faces)
    centres = vertices[faces].mean(axis=1)
    proper = np.linalg.norm(normals, axis=1) > 0.5  # the pole rows of a UV sphere are degenerate
    assert (np.einsum("ij,ij->i", normals[proper], centres[proper]) > 0).all()
    assert np.allclose(face_normals(vertices, faces[:, ::-1])[proper], normals[proper])


def test_camera_axes_are_right_handed():
    for yaw, pitch in ((0.0, 0.0), (30.0, -10.0), (-25.0, 8.0)):
        back, right, up = camera_axes(yaw, pitch)
        assert np.allclose(np.cross(right, up), back)
        assert up[1] > 0.9


def test_shot_sees_only_the_near_side_with_the_requested_spacing():
    from scipy.spatial import cKDTree

    vertices, faces = sphere()
    aim = np.array([0.0, 0.0, 80.0])
    scan = shoot(vertices, faces, Shot(distance=0.5, spacing=1.5, noise=0.0, depth_step=0.0), aim, seed=1)
    # yaw = pitch = 0: the camera frame is the mesh frame moved by the camera position.
    mesh = scan.points + np.array([0.0, 0.0, 580.0])
    assert np.allclose(np.linalg.norm(mesh, axis=1), 80.0, atol=0.2)
    assert mesh[:, 2].min() > 0.0
    near = mesh[mesh[:, 2] > 70.0]
    spacing = np.median(cKDTree(near).query(near, 2)[0][:, 1])
    assert 0.8 < spacing < 1.6
    assert np.all(vertices[scan.vertex][:, 2] > -5.0)


def test_shot_noise_follows_the_squared_distance():
    vertices, faces = plane(size=60.0)
    for distance in (0.4, 0.7):
        scan = shoot(vertices, faces, Shot(distance=distance, spacing=1.0, depth_step=0.0), np.zeros(3), seed=2)
        depth = scan.points[:, 2] + 1000.0 * distance
        assert np.std(depth) == pytest.approx(0.86 * distance**2, rel=0.15)


def test_depth_is_rounded_and_hidden_faces_leave_holes():
    vertices, faces = sphere()
    aim = np.array([0.0, 0.0, 80.0])
    scan = shoot(vertices, faces, Shot(distance=0.5, noise=0.0), aim, seed=3)
    along = -scan.points[:, 2]
    assert np.allclose(along, np.round(along, 1), atol=1e-6)
    front_cap = (vertices[faces][:, :, 2] > 60.0).all(axis=1)
    holed = shoot(vertices, faces, Shot(distance=0.5, noise=0.0), aim, seed=3, hidden=front_cap)
    seen = holed.points + np.array([0.0, 0.0, 580.0])
    # The cap is empty and the back of the sphere does not show through it.
    assert seen[:, 2].max() < 64.0  # triangles crossing z = 60 stay
    assert seen[:, 2].min() > 0.0
    assert len(holed.points) < len(scan.points)


def test_turned_shot_keeps_the_scene_rigid():
    vertices, faces = sphere()
    aim = np.array([0.0, 0.0, 80.0])
    scan = shoot(vertices, faces, Shot(distance=0.6, yaw=30.0, pitch=10.0, noise=0.0, depth_step=0.0), aim, seed=4)
    back, right, up = camera_axes(30.0, 10.0)
    camera = aim + 600.0 * back
    mesh = camera + scan.points @ np.vstack((right, up, back))
    assert np.allclose(np.linalg.norm(mesh, axis=1), 80.0, atol=0.2)
    assert (mesh @ back).min() > 0.0


def test_shot_rejects_distances_in_millimetres():
    vertices, faces = plane()
    with pytest.raises(ValueError, match="metres"):
        shoot(vertices, faces, Shot(distance=500.0), np.zeros(3))
