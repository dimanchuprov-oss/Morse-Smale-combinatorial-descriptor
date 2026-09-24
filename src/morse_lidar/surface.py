"""Surface reconstruction backends for LiDAR point clouds."""

from __future__ import annotations

import numpy as np

from .mesh import TriMesh
from .pointcloud import PointCloud


def _unique_projected_points(
    cloud: PointCloud,
    horizontal_axes: tuple[int, int],
) -> np.ndarray:
    """Collapse points sharing one horizontal position to their median."""
    horizontal = cloud.points[:, horizontal_axes]
    unique, inverse = np.unique(horizontal, axis=0, return_inverse=True)
    if len(unique) == len(cloud.points):
        return cloud.points.copy()
    inverse = np.asarray(inverse).ravel()
    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    return np.asarray([np.median(cloud.points[group], axis=0) for group in groups])


def reconstruct_delaunay(
    cloud: PointCloud,
    horizontal_axes: tuple[int, int] = (0, 1),
    scalar_axis: int = 2,
) -> TriMesh:
    """Build an open 2.5D surface by Delaunay triangulation in XY."""
    if len(set(horizontal_axes + (scalar_axis,))) != 3:
        raise ValueError("horizontal and scalar axes must be distinct")
    points = _unique_projected_points(cloud, horizontal_axes)
    try:
        from scipy.spatial import Delaunay
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional signal extra for Delaunay reconstruction") from error
    if len(points) < 3:
        raise ValueError("at least three projected points are required")
    triangulation = Delaunay(points[:, horizontal_axes])
    faces = np.asarray(triangulation.simplices, dtype=np.int64)
    # SciPy does not guarantee a consistent winding. Orient every triangle so
    # its normal points towards +scalar_axis; signed curvature relies on it.
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = normals[:, scalar_axis] < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return TriMesh(points, faces, points[:, scalar_axis])


def reconstruct_poisson(
    cloud: PointCloud,
    depth: int = 8,
    density_quantile: float = 0.02,
) -> TriMesh:
    """Reconstruct a 3D surface with Open3D Poisson reconstruction.

    Trimming low-density vertices (``density_quantile > 0``) removes the
    hallucinated surface far from the data but opens the mesh; pass ``0`` to
    keep the watertight result required by the TTK backend.
    """
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not np.isfinite(density_quantile) or not 0 <= density_quantile < 1:
        raise ValueError("density_quantile must be in [0, 1)")
    try:
        import open3d as o3d
    except (ModuleNotFoundError, ImportError) as error:
        raise RuntimeError(
            "Open3D is unavailable; install the lidar extra and its native macOS libusb dependency"
        ) from error
    if len(cloud.points) < 30:
        raise ValueError("Poisson reconstruction requires at least 30 points")
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.points))
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(min(30, len(cloud.points) - 1))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    density = np.asarray(densities)
    if density_quantile:
        threshold = float(np.quantile(density, density_quantile))
        mesh.remove_vertices_by_mask(density < threshold)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Poisson reconstruction produced an empty mesh")
    return TriMesh(vertices, faces, vertices[:, 2])


def reconstruct_surface(cloud: PointCloud, method: str = "delaunay", **kwargs: object) -> TriMesh:
    if method == "delaunay":
        return reconstruct_delaunay(cloud, **kwargs)
    if method == "poisson":
        return reconstruct_poisson(cloud, **kwargs)
    raise ValueError(f"unknown reconstruction method: {method}")
