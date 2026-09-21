"""LiDAR point-cloud input and projection to a regular depth map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point cloud must have shape (n, 3)")
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("point cloud must contain finite points")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class DepthRaster:
    depth: np.ndarray
    coverage: float
    bounds: tuple[float, float, float, float]


def _load_text_xyz(path: Path, delimiter: str | None = None) -> PointCloud:
    data = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError("text point cloud must contain at least x, y, z columns")
    return PointCloud(data[:, :3])


def _load_ascii_ply(path: Path) -> PointCloud:
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header has no end_header marker")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if not header or header[0] != "ply" or not any(line == "format ascii 1.0" for line in header):
            raise RuntimeError("binary PLY requires the optional lidar extra")
        vertex_line = next((line for line in header if line.startswith("element vertex ")), None)
        if vertex_line is None:
            raise ValueError("PLY has no vertex element")
        count = int(vertex_line.split()[-1])
        points = []
        for _ in range(count):
            values = handle.readline().decode("ascii").split()
            if len(values) < 3:
                raise ValueError("PLY vertex has fewer than three coordinates")
            points.append([float(values[0]), float(values[1]), float(values[2])])
    return PointCloud(np.asarray(points, dtype=float))


def _load_ascii_pcd(path: Path) -> PointCloud:
    with path.open("r", encoding="ascii") as handle:
        header: list[str] = []
        for line in handle:
            stripped = line.strip()
            header.append(stripped)
            if stripped.lower().startswith("data "):
                break
        data_mode = header[-1].lower().split()[-1]
        if data_mode != "ascii":
            raise RuntimeError("binary PCD requires the optional lidar extra")
        fields = next((line.split()[1:] for line in header if line.upper().startswith("FIELDS ")), None)
        if fields is None or not {"x", "y", "z"}.issubset(fields):
            raise ValueError("PCD must contain x, y, z fields")
        indices = [fields.index(name) for name in ("x", "y", "z")]
        values = [[float(item) for item in line.split()] for line in handle if line.strip()]
    array = np.asarray(values, dtype=float)
    return PointCloud(array[:, indices])


def _load_open3d(path: Path) -> PointCloud:
    try:
        import open3d as o3d
    except (ModuleNotFoundError, ImportError) as error:
        raise RuntimeError("install the optional lidar extra for binary PLY/PCD input") from error
    cloud = o3d.io.read_point_cloud(str(path))
    return PointCloud(np.asarray(cloud.points))


def load_point_cloud(path: str | Path) -> PointCloud:
    """Load PLY, PCD, LAS/LAZ, XYZ, CSV, or NumPy point arrays."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return PointCloud(np.load(source))
    if suffix == ".ply":
        try:
            return _load_ascii_ply(source)
        except RuntimeError:
            return _load_open3d(source)
    if suffix == ".pcd":
        try:
            return _load_ascii_pcd(source)
        except RuntimeError:
            return _load_open3d(source)
    if suffix in {".las", ".laz"}:
        try:
            import laspy
        except ModuleNotFoundError as error:
            raise RuntimeError("install the optional lidar extra for LAS/LAZ input") from error
        las = laspy.read(source)
        return PointCloud(np.column_stack((las.x, las.y, las.z)))
    if suffix in {".xyz", ".txt"}:
        return _load_text_xyz(source)
    if suffix == ".csv":
        return _load_text_xyz(source, delimiter=",")
    raise ValueError(f"unsupported point-cloud format: {suffix or '<none>'}")


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    origin = cloud.points.min(axis=0)
    keys = np.floor((cloud.points - origin) / voxel_size).astype(np.int64)
    _, selected = np.unique(keys, axis=0, return_index=True)
    return PointCloud(cloud.points[np.sort(selected)])


def _fill_nearest(field: np.ndarray) -> np.ndarray:
    if np.isfinite(field).all():
        return field
    try:
        from scipy.ndimage import distance_transform_edt
    except ModuleNotFoundError:
        result = field.copy()
        for _ in range(max(result.shape)):
            missing = ~np.isfinite(result)
            if not missing.any():
                return result
            padded = np.pad(result, 1, mode="edge")
            candidates = [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
            for candidate in candidates:
                take = missing & np.isfinite(candidate)
                result[take] = candidate[take]
        raise ValueError("unable to fill depth raster; install the signal extra")
    missing = ~np.isfinite(field)
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return field[tuple(nearest)]


def rasterize_depth(
    cloud: PointCloud,
    rows: int = 256,
    cols: int = 256,
    horizontal_axes: tuple[int, int] = (0, 1),
    depth_axis: int = 2,
) -> DepthRaster:
    """Project points to a median depth raster and fill uncovered cells."""
    if rows < 3 or cols < 3:
        raise ValueError("raster dimensions must be at least 3x3")
    if len(set(horizontal_axes + (depth_axis,))) != 3:
        raise ValueError("horizontal and depth axes must be distinct")
    points = cloud.points
    horizontal = points[:, horizontal_axes]
    lower = horizontal.min(axis=0)
    upper = horizontal.max(axis=0)
    span = np.maximum(upper - lower, np.finfo(float).eps)
    coordinates = np.floor((horizontal - lower) / span * np.array([cols - 1, rows - 1])).astype(int)
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, cols - 1)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, rows - 1)
    buckets: list[list[float]] = [[] for _ in range(rows * cols)]
    for (x, y), depth in zip(coordinates, points[:, depth_axis]):
        buckets[y * cols + x].append(float(depth))
    raster = np.full((rows, cols), np.nan, dtype=float)
    for index, values in enumerate(buckets):
        if values:
            raster[index // cols, index % cols] = float(np.median(values))
    coverage = float(np.isfinite(raster).mean())
    if coverage == 0:
        raise ValueError("point cloud produced an empty depth raster")
    return DepthRaster(_fill_nearest(raster), coverage, (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])))
