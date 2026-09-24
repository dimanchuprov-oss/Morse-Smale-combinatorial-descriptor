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
    # (x, y) distance between neighbouring raster samples, in point units.
    spacing: tuple[float, float] = (1.0, 1.0)


def _load_text_xyz(path: Path, delimiter: str | None = None) -> PointCloud:
    data = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError("text point cloud must contain at least x, y, z columns")
    return PointCloud(data[:, :3])


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _load_ply(path: Path) -> PointCloud:
    """Read the vertex element of an ASCII or binary PLY file with NumPy only.

    Supports the common case written by scanning apps (including iPhone LiDAR
    exporters): the vertex element comes first and has scalar properties.
    Other layouts raise ``RuntimeError`` so the caller can fall back to Open3D.
    """
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header has no end_header marker")
            decoded = line.decode("ascii", errors="replace").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        body = handle.read()
    if not header or header[0] != "ply":
        raise ValueError("not a PLY file")
    format_line = next((line.split() for line in header if line.startswith("format ")), None)
    if format_line is None:
        raise ValueError("PLY header has no format line")
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            elements.append((parts[1], int(parts[2]), []))
        elif parts[:1] == ["property"] and elements:
            if parts[1] == "list":
                elements[-1][2].append(("list", parts[-1]))
            else:
                elements[-1][2].append((parts[1], parts[2]))
    if not elements or elements[0][0] != "vertex":
        raise RuntimeError("PLY vertex element is not first; the lidar extra is required")
    _, count, properties = elements[0]
    names = [name for _, name in properties]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex element must contain x, y, z")
    indices = [names.index(axis) for axis in ("x", "y", "z")]
    if format_line[1] == "ascii":
        rows = [row for row in body.decode("ascii").splitlines() if row.strip()][:count]
        if len(rows) < count:
            raise ValueError("PLY file has fewer vertices than declared")
        fields = [row.split()[: len(properties)] for row in rows]
        if any(len(row) <= max(indices) for row in fields):
            raise ValueError("PLY vertex has fewer than three coordinates")
        values = np.asarray([[float(row[index]) for index in indices] for row in fields], dtype=float)
        return PointCloud(values)
    if any(kind == "list" or kind not in _PLY_TYPES for kind, _ in properties):
        raise RuntimeError("PLY vertex element has unsupported properties; the lidar extra is required")
    endian = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(format_line[1])
    if endian is None:
        raise ValueError(f"unknown PLY format: {format_line[1]}")
    dtype = np.dtype([(name, endian + _PLY_TYPES[kind]) for kind, name in properties])
    if len(body) < count * dtype.itemsize:
        raise ValueError("PLY file has fewer vertices than declared")
    vertices = np.frombuffer(body, dtype=dtype, count=count)
    return PointCloud(np.column_stack([vertices[axis].astype(float) for axis in ("x", "y", "z")]))


def _load_ascii_pcd(path: Path) -> PointCloud:
    with path.open("r", encoding="ascii") as handle:
        header: list[str] = []
        for line in handle:
            stripped = line.strip()
            header.append(stripped)
            if stripped.lower().startswith("data "):
                break
        if not header or not header[-1].lower().startswith("data "):
            raise ValueError("PCD header has no DATA declaration")
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
            return _load_ply(source)
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
    except (ModuleNotFoundError, ImportError):
        result = field.copy()
        # Propagation can need rows + columns - 2 steps when the only known
        # sample is in a corner.  A progress guard also handles pathological
        # inputs without looping forever.
        for _ in range(result.size):
            missing = ~np.isfinite(result)
            if not missing.any():
                return result
            before = int(missing.sum())
            padded = np.pad(result, 1, mode="edge")
            candidates = [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
            for candidate in candidates:
                take = missing & np.isfinite(candidate)
                result[take] = candidate[take]
            if int((~np.isfinite(result)).sum()) >= before:
                break
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
    """Project points to a median depth raster and fill uncovered cells.

    Sample ``(row, col)`` sits at ``lower + (col, row) * spacing``, so the
    raster can be meshed with its true metric spacing.
    """
    if rows < 3 or cols < 3:
        raise ValueError("raster dimensions must be at least 3x3")
    if len(set(horizontal_axes + (depth_axis,))) != 3:
        raise ValueError("horizontal and depth axes must be distinct")
    points = cloud.points
    horizontal = points[:, horizontal_axes]
    lower = horizontal.min(axis=0)
    upper = horizontal.max(axis=0)
    span = np.maximum(upper - lower, np.finfo(float).eps)
    spacing = span / np.array([cols - 1, rows - 1])
    coordinates = np.rint((horizontal - lower) / spacing).astype(np.int64)
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, cols - 1)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, rows - 1)
    cells = coordinates[:, 1] * cols + coordinates[:, 0]
    depth = points[:, depth_axis]
    ordered = np.lexsort((depth, cells))
    sorted_cells, sorted_depth = cells[ordered], depth[ordered]
    occupied, starts, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
    medians = 0.5 * (sorted_depth[starts + (counts - 1) // 2] + sorted_depth[starts + counts // 2])
    raster = np.full(rows * cols, np.nan, dtype=float)
    raster[occupied] = medians
    raster = raster.reshape(rows, cols)
    coverage = float(np.isfinite(raster).mean())
    if coverage == 0:
        raise ValueError("point cloud produced an empty depth raster")
    return DepthRaster(
        _fill_nearest(raster),
        coverage,
        (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])),
        (float(spacing[0]), float(spacing[1])),
    )


def set_up_axis(cloud: PointCloud, up: str) -> PointCloud:
    """Rotate the cloud so that the given axis becomes +Z (right-handed).

    ARKit/iPhone exports are Y-up; this package treats Z as height.
    """
    permutations = {"z": [0, 1, 2], "y": [2, 0, 1], "x": [1, 2, 0]}
    if up not in permutations:
        raise ValueError("up axis must be one of x, y, z")
    return PointCloud(cloud.points[:, permutations[up]])


def align_pca(cloud: PointCloud) -> PointCloud:
    """Centre the cloud and rotate its principal axes onto X, Y, Z.

    The direction of least variance becomes Z, which makes the height field
    independent of the scanner pose for roughly flat or elongated objects.
    The new Z keeps the orientation of the input +Z (so a dome stays a dome;
    set the up axis first with :func:`set_up_axis`), X is signed by the third
    moment and Y completes a right-handed frame, so the result is deterministic.
    """
    centred = cloud.points - cloud.points.mean(axis=0)
    if len(centred) < 3:
        raise ValueError("PCA alignment needs at least three points")
    _, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    basis = eigenvectors[:, ::-1].copy()  # largest variance first, smallest -> Z
    if basis[2, 2] < 0:  # new Z must not point against the input "up"
        basis[:, 2] *= -1.0
    if ((centred @ basis[:, 0]) ** 3).mean() < 0:
        basis[:, 0] *= -1.0
    basis[:, 1] = np.cross(basis[:, 2], basis[:, 0])
    return PointCloud(centred @ basis)


def remove_dominant_plane(
    cloud: PointCloud,
    distance: float,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[PointCloud, int]:
    """Drop the largest planar patch (floor, table, wall) with RANSAC.

    Returns the remaining points and how many were removed.
    """
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("plane distance must be positive")
    points = cloud.points
    if len(points) < 4:
        return cloud, 0
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        mask = np.abs((points - sample[0]) @ (normal / length)) <= distance
        if mask.sum() > best_mask.sum():
            best_mask = mask
    remaining = points[~best_mask]
    if len(remaining) == 0:
        raise ValueError("plane removal left no points")
    return PointCloud(remaining), int(best_mask.sum())
