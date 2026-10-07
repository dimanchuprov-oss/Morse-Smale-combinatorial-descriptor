"""Verify that two 3D head scans show the same person, from any viewing angle.

Two scans (full heads or partial views taken from arbitrary directions) are
compared in three steps:

1. Both clouds are reduced to the head: the top of the cloud along the up axis,
   a fixed height below it and a fixed radius around the vertical head axis.
   Small fragments and body parts that do not touch the head (shoulders, chest)
   are dropped. The head is moved so that its axis is the Z axis and its top is
   at ``z = 0``.
2. The scans are rigidly registered in both directions. ICP is started from a
   grid of rotations about the vertical axis, small tilts and shifts; at every
   stage only point pairs closer than a shrinking distance gate are used, so a
   small overlap is not pulled towards the parts that only one scan sees. The
   pose under which most points of one scan lie on the other wins. The viewing
   angles therefore do not need to be known.
3. Scores are computed on the registered pair:

   * ``geometric``: RMS point-to-plane distance between the scans over the
     surface where they come within ``GATE`` of each other, averaged over both
     directions;
   * ``shared_cm2``: area of that common surface, estimated from the point
     density of each scan. Small scores on a small common area are not
     evidence: a patch fits many faces;
   * ``topological``: bottleneck distance between the lower-star persistence
     diagrams of the radial height field ``r(theta, z)`` of both heads, taken
     over the cylindrical cells seen by both scans (see :func:`radial_map`);
   * ``overlap``: fraction of probe points within 3 mm of the gallery.

The geometric score is the decision baseline; the topological one shows how
the project's persistence descriptor behaves on faces. Neither is a trained
biometric system: thresholds must be estimated on repeat scans of real people.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mesh import grid_mesh
from .pointcloud import PointCloud, set_up_axis, voxel_downsample

# Roughly the middle of an adult head below its top; normals point away from it.
HEAD_CENTRE = np.array([0.0, 0.0, -110.0])
# Sanity limits for the head section 50-120 mm below the top: the widest adult
# head with hair is well under 350 mm, while a section through a body lying
# along the wrong axis is about a metre long.
MAX_SECTION_WIDTH = 350.0
MIN_HEAD_HEIGHT = 120.0
MAX_SLOPE_TOLERANCE = 30.0
# Pieces closer than FRAGMENT_LINK mm form one part of the scan. Parts with less
# than MIN_FRAGMENT of the points are noise; parts that lie entirely more than
# DETACHED_BELOW mm under the top are the shoulders or the chest seen below the chin.
FRAGMENT_LINK = 6.0
MIN_FRAGMENT = 0.03
DETACHED_BELOW = 110.0
# Scores compare the surfaces where the registered scans come within GATE mm.
GATE = 5.0
# A pose is judged by the fraction of points within INLIER_DISTANCE mm of the other scan.
INLIER_DISTANCE = 2.5
# The surface a point stands for is estimated from its AREA_NEIGHBOURS nearest
# neighbours, so that sparse and dense scans measure the same area.
AREA_NEIGHBOURS = 12
# Bumped whenever the crop or the scores change: calibrations of another version
# do not apply.
METHOD_VERSION = 2
# Start shifts (mm, gallery frame) for registration: the centres of two partial
# views of one head can lie several centimetres apart.
SHIFTS = ((0.0, 0.0, 0.0), (40.0, 0.0, 0.0), (-40.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, -40.0, 0.0),
          (0.0, 0.0, 30.0), (0.0, 0.0, -30.0))  # fmt: skip


def _kd_tree(points: np.ndarray) -> Any:
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional signal extra: python -m pip install '.[signal]'") from error
    return cKDTree(points)


def upright(points: np.ndarray, up_axis: str = "z", scale: float = 1.0) -> np.ndarray:
    """Points in millimetres with the given up axis turned to +Z."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    return set_up_axis(PointCloud(np.asarray(points, dtype=float) * scale), up_axis).points


@dataclass(frozen=True)
class Head:
    """Head points in millimetres, Z up, head axis on the Z axis, top at z = 0."""

    points: np.ndarray
    # Where the canonical origin lies in the upright input frame.
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Fraction of the cropped points dropped as fragments or detached body parts.
    dropped: float = 0.0


def locate_head(cloud: np.ndarray) -> tuple[np.ndarray, float]:
    """Horizontal head axis and top height of an upright cloud in millimetres."""
    top = float(np.quantile(cloud[:, 2], 0.999))
    band = cloud[(cloud[:, 2] > top - 120.0) & (cloud[:, 2] < top - 50.0)]
    if len(band) < 50:
        raise ValueError("too few points below the top of the cloud to find the head; check --up-axis and --units")
    low, high = np.quantile(band[:, :2], [0.02, 0.98], axis=0)
    if (high - low).max() > MAX_SECTION_WIDTH:
        raise ValueError(
            f"the section below the top is {(high - low).max():.0f} mm wide, not a head; "
            "the up axis or the units are probably wrong"
        )
    return _head_axis(band[:, :2]), top


def _attached(points: np.ndarray) -> np.ndarray:
    """Mask of the parts of a canonical head that belong to the head itself.

    Parts are groups of points linked by gaps under ``FRAGMENT_LINK``. Small
    parts are scanner noise; parts lying entirely more than ``DETACHED_BELOW``
    under the top are shoulders or chest, whose shape depends on clothes and
    posture rather than on the person.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional signal extra: python -m pip install '.[signal]'") from error
    pairs = _kd_tree(points).query_pairs(FRAGMENT_LINK, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    highest = np.full(len(sizes), -np.inf)
    np.maximum.at(highest, labels, points[:, 2])
    return (sizes[labels] >= max(50, MIN_FRAGMENT * len(points))) & (highest[labels] > -DETACHED_BELOW)


def extract_head(
    points: np.ndarray,
    up_axis: str = "z",
    scale: float = 1.0,
    height: float = 230.0,
    radius: float = 140.0,
    voxel: float = 2.0,
) -> Head:
    """Cut the head out of a body or head scan.

    ``scale`` converts input units to millimetres (1000 for metres). The head
    axis comes from the horizontal section 50-120 mm below the top, which lies
    above the shoulders for an upright person. Noise fragments and body parts
    that do not touch the head are dropped (see :func:`_attached`).
    """
    if height <= 0 or radius <= 0 or voxel <= 0:
        raise ValueError("height, radius and voxel must be positive")
    cloud = upright(points, up_axis, scale)
    centre, top = locate_head(cloud)
    distance = np.hypot(cloud[:, 0] - centre[0], cloud[:, 1] - centre[1])
    keep = (cloud[:, 2] > top - height) & (cloud[:, 2] <= top) & (distance < radius)
    if keep.sum() < 100:
        raise ValueError("head region contains fewer than 100 points")
    extent = top - float(np.quantile(cloud[keep, 2], 0.01))
    if extent < min(MIN_HEAD_HEIGHT, 0.7 * height):
        raise ValueError(f"the head region is only {extent:.0f} mm tall; the up axis or the units are probably wrong")
    offset = np.array([centre[0], centre[1], top])
    head = voxel_downsample(PointCloud(cloud[keep] - offset), voxel).points
    attached = _attached(head)
    if attached.sum() < 100:
        raise ValueError("head region contains fewer than 100 points after removing fragments")
    return Head(head[attached], (float(offset[0]), float(offset[1]), float(offset[2])), float(1.0 - attached.mean()))


def _head_axis(section: np.ndarray) -> np.ndarray:
    """Centre of a horizontal head section, also when only an arc is visible.

    A partial scan sees one side of the head, so the mean of its points lies
    several centimetres in front of the axis. A least-squares circle (Kasa fit)
    recovers the axis from an arc; implausible fits fall back to the median.
    """
    median = np.median(section, axis=0)
    x, y = (section - median).T
    design = np.column_stack((x, y, np.ones_like(x)))
    (a, b, c), *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    centre = np.array([a / 2.0, b / 2.0])
    radius = np.sqrt(max(c + centre @ centre, 0.0))
    if not 50.0 <= radius <= 130.0 or np.linalg.norm(centre) > 130.0:
        return median
    return median + centre


def estimate_normals(points: np.ndarray, centre: np.ndarray, neighbours: int = 16) -> np.ndarray:
    """Unit normals by local PCA, oriented away from ``centre``."""
    count = min(neighbours, len(points))
    if count < 3:
        raise ValueError("normals need at least three points")
    _, index = _kd_tree(points).query(points, count)
    local = points[index] - points[index].mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
    normals = vectors[:, :, 0]
    flip = np.einsum("ij,ij->i", normals, points - centre) < 0
    normals[flip] *= -1.0
    return normals


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _rotation_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def view_direction(yaw: float, pitch: float = 0.0, front: float = 0.0) -> np.ndarray:
    """Unit vector from the head towards a camera at the given angles (radians).

    ``front`` is the azimuth of the face; ``yaw`` turns the camera around the
    head, positive ``pitch`` raises it.
    """
    azimuth = front + yaw
    return np.array([np.cos(pitch) * np.cos(azimuth), np.cos(pitch) * np.sin(azimuth), np.sin(pitch)])


def visible_from(
    points: np.ndarray,
    direction: np.ndarray,
    normals: np.ndarray | None = None,
    cell: float = 2.0,
    tolerance: float = 6.0,
    min_cosine: float = 0.1,
) -> np.ndarray:
    """Mask of points seen by an orthographic camera looking along ``-direction``.

    A point must lie within ``tolerance`` of the nearest surface over its 3x3
    neighbourhood of image cells. The neighbourhood matters: in a sparse cloud
    a single cell can be empty on the front surface and let a point of the back
    of the head through. With ``normals`` a point must also face the camera,
    and the tolerance grows with the slope of its surface: over the 3x3
    neighbourhood a visible surface tilted by theta rises by up to
    ``2 sqrt(2) cell tan(theta)``, which would otherwise cut away visible
    cheeks and the sides of the nose.
    """
    try:
        from scipy.ndimage import maximum_filter
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional signal extra: python -m pip install '.[signal]'") from error
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    depth = points @ direction
    cells = np.floor(np.column_stack((points @ right, points @ up)) / cell).astype(np.int64)
    cells -= cells.min(axis=0)
    shape = tuple(cells.max(axis=0) + 1)
    nearest = np.full(shape, -np.inf)
    np.maximum.at(nearest, (cells[:, 0], cells[:, 1]), depth)
    nearest = maximum_filter(nearest, size=3, mode="constant", cval=-np.inf)
    allowed = np.full(len(points), tolerance)
    facing = np.ones(len(points), dtype=bool)
    if normals is not None:
        cosine = normals @ direction
        facing = cosine > min_cosine
        slope = np.sqrt(np.maximum(1.0 - cosine**2, 0.0)) / np.maximum(cosine, min_cosine)
        allowed += np.minimum(2.0 * np.sqrt(2.0) * cell * slope, MAX_SLOPE_TOLERANCE)
    return facing & (depth >= nearest[cells[:, 0], cells[:, 1]] - allowed)


def simulate_view(
    cloud: np.ndarray,
    yaw: float,
    pitch: float = 0.0,
    front: float = 0.0,
    normals: np.ndarray | None = None,
    noise: float = 0.3,
    keep: float = 0.6,
    seed: int = 0,
    scramble: bool = True,
) -> np.ndarray:
    """What a scanner at the given angles sees of an upright scan, in a random pose.

    ``cloud`` is a whole upright scan in millimetres (not a cut-out head), so
    the result has to go through :func:`extract_head` like a real probe.
    Normals are estimated around the head when not given. Gaussian noise (mm)
    and random subsampling make the probe differ from the gallery point by
    point; ``scramble`` applies a random rigid motion so that registration has
    to recover the pose from scratch.
    """
    rng = np.random.default_rng(seed)
    if normals is None:
        centre, top = locate_head(cloud)
        normals = estimate_normals(cloud, np.array([centre[0], centre[1], top + HEAD_CENTRE[2]]))
    seen = cloud[visible_from(cloud, view_direction(yaw, pitch, front), normals)]
    seen = seen[rng.random(len(seen)) < keep]
    seen = seen + rng.normal(0.0, noise, seen.shape)
    if scramble:
        rotation = rotation_z(rng.uniform(-np.pi, np.pi)) @ rotation_y(np.radians(rng.uniform(-8.0, 8.0)))
        seen = seen @ rotation.T + rng.uniform(-60.0, 60.0, 3)
    return seen


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, target_mean - rotation @ source_mean


@dataclass(frozen=True)
class Enrolled:
    """A head prepared for matching: points, normals, surface per point (mm^2) and a KD-tree."""

    points: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    tree: Any = field(repr=False)


def _point_areas(points: np.ndarray, tree: Any) -> np.ndarray:
    """Surface each point stands for, pi r^2 / (k + 1/2) with r the distance to the k-th neighbour.

    Unlike a fixed area per point this does not depend on how densely the
    scanner sampled the face (within a few per cent from 1.5 to 3.5 mm spacing).
    """
    count = min(AREA_NEIGHBOURS, len(points) - 1)
    distance, _ = tree.query(points, count + 1)
    return np.pi * distance[:, -1] ** 2 / (count + 0.5)


def enroll(head: Head | np.ndarray) -> Enrolled:
    points = head.points if isinstance(head, Head) else np.asarray(head, dtype=float)
    if len(points) < 30:
        raise ValueError("a head needs at least 30 points")
    tree = _kd_tree(points)
    return Enrolled(points, estimate_normals(points, HEAD_CENTRE), _point_areas(points, tree), tree)


def _gated_point_to_point(
    source: np.ndarray, gallery: Enrolled, rotation: np.ndarray, translation: np.ndarray, gates: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse ICP: Kabsch steps on the pairs closer than each gate in turn."""
    for gate in gates:
        for _ in range(3):
            moved = source @ rotation.T + translation
            # The bound lets the tree skip far branches; missing neighbours come back as inf.
            distance, index = gallery.tree.query(moved, distance_upper_bound=gate)
            inliers = distance < gate
            if inliers.sum() < 10:
                break
            step_rotation, step_translation = _kabsch(moved[inliers], gallery.points[index[inliers]])
            rotation, translation = step_rotation @ rotation, step_rotation @ translation + step_translation
    return rotation, translation


def _gated_point_to_plane(
    source: np.ndarray, gallery: Enrolled, rotation: np.ndarray, translation: np.ndarray, gates: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Fine ICP: point-to-plane steps on the pairs closer than each gate in turn.

    A fixed trimming ratio (keep the closest 80%) biases the pose when the scans
    share less than that fraction of their surface; a distance gate does not.
    """
    for gate in gates:
        for _ in range(8):
            moved = source @ rotation.T + translation
            distance, index = gallery.tree.query(moved, distance_upper_bound=gate)
            inliers = distance < gate
            if inliers.sum() < 30:
                break
            moved, target, normal = moved[inliers], gallery.points[index[inliers]], gallery.normals[index[inliers]]
            # Linearised (R p + t - q) . n = 0 for a small rotation vector w:
            # w . (p x n) + t . n = (q - p) . n.
            system = np.column_stack((np.cross(moved, normal), normal))
            solution, *_ = np.linalg.lstsq(system, np.einsum("ij,ij->i", target - moved, normal), rcond=None)
            step_rotation, step_translation = _rotation_vector(solution[:3]), solution[3:]
            rotation, translation = step_rotation @ rotation, step_rotation @ translation + step_translation
            if np.linalg.norm(step_translation) < 1e-3 and np.trace(step_rotation) > 3.0 - 1e-8:
                break
    return rotation, translation


def _inliers(points: np.ndarray, gallery: Enrolled, rotation: np.ndarray, translation: np.ndarray) -> float:
    distance, _ = gallery.tree.query(points @ rotation.T + translation, distance_upper_bound=INLIER_DISTANCE)
    return float(np.mean(distance < INLIER_DISTANCE))


def _upper_centre(points: np.ndarray) -> np.ndarray:
    """Centroid of the part within 150 mm under the top, which every view of a face shares."""
    upper = points[points[:, 2] > -150.0]
    return (upper if len(upper) >= 30 else points).mean(axis=0)


@dataclass(frozen=True)
class Registration:
    rotation: np.ndarray
    translation: np.ndarray
    # Fraction of the moved points within INLIER_DISTANCE of the target.
    inliers: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation

    def inverse(self, inliers: float) -> Registration:
        return Registration(self.rotation.T, -self.rotation.T @ self.translation, inliers)


def register(
    probe: np.ndarray,
    gallery: Enrolled | np.ndarray,
    yaw_step: float = 15.0,
    tilts: tuple[float, ...] = (-15.0, 0.0, 15.0),
    shifts: tuple[tuple[float, float, float], ...] = SHIFTS,
    refine: int = 8,
    coarse_gates: tuple[float, ...] = (25.0, 15.0, 10.0),
    fine_gates: tuple[float, ...] = (8.0, 5.0, 3.5, 2.5, 2.0),
    coarse_sample: int = 300,
    fine_sample: int = 4000,
    seed: int = 0,
) -> Registration:
    """Rigidly align ``probe`` to ``gallery`` (both canonical heads).

    Starts combine every rotation about the vertical axis (``yaw_step``
    degrees) with ``tilts`` about the probe's y axis (a nod or a roll depending
    on the probe pose), turning the probe about the centre of its upper part,
    placing that centre on the gallery's and adding ``shifts``. Coarse
    point-to-point ICP runs from every start on a sample of the probe; the
    ``refine`` starts with most inliers are refined by point-to-plane ICP on a
    larger sample, and the pose with the largest fraction of all probe points
    within ``INLIER_DISTANCE`` of the gallery wins.
    """
    gallery = gallery if isinstance(gallery, Enrolled) else enroll(gallery)
    probe = np.asarray(probe, dtype=float)
    if len(probe) < 30:
        raise ValueError("a probe head needs at least 30 points")
    rng = np.random.default_rng(seed)
    coarse = probe if len(probe) <= coarse_sample else probe[rng.choice(len(probe), coarse_sample, replace=False)]
    fine = probe if len(probe) <= fine_sample else probe[rng.choice(len(probe), fine_sample, replace=False)]
    probe_centre, gallery_centre = _upper_centre(probe), _upper_centre(gallery.points)
    starts = []
    for tilt in np.radians(tilts):
        for yaw in np.radians(np.arange(0.0, 360.0, yaw_step)):
            rotation = rotation_z(yaw) @ rotation_y(tilt)
            for shift in shifts:
                translation = gallery_centre + np.asarray(shift) - rotation @ probe_centre
                rotation_, translation_ = _gated_point_to_point(coarse, gallery, rotation, translation, coarse_gates)
                starts.append((_inliers(coarse, gallery, rotation_, translation_), rotation_, translation_))
    starts.sort(key=lambda item: -item[0])
    best: Registration | None = None
    for _, rotation, translation in starts[:refine]:
        rotation, translation = _gated_point_to_plane(fine, gallery, rotation, translation, fine_gates)
        candidate = Registration(rotation, translation, _inliers(probe, gallery, rotation, translation))
        if best is None or candidate.inliers > best.inliers:
            best = candidate
    assert best is not None
    return best


def radial_map(points: np.ndarray, theta_bins: int = 180, z_cell: float = 2.5, height: float = 250.0) -> np.ndarray:
    """Median distance from the head axis on a (z, theta) grid; NaN where empty.

    Points above the top or more than ``height`` below it are ignored.
    """
    rows = int(np.ceil(height / z_cell))
    inside = (points[:, 2] <= 0.0) & (points[:, 2] > -height)
    points = points[inside]
    theta = np.arctan2(points[:, 1], points[:, 0])
    column = np.clip(((theta + np.pi) / (2 * np.pi) * theta_bins).astype(np.int64), 0, theta_bins - 1)
    row = np.clip((-points[:, 2] / z_cell).astype(np.int64), 0, rows - 1)
    radius = np.hypot(points[:, 0], points[:, 1])
    cell = row * theta_bins + column
    order = np.lexsort((radius, cell))
    cell, radius = cell[order], radius[order]
    occupied, starts, counts = np.unique(cell, return_index=True, return_counts=True)
    result = np.full(rows * theta_bins, np.nan)
    result[occupied] = 0.5 * (radius[starts + (counts - 1) // 2] + radius[starts + counts // 2])
    return result.reshape(rows, theta_bins)


def _morphology(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Closing then opening; edge padding keeps the border of the map intact."""
    from scipy.ndimage import binary_closing, binary_opening

    inner = (slice(iterations, -iterations), slice(iterations, -iterations))
    for operation in (binary_closing, binary_opening):
        mask = operation(np.pad(mask, iterations, mode="edge"), iterations=iterations)[inner]
    return mask


def _shared_fields(first: np.ndarray, second: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter, label
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional signal extra: python -m pip install '.[signal]'") from error
    shared = np.isfinite(first) & np.isfinite(second)
    if not shared.any():
        raise ValueError("the scans do not overlap on the radial map")
    # Put the azimuth seam opposite the shared region, so that the result does
    # not depend on where the face points in the input frame.
    angle = (np.nonzero(shared)[1] + 0.5) / shared.shape[1] * 2 * np.pi
    mean = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) % (2 * np.pi)
    shift = shared.shape[1] // 2 - int(mean / (2 * np.pi) * shared.shape[1])
    first, second, shared = (np.roll(values, shift, axis=1) for values in (first, second, shared))
    components, count = label(_morphology(shared))
    if count == 0:
        raise ValueError("the scans do not overlap on the radial map")
    largest = components == 1 + int(np.argmax(np.bincount(components.ravel())[1:]))
    rows, cols = np.nonzero(largest)
    window = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    mask = largest[window]
    fields = []
    for values in (first[window], second[window]):
        known = mask & np.isfinite(values)
        nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
        fields.append(gaussian_filter(values[tuple(nearest)], sigma))
    return fields[0], fields[1], float(largest.mean())


def radial_bottleneck(probe: np.ndarray, gallery: np.ndarray, sigma: float = 2.0) -> tuple[float, float]:
    """Bottleneck distance (mm) between radial-height persistence diagrams.

    Both diagrams are computed on the same window of the radial map: the
    largest region seen by both scans, with gaps filled from the nearest seen
    cell. Returns the distance and the fraction of the map that was compared.
    """
    from .compare import bottleneck
    from .persistence import critical_pairs, persistence_diagram

    first, second, fraction = _shared_fields(radial_map(probe), radial_map(gallery), sigma)
    if min(first.shape) < 3:
        raise ValueError("overlap region is too small for a persistence diagram")
    diagrams = [persistence_diagram(critical_pairs(grid_mesh(values))) for values in (first, second)]
    distance = max(
        bottleneck(diagrams[0].get(dimension, []), diagrams[1].get(dimension, []))
        for dimension in set(diagrams[0]) | set(diagrams[1])
    )
    return float(distance), fraction


@dataclass(frozen=True)
class Match:
    geometric: float
    shared_cm2: float
    topological: float | None
    overlap: float
    radial_fraction: float | None
    registration: Registration = field(repr=False)
    topology_error: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "geometric_mm": self.geometric,
            "shared_cm2": self.shared_cm2,
            "topological_mm": self.topological,
            "overlap": self.overlap,
            "radial_fraction": self.radial_fraction,
            "topology_error": self.topology_error,
        }


def _surface_distance(points: np.ndarray, other: Enrolled) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the nearest point of ``other`` and to its tangent plane there."""
    distance, index = other.tree.query(points)
    return distance, np.abs(np.einsum("ij,ij->i", points - other.points[index], other.normals[index]))


def _pair_scores(probe: Enrolled, gallery: Enrolled, registration: Registration) -> tuple[float, float, float]:
    """Geometric score (mm), shared area (cm^2) and overlap of a registered pair."""
    moved = registration.apply(probe.points)
    moved = Enrolled(moved, probe.normals @ registration.rotation.T, probe.areas, _kd_tree(moved))
    rms, areas = [], []
    for source, target in ((moved, gallery), (gallery, moved)):
        distance, plane = _surface_distance(source.points, target)
        near = distance < GATE
        if not near.any():
            raise ValueError("the scans do not overlap after registration")
        rms.append(float(np.sqrt(np.mean(plane[near] ** 2))))
        areas.append(float(source.areas[near].sum()) / 100.0)
    forward, _ = gallery.tree.query(moved.points)
    return float(np.mean(rms)), float(np.mean(areas)), float(np.mean(forward < 3.0))


def match(probe: Enrolled | np.ndarray, gallery: Enrolled | np.ndarray, topology: bool = True, seed: int = 0) -> Match:
    """Register two canonical heads and score the pair.

    The probe is registered to the gallery and the gallery to the probe; the
    pose that puts more probe points on the gallery is kept, so a partial view
    is matched as well from either side. ``geometric`` and ``shared_cm2``
    average both directions, but since the pose is chosen by the probe's
    inliers, swapping the arguments can change them slightly.
    """
    probe = probe if isinstance(probe, Enrolled) else enroll(probe)
    gallery = gallery if isinstance(gallery, Enrolled) else enroll(gallery)
    forward = register(probe.points, gallery, seed=seed)
    backward = register(gallery.points, probe, seed=seed)
    backward = backward.inverse(
        _inliers(probe.points, gallery, backward.rotation.T, -backward.rotation.T @ backward.translation)
    )
    registration = forward if forward.inliers >= backward.inliers else backward
    geometric, shared, overlap = _pair_scores(probe, gallery, registration)
    topological = fraction = error = None
    if topology:
        # A failed persistence comparison (no shared region) must not hide a valid geometric score.
        try:
            topological, fraction = radial_bottleneck(registration.apply(probe.points), gallery.points)
        except ValueError as failure:
            error = str(failure)
    return Match(geometric, shared, topological, overlap, fraction, registration, error)


def equal_error_rate(genuine: Iterable[float], impostor: Iterable[float]) -> tuple[float, float]:
    """Equal error rate and its threshold for "same person if score <= threshold"."""
    genuine, impostor = np.asarray(list(genuine), float), np.asarray(list(impostor), float)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("need genuine and impostor scores")
    scores = np.unique(np.concatenate((genuine, impostor)))
    # Candidate thresholds lie between neighbouring scores, so that a clean
    # separation yields the middle of the gap rather than one of its edges.
    candidates = np.concatenate(([scores[0] - 1.0], 0.5 * (scores[1:] + scores[:-1]), [scores[-1] + 1.0]))
    false_reject = (genuine[None, :] > candidates[:, None]).mean(axis=1)
    false_accept = (impostor[None, :] <= candidates[:, None]).mean(axis=1)
    gap = np.abs(false_reject - false_accept)
    tied = np.flatnonzero(gap == gap.min())
    best = tied[len(tied) // 2]
    return float(0.5 * (false_reject[best] + false_accept[best])), float(candidates[best])
