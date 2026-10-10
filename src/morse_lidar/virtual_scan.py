"""A virtual single-shot depth scanner for triangle meshes.

Turns a face mesh into what a structured-light scanner such as the Revopoint
Range sees in one shot, so that tracked meshes with known expressions can be
compared like real scans:

* a pinhole camera at ``distance`` from the aim point, turned by ``yaw`` and
  ``pitch`` around the head, sees only the nearest surface along each ray;
* samples lie ``spacing`` mm apart at the aim point (1.2-2.2 mm on the real
  files) and surfaces seen more obliquely than ``max_angle`` are lost;
* depth noise grows with the square of the distance, sigma = 0.86 z^2 mm with z
  in metres (0.14 mm at 0.4 m, 0.42 mm at 0.7 m, as measured on the scanner),
  and the exported depth is rounded to 0.1 mm;
* parts the scanner does not return (brows, hair) can be hidden: they leave
  holes instead of letting the surface behind them through.

Points come out in the camera frame, Y up, the camera at the origin looking
along -Z, in millimetres: the same convention as the real exports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Depth noise sigma = NOISE * z^2 mm with z in metres (bench measurement of 2026-10-07).
NOISE = 0.86
# The scanner exports depth rounded to this step, mm.
DEPTH_STEP = 0.1
# Surface samples per scanner pixel: enough that every pixel sees its nearest surface.
OVERSAMPLING = 16


@dataclass(frozen=True)
class Shot:
    """Where the camera stands and how it samples, relative to the aim point of the head."""

    distance: float = 0.5  # metres from the camera to the aim point
    yaw: float = 0.0  # degrees around the up axis; 0 looks straight at the face (+Z)
    pitch: float = 0.0  # degrees; positive raises the camera
    spacing: float = 1.7  # mm between samples at the aim point
    noise: float = NOISE
    max_angle: float = 75.0  # degrees between the surface normal and the ray
    depth_step: float = DEPTH_STEP


@dataclass(frozen=True)
class Scan:
    """Points in the camera frame (mm, Y up, camera looking along -Z) and the mesh vertex each one shows."""

    points: np.ndarray
    vertex: np.ndarray


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals of the triangles, turned so that they point away from the mesh centroid on the whole.

    The winding of a head mesh is consistent, so one global flip settles the
    orientation of every face.
    """
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(length, 1e-12)[:, None]
    outward = np.einsum("ij,ij->i", normals, triangles.mean(axis=1) - vertices.mean(axis=0))
    return normals if np.sum(length * outward) >= 0 else -normals


def sample_surface(
    vertices: np.ndarray, faces: np.ndarray, spacing: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform random points on the mesh, one per ``spacing``^2 mm^2 on average.

    Returns the points, the triangle each lies on and its barycentric weights.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    triangles = vertices[faces].astype(float)
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    count = max(1, int(np.ceil(area.sum() / spacing**2)))
    chosen = rng.choice(len(faces), count, p=area / area.sum())
    root, second = np.sqrt(rng.random(count)), rng.random(count)
    weights = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
    return np.einsum("nk,nki->ni", weights, triangles[chosen]), chosen, weights


def ideal(vertices: np.ndarray, faces: np.ndarray, spacing: float = 1.0, seed: int = 0) -> Scan:
    """The whole surface sampled every ``spacing`` mm, in the mesh frame: no camera, holes or noise."""
    points, chosen, weights = sample_surface(vertices, faces, spacing, np.random.default_rng(seed))
    return Scan(points, faces[chosen, np.argmax(weights, axis=1)])


def camera_axes(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (towards the camera, right, up) of a camera turned by ``yaw`` and ``pitch`` degrees."""
    yaw, pitch = np.radians(yaw), np.radians(pitch)
    back = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = np.cross([0.0, 1.0, 0.0], back)
    right /= np.linalg.norm(right)
    return back, right, np.cross(back, right)


def shoot(
    vertices: np.ndarray,
    faces: np.ndarray,
    shot: Shot,
    aim: np.ndarray,
    seed: int = 0,
    hidden: np.ndarray | None = None,
) -> Scan:
    """One scanner shot of the mesh aimed at ``aim`` (mm, mesh frame; Y up, face towards +Z).

    ``hidden`` marks triangles the scanner does not return (brows, hair):
    their pixels stay empty.
    """
    rng = np.random.default_rng(seed)
    if not 0.05 <= shot.distance <= 5.0:
        raise ValueError("the shot distance is in metres")
    back, right, up = camera_axes(shot.yaw, shot.pitch)
    camera = np.asarray(aim, dtype=float) + 1000.0 * shot.distance * back
    points, chosen, weights = sample_surface(vertices, faces, shot.spacing / np.sqrt(OVERSAMPLING), rng)
    ray = points - camera
    depth = -ray @ back
    if (depth <= 0).any():
        raise ValueError("the camera stands inside the mesh")
    pixel = shot.spacing / (1000.0 * shot.distance)
    column = np.floor((ray @ right) / depth / pixel).astype(np.int64)
    row = np.floor((ray @ up) / depth / pixel).astype(np.int64)
    # Z-buffer: the nearest sample of every pixel wins, whatever it faces.
    order = np.lexsort((depth, column, row))
    keys = np.stack((row[order], column[order]), axis=1)
    first = np.ones(len(order), dtype=bool)
    first[1:] = (keys[1:] != keys[:-1]).any(axis=1)
    winner = order[first]
    normals = face_normals(vertices.astype(float), faces)[chosen[winner]]
    distance = np.linalg.norm(ray[winner], axis=1)
    facing = -np.einsum("ij,ij->i", normals, ray[winner]) / distance > np.cos(np.radians(shot.max_angle))
    if hidden is not None:
        facing &= ~np.asarray(hidden, dtype=bool)[chosen[winner]]
    winner, distance = winner[facing], distance[facing]
    # Noise along the ray, then the exported depth is rounded.
    sigma = shot.noise * (depth[winner] / 1000.0) ** 2
    stretched = (distance + rng.normal(0.0, 1.0, len(winner)) * sigma) / distance
    seen = camera + ray[winner] * stretched[:, None]
    if shot.depth_step > 0:
        along = -(seen - camera) @ back
        seen = camera + (seen - camera) * (np.round(along / shot.depth_step) * shot.depth_step / along)[:, None]
    relative = seen - camera
    camera_frame = np.column_stack((relative @ right, relative @ up, relative @ back))
    return Scan(camera_frame, faces[chosen[winner], np.argmax(weights[winner], axis=1)])
