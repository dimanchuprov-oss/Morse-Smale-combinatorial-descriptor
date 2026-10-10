"""Tracked face meshes of the Multiface dataset for expression experiments.

Multiface (Meta, CC-BY-NC 4.0, arXiv 2207.11243) films 13 people performing
facial expressions and tracks one mesh topology of 7,306 vertices through
every frame. Ten people follow the v1 script, whose expressions are named
(``E001_Neutral_Eyes_Open`` and so on); the other three follow the v2 range
of motion script and are not used here.

Only the tracked meshes are fetched. Every archive is streamed from the
public bucket, checked against the published MD5 sum and reduced to the
vertices of each frame without the head pose: millimetres, Y up, the face
looking towards +Z. Nothing of the archive is written to disk. File names
coming from the server are checked before use, and no shell is involved.

The data may not be redistributed: keep the destination folder out of the
repository and cite the technical report in anything built on it.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import re
import tarfile
import threading
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities"
# People recorded with the v1 script, whose expressions are named.
PEOPLE = ("002539136", "002643814", "002757580", "002914589", "2183941",
          "5372021", "6674443", "6795937", "7889059", "8870559")  # fmt: skip
VERTICES = 7306
# Expression classes and the v1 segments that show them, in order of preference:
# not every person has every segment.
CLASSES: dict[str, tuple[str, ...]] = {
    "neutral": ("E001_Neutral_Eyes_Open",),
    "eyes_closed": ("E003_Neutral_Eyes_Closed", "E074_Blink"),
    "smile_closed": ("E008_Smile_Mouth_Closed",),
    "smile_open": ("E009_Smile_Mouth_Open",),
    "smile_wide": ("E012_Jaw_Open_Huge_Smile", "E010_Smile_Stretched"),
    "mouth_open": ("E004_Relaxed_Mouth_Open",),
    "brows_up": ("E022_Raise_Inner_Eyebrows", "E006_Jaw_Drop_Brows_Up"),
    "frown": ("E019_Frown", "E021_Pressed_Lips_Brows_Down"),
    "cheeks_lips": ("E057_Cheeks_Puffed", "E061_Lips_Puffed"),
    "speech": ("SEN_are_you_looking_for_employment", "SEN_how_do_oysters_make_pearls",
               "SEN_have_you_got_our_keys_handy"),  # fmt: skip
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}")
FRAME_FILE = re.compile(r"(\d{1,8})(\.bin|_transform\.txt|\.obj)")
# Larger archive members are not frames of a tracked mesh and are not read.
MAX_MEMBER = 8 * 1024 * 1024
# Bounds against a broken or hostile server: the largest v1 archive is 1.6 GB
# and the longest segment has a few thousand frames.
MAX_ARCHIVE = 3 * 1024**3
MAX_FRAMES = 20_000
TIMEOUT = 60.0


class DownloadError(OSError):
    """A transfer that ended early, sent more than announced or failed its MD5 check; worth retrying."""


# Network failures worth another attempt: urllib raises OSError subclasses,
# http.client raises IncompleteRead and other HTTPException, which are not.
RETRIED = (OSError, http.client.HTTPException, tarfile.ReadError)


@dataclass(frozen=True)
class Segment:
    """Frames of one recorded expression: vertices without the head pose and the poses themselves."""

    name: str
    frames: np.ndarray  # (F,) frame numbers
    vertices: np.ndarray  # (F, VERTICES, 3) float32, mm, Y up, face towards +Z
    headpose: np.ndarray  # (F, 3, 4) float32: tracked = R @ vertex + t

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with partial.open("wb") as handle:
            np.savez(handle, name=self.name, frames=self.frames, vertices=self.vertices, headpose=self.headpose)
        os.replace(partial, path)

    @classmethod
    def load(cls, path: Path) -> Segment:
        with np.load(path, allow_pickle=False) as data:
            return cls(str(data["name"]), data["frames"], data["vertices"], data["headpose"])


def _check_name(name: str) -> str:
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"unexpected file name from the server: {name!r}")
    return name


def _url(person: str, name: str) -> str:
    return f"{ROOT}/{_check_name(person)}/{_check_name(name)}"


def parse_checksums(text: str) -> dict[str, str]:
    """``name -> md5`` from a CHECKSUM listing (``md5  name`` per line); malformed lines are skipped."""
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{32}", parts[0]) and SAFE_NAME.fullmatch(parts[1]):
            sums[parts[1]] = parts[0]
    return sums


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turns a redirect into an HTTP error, so that no request leaves the dataset (or https)."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _get(url: str, timeout: float = TIMEOUT) -> io.BufferedIOBase:
    if not url.startswith(ROOT + "/"):
        raise ValueError(f"refusing to fetch outside the dataset: {url}")
    response = _OPENER.open(url, timeout=timeout)
    if not response.geturl().startswith(ROOT + "/"):
        response.close()
        raise ValueError(f"the server answered from outside the dataset: {response.geturl()}")
    return response


def _retrying(action: str, attempt: int, retries: int, error: BaseException) -> None:
    if attempt == retries - 1:
        raise error
    print(f"retry {action}: {type(error).__name__}: {error}", flush=True)
    time.sleep(5.0 * (attempt + 1))


def fetch_checksums(person: str, retries: int = 3) -> dict[str, str]:
    for attempt in range(retries):
        try:
            with _get(_url(person, "CHECKSUM")) as response:
                return parse_checksums(response.read(16 * 1024 * 1024).decode("utf-8", "replace"))
        except RETRIED as error:
            _retrying(f"{person}/CHECKSUM", attempt, retries, error)
    raise AssertionError("unreachable")


def resolve(checksums: dict[str, str], classes: Iterable[str]) -> dict[str, str]:
    """Segment chosen for each class: the first preferred segment the person has."""
    chosen = {}
    for label in classes:
        for segment in CLASSES[label]:
            if f"tracked_mesh--{segment}.tar" in checksums:
                chosen[label] = segment
                break
    return chosen


class _Hashing(io.RawIOBase):
    """Read-through wrapper that hashes and counts every byte read from ``stream``, up to ``limit``."""

    def __init__(self, stream: io.BufferedIOBase, limit: int = MAX_ARCHIVE) -> None:
        self.stream, self.md5, self.size, self.limit = stream, hashlib.md5(), 0, limit

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        data = self.stream.read(len(buffer))
        buffer[: len(data)] = data
        self.md5.update(data)
        self.size += len(data)
        if self.size > self.limit:
            raise DownloadError(f"the server sent more than {self.limit} bytes")
        return len(data)

    def drain(self) -> None:
        while self.read(1 << 20):
            pass


def _obj_faces(text: str) -> np.ndarray:
    faces = [[int(token.split("/")[0]) - 1 for token in line.split()[1:]] for line in text.splitlines()
             if line.startswith("f ")]  # fmt: skip
    if not faces or any(len(face) != 3 for face in faces):
        raise ValueError("the tracked mesh is expected to consist of triangles")
    faces = np.asarray(faces, dtype=np.int32)
    if faces.min() < 0 or faces.max() >= VERTICES:
        raise ValueError("a face refers to a vertex outside the tracked mesh")
    return faces


def read_segment(stream: io.RawIOBase, segment: str, want_faces: bool = False) -> tuple[Segment, np.ndarray | None]:
    """Frames of a ``tracked_mesh--<segment>.tar`` stream, and the mesh faces if ``want_faces``.

    Only ``<frame>.bin`` (float32 vertices without the head pose),
    ``<frame>_transform.txt`` (the 3x4 head pose) and, for the faces, the first
    ``<frame>.obj`` are read; members are matched by their last two path parts
    and never extracted.
    """
    vertices: dict[int, np.ndarray] = {}
    poses: dict[int, np.ndarray] = {}
    faces = None
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            parts = member.name.replace("\\", "/").split("/")
            found = FRAME_FILE.fullmatch(parts[-1])
            if not member.isfile() or len(parts) < 2 or parts[-2] != segment or found is None:
                continue
            if member.size > MAX_MEMBER:
                continue
            kind, frame = found.group(2), int(found.group(1))
            if kind == ".obj" and not (want_faces and faces is None):
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            data = handle.read()
            if kind == ".bin":
                if len(data) != VERTICES * 12:
                    raise ValueError(f"{member.name}: {len(data)} bytes, expected {VERTICES * 12}")
                if len(vertices) >= MAX_FRAMES:
                    raise ValueError(f"more than {MAX_FRAMES} frames in one segment")
                vertices[frame] = np.frombuffer(data, dtype="<f4").reshape(VERTICES, 3)
                if not np.isfinite(vertices[frame]).all():
                    raise ValueError(f"{member.name}: vertices are not finite")
            elif kind == "_transform.txt":
                pose = np.array(data.decode("ascii").split(), dtype=np.float32)
                if pose.shape != (12,):
                    raise ValueError(f"{member.name}: a head pose has 12 numbers")
                poses[frame] = pose.reshape(3, 4)
            else:
                faces = _obj_faces(data.decode("ascii"))
    frames = np.array(sorted(vertices), dtype=np.int64)
    if len(frames) == 0:
        raise ValueError(f"no tracked frames of {segment} in the archive")
    missing = np.full((3, 4), np.nan, dtype=np.float32)
    result = Segment(segment, frames, np.stack([vertices[f] for f in frames]),
                     np.stack([poses.get(int(f), missing) for f in frames]))  # fmt: skip
    return result, faces


def fetch_segment(person: str, segment: str, md5: str, want_faces: bool = False,
                  retries: int = 3) -> tuple[Segment, np.ndarray | None]:  # fmt: skip
    """Stream one tracked-mesh archive and check its length and MD5 sum; retried on network errors.

    A connection that closes early can look like a clean end of the stream, and
    a tar read in stream mode can stop at a member boundary without an error,
    so the byte count is compared with ``Content-Length``.
    """
    name = f"tracked_mesh--{segment}.tar"
    for attempt in range(retries):
        try:
            with _get(_url(person, name)) as response:
                announced = response.headers.get("Content-Length", "")
                length = int(announced) if announced.isdigit() else None
                if length is not None and length > MAX_ARCHIVE:
                    raise ValueError(f"{person}/{name}: {length} bytes is more than an archive of meshes can be")
                stream = _Hashing(response, MAX_ARCHIVE if length is None else length)
                result = read_segment(stream, segment, want_faces)
                stream.drain()
            if length is not None and stream.size != length:
                raise DownloadError(f"{person}/{name}: received {stream.size} of {length} bytes")
            if stream.md5.hexdigest() != md5:
                raise DownloadError(f"{person}/{name}: MD5 {stream.md5.hexdigest()} does not match {md5}")
            return result
        except RETRIED as error:
            _retrying(f"{person}/{name}", attempt, retries, error)
    raise AssertionError("unreachable")


def segment_path(dest: Path, person: str, label: str) -> Path:
    return dest / _check_name(person) / f"{label}.npz"


def fetch(dest: Path, people: Iterable[str] = PEOPLE, classes: Iterable[str] = tuple(CLASSES),
          workers: int = 4) -> dict[str, dict[str, str]]:  # fmt: skip
    """Download the chosen classes of the chosen people into ``dest``; existing files are kept.

    ``dest/<person>/<class>.npz`` holds a :class:`Segment`, ``dest/<person>/manifest.json``
    the chosen segments and their MD5 sums, and ``dest/faces.npy`` the shared
    triangles (checked to be the same for every person).
    """
    dest.mkdir(parents=True, exist_ok=True)
    classes = list(classes)
    unknown = sorted(set(classes) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown classes: {', '.join(unknown)}")
    faces_path = dest / "faces.npy"
    plan = {}
    jobs = []
    for person in people:
        checksums = fetch_checksums(_check_name(person))
        chosen = resolve(checksums, classes)
        plan[person] = chosen
        manifest_path = dest / person / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest.update({label: {"segment": segment, "md5": checksums[f"tracked_mesh--{segment}.tar"]}
                         for label, segment in chosen.items()})  # fmt: skip
        (dest / person).mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        missing = [label for label in chosen if not segment_path(dest, person, label).exists()]
        for label in missing:
            # The first archive of each person also supplies the faces, to check the shared topology.
            entry = manifest[label]
            jobs.append((person, label, entry["segment"], entry["md5"], label == missing[0]))
    topology = threading.Lock()

    def run(job: tuple[str, str, str, str, bool]) -> str:
        person, label, segment, md5, want_faces = job
        result, faces = fetch_segment(person, segment, md5, want_faces)
        if faces is not None:
            with topology:
                if faces_path.exists():
                    if not np.array_equal(np.load(faces_path), faces):
                        raise ValueError(f"{person}: the mesh topology differs from the other people")
                else:
                    partial = faces_path.with_name(faces_path.name + ".partial")
                    with partial.open("wb") as handle:
                        np.save(handle, faces)
                    os.replace(partial, faces_path)
        result.save(segment_path(dest, person, label))
        return f"{person} {label:<13} {segment}: {len(result.frames)} frames"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for line in pool.map(run, jobs):
            print(line, flush=True)
    return plan


def load_people(dest: Path, classes: Iterable[str] | None = None) -> dict[str, dict[str, Segment]]:
    """Every downloaded person with a neutral segment: ``person -> class -> Segment``."""
    labels = list(CLASSES if classes is None else classes)
    people = {}
    # Other folders (notebook checkpoints, old copies) are not people.
    for folder in sorted(path for path in dest.iterdir() if path.is_dir() and SAFE_NAME.fullmatch(path.name)):
        segments = {label: Segment.load(segment_path(dest, folder.name, label))
                    for label in labels if segment_path(dest, folder.name, label).exists()}  # fmt: skip
        if "neutral" in segments:
            people[folder.name] = segments
    return people


def peak_frame(segment: Segment, reference: np.ndarray, weights: np.ndarray | None = None) -> int:
    """Index of the frame that departs most from ``reference`` (RMS over the weighted vertices).

    The head pose is already removed, so the departure is the expression itself.
    """
    offsets = np.linalg.norm(segment.vertices.astype(float) - reference[None], axis=2)
    weights = np.ones(VERTICES) if weights is None else np.asarray(weights, dtype=float)
    return int(np.argmax(np.sqrt((offsets**2 * weights).sum(axis=1) / weights.sum())))
