"""What facial expressions cost the face matcher, measured on Multiface meshes.

Steps M1-M2 of the Multiface plan. Every person is enrolled from a virtual
frontal shot of a neutral frame (0.5 m, 1.7 mm spacing). Probes are virtual
shots of the peak frame of each expression class, taken from a random pose
(yaw within +-30 degrees, pitch within +-10, 0.4-0.7 m away, 1.2-2.2 mm
spacing); the neutral probe is another frame of the neutral recording. Eyes,
brows and hair leave holes, as on the real scanner (see :func:`regions`).
The mouth is left as tracked: the mesh continues the lips into a pocket, so
an open mouth shows its walls where a real scanner would see teeth and
tongue. Every probe is matched against every enrolled person.

Methods:

* ``b0``: the current matcher (:func:`morse_lidar.face.match`) on the heads
  cut out of the shots;
* ``b1``: the same on the nose, the bridge of the nose and the forehead only.
  The region comes from the mesh labels, so this is an upper bound for a mask
  that a real system has to find on the scan;
* ``b2``: each person is enrolled from three shots (neutral, open smile, open
  mouth) and a probe gets the score of the closest; a probe is never compared
  with a template of its own class.
* ``b3`` (M4): nose/bridge and forehead found from scan XYZ alone. Mesh labels
  are used only for an independent mask diagnostic, never by the detector.

The scores are optimistic in one respect: the probe and the gallery come from
one recording session, so the neutral probe shows only what the scanner and
the pose change, not what a different day does.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .face import GATE, _surface_distance, enroll, equal_error_rate, extract_head, match, upright
from .face_regions import REGION_VERSION, locate_regions, plot_regions
from .multiface import VERTICES, Segment, load_people, peak_frame
from .pointcloud import PointCloud, voxel_downsample
from .virtual_scan import Scan, Shot, shoot

BASELINES = ("b0", "b1", "b2")
METHODS = (*BASELINES, "b3")
# Classes the b2 gallery is enrolled from, besides the neutral one.
B2_EXTRA = ("smile_open", "mouth_open")
GALLERY_SHOT = Shot(distance=0.5, spacing=1.7)
VOXEL = 2.0
# The tracked mesh closes each eye with a few large triangles at the level of
# the lids (edges over EYE_EDGE mm, where the skin around has 3-5 mm). The real
# scanner returns nothing there, so they become holes.
EYE_EDGE = 7.0
PAIR_FIELDS = ["method", "probe", "probe_person", "probe_class", "gallery", "gallery_person", "genuine",
               "geometric_mm", "shared_cm2", "overlap", "error"]  # fmt: skip


@dataclass(frozen=True)
class Regions:
    """Labels of the shared Multiface topology, from the mean neutral face (mm, Y up, face towards +Z)."""

    eye_covers: np.ndarray  # faces closing the eye openings
    brows: np.ndarray  # vertices
    hair: np.ndarray  # vertices
    rigid: np.ndarray  # vertices of the nose, its bridge and the forehead
    face: np.ndarray  # vertices from the forehead to the chin, used to find the peak frame
    eyes: np.ndarray  # (2, 3) eye centres
    nose_tip: int

    def hidden(self, faces: np.ndarray) -> np.ndarray:
        """Triangles the scanner does not return: the eyes and anything touching brows or hair."""
        return self.eye_covers | (self.brows | self.hair)[faces].any(axis=1)


def _areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[..., faces, :]
    return 0.5 * np.linalg.norm(np.cross(triangles[..., 1, :] - triangles[..., 0, :],
                                         triangles[..., 2, :] - triangles[..., 0, :]), axis=-1)  # fmt: skip


def _largest_patch(chosen: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The largest group of ``chosen`` triangles connected through shared edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    _, edge = np.unique(edges, axis=0, return_inverse=True)
    order = np.argsort(edge, kind="stable")
    shared = edge[order][1:] == edge[order][:-1]
    first, second = owner[order][:-1][shared], owner[order][1:][shared]
    keep = chosen[first] & chosen[second]
    graph = coo_matrix((np.ones(keep.sum()), (first[keep], second[keep])), shape=(len(faces), len(faces)))
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label[chosen], minlength=label.max() + 1)
    return chosen & (label == np.argmax(sizes))


def _nose_tip(vertices: np.ndarray) -> int:
    x, y, z = vertices.T
    central = (np.abs(x) < 25.0) & (np.abs(y - np.median(y)) < 80.0)
    return int(np.flatnonzero(central)[np.argmax(z[central])])


def _eye_candidates(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Triangles around the eyes with an edge over ``EYE_EDGE``: the eye covers and a few stray ones."""
    triangles = vertices[faces]
    longest = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max(axis=1)
    offset = triangles.mean(axis=1) - vertices[_nose_tip(vertices)]
    across = np.abs(offset[:, 0])
    return (across > 8.0) & (across < 60.0) & (offset[:, 1] > 15.0) & (offset[:, 1] < 60.0) & (offset[:, 2] > -60.0) \
        & (longest > EYE_EDGE)


def regions(neutrals: np.ndarray, faces: np.ndarray) -> Regions:
    """Label the topology from the neutral meshes of several people, shape (people, VERTICES, 3).

    The eyes are the triangles with long edges near the eyes in at least half
    of the people (``EYE_EDGE``; averaging the meshes first would shorten
    those edges), the largest connected patch on each side. Everything else is
    placed on the mean mesh relative to the nose tip and the eye centres (area
    centroids of the patches) with fixed margins in mm.
    """
    neutrals = np.asarray(neutrals, dtype=float).reshape(-1, VERTICES, 3)
    votes = np.mean([_eye_candidates(vertices, faces) for vertices in neutrals], axis=0)
    neutral = neutrals.mean(axis=0)
    x, y, z = neutral.T
    nose_tip = _nose_tip(neutral)
    nose = neutral[nose_tip]
    triangles = neutral[faces]
    offset = triangles.mean(axis=1) - nose
    area = _areas(neutral, faces)
    covers, eyes = np.zeros(len(faces), dtype=bool), []
    for side in (offset[:, 0] < 0.0, offset[:, 0] > 0.0):
        if not (side & (votes >= 0.5)).any():
            raise ValueError("no eye found on this side of the face")
        patch = _largest_patch(side & (votes >= 0.5), faces)
        covers |= patch
        eyes.append((area[patch, None] * triangles[patch].mean(axis=1)).sum(axis=0) / area[patch].sum())
    eyes = np.array(eyes)
    if not (eyes[:, 1] > nose[1] + 15.0).all() or not (eyes[:, 2] < nose[2] - 10.0).all():
        raise ValueError("the meshes are expected in millimetres with Y up and the face towards +Z")
    eye_y, half = float(eyes[:, 1].mean()), 0.5 * float(abs(eyes[1, 0] - eyes[0, 0]))
    front = z > nose[2] - 75.0
    across = np.abs(x - nose[0])
    brows = front & (y > eye_y + 9.0) & (y < eye_y + 28.0) & (across > 6.0) & (across < half + 26.0)
    hair = (y > eye_y + 72.0) | ((z < nose[2] - 140.0) & (y > eye_y - 40.0))
    nose_part = (across < 18.0) & (y > nose[1] - 8.0) & (y < eye_y + 28.0) & (z > nose[2] - 40.0)
    forehead = front & (across < half + 15.0) & (y > eye_y + 28.0) & (y < eye_y + 65.0)
    rigid = (nose_part | forehead) & ~brows
    face = front & (y > eye_y - 120.0) & (y < eye_y + 65.0)
    return Regions(covers, brows, hair, rigid, face, eyes, nose_tip)


@dataclass(frozen=True)
class Shots:
    """Virtual scans of one person: the gallery shot(s) and one probe per class."""

    person: str
    # A shot that failed holds the error message instead of a scan.
    gallery: dict[str, Scan | str]
    probes: dict[str, Scan | str]


def _seed(*parts: object) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def probe_shot(rng: np.random.Generator) -> Shot:
    return Shot(distance=float(rng.uniform(0.4, 0.7)), yaw=float(rng.uniform(-30.0, 30.0)),
                pitch=float(rng.uniform(-10.0, 10.0)), spacing=float(rng.uniform(1.2, 2.2)))  # fmt: skip


def choose_frames(segments: dict[str, Segment], labels: Regions) -> dict[str, tuple[str, int]]:
    """``class -> (class of the source segment, frame index)`` of the gallery and the probes.

    The gallery is the neutral frame closest to the median neutral shape; the
    neutral probe is the neutral frame furthest from it in time; every other
    probe is the peak of its class. Keys ``gallery:<class>`` hold the b2
    templates: the peaks of those classes.
    """
    neutral = segments["neutral"]
    reference = np.median(neutral.vertices.astype(float), axis=0)
    departure = np.linalg.norm(neutral.vertices - reference[None], axis=2).mean(axis=1)
    gallery = int(np.argmin(departure))
    weights = labels.face.astype(float)
    chosen = {"gallery:neutral": ("neutral", gallery),
              "neutral": ("neutral", int(np.argmax(np.abs(neutral.frames - neutral.frames[gallery]))))}  # fmt: skip
    for label, segment in segments.items():
        if label != "neutral":
            chosen[label] = (label, peak_frame(segment, reference, weights))
    for label in B2_EXTRA:
        if label in segments:
            chosen[f"gallery:{label}"] = chosen[label]
    return chosen


def take_shots(person: str, segments: dict[str, Segment], labels: Regions, faces: np.ndarray, seed: int = 0) -> Shots:
    hidden = labels.hidden(faces)
    gallery: dict[str, Scan | str] = {}
    probes: dict[str, Scan | str] = {}
    for key, (label, index) in choose_frames(segments, labels).items():
        vertices = segments[label].vertices[index].astype(float)
        aim = vertices[labels.nose_tip]
        template = key.startswith("gallery:")
        shot = GALLERY_SHOT if template else probe_shot(np.random.default_rng(_seed(seed, person, key, "pose")))
        try:
            scan: Scan | str = shoot(vertices, faces, shot, aim, _seed(seed, person, key), hidden)
        except ValueError as error:
            scan = f"{type(error).__name__}: {error}"
        (gallery if template else probes)[key.split(":", 1)[1] if template else key] = scan
    return Shots(person, gallery, probes)


def _head(scan: Scan, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical head points for matching, plus every scan point in that frame and its vertex."""
    head = extract_head(scan.points, up_axis="y", voxel=VOXEL)
    canonical = upright(scan.points, "y") - np.asarray(head.offset)
    if mask is None:
        return head.points, canonical, scan.vertex
    keep = mask[scan.vertex]
    if keep.sum() < 30:
        raise ValueError("fewer than 30 points of the scan lie in the region")
    return voxel_downsample(PointCloud(canonical[keep]), VOXEL).points, canonical[keep], scan.vertex[keep]


def _head_or_error(scan: Scan | str, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | str:
    if isinstance(scan, str):
        return scan
    try:
        return _head(scan, mask)
    except ValueError as error:
        return f"{type(error).__name__}: {error}"


def _automatic_heads(shots: dict[str, Shots], labels: Regions, out: Path, plot: bool) -> tuple[dict, dict]:
    """Detect first, then compare the resulting masks with oracle labels for diagnostics."""
    from .face import _kd_tree

    heads, records, pictures, arrays = {}, {}, {}, {}
    for person, shot in shots.items():
        for kind, scans in (("gallery", shot.gallery), ("probe", shot.probes)):
            for key, scan in scans.items():
                name, task_key = f"{person}/{kind}/{key}", (person, kind, key)
                if isinstance(scan, str):
                    heads[task_key], records[name] = scan, {"error": scan}
                    continue
                points = np.empty((0, 3))
                try:
                    head = extract_head(scan.points, up_axis="y", voxel=VOXEL)
                    points = head.points
                    region = locate_regions(points)  # only XYZ crosses this boundary
                    selected = points[region.mask]
                    heads[task_key] = (selected, selected, np.empty(0, dtype=int))
                    records[name] = {**region.as_dict(), "error": None}
                    pictures[name] = (points, region)
                    arrays[f"{name}/points"] = points
                    arrays[f"{name}/nose"] = region.nose
                    arrays[f"{name}/forehead"] = region.forehead
                except ValueError as error:
                    message = f"{type(error).__name__}: {error}"
                    heads[task_key], records[name] = message, {"error": message}
                    pictures[name] = (points, message)
                    continue
                # Evaluation only: changing/removing these labels cannot change the mask.
                canonical = upright(scan.points, "y") - np.asarray(head.offset)
                _, nearest = _kd_tree(canonical).query(points)
                oracle = labels.rigid[scan.vertex[nearest]]
                intersection = int((oracle & region.mask).sum())
                union = int((oracle | region.mask).sum())
                records[name].update(oracle_iou=intersection / max(1, union),
                                     oracle_precision=intersection / int(region.mask.sum()),
                                     oracle_recall=intersection / max(1, int(oracle.sum())))
                tip_samples = canonical[scan.vertex == labels.nose_tip]
                records[name]["oracle_tip_distance_mm"] = (
                    float(np.linalg.norm(tip_samples - region.nose_tip, axis=1).min()) if len(tip_samples) else None)
    report = {"version": REGION_VERSION, "scans": records, "total": len(records),
              "failed": sum(record["error"] is not None for record in records.values()),
              "note": "oracle masks evaluate the detector; they are not detector inputs"}
    if plot:
        report["plots"] = plot_regions(pictures, out)
    np.savez_compressed(out / "auto_regions.npz", **arrays)
    (out / "auto_regions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return heads, report


def pairs_for(method: str, probes: dict[str, list[str]], galleries: dict[str, list[str]]) -> list[tuple[str, ...]]:
    """``(probe person, probe class, gallery person, template)`` the method compares.

    ``probes`` and ``galleries`` map a person to their probe classes and
    template classes. b2 uses several templates per person but never one of
    the probe's own class.
    """
    templates = ("neutral", *B2_EXTRA) if method == "b2" else ("neutral",)
    return [(person, label, other, key) for person, labels in probes.items() for label in labels
            for other, keys in galleries.items() for key in templates
            if key in keys and not (key == label and key != "neutral")]  # fmt: skip


def _match_task(task: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]) -> dict[str, Any]:
    """Score one probe against one template; for genuine pairs also the residual of every probe point."""
    probe, gallery, raw, vertex, residuals = task
    try:
        result = match(probe, gallery, topology=False)
    except Exception as error:  # noqa: BLE001 - one failed pair must not lose the benchmark
        return {"error": f"{type(error).__name__}: {error}"}
    row: dict[str, Any] = {"geometric_mm": result.geometric, "shared_cm2": result.shared_cm2,
                           "overlap": result.overlap, "error": None}  # fmt: skip
    if residuals:
        distance, plane = _surface_distance(result.registration.apply(raw), enroll(gallery))
        near = distance < GATE
        row["residual"] = (vertex[near], plane[near])
    return row


def run(data: Path, out: Path, methods: tuple[str, ...] = BASELINES, people: list[str] | None = None,
        jobs: int = 0, seed: int = 0, plot: bool = True, regions_only: bool = False) -> dict[str, Any]:  # fmt: skip
    """Shoot, match and summarise; writes ``bench_pairs.csv``, ``bench_summary.json`` and plots to ``out``."""
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    if not methods or jobs < 0:
        raise ValueError("choose at least one method and a nonnegative worker count")
    if regions_only and "b3" not in methods:
        raise ValueError("--regions-only requires b3")
    out.mkdir(parents=True, exist_ok=True)
    print("Loading Multiface and generating reproducible shots", file=sys.stderr, flush=True)
    loaded = load_people(data)
    if people:
        missing = sorted(set(people) - set(loaded))
        if missing:
            raise ValueError(f"not downloaded (or without a neutral segment): {', '.join(missing)}")
        loaded = {person: loaded[person] for person in people}
    if len(loaded) < 2:
        raise ValueError("need at least two downloaded people")
    faces = np.load(data / "faces.npy")
    labels = _labels(loaded, faces)
    shots = {person: take_shots(person, segments, labels, faces, seed) for person, segments in loaded.items()}
    # b0 and b2 share the unmasked heads, so a pair they both need is matched once.
    masks = {"b0": "full", "b1": "rigid", "b2": "full", "b3": "auto"}
    heads = {mask: {(person, kind, key): _head_or_error(scan, labels.rigid if mask == "rigid" else None)
                    for person, shot in shots.items()
                    for kind, scans in (("gallery", shot.gallery), ("probe", shot.probes))
                    for key, scan in scans.items()}
             for mask in {masks[method] for method in methods} - {"auto"}}  # fmt: skip
    diagnostics = None
    if "b3" in methods:
        print("Finding nose and forehead from scan XYZ", file=sys.stderr, flush=True)
        heads["auto"], diagnostics = _automatic_heads(shots, labels, out, plot)
        print(f"M4 regions: {diagnostics['total'] - diagnostics['failed']}/{diagnostics['total']} scans",
              file=sys.stderr, flush=True)
    if regions_only:
        return diagnostics
    probes = {person: list(shot.probes) for person, shot in shots.items()}
    galleries = {person: list(shot.gallery) for person, shot in shots.items()}
    wanted = {method: pairs_for(method, probes, galleries) for method in methods}
    tasks, results = {}, {}
    for method, pairs in wanted.items():
        for person, label, other, key in pairs:
            task_key = (masks[method], person, label, other, key)
            probe, gallery = heads[masks[method]][(person, "probe", label)], heads[masks[method]][(other, "gallery", key)]
            if isinstance(probe, str) or isinstance(gallery, str):
                results[task_key] = {"error": probe if isinstance(probe, str) else gallery}
            elif task_key not in tasks:
                residuals = masks[method] == "full" and person == other and key == "neutral"
                tasks[task_key] = (probe[0], gallery[0], probe[1], probe[2], residuals)
    workers = jobs or max(1, (os.cpu_count() or 2) - 1)
    started = time.monotonic()
    print(f"Matching {len(tasks)} pairs with {workers} workers; {len(results)} preparation failures",
          file=sys.stderr, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with (out / "bench_checkpoint.jsonl").open("w", encoding="utf-8") as checkpoint:
            for done, (key, result) in enumerate(zip(tasks, pool.map(_match_task, tasks.values(), chunksize=4)), 1):
                results[key] = result
                checkpoint.write(json.dumps({"key": key, **{k: v for k, v in result.items() if k != "residual"}}) + "\n")
                if done % 25 == 0 or done == len(tasks):
                    checkpoint.flush()
                    progress = {"completed": done, "total": len(tasks), "elapsed_s": time.monotonic() - started}
                    (out / "bench_progress.json").write_text(json.dumps(progress), encoding="utf-8")
                    print(f"Pairs {done}/{len(tasks)} ({progress['elapsed_s'] / 60:.1f} min)", file=sys.stderr, flush=True)
    maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (_, _, label, _, _), result in results.items():
        residual = result.pop("residual", None)
        if residual is not None:
            total, count = maps.setdefault(label, (np.zeros(VERTICES), np.zeros(VERTICES)))
            np.add.at(total, residual[0], residual[1])
            np.add.at(count, residual[0], 1.0)
    rows: list[dict[str, Any]] = []
    failed = {}
    for method, pairs in wanted.items():
        failed[method] = sum(1 for pair in pairs if results[(masks[method], *pair)].get("error"))
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for person, label, other, key in pairs:
            grouped.setdefault((person, label, other), []).append(
                {"method": method, "probe": f"{person}/{label}", "probe_person": person, "probe_class": label,
                 "gallery": f"{other}/{key}", "gallery_person": other, "genuine": person == other,
                 **results[(masks[method], person, label, other, key)]})  # fmt: skip
        # b2 enrols a person from several templates: the closest one counts.
        for candidates in grouped.values():
            scored = [row for row in candidates if row.get("geometric_mm") is not None]
            rows.append(min(scored, key=lambda row: row["geometric_mm"]) if scored else candidates[0])
    summary = summarise(rows, labels, loaded, failed)
    summary["settings"] = {"seed": seed, "methods": list(methods), "voxel_mm": VOXEL,
                           "auto_region_version": REGION_VERSION if "b3" in methods else None}
    if diagnostics is not None:
        summary["automatic_regions"] = {"total": diagnostics["total"], "failed": diagnostics["failed"],
                                        "report": "auto_regions.json"}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bench_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in PAIR_FIELDS} for row in rows)
    error_maps = {label: (total / np.maximum(count, 1.0), count) for label, (total, count) in maps.items()}
    np.savez(out / "bench_error_maps.npz", **{label: values for label, (values, _) in error_maps.items()})
    if plot:
        summary["plots"] = plots(rows, summary, error_maps, loaded, labels, faces, out)
    (out / "bench_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _labels(loaded: dict[str, dict[str, Segment]], faces: np.ndarray) -> Regions:
    """Regions from the neutral faces of everybody (the topology is shared)."""
    return regions(np.array([np.median(segments["neutral"].vertices, axis=0) for segments in loaded.values()]), faces)


def _ordered(labels: set[str]) -> list[str]:
    return sorted(labels, key=lambda label: (label != "neutral", label))


def summarise(rows: list[dict[str, Any]], labels: Regions, loaded: dict[str, dict[str, Segment]],
              failed: dict[str, int] | None = None) -> dict[str, Any]:  # fmt: skip
    """Per method and class: genuine and impostor scores, closed-set rank-1, EER and the cost of the expression.

    A probe is recognised when its own person scores strictly lower than every
    other person it could be scored against; a probe whose own pair failed
    counts as not recognised. ``failed`` counts the failed comparisons of each
    method before b2 keeps the closest template (default: the failed rows).
    """
    summary: dict[str, Any] = {"people": sorted(loaded), "methods": {}, "regions": {
        "rigid_vertices": int(labels.rigid.sum()), "brow_vertices": int(labels.brows.sum()),
        "hair_vertices": int(labels.hair.sum()), "eye_faces": int(labels.eye_covers.sum())}}  # fmt: skip
    for method in sorted({row["method"] for row in rows}):
        everything = [row for row in rows if row["method"] == method]
        per_class: dict[str, Any] = {}
        for label in _ordered({row["probe_class"] for row in everything}):
            mine = [row for row in everything if row["probe_class"] == label]
            scored = [row for row in mine if row.get("geometric_mm") is not None]
            genuine = [row["geometric_mm"] for row in scored if row["genuine"]]
            impostor = [row["geometric_mm"] for row in scored if not row["genuine"]]
            ranks = []
            for probe in sorted({row["probe"] for row in mine}):
                own = [row["geometric_mm"] for row in scored if row["probe"] == probe and row["genuine"]]
                rivals = [row["geometric_mm"] for row in scored if row["probe"] == probe and not row["genuine"]]
                # An unscored rival must not turn an incomplete search into a hit.
                expected = [row for row in mine if row["probe"] == probe and not row["genuine"]]
                ranks.append(bool(own) and bool(rivals) and len(rivals) == len(expected)
                             and all(own[0] < score for score in rivals))
            entry: dict[str, Any] = {"probes": len(ranks), "rank1": float(np.mean(ranks)) if ranks else None,
                                     "genuine_median": float(np.median(genuine)) if genuine else None,
                                     "genuine_max": max(genuine) if genuine else None,
                                     "impostor_min": min(impostor) if impostor else None,
                                     "impostor_median": float(np.median(impostor)) if impostor else None,
                                     "failed_rows": len(mine) - len(scored)}  # fmt: skip
            if genuine and impostor:
                entry["eer"] = equal_error_rate(genuine, impostor)[0]
            per_class[label] = entry
        base = per_class.get("neutral", {}).get("genuine_median")
        for entry in per_class.values():
            if base is not None and entry["genuine_median"] is not None:
                entry["cost_mm"] = entry["genuine_median"] - base
        expressive = [row for row in everything if row["probe_class"] != "neutral" and row.get("geometric_mm") is not None]
        genuine = [row["geometric_mm"] for row in expressive if row["genuine"]]
        impostor = [row["geometric_mm"] for row in expressive if not row["genuine"]]
        others = [entry for label, entry in per_class.items() if label != "neutral"]
        probes = sum(entry["probes"] for entry in others)
        summary["methods"][method] = {
            "classes": per_class,
            "expressive_rank1": sum(entry["rank1"] * entry["probes"] for entry in others) / probes if probes else None,
            "expressive_eer": equal_error_rate(genuine, impostor)[0] if genuine and impostor else None,
            "eer_is_conditional_on_success": any(row.get("geometric_mm") is None for row in everything
                                                  if row["probe_class"] != "neutral"),
            "scored_rows": sum(row.get("geometric_mm") is not None for row in everything),
            "total_rows": len(everything),
            "failed_pairs": (failed or {}).get(method, sum(1 for row in everything if row.get("error"))),
        }
    return summary


def plots(rows: list[dict[str, Any]], summary: dict[str, Any], error_maps: dict[str, tuple[np.ndarray, np.ndarray]],
          loaded: dict[str, dict[str, Segment]], labels: Regions, faces: np.ndarray, out: Path) -> list[str]:  # fmt: skip
    from .face_cli import GENUINE, IMPOSTOR, INK, MUTED, _pyplot

    plt = _pyplot()
    written = []
    methods = list(summary["methods"])
    classes = _ordered({label for entry in summary["methods"].values() for label in entry["classes"]})
    figure, axes = plt.subplots(1, len(methods), figsize=(max(6.5, 4.2 * len(methods)), 5.0), sharey=True,
                                squeeze=False)  # fmt: skip
    for axis, method in zip(axes[0], methods):
        for position, label in enumerate(classes):
            mine = [row for row in rows if row["method"] == method and row["probe_class"] == label
                    and row.get("geometric_mm") is not None]  # fmt: skip
            impostor = [row["geometric_mm"] for row in mine if not row["genuine"]]
            genuine = [row["geometric_mm"] for row in mine if row["genuine"]]
            axis.scatter(impostor, np.full(len(impostor), position) + 0.12, s=9, color=IMPOSTOR, alpha=0.45, lw=0)
            axis.scatter(genuine, np.full(len(genuine), position) - 0.12, s=16, color=GENUINE, lw=0)
        # The panels share the y axis, so tick labels would show one method's rates on all of
        # them: each panel writes its own rate at the right end of the row.
        entries = summary["methods"][method]["classes"]
        for position, label in enumerate(classes):
            rate = entries.get(label, {}).get("rank1")
            axis.text(1.01, position, "—" if rate is None else f"{rate * 100:.0f}%", transform=axis.get_yaxis_transform(),
                      va="center", ha="left", fontsize=8, color=INK)  # fmt: skip
        axis.set_yticks(range(len(classes)), classes)
        axis.invert_yaxis()
        axis.set_xlabel("RMS расстояние, мм")
        rank1 = summary["methods"][method]["expressive_rank1"]
        axis.set_title(f"{method}: узнано с мимикой " + ("—" if rank1 is None else f"{rank1 * 100:.0f}%"),
                       color=INK, loc="left", fontsize=10)  # fmt: skip
        axis.grid(axis="x", color="#e6e9f0", lw=0.6)
    handles = [axes[0][0].scatter([], [], s=16, color=GENUINE), axes[0][0].scatter([], [], s=16, color=IMPOSTOR)]
    figure.legend(handles, ["тот же человек", "другой человек"], loc="upper right", ncol=2, frameon=False,
                  fontsize=8, labelcolor=MUTED)  # fmt: skip
    figure.suptitle("Мимика против нейтрального эталона\nсправа у строки — доля узнанных среди всех людей (rank-1)",
                    color=INK, x=0.01, ha="left", fontsize=10)  # fmt: skip
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    figure.savefig(out / "bench_scores.png", dpi=150)
    plt.close(figure)
    written.append("bench_scores.png")
    if error_maps:
        neutral = np.mean([np.median(segments["neutral"].vertices, axis=0) for segments in loaded.values()], axis=0)
        shown = [label for label in classes if label in error_maps]
        columns = 5
        rows_count = int(np.ceil(len(shown) / columns))
        figure, axes = plt.subplots(rows_count, columns, figsize=(2.6 * columns, 3.3 * rows_count), squeeze=False)
        front = neutral[:, 2] > neutral[labels.nose_tip, 2] - 90.0
        visible = front[faces].all(axis=1)
        image = None
        for axis, label in zip(axes.ravel(), shown):
            values, count = error_maps[label]
            colour = np.where(count[faces].min(axis=1) > 0, values[faces].mean(axis=1), np.nan)[visible]
            image = axis.tripcolor(neutral[:, 0], neutral[:, 1], faces[visible], facecolors=colour, cmap="magma_r",
                                   vmin=0.0, vmax=2.0)  # fmt: skip
            axis.set_title(label, color=INK, fontsize=9)
            axis.set_aspect("equal")
            axis.axis("off")
        for axis in axes.ravel()[len(shown):]:
            axis.axis("off")
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, label="среднее |расстояние| до эталона, мм")
        figure.suptitle("Где мимика расходится с нейтральным эталоном (b0, свои пары)", color=INK, x=0.01, ha="left")
        figure.savefig(out / "bench_error_maps.png", dpi=150)
        plt.close(figure)
        written.append("bench_error_maps.png")
    return written
