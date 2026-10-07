"""Command line for face verification across viewing angles.

Run the angle experiment on full head scans, identify extra real scans and plot
the comparison::

    morse-lidar-face experiment --person A=a.ply --person B=b.ply --probe x.ply \\
        --up-axis y --front-axis +z --out results/faces

Compare every pair of real scans of named people (the person is the file name
without the trailing scan number: Ivan1.ply, Ivan2.ply -> Ivan)::

    morse-lidar-face matrix scans/*.ply --up-axis y --out results/matrix

Identify a new scan against enrolled people, with thresholds from a matrix or
an experiment::

    morse-lidar-face compare probe.ply --gallery A=a.ply --gallery B=b.ply --up-axis y \\
        --calibration results/matrix/face_matrix_summary.json

The experiment splits every full scan into two disjoint halves of its points.
One half is enrolled. Views of the other half, as a scanner would see them from
each yaw/pitch, are scrambled by a random rigid motion, cut to the head like a
real probe and matched against every enrolled person. Probes from the same
capture share hair, expression and scan artefacts with the gallery, so the
genuine scores are optimistic: real thresholds need repeat scans of several
people on different occasions. Negative angles must be passed with ``=``,
for example ``--yaws=-90:90:15``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .face import (
    GATE,
    HEAD_CENTRE,
    METHOD_VERSION,
    Enrolled,
    _kd_tree,
    enroll,
    equal_error_rate,
    estimate_normals,
    extract_head,
    locate_head,
    match,
    rotation_z,
    simulate_view,
    upright,
)
from .pointcloud import PointCloud, load_point_cloud, set_up_axis, voxel_downsample

UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
GENUINE, IMPOSTOR, REAL, INK, MUTED = "#355DA6", "#E4572E", "#2A9D8F", "#1d2433", "#6b7a99"
FIELDS = ["probe", "kind", "yaw", "pitch", "gallery", "genuine", "geometric_mm", "shared_cm2", "topological_mm",
          "overlap", "radial_fraction", "topology_error", "error"]  # fmt: skip
MATRIX_FIELDS = ["scan_a", "scan_b", "person_a", "person_b", "genuine", "geometric_mm", "shared_cm2", "reliable",
                 "topological_mm", "overlap", "radial_fraction", "topology_error", "error"]  # fmt: skip
# Pairs sharing less surface are reported but not trusted. 100 cm^2 is roughly
# half of a face from the forehead to the chin; on the scans in biometrics/ the
# error rate falls to zero from there (see eer_by_min_shared_cm2 of the matrix
# summary). It was chosen on those scans and has to be checked on new ones.
MIN_SHARED_CM2 = 100.0


def _labelled(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise SystemExit(f"{option} expects LABEL=PATH, got {value!r}")
        if label in result:
            raise SystemExit(f"duplicate {option} label {label!r}")
        result[label] = Path(path)
    return result


def _front_azimuth(front_axis: str, up_axis: str) -> float:
    """Azimuth of the face direction after the up axis is turned to +Z."""
    if len(front_axis) != 2 or front_axis[0] not in "+-" or front_axis[1] not in "xyz":
        raise SystemExit("--front-axis must look like +z or -x")
    if front_axis[1] == up_axis:
        raise SystemExit("--front-axis must be horizontal (differ from --up-axis)")
    vector = np.zeros((1, 3))
    vector[0, "xyz".index(front_axis[1])] = 1.0 if front_axis[0] == "+" else -1.0
    turned = set_up_axis(PointCloud(vector), up_axis).points[0]
    return float(math.atan2(turned[1], turned[0]))


def _angles(text: str, option: str, limit: float) -> list[float]:
    """START:STOP:STEP (inclusive) or a comma-separated list, in degrees."""
    try:
        if ":" in text:
            start, stop, step = (float(item) for item in text.split(":"))
            if not step > 0 or start > stop:
                raise ValueError
            values = list(np.arange(start, stop + step / 2, step))
        else:
            values = [float(item) for item in text.split(",") if item.strip()]
    except ValueError:
        raise SystemExit(f"{option} expects START:STOP:STEP with STEP > 0 and START <= STOP, or a list") from None
    if not values or not all(math.isfinite(value) and abs(value) <= limit for value in values):
        raise SystemExit(f"{option} needs at least one angle within ±{limit:g} degrees")
    return sorted({round(float(value), 6) for value in values})


def _upright(path: Path, args: argparse.Namespace) -> np.ndarray:
    return upright(load_point_cloud(path).points, args.up_axis, UNITS[args.units])


def _enroll(cloud: np.ndarray) -> Enrolled:
    return enroll(extract_head(cloud, up_axis="z"))


def _score(
    probe: np.ndarray | Enrolled, gallery: Enrolled, args: argparse.Namespace, seed: int
) -> tuple[dict[str, Any], Any]:
    result = match(probe, gallery, topology=not args.no_topology, seed=seed)
    return {**result.as_dict(), "error": None}, result


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {"up_axis": args.up_axis, "units": args.units, "method": METHOD_VERSION}


def _calibration(args: argparse.Namespace) -> dict[str, Any]:
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if "geometric_mm" not in calibration:
        raise SystemExit(f"{args.calibration} has no geometric_mm statistics; is it a face summary?")
    if "settings" not in calibration:
        raise SystemExit(f"{args.calibration} records no settings; regenerate it with this version")
    if calibration["settings"].get("method") != METHOD_VERSION:
        raise SystemExit(f"calibration was made with an older version of the method (scores and crop changed); "
                         f"regenerate {args.calibration}")  # fmt: skip
    if calibration["settings"] != _settings(args):
        raise SystemExit(f"calibration was made with {calibration['settings']}, not {_settings(args)}")
    return calibration


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.calibration and args.threshold is not None:
        raise SystemExit("pass either --calibration or --threshold, not both")
    calibration = _calibration(args) if args.calibration else {}
    min_area = args.min_area if args.min_area is not None else calibration.get("calibrated_min_shared_cm2", 0.0)
    galleries = {label: _enroll(_upright(path, args)) for label, path in _labelled(args.gallery, "--gallery").items()}
    probe = enroll(extract_head(_upright(Path(args.probe), args), up_axis="z"))
    scores: dict[str, Any] = {}
    for label, gallery in galleries.items():
        try:
            scores[label] = _score(probe, gallery, args, args.seed)[0]
        except ValueError as error:  # one enrolled scan that cannot be matched must not stop the others
            scores[label] = {"error": str(error)}
    # Only pairs with enough common surface take part in the ranking and the decision.
    usable = {label: score for label, score in scores.items()
              if score.get("error") is None and score["shared_cm2"] >= min_area}  # fmt: skip
    result: dict[str, Any] = {"probe": str(args.probe), "scores": scores, "min_shared_cm2": min_area}
    if not usable:
        result.update(best_match=None, decision="insufficient overlap",
                      note=f"no enrolled scan shares {min_area:g} cm2 of surface with the probe")  # fmt: skip
        return result
    best = min(usable, key=lambda label: usable[label]["geometric_mm"])
    result["best_match"] = best
    if len(usable) > 1:
        ordered = sorted(score["geometric_mm"] for score in usable.values())
        result["margin_mm"] = ordered[1] - ordered[0]
    if calibration:
        statistics = calibration["geometric_mm"]
        result["calibration"] = {key: statistics[key] for key in ("genuine_max", "impostor_min", "threshold")}
        result["decision"] = _decision(usable[best]["geometric_mm"], statistics)
    elif args.threshold is not None:
        result["threshold_mm"] = args.threshold
        result["decision"] = "same" if usable[best]["geometric_mm"] <= args.threshold else "different"
    else:
        result["decision"] = None
        result["note"] = "pass --calibration (a matrix or experiment summary) or --threshold to get a decision"
    return result


def experiment(args: argparse.Namespace) -> dict[str, Any]:
    people = _labelled(args.person, "--person")
    if len(people) < 2:
        raise SystemExit("the experiment needs at least two --person scans (impostor pairs)")
    yaws, pitches = _angles(args.yaws, "--yaws", 180.0), _angles(args.pitches, "--pitches", 80.0)
    front = _front_azimuth(args.front_axis or ("+x" if args.up_axis == "z" else "+z"), args.up_axis)
    example_yaw = min(yaws, key=lambda yaw: abs(yaw - args.example_yaw))
    plot_pitch = min(pitches, key=abs)
    clouds = {label: _upright(path, args) for label, path in people.items()}
    galleries = {label: _enroll(cloud[0::2]) for label, cloud in clouds.items()}
    rows: list[dict[str, Any]] = []
    examples: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    turn = rotation_z(-front).T  # face along +x, so that x-z is the profile
    for person, cloud in clouds.items():
        source = voxel_downsample(PointCloud(cloud[1::2]), 2.0).points
        centre, top = locate_head(source)
        normals = estimate_normals(source, np.array([centre[0], centre[1], top + HEAD_CENTRE[2]]))
        for pitch in pitches:
            for yaw in yaws:
                seed = args.seed + len(rows)
                base = {"probe": person, "kind": "simulated", "yaw": yaw, "pitch": pitch}
                try:
                    view = simulate_view(source, math.radians(yaw), math.radians(pitch), front, normals, seed=seed)
                    probe = extract_head(view, up_axis="z").points
                except ValueError as error:
                    rows.extend({**base, "gallery": label, "genuine": person == label, "error": str(error)}
                                for label in galleries)  # fmt: skip
                    continue
                for label, gallery in galleries.items():
                    row = {**base, "gallery": label, "genuine": person == label}
                    try:
                        score, result = _score(probe, gallery, args, seed)
                    except ValueError as error:
                        rows.append({**row, "error": str(error)})
                        continue
                    rows.append({**row, **score})
                    if yaw == example_yaw and pitch == plot_pitch:
                        examples[(person, label)] = (result.registration.apply(probe) @ turn, gallery.points @ turn)
    if args.probe:
        # Real probes are identified like `compare` does: against whole scans.
        whole = {label: _enroll(cloud) for label, cloud in clouds.items()}
        for path in args.probe:
            base = {"probe": Path(path).name, "kind": "real", "yaw": None, "pitch": None, "genuine": None}
            try:
                probe = extract_head(_upright(Path(path), args), up_axis="z").points
            except (ValueError, OSError) as error:
                rows.extend({**base, "gallery": label, "error": str(error)} for label in whole)
                continue
            for label, gallery in whole.items():
                try:
                    rows.append({**base, "gallery": label, **_score(probe, gallery, args, args.seed)[0]})
                except ValueError as error:
                    rows.append({**base, "gallery": label, "error": str(error)})
    summary = _summarise(rows, args, yaws, pitches)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "face_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)
    (output / "face_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_plots:
        summary["plots"] = _plot(rows, summary, examples, output, example_yaw, plot_pitch)
    return summary


def _person(path: Path) -> str:
    """Person label from a file name: the trailing number (and ' _-' before it) is the scan number."""
    return path.stem.rstrip("0123456789").rstrip(" _-") or path.stem


def _match_pair(task: tuple[np.ndarray, np.ndarray, bool, int]) -> dict[str, Any]:
    """Score one pair of canonical heads; module level so that worker processes can run it."""
    first, second, topology, seed = task
    try:
        result = match(first, second, topology=topology, seed=seed)
    except Exception as error:  # noqa: BLE001 - one broken pair must not lose the whole matrix
        return {"error": f"{type(error).__name__}: {error}"}
    pose = {"rotation": result.registration.rotation.tolist(), "translation": result.registration.translation.tolist()}
    return {**result.as_dict(), "error": None, "pose": pose}


def _statistics(genuine: list[float], impostor: list[float]) -> dict[str, float] | None:
    if not genuine or not impostor:
        return None
    rate, threshold = equal_error_rate(genuine, impostor)
    return {
        "eer": rate,
        "threshold": threshold,
        "genuine_max": max(genuine),
        "impostor_min": min(impostor),
        "genuine_median": float(np.median(genuine)),
        "impostor_median": float(np.median(impostor)),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
    }


def _scores_of(rows: list[dict[str, Any]], score: str, reliable_only: bool) -> tuple[list[float], list[float]]:
    chosen = [row for row in rows if row.get(score) is not None and (row["reliable"] or not reliable_only)]
    return [row[score] for row in chosen if row["genuine"]], [row[score] for row in chosen if not row["genuine"]]


def _nearest(
    rows: list[dict[str, Any]], scans: list[str], min_area: float, statistics: dict[str, float] | None
) -> dict[str, Any]:
    """Identification of every scan that has a same-person partner.

    ``nearest``/``correct``: the closest other scan over all pairs (closed set).
    ``trusted``: what ``compare`` would answer with the area limit and the
    calibrated decision: the closest scan among the pairs sharing ``min_area``.
    """
    result: dict[str, Any] = {}
    for scan in scans:
        own = [row for row in rows if scan in (row["scan_a"], row["scan_b"])]
        same = [row for row in own if row["genuine"]]
        if not same:
            continue
        best = min(own, key=lambda row: row["geometric_mm"])
        other = [row for row in own if not row["genuine"]]
        rival = min(other, key=lambda row: row["geometric_mm"]) if other else None
        usable = [row for row in own if row["shared_cm2"] >= min_area]
        trusted = None
        if usable and statistics is not None:
            pick = min(usable, key=lambda row: row["geometric_mm"])
            trusted = {
                "nearest": pick["scan_b"] if pick["scan_a"] == scan else pick["scan_a"],
                "geometric_mm": pick["geometric_mm"],
                "same_person": pick["genuine"],
                "decision": _decision(pick["geometric_mm"], statistics),
            }
        result[scan] = {
            "nearest": best["scan_b"] if best["scan_a"] == scan else best["scan_a"],
            "correct": best["genuine"],
            "best_same_mm": min(row["geometric_mm"] for row in same),
            "best_same_shared_cm2": min(same, key=lambda row: row["geometric_mm"])["shared_cm2"],
            "best_other": None if rival is None else (rival["scan_b"] if rival["scan_a"] == scan else rival["scan_a"]),
            "best_other_mm": None if rival is None else rival["geometric_mm"],
            "trusted": trusted,
        }
    return result


def _trusted_outcomes(nearest: dict[str, Any]) -> dict[str, int]:
    """Counts of what `compare` would answer for every scan against all the others."""
    counts = {"identified": 0, "wrong_person": 0, "uncertain": 0, "rejected": 0, "no_pair_with_enough_surface": 0}
    for entry in nearest.values():
        trusted = entry["trusted"]
        if trusted is None:
            counts["no_pair_with_enough_surface"] += 1
        elif trusted["decision"] == "same":
            counts["identified" if trusted["same_person"] else "wrong_person"] += 1
        else:
            counts["uncertain" if trusted["decision"] == "uncertain" else "rejected"] += 1
    return counts


def _decisions(rows: list[dict[str, Any]], min_area: float, statistics: dict[str, float]) -> dict[str, int]:
    """Pair decisions under the area limit and the calibrated same/uncertain/different rule of `compare`."""
    counts = {f"{kind}_{decision}": 0 for kind in ("same_person", "different_people")
              for decision in ("same", "uncertain", "different")}  # fmt: skip
    counts["undecided_small_overlap"] = 0
    for row in rows:
        if row["shared_cm2"] < min_area:
            counts["undecided_small_overlap"] += 1
            continue
        kind = "same_person" if row["genuine"] else "different_people"
        counts[f"{kind}_{_decision(row['geometric_mm'], statistics)}"] += 1
    return counts


def matrix(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for path in args.scans]
    names = [path.stem for path in paths]
    if len(set(names)) != len(names):
        raise SystemExit("scan file names must be unique (the name is the scan label)")
    person = {path.stem: _person(path) for path in paths}
    heads: dict[str, np.ndarray] = {}
    scans: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            head = extract_head(_upright(path, args), up_axis="z")
            heads[path.stem] = head.points
            scans[path.stem] = {"person": person[path.stem], "points": len(head.points),
                                "dropped_fraction": head.dropped, "error": None}  # fmt: skip
        except (ValueError, OSError) as error:
            scans[path.stem] = {"person": person[path.stem], "points": 0, "dropped_fraction": None, "error": str(error)}
    order = sorted(heads, key=lambda name: (person[name], name))
    if len(order) < 2:
        raise SystemExit("need at least two scans with a usable head")
    pairs = list(itertools.combinations(order, 2))
    tasks = [(heads[first], heads[second], not args.no_topology, args.seed) for first, second in pairs]
    jobs = args.jobs or max(1, min(len(tasks), (os.cpu_count() or 2) - 1))
    if jobs == 1:
        results = [_match_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_match_pair, tasks))
    rows: list[dict[str, Any]] = []
    poses: dict[tuple[str, str], dict[str, Any]] = {}
    for (first, second), result in zip(pairs, results):
        pose = result.pop("pose", None)
        if pose is not None:
            poses[(first, second)] = pose
        shared = result.get("shared_cm2")
        rows.append({"scan_a": first, "scan_b": second, "person_a": person[first], "person_b": person[second],
                     "genuine": person[first] == person[second],
                     "reliable": shared is not None and shared >= args.min_area, **result})  # fmt: skip
    for row in rows:
        if row.get("error") is not None:
            print(f"warning: {row['scan_a']} - {row['scan_b']}: {row['error']}", file=sys.stderr)
    summary = _matrix_summary(rows, scans, order, person, args)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "face_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in MATRIX_FIELDS} for row in rows)
    (output / "face_matrix_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")  # fmt: skip
    if not args.no_plots:
        summary["plots"] = _matrix_plots(rows, summary, heads, poses, order, person, output)
    return summary


def _matrix_summary(
    rows: list[dict[str, Any]],
    scans: dict[str, dict[str, Any]],
    order: list[str],
    person: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    people: dict[str, list[str]] = {}
    for scan in order:
        people.setdefault(person[scan], []).append(scan)
    summary: dict[str, Any] = {
        "settings": _settings(args),
        "scans": scans,
        "people": people,
        "min_shared_cm2": args.min_area,
        "pairs": {
            "genuine": sum(row["genuine"] for row in valid),
            "impostor": sum(not row["genuine"] for row in valid),
            "failed": len(rows) - len(valid),
            "reliable": sum(row["reliable"] for row in valid),
        },
        "note": "thresholds and the area limit are estimated on the same scans they are applied to, and on few "
        "people; read them as a description of this data set, not as biometric accuracy",
    }
    for group, reliable_only in (("all_pairs", False), ("reliable_pairs", True)):
        summary[group] = {}
        for score in ("geometric_mm", "topological_mm"):
            statistics = _statistics(*_scores_of(valid, score, reliable_only))
            if statistics is not None:
                summary[group][score] = statistics
    # `compare --calibration` reads the statistics and the area limit they were made with.
    trusted = summary["reliable_pairs"].get("geometric_mm")
    summary["calibrated_min_shared_cm2"] = args.min_area
    if trusted is None:
        trusted = summary["all_pairs"].get("geometric_mm")
        summary["calibrated_min_shared_cm2"] = 0.0
        print(f"warning: no same-person and different-person pairs share {args.min_area:g} cm2; "
              "the calibration uses all pairs", file=sys.stderr)  # fmt: skip
    limit = summary["calibrated_min_shared_cm2"]
    if trusted is not None:
        summary["geometric_mm"] = trusted
    nearest = _nearest(valid, order, limit, trusted)
    summary["identification"] = {
        "scans": len(nearest),
        "rank1_correct": sum(entry["correct"] for entry in nearest.values()),
        "with_area_limit": _trusted_outcomes(nearest) if trusted is not None else None,
        "nearest": nearest,
    }
    if trusted is not None:
        summary["decisions"] = _decisions(valid, limit, trusted)
    curve = []
    areas = sorted(row["shared_cm2"] for row in valid)
    for limit in np.arange(0.0, (areas[-1] if areas else 0.0) + 10.0, 10.0):
        genuine = [row["geometric_mm"] for row in valid if row["genuine"] and row["shared_cm2"] >= limit]
        impostor = [row["geometric_mm"] for row in valid if not row["genuine"] and row["shared_cm2"] >= limit]
        point: dict[str, Any] = {"min_shared_cm2": float(limit), "genuine_pairs": len(genuine),
                                 "impostor_pairs": len(impostor), "eer": None}  # fmt: skip
        if genuine and impostor:
            point["eer"] = equal_error_rate(genuine, impostor)[0]
        curve.append(point)
    summary["eer_by_min_shared_cm2"] = curve
    return summary


def _summarise(
    rows: list[dict[str, Any]], args: argparse.Namespace, yaws: list[float], pitches: list[float]
) -> dict[str, Any]:
    simulated = [row for row in rows if row["kind"] == "simulated" and row.get("error") is None]
    summary: dict[str, Any] = {
        "settings": _settings(args),
        "people": sorted({row["gallery"] for row in rows}),
        "yaws": yaws,
        "pitches": pitches,
        "genuine_pairs": sum(row["genuine"] for row in simulated),
        "impostor_pairs": sum(not row["genuine"] for row in simulated),
        "failed_pairs": sum(row.get("error") is not None for row in rows),
        "note": "genuine probes come from the same capture as the gallery; thresholds are estimated on the same "
        "data and on few people, so they are optimistic",
    }
    for score in ("geometric_mm", "topological_mm"):
        genuine = [row[score] for row in simulated if row["genuine"] and row.get(score) is not None]
        impostor = [row[score] for row in simulated if not row["genuine"] and row.get(score) is not None]
        if genuine and impostor:
            rate, threshold = equal_error_rate(genuine, impostor)
            summary[score] = {
                "eer": rate,
                "threshold": threshold,
                "genuine_max": max(genuine),
                "impostor_min": min(impostor),
                "genuine_median": float(np.median(genuine)),
                "impostor_median": float(np.median(impostor)),
            }
    real: dict[str, Any] = {}
    failures: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["kind"] != "real":
            continue
        if row.get("error") is not None:
            failures.setdefault(row["probe"], {})[row["gallery"]] = row["error"]
            continue
        real.setdefault(row["probe"], {})[row["gallery"]] = {
            key: row.get(key) for key in ("geometric_mm", "shared_cm2", "topological_mm", "overlap")
        }
    for probe, scores in list(real.items()):
        best = min(scores, key=lambda label: scores[label]["geometric_mm"])
        ordered = sorted(score["geometric_mm"] for score in scores.values())
        real[probe] = {
            "scores": scores,
            "best_match": best,
            "decision": (
                _decision(scores[best]["geometric_mm"], summary["geometric_mm"]) if "geometric_mm" in summary else None
            ),
            "margin_mm": ordered[1] - ordered[0] if len(ordered) > 1 else None,
            "failed": failures.get(probe, {}),
        }
    for probe, failed in failures.items():
        real.setdefault(
            probe, {"scores": {}, "best_match": None, "decision": None, "margin_mm": None, "failed": failed}
        )
    summary["real_probes"] = real
    return summary


def _decision(score: float, statistics: dict[str, float]) -> str:
    """'same' within the simulated genuine range and below every impostor,
    'different' at or above the closest impostor, 'uncertain' in between."""
    low, high = sorted((statistics["genuine_max"], statistics["impostor_min"]))
    if score <= low:
        return "same"
    if score >= high:
        return "different"
    return "uncertain"


def _band(axis: Any, statistics: dict[str, float], vertical: bool) -> None:
    low, high = sorted((statistics["genuine_max"], statistics["impostor_min"]))
    span = axis.axvspan if vertical else axis.axhspan
    span(low, high, color="#eef1f6", zorder=0, lw=0)
    line = axis.axvline if vertical else axis.axhline
    line(statistics["threshold"], color=MUTED, lw=1, ls=":")


def _label_line_ends(axis: Any, ends: list[tuple[tuple[float, float], str]]) -> None:
    """Direct labels at the right end of each line, pushed apart vertically."""
    if not ends:
        return
    high = max(point[1] for point, _ in ends)
    gap = 0.045 * max(high, 1e-9)
    left, right = axis.get_xlim()
    placed: list[float] = []
    for (x, y), text in sorted(ends, key=lambda item: item[0][1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        axis.annotate(text, (x, y), xytext=(x + 0.015 * (right - left), y_text), textcoords="data", va="center",
                      fontsize=8, color=INK, annotation_clip=False)  # fmt: skip
    axis.set_xlim(right=right + 0.08 * (right - left))


def _plot(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    examples: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    output: Path,
    example_yaw: float,
    plot_pitch: float,
) -> list[str]:
    plt = _pyplot()
    written = []
    valid = [row for row in rows if row["kind"] == "simulated" and row.get("error") is None]
    scores = [
        ("geometric_mm", "Геометрия: RMS после совмещения, мм"),
        ("topological_mm", "Топология: bottleneck радиальной карты, мм"),
    ]
    scores = [item for item in scores if item[0] in summary]
    if not scores:
        print("warning: no genuine/impostor scores, plots skipped", file=sys.stderr)
        return written

    level = [row for row in valid if row["pitch"] == plot_pitch]
    fig, axes = plt.subplots(1, len(scores), figsize=(6.2 * len(scores), 4.2), squeeze=False)
    for axis, (score, title) in zip(axes[0], scores):
        ends: list[tuple[tuple[float, float], str]] = []
        for person in summary["people"]:
            others = "/".join(label for label in summary["people"] if label != person)
            for genuine, colour, style, marker, target in ((True, GENUINE, "-", "o", person),
                                                           (False, IMPOSTOR, "--", "x", others)):  # fmt: skip
                points = sorted((row["yaw"], row[score]) for row in level
                                if row["probe"] == person and row["genuine"] == genuine and row[score] is not None)  # fmt: skip
                if points:
                    axis.plot(*zip(*points), color=colour, lw=2 if genuine else 1.2, ls=style, marker=marker, ms=5)
                    ends.append((points[-1], f"{person}→{target}"))
        _label_line_ends(axis, ends)
        _band(axis, summary[score], vertical=False)
        threshold = summary[score]["threshold"]
        axis.text(axis.get_xlim()[0], threshold, f" порог {threshold:.2f}", va="bottom", color=MUTED, fontsize=8)
        axis.set_title(f"{title}\nEER = {summary[score]['eer']:.0%} (все ракурсы), на графике наклон камеры "
                       f"{plot_pitch:+.0f}°", color=INK, loc="left")  # fmt: skip
        axis.set_xlabel("Угол съёмки относительно лица (yaw), °")
        yaws = summary["yaws"]
        axis.set_xticks(np.linspace(min(yaws), max(yaws), min(len(yaws), 7)))
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#e6e9f0", lw=0.8)
    handles = [
        plt.Line2D([], [], color=GENUINE, lw=2, marker="o", label="тот же человек"),
        plt.Line2D([], [], color=IMPOSTOR, lw=1.2, ls="--", marker="x", label="другой человек"),
        plt.Rectangle((0, 0), 1, 1, color="#eef1f6", label="зона неуверенности"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    path = output / "face_scores_vs_angle.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    real = summary.get("real_probes", {})
    fig, axes = plt.subplots(1, len(scores), figsize=(6.2 * len(scores), 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for axis, (score, title) in zip(axes[0], scores):
        for position, (genuine, colour, marker, name) in enumerate(
            ((True, GENUINE, "o", "тот же человек"), (False, IMPOSTOR, "x", "другой человек"))
        ):
            values = [row[score] for row in valid if row["genuine"] == genuine and row[score] is not None]
            axis.scatter(values, position + rng.uniform(-0.18, 0.18, len(values)), color=colour, marker=marker, s=26,
                         label=f"{name} (n={len(values)})", linewidths=1.2)  # fmt: skip
        for index, result in enumerate(real.values()):
            for label, value in result["scores"].items():
                if value[score] is None:
                    continue
                height = 2 + 0.2 * index
                axis.scatter(value[score], height, color=REAL, marker="D", s=40, edgecolors="white")
                axis.annotate(label, (value[score], height), textcoords="offset points", xytext=(0, 7), ha="center",
                              fontsize=8, color=INK)  # fmt: skip
        _band(axis, summary[score], vertical=True)
        axis.set_yticks(range(3 if real else 2), ["тот же", "другой", "реальная проба"][: 3 if real else 2])
        axis.set_title(title, color=INK, loc="left")
        axis.set_xlim(left=0)
        axis.grid(axis="x", color="#e6e9f0", lw=0.8)
    axes[0][0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    path = output / "face_distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    people = summary["people"][:2]
    if not all((person, label) in examples for person in people for label in people):
        print("warning: some example registrations failed, face_registration.png skipped", file=sys.stderr)
        return written
    fig, axes = plt.subplots(len(people), len(people), figsize=(3.6 * len(people) + 1.2, 3.8 * len(people)),
                             squeeze=False, layout="constrained")  # fmt: skip
    for row, person in enumerate(people):
        for col, label in enumerate(people):
            axis = axes[row][col]
            probe, gallery = examples[(person, label)]
            distance, _ = _kd_tree(gallery).query(probe)
            axis.scatter(gallery[:, 0], gallery[:, 2], s=0.4, color="#c9d3e6", rasterized=True)
            shown = axis.scatter(probe[:, 0], probe[:, 2], s=0.8, c=np.minimum(distance, 10.0), cmap="Blues_r", vmin=0,
                                 vmax=10, rasterized=True)  # fmt: skip
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            verdict = "тот же" if person == label else "другой"
            near = distance[distance < GATE]
            rms = float(np.sqrt(np.mean(near**2))) if len(near) else float("nan")
            axis.set_title(f"проба {person} → эталон {label} ({verdict})\nRMS от пробы до эталона в зоне {GATE:g} мм: "
                           f"{rms:.2f} мм",
                           color=INK, fontsize=9, loc="left")  # fmt: skip
    fig.colorbar(shown, ax=axes, shrink=0.6, label="расстояние до эталона, мм")
    fig.suptitle(f"Совмещение пробы, снятой под углом {example_yaw:+.0f}° (наклон {plot_pitch:+.0f}°), с эталонами "
                 "(профиль, лицо справа)", color=INK, x=0.02, ha="left")  # fmt: skip
    path = output / "face_registration.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path.name)
    return written


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("plots need matplotlib: python -m pip install '.[plot]' or pass --no-plots") from error
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip
    return plt


def _matrix_plots(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    heads: dict[str, np.ndarray],
    poses: dict[tuple[str, str], dict[str, Any]],
    order: list[str],
    person: dict[str, str],
    output: Path,
) -> list[str]:
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    valid = [row for row in rows if row.get("geometric_mm") is not None]
    if not valid:
        print("warning: no scored pairs, plots skipped", file=sys.stderr)
        return []
    min_area = summary["calibrated_min_shared_cm2"]
    trusted = summary.get("geometric_mm")
    # One hue, dark = surfaces close together (similar faces).
    ramp = LinearSegmentedColormap.from_list("close", ["#1f3b73", "#355DA6", "#8fb0de", "#eef3fb"])
    written = []

    # 1. Heat map of all pairs.
    size = len(order)
    values = np.full((size, size), np.nan)
    shared = np.full((size, size), np.nan)
    for row in valid:
        i, j = order.index(row["scan_a"]), order.index(row["scan_b"])
        values[i, j] = values[j, i] = row["geometric_mm"]
        shared[i, j] = shared[j, i] = row["shared_cm2"]
    high = float(np.nanquantile(values, 0.95))
    fig, axis = plt.subplots(figsize=(0.55 * size + 2.6, 0.55 * size + 1.6))
    shown = axis.imshow(values, cmap=ramp, vmin=0.0, vmax=high, interpolation="nearest")
    for i in range(size):
        for j in range(size):
            if i == j or np.isnan(values[i, j]):
                continue
            if shared[i, j] < min_area:
                axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor="#b7c0d1",
                                         lw=0))  # fmt: skip
            dark = values[i, j] < 0.45 * high
            axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if dark else INK)  # fmt: skip
    for k in range(1, size):
        if person[order[k]] != person[order[k - 1]]:
            axis.axhline(k - 0.5, color="white", lw=3)
            axis.axvline(k - 0.5, color="white", lw=3)
    axis.set_xticks(range(size), order, rotation=45, ha="right")
    axis.set_yticks(range(size), order)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(shown, ax=axis, shrink=0.75)
    bar.set_label("расхождение поверхностей (RMS), мм; темнее — ближе")
    axis.set_title(f"Все пары сканов. Белые линии разделяют людей;\nштриховка: общая поверхность меньше {min_area:g} "
                   "см², решение не принимается", color=INK, loc="left", fontsize=9)  # fmt: skip
    fig.tight_layout()
    path = output / "face_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    # 2. Score against the shared area: why small overlaps are not trusted.
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    largest = max(row["shared_cm2"] for row in valid)
    axis.axvspan(0, min_area, color="#eef1f6", lw=0, zorder=0)
    axis.text(min_area / 2, 0.02, f"меньше {min_area:g} см²:\nрешение не принимается", transform=axis.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=8, color=MUTED)  # fmt: skip
    for genuine, colour, marker, label in (
        (False, IMPOSTOR, "x", "разные люди"),
        (True, GENUINE, "o", "тот же человек"),
    ):
        chosen = [row for row in valid if row["genuine"] == genuine]
        axis.scatter([row["shared_cm2"] for row in chosen], [row["geometric_mm"] for row in chosen], color=colour,
                     marker=marker, s=42 if genuine else 30, linewidths=1.4 if not genuine else 0.8,
                     edgecolors="white" if genuine else None, label=f"{label} (n={len(chosen)})", zorder=3)  # fmt: skip
    # Direct labels for the same-person pairs, nudged down where they would overlap.
    placed: list[tuple[float, float]] = []
    span = max(row["geometric_mm"] for row in valid)
    for row in sorted((row for row in valid if row["genuine"]), key=lambda row: -row["geometric_mm"]):
        x, y = row["shared_cm2"], row["geometric_mm"]
        y_text = y + 0.012 * span
        while any(abs(x - px) < 0.16 * largest and abs(y_text - py) < 0.035 * span for px, py in placed):
            y_text -= 0.035 * span
        placed.append((x, y_text))
        axis.annotate(f"{row['scan_a']}–{row['scan_b']}", (x, y), xytext=(x + 0.012 * largest, y_text),
                      textcoords="data", fontsize=7, color=INK, va="center")  # fmt: skip
    if trusted is not None:
        axis.axhline(trusted["threshold"], color=MUTED, lw=1, ls=":")
        axis.text(largest * 1.02, trusted["threshold"], f"порог\n{trusted['threshold']:.2f} мм", va="center",
                  fontsize=8, color=MUTED)  # fmt: skip
    groups = summary["all_pairs"].get("geometric_mm"), summary["reliable_pairs"].get("geometric_mm")
    subtitle = " · ".join(f"{name}: EER {stats['eer']:.0%}" for name, stats in
                          zip(("все пары", f"от {min_area:g} см²"), groups) if stats)  # fmt: skip
    axis.set_title(f"Чем больше общая поверхность, тем надёжнее оценка\n{subtitle}", color=INK, loc="left")
    axis.set_xlabel("общая поверхность двух сканов после совмещения, см²")
    axis.set_ylabel("расхождение поверхностей (RMS), мм")
    axis.set_xlim(0, largest * 1.12)
    axis.set_ylim(bottom=0)
    axis.grid(color="#e6e9f0", lw=0.8)
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = output / "face_score_vs_area.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    # 3. Closest scan of the same person against the closest scan of anyone else.
    nearest = summary["identification"]["nearest"]
    listed = [scan for scan in order if scan in nearest]
    if listed:
        fig, axis = plt.subplots(figsize=(8.2, 0.42 * len(listed) + 1.9))
        for position, scan in enumerate(listed):
            entry = nearest[scan]
            same, other = entry["best_same_mm"], entry["best_other_mm"]
            if other is not None:
                axis.plot([same, other], [position, position], color="#d5dbe6", lw=2, zorder=1)
                axis.scatter(other, position, color=IMPOSTOR, marker="x", s=36, linewidths=1.5, zorder=3)
                axis.annotate(entry["best_other"], (other, position), textcoords="offset points", xytext=(6, -3),
                              fontsize=7, color=MUTED)  # fmt: skip
            hollow = entry["best_same_shared_cm2"] < min_area
            axis.scatter(same, position, s=48, zorder=4, color="white" if hollow else GENUINE, edgecolors=GENUINE,
                         linewidths=1.6)  # fmt: skip
            axis.text(1.01, position, "верно" if entry["correct"] else "ошибка", transform=axis.get_yaxis_transform(),
                      va="center", fontsize=8, color=INK)  # fmt: skip
        axis.set_yticks(range(len(listed)), listed)
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.grid(axis="x", color="#e6e9f0", lw=0.8)
        axis.set_xlabel("расхождение с ближайшим сканом, мм")
        hollow_label = f"то же, но общая поверхность < {min_area:g} см²"
        handles = [
            plt.Line2D([], [], color=GENUINE, marker="o", ls="", ms=7, label="ближайший скан того же человека"),
            plt.Line2D([], [], color=GENUINE, marker="o", ls="", ms=7, mfc="white", label=hollow_label),
            plt.Line2D(
                [], [], color=IMPOSTOR, marker="x", ls="", ms=7, mew=1.5, label="ближайший скан другого человека"
            ),
        ]
        fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=8)
        identification = summary["identification"]
        title = (f"Ближайший скан — тот же человек: {identification['rank1_correct']} из {identification['scans']} "
                 "(синяя точка левее крестика)")  # fmt: skip
        outcomes = identification.get("with_area_limit")
        if outcomes:
            undecided = outcomes["uncertain"] + outcomes["rejected"] + outcomes["no_pair_with_enough_surface"]
            title += (f"\nпо правилу «от {summary['calibrated_min_shared_cm2']:g} см² общей поверхности»: узнаны "
                      f"{outcomes['identified']}, приняты за другого {outcomes['wrong_person']}, без решения {undecided}")  # fmt: skip
        axis.set_title(title, color=INK, loc="left", fontsize=9)
        fig.tight_layout(rect=(0, 0.9 / (0.42 * len(listed) + 1.9), 1, 1))
        path = output / "face_nearest.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path.name)

    # 4. Residual maps of telling pairs.
    genuine = [row for row in valid if row["genuine"]]
    impostor = [row for row in valid if not row["genuine"]]
    reliable_impostor = [row for row in impostor if row["reliable"]] or impostor
    chosen = []
    if genuine:
        chosen.append((min(genuine, key=lambda row: row["geometric_mm"]), "самая похожая пара одного человека"))
        chosen.append((max(genuine, key=lambda row: row["geometric_mm"]), "самая непохожая пара одного человека"))
    if reliable_impostor:
        chosen.append((min(reliable_impostor, key=lambda row: row["geometric_mm"]), "самая похожая пара разных людей"))
    chosen = [(row, caption) for row, caption in chosen if (row["scan_a"], row["scan_b"]) in poses]
    if chosen:
        fig, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen) + 1.0, 4.6), squeeze=False,
                                 layout="constrained")  # fmt: skip
        error_ramp = LinearSegmentedColormap.from_list("error", ["#eef3fb", "#8fb0de", "#355DA6", "#1f3b73"])
        for axis, (row, caption) in zip(axes[0], chosen):
            pose = poses[(row["scan_a"], row["scan_b"])]
            rotation, translation = np.array(pose["rotation"]), np.array(pose["translation"])
            probe = heads[row["scan_a"]] @ rotation.T + translation
            gallery = heads[row["scan_b"]]
            facing = estimate_normals(gallery, HEAD_CENTRE)[:, :2].mean(axis=0)
            facing = np.append(facing / max(np.linalg.norm(facing), 1e-9), 0.0)
            lateral = np.cross([0.0, 0.0, 1.0], facing)
            distance, _ = _kd_tree(gallery).query(probe)
            order_ = np.argsort(probe @ facing)
            axis.scatter(gallery @ lateral, gallery[:, 2], s=0.6, color="#dfe4ee", rasterized=True)
            shown = axis.scatter((probe @ lateral)[order_], probe[order_, 2], s=1.2, c=np.minimum(distance, GATE)[order_],
                                 cmap=error_ramp, vmin=0, vmax=GATE, rasterized=True)  # fmt: skip
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(f"{caption}\n{row['scan_a']} → {row['scan_b']}: {row['geometric_mm']:.2f} мм, "
                           f"{row['shared_cm2']:.0f} см²", color=INK, fontsize=9, loc="left")  # fmt: skip
        fig.colorbar(shown, ax=axes, shrink=0.6, label=f"расстояние до второго скана, мм (обрезано на {GATE:g})")
        fig.suptitle("Где поверхности расходятся (вид спереди; серым — второй скан)", color=INK, x=0.02, ha="left")
        path = output / "face_examples.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path.name)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "experiment", "matrix"):
        command = commands.add_parser(name)
        command.add_argument("--up-axis", choices=("x", "y", "z"), required=True,
                             help="vertical axis of the scans (Revo Scan and iPhone exports: y)")  # fmt: skip
        command.add_argument("--units", choices=tuple(UNITS), default="mm", help="length units of the scans")
        command.add_argument("--no-topology", action="store_true", help="skip the persistence score (faster)")
        command.add_argument("--seed", type=int, default=0)
        if name == "compare":
            command.add_argument("probe", help="scan to identify")
            command.add_argument("--gallery", action="append", required=True,
                                 help="enrolled person LABEL=PATH (repeatable)")  # fmt: skip
            command.add_argument("--calibration",
                                 help="face_matrix_summary.json of a matrix or face_summary.json of an experiment: "
                                 "decide same/different/uncertain")  # fmt: skip
            command.add_argument("--threshold", type=float, help="instead of --calibration: same if RMS <= this, mm")
            command.add_argument("--min-area", type=float,
                                 help="cm2 of common surface below which a pair is not used (default: from the "
                                 "calibration, else 0)")  # fmt: skip
            command.add_argument("--output", help="write the JSON result here as well")
        elif name == "matrix":
            command.add_argument("scans", nargs="+",
                                 help="scans named PERSON<number>.ply, e.g. Ivan1.ply Ivan2.ply Olga1.ply")  # fmt: skip
            command.add_argument("--min-area", type=float, default=MIN_SHARED_CM2,
                                 help="cm2 of common surface needed to trust a pair (default %(default)g)")  # fmt: skip
            command.add_argument("--jobs", type=int, default=0, help="worker processes (default: CPU count - 1)")
            command.add_argument("--out", required=True, help="results folder")
            command.add_argument("--no-plots", action="store_true")
        else:
            command.add_argument("--person", action="append", required=True,
                                 help="full head scan LABEL=PATH (repeatable)")  # fmt: skip
            command.add_argument("--probe", action="append", help="extra real scan to identify (repeatable)")
            command.add_argument("--front-axis",
                                 help="direction the face looks in the input frame (default +z, or +x when Z is up)")  # fmt: skip
            command.add_argument("--yaws", default="-90:90:15",
                                 help="camera yaw angles, START:STOP:STEP or a list; write --yaws=-90:90:15")  # fmt: skip
            command.add_argument("--pitches", default="0,20", help="camera pitch angles, list; write --pitches=-20,0")
            command.add_argument("--example-yaw", type=float, default=60.0,
                                 help="yaw shown in the registration plot (nearest listed yaw)")  # fmt: skip
            command.add_argument("--out", required=True, help="results folder")
            command.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        result = {"compare": compare, "experiment": experiment, "matrix": matrix}[args.command](args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(f"morse-lidar-face: {error}") from error
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.command == "compare" and args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
