"""Command line for face verification across viewing angles.

Run the angle experiment on full head scans, identify extra real scans and plot
the comparison::

    morse-lidar-face experiment --person A=a.ply --person B=b.ply --probe x.ply \\
        --up-axis y --front-axis +z --out results/faces

Identify a new scan against enrolled people, with thresholds from an experiment::

    morse-lidar-face compare probe.ply --gallery A=a.ply --gallery B=b.ply --up-axis y \\
        --calibration results/faces/face_summary.json

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
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .face import (
    HEAD_CENTRE,
    Enrolled,
    _kd_tree,
    _trimmed_rms,
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
FIELDS = ["probe", "kind", "yaw", "pitch", "gallery", "genuine", "geometric_mm", "topological_mm", "overlap",
          "radial_fraction", "topology_error", "error"]  # fmt: skip


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


def _score(probe: np.ndarray, gallery: Enrolled, args: argparse.Namespace, seed: int) -> tuple[dict[str, Any], Any]:
    result = match(probe, gallery, topology=not args.no_topology, seed=seed)
    return {**result.as_dict(), "error": None}, result


def _settings(args: argparse.Namespace) -> dict[str, str]:
    return {"up_axis": args.up_axis, "units": args.units}


def _calibration(args: argparse.Namespace) -> dict[str, float]:
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if "geometric_mm" not in calibration:
        raise SystemExit(f"{args.calibration} has no geometric_mm statistics; is it a face_summary.json?")
    if "settings" not in calibration:
        raise SystemExit(f"{args.calibration} records no settings; regenerate it with this version")
    if calibration["settings"] != _settings(args):
        raise SystemExit(f"calibration was made with {calibration['settings']}, not {_settings(args)}")
    return calibration["geometric_mm"]


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.calibration and args.threshold is not None:
        raise SystemExit("pass either --calibration or --threshold, not both")
    statistics = _calibration(args) if args.calibration else None
    galleries = {label: _enroll(_upright(path, args)) for label, path in _labelled(args.gallery, "--gallery").items()}
    probe = extract_head(_upright(Path(args.probe), args), up_axis="z").points
    scores = {label: _score(probe, gallery, args, args.seed)[0] for label, gallery in galleries.items()}
    best = min(scores, key=lambda label: scores[label]["geometric_mm"])
    result: dict[str, Any] = {"probe": str(args.probe), "scores": scores, "best_match": best}
    if len(scores) > 1:
        ordered = sorted(score["geometric_mm"] for score in scores.values())
        result["margin_mm"] = ordered[1] - ordered[0]
    if statistics is not None:
        result["calibration"] = {key: statistics[key] for key in ("genuine_max", "impostor_min", "threshold")}
        result["decision"] = _decision(scores[best]["geometric_mm"], statistics)
    elif args.threshold is not None:
        result["threshold_mm"] = args.threshold
        result["decision"] = "same" if scores[best]["geometric_mm"] <= args.threshold else "different"
    else:
        result["decision"] = None
        result["note"] = "pass --calibration (face_summary.json of an experiment) or --threshold to get a decision"
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
            key: row.get(key) for key in ("geometric_mm", "topological_mm", "overlap")
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("plots need matplotlib: python -m pip install '.[plot]' or pass --no-plots") from error
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip
    written = []
    valid = [row for row in rows if row["kind"] == "simulated" and row.get("error") is None]
    scores = [
        ("geometric_mm", "Геометрия: RMS после совмещения, мм"),
        ("topological_mm", "Топология: bottleneck радиальной карты, мм"),
    ]
    scores = [item for item in scores if item[0] in summary]
    if not scores:
        print("warning: no genuine/impostor scores, plots skipped")
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
        print("warning: some example registrations failed, face_registration.png skipped")
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
            axis.set_title(f"проба {person} → эталон {label} ({verdict})\nRMS {_trimmed_rms(distance):.2f} мм",
                           color=INK, fontsize=9, loc="left")  # fmt: skip
    fig.colorbar(shown, ax=axes, shrink=0.6, label="расстояние до эталона, мм")
    fig.suptitle(f"Совмещение пробы, снятой под углом {example_yaw:+.0f}° (наклон {plot_pitch:+.0f}°), с эталонами "
                 "(профиль, лицо справа)", color=INK, x=0.02, ha="left")  # fmt: skip
    path = output / "face_registration.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path.name)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "experiment"):
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
                                 help="face_summary.json of an experiment: decide same/different/uncertain")  # fmt: skip
            command.add_argument("--threshold", type=float, help="instead of --calibration: same if RMS <= this, mm")
            command.add_argument("--output", help="write the JSON result here as well")
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
        result = compare(args) if args.command == "compare" else experiment(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(f"morse-lidar-face: {error}") from error
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.command == "compare" and args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
