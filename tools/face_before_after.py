"""Before/after chart: the face matcher of PR #4 against the current one, on the same scans.

The old scores (``face_pairs_pr4_method.csv``: probe, gallery, geometric_mm)
were computed with the code of PR #4 on biometrics/<Person><n>.ply; the new ones
are the ``face_pairs.csv`` of ``morse-lidar-face matrix``::

    python tools/face_before_after.py biometrics/face_matrix

The left panel repeats Dima's protocol: the first scan of every person is
enrolled, the other scans are probes.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from morse_lidar.face import equal_error_rate

GENUINE, IMPOSTOR, INK, MUTED, BEFORE = "#355DA6", "#E4572E", "#1d2433", "#6b7a99", "#b7c0d1"
DECISION = {"same": "тот же", "different": "другой", "uncertain": "не уверен", None: "мало общей поверхности"}


def person(scan: str) -> str:
    return re.sub(r"[\d _-]+$", "", scan) or scan


def load(folder: Path) -> tuple[dict, dict, dict, dict]:
    old: dict[tuple[str, str], float] = {}
    with (folder / "face_pairs_pr4_method.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            old[(row["probe"], row["gallery"])] = float(row["geometric_mm"])
    new: dict[tuple[str, str], float] = {}
    area: dict[tuple[str, str], float] = {}
    with (folder / "face_pairs.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["geometric_mm"]:
                continue
            for key in ((row["scan_a"], row["scan_b"]), (row["scan_b"], row["scan_a"])):
                new[key], area[key] = float(row["geometric_mm"]), float(row["shared_cm2"])
    summary = json.loads((folder / "face_matrix_summary.json").read_text(encoding="utf-8"))
    return old, new, area, summary


def decide(score: float, statistics: dict) -> str:
    low, high = sorted((statistics["genuine_max"], statistics["impostor_min"]))
    return "same" if score <= low else "different" if score >= high else "uncertain"


def main(folder: Path) -> Path:
    old, new, area, summary = load(folder)
    scans = sorted({scan for scan, _ in old}, key=lambda scan: (person(scan), scan))
    gallery = [next(scan for scan in scans if person(scan) == name) for name in dict.fromkeys(map(person, scans))]
    probes = [scan for scan in scans if scan not in gallery]
    limit, statistics = summary["calibrated_min_shared_cm2"], summary["geometric_mm"]

    def nearest(score: dict, scan: str, candidates: list[str]) -> tuple[float, str]:
        return min((score[(scan, other)], other) for other in candidates if other != scan)

    def rank1(score: dict, candidates_of) -> int:
        return sum(person(nearest(score, scan, candidates_of(scan))[1]) == person(scan) for scan in scans_of)

    scans_of = probes
    dima_old, dima_new = rank1(old, lambda _: gallery), rank1(new, lambda _: gallery)
    scans_of = [scan for scan in scans if sum(person(other) == person(scan) for other in scans) > 1]
    all_old, all_new = rank1(old, lambda _: scans), rank1(new, lambda _: scans)
    pairs = [(a, b) for i, a in enumerate(scans) for b in scans[i + 1 :]]
    symmetric_old = {pair: (old[pair] + old[pair[::-1]]) / 2 for pair in pairs}
    eer_old = equal_error_rate(*[[value for pair, value in symmetric_old.items() if (person(pair[0]) == person(pair[1])) == same]
                                 for same in (True, False)])[0]  # fmt: skip

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip
    fig = plt.figure(figsize=(12.4, 5.6))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.12)

    # Left: Dima's protocol, probe by probe.
    table = fig.add_subplot(grid[0])
    table.set_xlim(0, 10)
    table.set_ylim(len(probes) + 0.2, -1.4)
    table.axis("off")
    columns = ((0.0, "проба"), (1.9, "было (PR #4)"), (5.0, "стало"), (7.75, "решение сейчас"))
    for x, title in columns:
        table.text(x, -0.75, title, fontsize=9, color=MUTED, va="center")
    for row, probe in enumerate(probes):
        table.text(0.0, row, probe, va="center", fontsize=9.5, color=INK)
        for x, score in ((1.9, old), (5.0, new)):
            value, match = nearest(score, probe, gallery)
            right = person(match) == person(probe)
            table.scatter(x + 0.12, row, marker="o" if right else "x", s=46 if right else 40,
                          color=GENUINE if right else IMPOSTOR, linewidths=1.6, zorder=3)  # fmt: skip
            table.text(x + 0.42, row, f"{match}  {value:.2f} мм", va="center", fontsize=9, color=INK)
        usable = [(new[(probe, other)], other) for other in gallery if area[(probe, other)] >= limit]
        verdict = None
        if usable:
            value, match = min(usable)
            verdict = decide(value, statistics)
            text = f"{DECISION[verdict]}" + (f": {match}" if verdict == "same" else "")
        else:
            text = DECISION[None]
        table.text(7.75, row, text, va="center", fontsize=9, color=INK if verdict == "same" else MUTED)
        if row < len(probes) - 1:
            table.axhline(row + 0.5, color="#eef1f6", lw=0.8)
    table.set_title(f"Протокол Димы: эталон — первый скан каждого, {len(probes)} проб, ближайший эталон\n"
                    f"верный эталон первым: было {dima_old} из {len(probes)}, стало {dima_new} из {len(probes)}",
                    loc="left", color=INK, fontsize=10)  # fmt: skip
    table.scatter([], [], marker="o", color=GENUINE, label="тот же человек")
    table.scatter([], [], marker="x", color=IMPOSTOR, label="другой человек")
    table.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.1), ncol=2, fontsize=8.5)

    # Right: the summary numbers.
    bars = fig.add_subplot(grid[1])
    metrics = [
        (f"верный эталон первым (протокол Димы, {len(probes)} проб)", dima_old / len(probes), dima_new / len(probes),
         f"{dima_old}/{len(probes)}", f"{dima_new}/{len(probes)}"),
        (f"ближайший скан — тот же человек (все {len(scans_of)} сканов)", all_old / len(scans_of),
         all_new / len(scans_of),
         f"{all_old}/{len(scans_of)}", f"{all_new}/{len(scans_of)}"),
        ("верных решений «тот же / другой» при пороге EER (все пары)", 1 - eer_old,
         1 - summary["all_pairs"]["geometric_mm"]["eer"],
         f"{1 - eer_old:.0%}", f"{1 - summary['all_pairs']['geometric_mm']['eer']:.0%}"),
    ]  # fmt: skip
    for index, (label, before, after, before_text, after_text) in enumerate(metrics):
        y = 1.4 * (len(metrics) - 1 - index)
        bars.text(0.0, y + 0.52, label, va="bottom", fontsize=8.5, color=INK)
        bars.barh(y + 0.19, before, height=0.34, color=BEFORE)
        bars.barh(y - 0.19, after, height=0.34, color=GENUINE)
        bars.text(before + 0.015, y + 0.19, before_text, va="center", fontsize=8.5, color=MUTED)
        bars.text(after + 0.015, y - 0.19, after_text, va="center", fontsize=8.5, color=INK)
    bars.set_yticks([])
    bars.spines["left"].set_visible(False)
    bars.set_ylim(-0.6, 1.4 * (len(metrics) - 1) + 0.95)
    bars.set_xlim(0, 1.12)
    bars.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    bars.grid(axis="x", color="#eef1f6", lw=0.8)
    bars.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch

    bars.legend(handles=[Patch(color=BEFORE, label="было (PR #4)"), Patch(color=GENUINE, label="стало")],
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5)  # fmt: skip
    genuine_old = sorted(value for pair, value in symmetric_old.items() if person(pair[0]) == person(pair[1]))
    impostor_old = sorted(value for pair, value in symmetric_old.items() if person(pair[0]) != person(pair[1]))
    median = lambda values: values[len(values) // 2]
    stats = summary["all_pairs"]["geometric_mm"]
    bars.set_title(f"Те же {len(scans)} сканов, старый и новый метод\nмедиана свои / чужие: было "
                   f"{median(genuine_old):.1f} / {median(impostor_old):.1f} мм, стало {stats['genuine_median']:.2f} / "
                   f"{stats['impostor_median']:.2f} мм", loc="left", color=INK, fontsize=10)  # fmt: skip
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.12)
    path = folder / "face_before_after.png"
    fig.savefig(path, dpi=160)
    return path


if __name__ == "__main__":
    print(main(Path(sys.argv[1] if len(sys.argv) > 1 else "biometrics/face_matrix")))
