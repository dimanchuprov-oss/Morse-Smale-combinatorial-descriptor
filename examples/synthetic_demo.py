"""End-to-end demo on synthetic iPhone-like scans (no real data needed).

Writes Y-up binary PLY scans of a floor with an object on it, runs the CLI
on each and compares the descriptors:

* the same object scanned twice (different noise and sampling),
* the same object after a local change (an added bump),
* a different object.

Run:  python examples/synthetic_demo.py [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from morse_lidar.cli import main as describe
from morse_lidar.compare import compare_reports

NOISE = 0.003  # metres, roughly iPhone LiDAR after ARKit fusion


def _object(x: np.ndarray, z: np.ndarray, kind: str) -> np.ndarray:
    """Height (metres) of an object standing on a floor at y = 0."""
    height = 0.25 * np.exp(-((x - 0.1) ** 2 + z**2) / 0.02) + 0.18 * np.exp(-((x + 0.25) ** 2 + (z - 0.1) ** 2) / 0.01)
    if kind in {"changed", "different"}:
        height += 0.12 * np.exp(-((x - 0.1) ** 2 + (z + 0.3) ** 2) / 0.005)
    if kind == "different":
        height = 0.3 * np.exp(-(x**2 + z**2) / 0.05)
    return height


def write_scan(path: Path, kind: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = 60_000
    x, z = rng.uniform(-0.6, 0.6, count), rng.uniform(-0.6, 0.6, count)
    y = _object(x, z, kind) + NOISE * rng.normal(size=count)
    points = np.column_stack((x, y, z)).astype("<f4")  # ARKit: Y is up
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    path.write_bytes(header.encode("ascii") + points.tobytes())


def run(directory: Path) -> dict[str, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    scans = {"object_a": ("base", 1), "object_a_rescan": ("base", 2), "object_a_changed": ("changed", 3), "object_b": ("different", 4)}
    reports = {}
    for name, (kind, seed) in scans.items():
        write_scan(directory / f"{name}.ply", kind, seed)
        output = directory / f"{name}.json"
        describe(
            [
                "--input", str(directory / f"{name}.ply"), "--output", str(output),
                "--up-axis", "y", "--rows", "96", "--cols", "96", "--sigma", "1",
                "--persistence-threshold", "0.02",
            ]
        )
        reports[name] = json.loads(output.read_text())
    results = {
        other: compare_reports(reports["object_a"], reports[other])
        for other in ("object_a_rescan", "object_a_changed", "object_b")
    }
    return results


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_output")
    for other, result in run(target).items():
        print(f"object_a vs {other}: bottleneck={result['bottleneck']} persistent_counts_equal={result['persistent_counts_equal']}")
        print(f"    {result['persistent_counts']}")
