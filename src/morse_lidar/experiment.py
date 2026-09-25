"""Run a fixed-parameter repeat-scan experiment from a JSON manifest."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
import json
from pathlib import Path
import platform

from .cli import main as describe
from .compare import compare_reports


def run_experiment(manifest_path: Path, output: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("units") not in {"m", "cm", "mm"}:
        raise ValueError("manifest units must be m, cm or mm (no automatic conversion)")
    parameters = manifest["parameters"]
    if not isinstance(parameters, dict) or not {"up_axis", "geometry", "scalar", "persistence_threshold"} <= parameters.keys():
        raise ValueError("parameters must specify up_axis, geometry, scalar and persistence_threshold")
    if parameters["persistence_threshold"] is None:
        raise ValueError("experiment requires a fixed persistence_threshold and GUDHI")
    if {"input", "output", "raster_output", "periodic"} & parameters.keys():
        raise ValueError("input/output paths and periodic alias are not experiment parameters")
    scans = manifest["scans"]
    if len(scans) < 2 or any(not isinstance(s.get("object"), str) or not s["object"] for s in scans):
        raise ValueError("provide at least two scans with nonempty object labels")
    if len({s["object"] for s in scans}) == len(scans):
        raise ValueError("provide at least two repeats of one object")
    paths = [(manifest_path.parent / scan["path"]).resolve(strict=True) for scan in scans]
    if len(set(paths)) != len(paths):
        raise ValueError("each scan must have a distinct input path")
    arguments = [token for key, value in parameters.items() if value is not None for token in ("--" + key.replace("_", "-"), str(value))]
    # A new output directory prevents accidental replacement of prior runs.
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reports, inputs = [], []
    for index, (scan, path) in enumerate(zip(scans, paths)):
        target = output / f"scan_{index:03d}.json"
        digest = _sha256(path)
        with (output / f"scan_{index:03d}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            describe(["--input", str(path), "--output", str(target), *arguments])
        if _sha256(path) != digest:
            raise ValueError(f"input changed during processing: {path}")
        reports.append(json.loads(target.read_text(encoding="utf-8")))
        inputs.append({"index": index, "path": str(path), "object": scan["object"], "sha256": digest,
                       "descriptor": target.name})
    pairs = []
    for left, right in combinations(range(len(scans)), 2):
        pairs.append({"left": left, "right": right, "same_object": scans[left]["object"] == scans[right]["object"],
                      **compare_reports(reports[left], reports[right])})
    dependencies = {}
    for package in ("morse-lidar", "numpy", "scipy", "gudhi", "open3d"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    result = {"experiment_version": 1, "units": manifest["units"], "parameters": reports[0]["parameters"],
              "python": platform.python_version(), "dependencies": dependencies,
              "source_sha256": {path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
              "inputs": inputs, "pairs": pairs}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="new directory for this run")
    args = parser.parse_args(argv)
    result = run_experiment(args.manifest, args.output)
    print(json.dumps({"scans": len(result["inputs"]), "pairs": len(result["pairs"]), "results": str(args.output / "results.json")}))


if __name__ == "__main__":
    main()
