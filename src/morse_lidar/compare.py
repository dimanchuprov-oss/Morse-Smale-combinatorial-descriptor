"""Compare two JSON descriptors written by ``morse-lidar``.

The main distance is the bottleneck distance between persistence diagrams,
which is stable: it never exceeds the sup-norm difference of the two scalar
fields (Cohen-Steiner, Edelsbrunner, Harer 2007). Counts are reported
alongside for readability; they are not stable under noise without a
persistence threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _diagram(report: dict[str, Any], dimension: str) -> list[tuple[float, float]]:
    diagram = report.get("persistence_diagram")
    if diagram is None:
        raise ValueError("descriptor has no persistence_diagram; recompute it with the topology extra installed")
    return [(birth, math.inf if death is None else death) for birth, death in diagram.get(dimension, [])]


def bottleneck(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> float:
    """Bottleneck distance; essential (infinite) points must match in number."""
    try:
        import gudhi
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional topology extra: python -m pip install '.[topology]'") from error
    finite = [[point for point in diagram if math.isfinite(point[1])] for diagram in (first, second)]
    essential = [sorted(point[0] for point in diagram if not math.isfinite(point[1])) for diagram in (first, second)]
    if len(essential[0]) != len(essential[1]):
        return math.inf
    essential_distance = max((abs(left - right) for left, right in zip(*essential)), default=0.0)
    return max(float(gudhi.bottleneck_distance(finite[0], finite[1])), essential_distance)


def compare_reports(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    for report in (first, second):
        if report.get("schema_version") not in (None, 1):
            raise ValueError("unsupported descriptor schema_version")
    for key in ("schema_version", "parameters", "scalar", "geometry", "surface", "persistence_threshold"):
        if first.get(key) != second.get(key):
            raise ValueError(f"descriptors differ in {key!r}: {first.get(key)!r} vs {second.get(key)!r}")
    for name, report in (("first", first), ("second", second)):
        if report.get("persistence_diagram") is None:
            raise ValueError(f"{name} descriptor has no persistence_diagram; recompute it with GUDHI installed")
    dimensions = sorted(set(first["persistence_diagram"]) | set(second["persistence_diagram"]))
    distances = {dimension: bottleneck(_diagram(first, dimension), _diagram(second, dimension)) for dimension in dimensions}
    result: dict[str, Any] = {
        "bottleneck": {dimension: (None if math.isinf(value) else value) for dimension, value in distances.items()},
        "bottleneck_max": None if not distances or math.isinf(max(distances.values())) else max(distances.values()),
        "counts": {"first": first.get("counts"), "second": second.get("counts")},
    }
    if "persistent_counts" in first and "persistent_counts" in second:
        result["persistent_counts"] = {"first": first["persistent_counts"], "second": second["persistent_counts"]}
        result["persistent_counts_equal"] = first["persistent_counts"] == second["persistent_counts"]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", help="first descriptor JSON")
    parser.add_argument("second", help="second descriptor JSON")
    args = parser.parse_args(argv)
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.first, args.second)]
    try:
        result = compare_reports(*reports)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"morse-lidar-compare: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
