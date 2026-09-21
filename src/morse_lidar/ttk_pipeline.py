"""Run by ParaView's ``pvpython`` and serialize a real TTK MSC.

The script deliberately uses TTK's VTK Python wrappers directly. This avoids
depending on ParaView's GUI plugin auto-loading policy, which differs between
binary distributions.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import types


REQUIRED_CRITICAL_ARRAYS = ("CellId", "CellDimension")
REQUIRED_SEPARATRIX_ARRAYS = (
    "SourceId",
    "DestinationId",
    "SeparatrixId",
    "SeparatrixType",
)


def _load_ttk_class(module_name: str, class_name: str):
    """Import one wrapper without eagerly importing every optional TTK filter."""
    try:
        module = __import__(f"topologytoolkit.{module_name}", fromlist=[class_name])
    except ImportError:
        candidates = []
        for entry in sys.path:
            root = Path(entry)
            if any(root.glob(f"{module_name}.*")):
                candidates.append(str(root))
            package = root / "topologytoolkit"
            if package.is_dir() and any(package.glob(f"{module_name}.*")):
                candidates.append(str(package))
        configured = os.environ.get("TTK_PYTHON_DIR")
        if configured:
            candidates.insert(0, configured)
        if not candidates:
            raise
        package = types.ModuleType("topologytoolkit")
        package.__path__ = candidates
        sys.modules["topologytoolkit"] = package
        module = __import__(f"topologytoolkit.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)


def _array_names(attributes) -> list[str]:
    return [attributes.GetArrayName(index) for index in range(attributes.GetNumberOfArrays())]


def _schema(data) -> dict[str, list[str]]:
    return {
        "point_data": _array_names(data.GetPointData()),
        "cell_data": _array_names(data.GetCellData()),
    }


def _require_array(attributes, name: str):
    array = attributes.GetArray(name)
    if array is None:
        raise RuntimeError(f"TTK output is missing required array {name!r}; found {_array_names(attributes)!r}")
    return array


def _optional_value(attributes, name: str, index: int, default=None):
    array = attributes.GetArray(name)
    return default if array is None else array.GetTuple1(index)


def _ordered_polyline(data, cell_ids: list[int]) -> list[list[float]]:
    """Join TTK's line segments by shared VTK point ids."""
    adjacency: dict[int, list[int]] = defaultdict(list)
    for cell_id in cell_ids:
        cell = data.GetCell(cell_id)
        if cell.GetNumberOfPoints() != 2:
            raise RuntimeError("TTK 1-separatrix output contains a non-line cell")
        left, right = cell.GetPointId(0), cell.GetPointId(1)
        adjacency[left].append(right)
        adjacency[right].append(left)
    endpoints = sorted(point for point, adjacent in adjacency.items() if len(adjacent) == 1)
    if len(endpoints) != 2 or any(len(adjacent) > 2 for adjacent in adjacency.values()):
        raise RuntimeError("TTK separatrix segments do not form one simple polyline")
    ordered = [endpoints[0]]
    previous = None
    current = endpoints[0]
    while current != endpoints[1]:
        following = [point for point in adjacency[current] if point != previous]
        if len(following) != 1:
            raise RuntimeError("TTK separatrix polyline is disconnected")
        previous, current = current, following[0]
        ordered.append(current)
    return [[float(value) for value in data.GetPoint(point)] for point in ordered]


def _critical_points(data) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    arrays = data.GetPointData()
    for name in REQUIRED_CRITICAL_ARRAYS:
        _require_array(arrays, name)
    lookup: dict[tuple[int, int], int] = {}
    result = []
    for index in range(data.GetNumberOfPoints()):
        cell_id = int(_require_array(arrays, "CellId").GetTuple1(index))
        dimension = int(_require_array(arrays, "CellDimension").GetTuple1(index))
        lookup[(dimension, cell_id)] = index
        # TTK 1.4 uses CellDimension as the critical type on 2-manifolds.
        # Older releases additionally exposed the same values as CriticalType.
        critical_type = int(_optional_value(arrays, "CriticalType", index, dimension))
        result.append(
            {
                "id": index,
                "position": [float(value) for value in data.GetPoint(index)],
                "critical_type": critical_type,
                "cell_id": cell_id,
                "cell_dimension": dimension,
                "vertex_id": int(
                    _optional_value(
                        arrays,
                        "PLVertexIdentifier",
                        index,
                        _optional_value(arrays, "ttkVertexScalarField", index, -1),
                    )
                ),
                "scalar": float(
                    _optional_value(
                        arrays,
                        "Scalar",
                        index,
                        _optional_value(arrays, "ScalarField", index, float("nan")),
                    )
                ),
                "is_on_boundary": bool(_optional_value(arrays, "IsOnBoundary", index, 0)),
            }
        )
    return result, lookup


def _separatrices(data, critical_lookup: dict[tuple[int, int], int]) -> list[dict[str, object]]:
    arrays = data.GetCellData()
    required = {name: _require_array(arrays, name) for name in REQUIRED_SEPARATRIX_ARRAYS}
    grouped: dict[int, list[int]] = defaultdict(list)
    for cell_id in range(data.GetNumberOfCells()):
        grouped[int(required["SeparatrixId"].GetTuple1(cell_id))].append(cell_id)
    result = []
    for identifier, cells in sorted(grouped.items()):
        first = cells[0]
        raw_source = int(required["SourceId"].GetTuple1(first))
        raw_destination = int(required["DestinationId"].GetTuple1(first))
        separatrix_type = int(required["SeparatrixType"].GetTuple1(first))
        if any(
            int(required[name].GetTuple1(cell)) != int(required[name].GetTuple1(first))
            for cell in cells
            for name in REQUIRED_SEPARATRIX_ARRAYS
        ):
            raise RuntimeError(f"TTK arrays are inconsistent within separatrix {identifier}")
        destination_dimension = 0 if separatrix_type == 0 else 2
        source = critical_lookup.get((1, raw_source))
        destination = critical_lookup.get((destination_dimension, raw_destination))
        if source is None or destination is None:
            raise RuntimeError(
                "TTK SourceId/DestinationId did not match the critical-point "
                f"(CellDimension, CellId) keys for separatrix {identifier}"
            )
        result.append(
            {
                "id": identifier,
                "source_id": source,
                "destination_id": destination,
                "source_cell_id": raw_source,
                "destination_cell_id": raw_destination,
                "separatrix_type": separatrix_type,
                "kind": "u" if separatrix_type == 0 else "s",
                "points": _ordered_polyline(data, cells),
            }
        )
    return result


def _quadrangulation(data) -> dict[str, object]:
    cells: list[list[int]] = []
    for index in range(data.GetNumberOfCells()):
        cell = data.GetCell(index)
        cells.append([int(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())])
    point_data = data.GetPointData()
    arrays = {
        name: [point_data.GetArray(name).GetTuple1(i) for i in range(data.GetNumberOfPoints())]
        for name in _array_names(point_data)
        if point_data.GetArray(name) is not None and point_data.GetArray(name).GetNumberOfComponents() == 1
    }
    return {
        "points": [[float(value) for value in data.GetPoint(i)] for i in range(data.GetNumberOfPoints())],
        "cells": cells,
        "point_arrays": arrays,
    }


def main() -> None:
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader

    msc_class = _load_ttk_class("ttkMorseSmaleComplex", "ttkMorseSmaleComplex")
    quadrangulation_class = _load_ttk_class(
        "ttkMorseSmaleQuadrangulation", "ttkMorseSmaleQuadrangulation"
    )

    input_path, output_path = sys.argv[1:3]
    reader = vtkPolyDataReader()
    reader.SetFileName(input_path)
    reader.Update()

    complex_ = msc_class()
    complex_.SetInputConnection(reader.GetOutputPort())
    complex_.SetInputArrayToProcess(0, 0, 0, 0, "ScalarField")
    complex_.SetComputeCriticalPoints(True)
    complex_.SetComputeAscendingSeparatrices1(True)
    complex_.SetComputeDescendingSeparatrices1(True)
    complex_.SetComputeSaddleConnectors(False)
    complex_.Update()

    critical = complex_.GetOutputDataObject(0)
    separatrix_data = complex_.GetOutputDataObject(1)
    critical_points, lookup = _critical_points(critical)
    separatrices = _separatrices(separatrix_data, lookup)

    quadrangulation = quadrangulation_class()
    quadrangulation.SetInputConnection(0, complex_.GetOutputPort(0))
    quadrangulation.SetInputConnection(1, complex_.GetOutputPort(1))
    quadrangulation.SetInputConnection(2, reader.GetOutputPort())
    quadrangulation.Update()
    quadrangles = quadrangulation.GetOutputDataObject(0)

    payload = {
        "critical_points": critical_points,
        "separatrices": separatrices,
        "quadrangulation": _quadrangulation(quadrangles),
        "arrays": {
            "critical_points": _schema(critical),
            "separatrices": _schema(separatrix_data),
            "quadrangulation": _schema(quadrangles),
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
