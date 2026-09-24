"""Command-line entry point for a depth-map topology report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import numpy as np

from .critical import CriticalType
from .critical import analyze_morse
from .descriptor import Descriptor, build_descriptor, build_ttk_descriptor
from .intrinsic import intrinsic_scalar
from .mesh import TriMesh, grid_mesh, periodic_grid_mesh
from .persistence import annotate_persistence, critical_pairs, persistence_diagram, persistent_counts
from .pointcloud import (
    align_pca,
    load_point_cloud,
    rasterize_depth,
    remove_dominant_plane,
    set_up_axis,
    voxel_downsample,
)
from .preprocess import denoise_depth, load_depth
from .surface import reconstruct_surface
from .ttk_backend import run_ttk_msc

SCALARS = ("height", "gaussian-curvature", "mean-curvature", "shape-index")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="depth .npy or LiDAR point cloud (.ply/.pcd/.las/.laz/.xyz/.csv)")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--surface", choices=("open", "periodic"), default="open")
    parser.add_argument("--geometry", choices=("raster", "delaunay", "poisson"), default="raster")
    parser.add_argument("--msc-backend", choices=("heuristic", "ttk"), default="heuristic")
    parser.add_argument("--scalar", choices=SCALARS, default="height")
    parser.add_argument("--periodic", action="store_true", help="deprecated alias for --surface periodic")
    parser.add_argument("--median-size", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument(
        "--persistence-threshold",
        type=float,
        default=None,
        help="report critical point counts that survive this persistence (scalar units)",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--pixel-size", type=float, default=1.0, help="grid spacing of a depth .npy, in depth units")
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument(
        "--up-axis",
        choices=("x", "y", "z"),
        default="z",
        help="vertical axis of the input cloud (iPhone/ARKit exports are usually y)",
    )
    parser.add_argument("--align", choices=("none", "pca"), default="none", help="canonicalize the pose before meshing")
    parser.add_argument(
        "--remove-plane",
        type=float,
        default=None,
        metavar="DISTANCE",
        help="drop the dominant plane (floor/table) with RANSAC, inlier distance in point units",
    )
    parser.add_argument("--raster-output", default=None, help="optional .npy path for the rasterized depth")
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument(
        "--poisson-density-quantile",
        type=float,
        default=0.02,
        help="trim this fraction of lowest-density Poisson vertices; 0 keeps the mesh closed",
    )
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.periodic:
        args.surface = "periodic"
    for name in ("sigma", "pixel_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "pixel_size" and value == 0):
            parser.error(f"--{name.replace('_', '-')} must be a finite {'positive' if name == 'pixel_size' else 'non-negative'} number")
    for name in ("persistence_threshold", "voxel_size", "remove_plane"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0 or (name != "persistence_threshold" and value == 0)):
            parser.error(f"--{name.replace('_', '-')} must be a finite {'non-negative' if name == 'persistence_threshold' else 'positive'} number")
    if args.rows < 3 or args.cols < 3:
        parser.error("--rows and --cols must be at least 3")
    if args.geometry == "poisson" and args.poisson_depth < 1:
        parser.error("--poisson-depth must be a positive integer")
    if not 0 <= args.poisson_density_quantile < 1:
        parser.error("--poisson-density-quantile must be in [0, 1)")
    if args.surface == "periodic" and args.geometry != "raster":
        parser.error("--surface periodic is only supported with raster geometry")
    is_depth = args.input.lower().endswith(".npy")
    if not is_depth and args.pixel_size != 1.0:
        parser.error("--pixel-size applies to depth .npy input; point clouds carry their own metric spacing")
    if args.surface == "periodic" and args.scalar != "height":
        parser.error(
            "curvature scalars are not supported with --surface periodic: the seam triangles wrap "
            "coordinates and give wrong curvature there"
        )
    if is_depth and args.geometry != "raster":
        parser.error("--geometry delaunay/poisson requires a point-cloud input, not a depth .npy file")
    if is_depth and (args.up_axis != "z" or args.align != "none" or args.remove_plane is not None):
        parser.error("--up-axis, --align and --remove-plane apply to point clouds, not depth .npy input")
    if args.surface == "periodic" and not is_depth:
        print(
            "warning: --surface periodic glues opposite raster edges of a real scan into a torus; "
            "the seam creates artificial critical points",
            file=sys.stderr,
        )


def _build_mesh(args: argparse.Namespace, report: dict[str, Any]) -> tuple[TriMesh, np.ndarray | None]:
    if args.input.lower().endswith(".npy"):
        depth = load_depth(args.input)
        spacing = (args.pixel_size, args.pixel_size)
        report["raster_coverage"] = 1.0
    else:
        cloud = set_up_axis(load_point_cloud(args.input), args.up_axis)
        if args.remove_plane is not None:
            cloud, removed = remove_dominant_plane(cloud, args.remove_plane)
            report["plane_points_removed"] = removed
        if args.align == "pca":
            cloud = align_pca(cloud)
        if args.voxel_size is not None:
            cloud = voxel_downsample(cloud, args.voxel_size)
        report["point_count"] = len(cloud.points)
        if args.geometry != "raster":
            options = {"depth": args.poisson_depth, "density_quantile": args.poisson_density_quantile}
            return reconstruct_surface(cloud, method=args.geometry, **(options if args.geometry == "poisson" else {})), None
        raster = rasterize_depth(cloud, rows=args.rows, cols=args.cols)
        depth, spacing = raster.depth, raster.spacing
        report["raster_coverage"] = raster.coverage
        report["raster_spacing"] = list(spacing)
        if args.raster_output:
            np.save(args.raster_output, depth)
    depth = denoise_depth(depth, args.median_size, args.sigma)
    mesh = periodic_grid_mesh(depth, spacing) if args.surface == "periodic" else grid_mesh(depth, spacing)
    return mesh, depth


def _critical_point_json(point) -> dict[str, Any]:
    return {
        "id": point.identifier,
        "vertex_id": point.vertex,
        "kind": point.kind.value,
        "position": point.position,
        "cell_id": point.cell_id,
        "cell_dimension": point.cell_dimension,
        "scalar": point.scalar,
        "is_on_boundary": point.is_on_boundary,
        "saddle_multiplicity": point.saddle_multiplicity,
        "persistence": None if point.persistence is None or math.isinf(point.persistence) else point.persistence,
        "essential": point.persistence is not None and math.isinf(point.persistence),
    }


def _descriptor_json(descriptor: Descriptor, mesh: TriMesh, report: dict[str, Any]) -> None:
    points = []
    for point in descriptor.critical_points:
        if point.kind == CriticalType.REGULAR:
            continue
        item = _critical_point_json(point)
        if point.backend != "ttk":
            item["position"] = [float(value) for value in mesh.vertices[point.vertex]]
            item["scalar"] = float(mesh.values[point.vertex])
        points.append(item)
    report["critical_points"] = points
    report["separatrices"] = [
        {
            "id": arc.identifier,
            "source_id": arc.source,
            "destination_id": arc.target,
            "kind": arc.kind,
            "source_cell_id": arc.source_cell_id,
            "destination_cell_id": arc.destination_cell_id,
            "ends_on_boundary": arc.ends_on_boundary,
            "vertices": list(arc.vertices),
            "points": [list(point) for point in arc.points]
            or [[float(value) for value in mesh.vertices[vertex]] for vertex in arc.vertices],
        }
        for arc in descriptor.separatrices
    ]


def _persistence_json(
    descriptor: Descriptor, mesh: TriMesh, threshold: float | None, report: dict[str, Any]
) -> Descriptor:
    try:
        pairs = critical_pairs(mesh)
    except RuntimeError as error:  # GUDHI is an optional extra
        report["persistence_diagram"] = None
        report["persistence_note"] = str(error)
        if threshold is not None:
            raise SystemExit(f"--persistence-threshold needs GUDHI: {error}") from error
        return descriptor
    report["persistence_diagram"] = {
        str(dimension): [[birth, None if math.isinf(death) else death] for birth, death in points]
        for dimension, points in persistence_diagram(pairs).items()
    }
    heuristic = descriptor.backend == "heuristic"
    # TTK vertex ids are not guaranteed to be mesh vertex ids, so TTK points are
    # not annotated; persistent counts use the PL classification of the mesh.
    points = annotate_persistence(descriptor.critical_points if heuristic else tuple(analyze_morse(mesh)), pairs)
    annotated = (
        Descriptor(
            points,
            descriptor.separatrices,
            descriptor.euler_characteristic,
            descriptor.quadrilaterals,
            descriptor.colored_graph,
            descriptor.backend,
        )
        if heuristic
        else descriptor
    )
    if threshold is not None:
        counts = persistent_counts(points, pairs, threshold)
        report["persistence_threshold"] = threshold
        report["persistent_counts"] = counts
        report["persistent_minima"] = counts["min"]
        report["persistent_saddles"] = counts["saddle"]
        report["persistent_maxima"] = counts["max"]
    return annotated


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    report: dict[str, Any] = {}
    mesh, _ = _build_mesh(args, report)
    if args.scalar != "height":
        mesh = TriMesh(mesh.vertices, mesh.faces, intrinsic_scalar(mesh, args.scalar))
    expected_euler = mesh.euler_characteristic if mesh.closed else None
    ttk_result = None
    if args.msc_backend == "ttk":
        if not mesh.closed:
            raise SystemExit(
                "--msc-backend ttk requires a closed mesh for the mandatory trivalent graph; "
                "choose --surface periodic, --geometry poisson --poisson-density-quantile 0, "
                "or use the heuristic backend for open scans"
            )
        ttk_result = run_ttk_msc(mesh)
        descriptor = build_ttk_descriptor(ttk_result, mesh.euler_characteristic)
    else:
        descriptor = build_descriptor(mesh, euler_characteristic=expected_euler)
    descriptor = _persistence_json(descriptor, mesh, args.persistence_threshold, report)
    report.update(
        {
            "counts": descriptor.counts,
            "saddle_index_sum": descriptor.saddle_index_sum,
            "euler_characteristic": descriptor.euler_characteristic,
            "separatrix_count": len(descriptor.separatrices),
            "boundary_separatrix_count": sum(arc.ends_on_boundary for arc in descriptor.separatrices),
            "geometry": args.geometry,
            "surface": args.surface,
            "scalar": args.scalar,
            "boundary_vertices": len(mesh.boundary_vertices),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "msc_backend": args.msc_backend,
        }
    )
    _descriptor_json(descriptor, mesh, report)
    if ttk_result is not None:
        report["critical_point_count"] = len(ttk_result.critical_points)
        report["ttk_arrays"] = ttk_result.arrays
        report["quadrilaterals"] = [
            {"id": cell.identifier, "critical_points": cell.critical_points, "separatrices": cell.separatrices}
            for cell in descriptor.quadrilaterals
        ]
        if descriptor.colored_graph is None:  # pragma: no cover - TTK contract violation
            raise RuntimeError("TTK descriptor did not contain a colored graph")
        report["colored_graph"] = {
            "node_count": descriptor.colored_graph.node_count,
            "triangles": descriptor.colored_graph.triangles,
            "edges": [
                {"source": edge.left, "target": edge.right, "color": edge.color, "shared_primal_arc": edge.primal_edge}
                for edge in descriptor.colored_graph.edges
            ],
        }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    summary = {key: value for key, value in report.items() if key not in {"critical_points", "separatrices", "persistence_diagram", "colored_graph", "quadrilaterals", "ttk_arrays"}}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
