"""Command-line entry point for a depth-map topology report."""

from __future__ import annotations

import argparse
import json
import math

from .descriptor import build_descriptor, build_ttk_descriptor
from .intrinsic import intrinsic_scalar
from .mesh import TriMesh, grid_mesh, periodic_grid_mesh
from .persistence import depth_persistence, mesh_persistence
from .pointcloud import load_point_cloud, rasterize_depth, voxel_downsample
from .preprocess import denoise_depth, load_depth
from .surface import reconstruct_surface
from .ttk_backend import run_ttk_msc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="depth .npy or LiDAR point cloud (.ply/.pcd/.las/.laz/.xyz/.csv)")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--surface", choices=("open", "periodic"), default="open")
    parser.add_argument("--geometry", choices=("raster", "delaunay", "poisson"), default="raster")
    parser.add_argument("--msc-backend", choices=("heuristic", "ttk"), default="heuristic")
    parser.add_argument("--scalar", choices=("height", "gaussian-curvature", "mean-curvature", "shape-index"), default="height")
    parser.add_argument("--periodic", action="store_true", help="deprecated alias for --surface periodic")
    parser.add_argument("--median-size", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--persistence-threshold", type=float, default=None)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--raster-output", default=None, help="optional .npy path for the rasterized depth")
    parser.add_argument("--poisson-depth", type=int, default=8)
    args = parser.parse_args()
    if not math.isfinite(args.sigma) or args.sigma < 0:
        parser.error("--sigma must be a finite non-negative number")
    if args.persistence_threshold is not None and (
        not math.isfinite(args.persistence_threshold) or args.persistence_threshold < 0
    ):
        parser.error("--persistence-threshold must be a finite non-negative number")
    if args.rows < 3 or args.cols < 3:
        parser.error("--rows and --cols must be at least 3")
    if args.geometry == "poisson" and args.poisson_depth < 1:
        parser.error("--poisson-depth must be a positive integer")
    if args.surface == "periodic" and args.geometry != "raster":
        parser.error("--surface periodic is only supported with raster geometry")
    if args.periodic:
        args.surface = "periodic"
    if args.geometry != "raster" and args.input.lower().endswith(".npy"):
        raise SystemExit("--geometry mesh requires a point-cloud input, not a depth .npy file")
    depth = None
    mesh = None
    if args.input.lower().endswith(".npy"):
        depth = load_depth(args.input)
        point_count = None
        coverage = 1.0
    else:
        cloud = load_point_cloud(args.input)
        if args.voxel_size is not None:
            cloud = voxel_downsample(cloud, args.voxel_size)
        point_count = len(cloud.points)
        coverage = None
        if args.geometry == "raster":
            raster = rasterize_depth(cloud, rows=args.rows, cols=args.cols)
            depth = raster.depth
            coverage = raster.coverage
            if args.raster_output:
                import numpy as np

                np.save(args.raster_output, depth)
        else:
            mesh = reconstruct_surface(
                cloud,
                method=args.geometry,
                depth=args.poisson_depth,
            ) if args.geometry == "poisson" else reconstruct_surface(cloud, method=args.geometry)
    if depth is not None:
        depth = denoise_depth(depth, args.median_size, args.sigma)
        mesh = periodic_grid_mesh(depth) if args.surface == "periodic" else grid_mesh(depth)
    if mesh is None:  # pragma: no cover - all current input branches construct one
        raise RuntimeError("failed to construct a surface mesh")
    if args.scalar != "height":
        mesh = TriMesh(mesh.vertices, mesh.faces, intrinsic_scalar(mesh, args.scalar))
    expected_euler = mesh.euler_characteristic if mesh.closed else None
    ttk_result = None
    if args.msc_backend == "ttk":
        if not mesh.closed:
            raise SystemExit(
                "--msc-backend ttk requires a closed mesh for the mandatory trivalent graph; "
                "choose --surface periodic or close the facial boundary explicitly"
            )
        ttk_result = run_ttk_msc(mesh)
        descriptor = build_ttk_descriptor(ttk_result, mesh.euler_characteristic)
    else:
        descriptor = build_descriptor(mesh, euler_characteristic=expected_euler)
    report = {
        "counts": descriptor.counts,
        "euler_characteristic": descriptor.euler_characteristic,
        "separatrix_count": len(descriptor.separatrices),
        "geometry": args.geometry,
        "surface": args.surface,
        "boundary_vertices": len(mesh.boundary_vertices),
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "msc_backend": args.msc_backend,
    }
    if ttk_result is not None:
        report["critical_point_count"] = len(ttk_result.critical_points)
        report["ttk_arrays"] = ttk_result.arrays
        report["critical_points"] = [
            {
                "id": point.identifier,
                "vertex_id": point.vertex,
                "kind": point.kind.value,
                "position": point.position,
                "cell_id": point.cell_id,
                "cell_dimension": point.cell_dimension,
                "scalar": point.scalar,
                "is_on_boundary": point.is_on_boundary,
            }
            for point in descriptor.critical_points
        ]
        report["separatrices"] = [
            {
                "id": arc.identifier,
                "source_id": arc.source,
                "destination_id": arc.target,
                "kind": arc.kind,
                "source_cell_id": arc.source_cell_id,
                "destination_cell_id": arc.destination_cell_id,
                "points": arc.points,
            }
            for arc in descriptor.separatrices
        ]
        report["quadrilaterals"] = [
            {
                "id": cell.identifier,
                "critical_points": cell.critical_points,
                "separatrices": cell.separatrices,
            }
            for cell in descriptor.quadrilaterals
        ]
        if descriptor.colored_graph is None:  # pragma: no cover - TTK contract violation
            raise RuntimeError("TTK descriptor did not contain a colored graph")
        report["colored_graph"] = {
            "node_count": descriptor.colored_graph.node_count,
            "triangles": descriptor.colored_graph.triangles,
            "edges": [
                {
                    "source": edge.left,
                    "target": edge.right,
                    "color": edge.color,
                    "shared_primal_arc": edge.primal_edge,
                }
                for edge in descriptor.colored_graph.edges
            ],
        }
    if coverage is not None:
        report["raster_coverage"] = coverage
    if point_count is not None:
        report["point_count"] = point_count
    if args.persistence_threshold is not None:
        persistence = depth_persistence(depth) if depth is not None and args.scalar == "height" else mesh_persistence(mesh)
        report["persistent_minima"] = len(
            persistence["minima"].significant(args.persistence_threshold, dimension=0)
        )
        report["persistent_maxima"] = len(
            persistence["maxima"].significant(args.persistence_threshold, dimension=0)
        )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
