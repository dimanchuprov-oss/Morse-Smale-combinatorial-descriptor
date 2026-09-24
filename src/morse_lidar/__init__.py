"""Research tools for Morse-Smale descriptors of LiDAR surfaces."""

from .critical import CriticalPoint, CriticalType, analyze_morse
from .descriptor import (
	ColoredEdge,
	ColoredGraph,
	Descriptor,
	MorseSmaleQuadrilateral,
	build_descriptor,
	build_ttk_descriptor,
)
from .mesh import TriMesh, grid_mesh, periodic_grid_mesh
from .pointcloud import (
    DepthRaster,
    PointCloud,
    align_pca,
    load_point_cloud,
    rasterize_depth,
    remove_dominant_plane,
    set_up_axis,
    voxel_downsample,
)
from .surface import reconstruct_delaunay, reconstruct_poisson, reconstruct_surface
from .intrinsic import gaussian_curvature, intrinsic_scalar, mean_curvature, shape_index
from .persistence import (
    CriticalPair,
    annotate_persistence,
    critical_pairs,
    mesh_persistence,
    persistence_diagram,
    persistent_counts,
    simplicial_persistence,
)
from .compare import bottleneck, compare_reports
from .ttk_backend import TTKResult, TTKSeparatrix, run_ttk_msc

__all__ = [
	"CriticalPoint",
	"CriticalType",
	"Descriptor",
	"ColoredEdge",
	"ColoredGraph",
	"MorseSmaleQuadrilateral",
	"TriMesh",
	"analyze_morse",
	"build_descriptor",
	"build_ttk_descriptor",
	"periodic_grid_mesh",
	"grid_mesh",
	"DepthRaster",
	"PointCloud",
	"load_point_cloud",
	"rasterize_depth",
	"voxel_downsample",
	"align_pca",
	"remove_dominant_plane",
	"set_up_axis",
	"reconstruct_delaunay",
	"reconstruct_poisson",
	"reconstruct_surface",
	"gaussian_curvature",
	"intrinsic_scalar",
	"mean_curvature",
	"shape_index",
	"mesh_persistence",
	"simplicial_persistence",
	"CriticalPair",
	"annotate_persistence",
	"critical_pairs",
	"persistence_diagram",
	"persistent_counts",
	"bottleneck",
	"compare_reports",
	"TTKResult",
	"TTKSeparatrix",
	"run_ttk_msc",
]
