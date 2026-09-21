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
from .pointcloud import DepthRaster, PointCloud, load_point_cloud, rasterize_depth, voxel_downsample
from .surface import reconstruct_delaunay, reconstruct_poisson, reconstruct_surface
from .intrinsic import gaussian_curvature, intrinsic_scalar, mean_curvature, shape_index
from .persistence import mesh_persistence, simplicial_persistence
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
	"reconstruct_delaunay",
	"reconstruct_poisson",
	"reconstruct_surface",
	"gaussian_curvature",
	"intrinsic_scalar",
	"mean_curvature",
	"shape_index",
	"mesh_persistence",
	"simplicial_persistence",
	"TTKResult",
	"TTKSeparatrix",
	"run_ttk_msc",
]
