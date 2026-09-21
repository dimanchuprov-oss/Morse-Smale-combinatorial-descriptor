# Morse-LiDAR

Research scaffold for extracting a Morse-Smale combinatorial descriptor from a
LiDAR depth surface.

## Design

The pipeline supports both open rectangular surfaces and closed periodic test
surfaces. Open surfaces use an ordinary triangulation and mask boundary
vertices during Morse classification; periodic surfaces are useful for torus
fixtures and wrapped scans. Critical points are classified from connected
components of the lower and upper links, not from the number of higher/lower
neighbors.

The package deliberately keeps persistence simplification separate from mesh
construction. The built-in simplifier is a small deterministic baseline for
experimentation; production data should use TTK `TopologicalSimplification` or
GUDHI and import its cancelled critical pairs.

## Quick start

```bash
python -m pip install -e '.[dev,topology]'
pytest
python -m morse_lidar.cli --input depth.npy --output descriptor.json --surface open
```

`depth.npy` must contain a 2D finite NumPy array. The CLI reports topology
violations instead of silently producing a descriptor from an invalid mesh.

Real point clouds are also accepted. ASCII PLY/PCD and XYZ/CSV work with the
base installation; binary PLY/PCD and LAS/LAZ require the optional LiDAR extra:

```bash
python -m pip install -e '.[lidar]'
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
	--periodic --rows 256 --cols 256 --voxel-size 0.002 \
	--raster-output depth.npy
```

The projection is an axis-aligned median depth raster. `raster_coverage`,
`point_count`, `surface`, and `boundary_vertices` are written to the JSON
report. Use `--surface open` for facial scans; `--periodic` is retained as a
backward-compatible alias for synthetic torus data. Open-surface descriptors
do not report the closed-manifold Euler invariant.

To build a triangular surface directly from XYZ/PLY/PCD points, use Delaunay
geometry instead of rasterization:

```bash
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
	--geometry delaunay --voxel-size 0.002
```

For a pose-robust scalar field, replace camera height with intrinsic geometry:

```bash
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
	--geometry delaunay --scalar gaussian-curvature
```

Available scalar fields are `height`, `gaussian-curvature`,
`mean-curvature`, and `shape-index`. Curvature fields are computed on the
triangular mesh before Morse classification, including boundary-aware angle
deficits for open surfaces.

Persistence filtering also works on reconstructed meshes:

```bash
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
	--geometry delaunay --scalar gaussian-curvature \
	--persistence-threshold 0.001
```

Raster geometry uses GUDHI's cubical complex; Delaunay and Poisson geometry use
a GUDHI simplicial complex with lower-star vertex filtration. Maxima are
computed by running the corresponding scalar field with inverted values.

This is a 2.5D surface: triangles are built in the selected XY projection and
retain the measured Z coordinates. For a full 3D reconstruction, install the
LiDAR extra and use Open3D Poisson reconstruction:

```bash
python -m pip install -e '.[lidar]'
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
	--geometry poisson --poisson-depth 8
```

On macOS, Open3D may additionally require the native `libusb` library. Poisson
reconstruction requires at least 30 points and may produce a closed mesh. Its
Euler characteristic is computed from the reconstructed mesh rather than
assumed to be the torus value.

For cubical persistence, install the optional GUDHI backend and pass a noise
threshold:

```bash
python -m pip install -e '.[topology]'
python -m morse_lidar.cli --input depth.npy --output descriptor.json \
	--periodic --persistence-threshold 0.02
```

TTK is distributed most reliably with ParaView rather than PyPI. The
`morse_lidar.ttk_backend` adapter launches ParaView's Python interpreter as a
subprocess. It is intentionally separate from GUDHI:
GUDHI supplies persistence intervals, while TTK supplies the geometric
Morse-Smale complex and separatrices.

When ParaView with the TTK Python wrappers is installed, select the real
backend explicitly. Override the executable location with
`MORSE_LIDAR_PVPYTHON=/path/to/pvpython` when it is not on `PATH`:

```bash
MORSE_LIDAR_PVPYTHON=/path/to/pvpython python -m morse_lidar.cli \
    --input depth.npy --output descriptor.json --surface periodic \
    --scalar gaussian-curvature --msc-backend ttk
```

This checkout also includes a project-local launcher at `tools/pvpython-ttk`
when the local `.tools` runtime has been provisioned.

The CLI refuses to silently fall back to the heuristic backend when `ttk` is
requested. The TTK path supplies the descriptor's critical points, geometric
separatrix polylines, Morse-Smale quadrangulation, and colored dual graph; it
does not run the heuristic classifier or tracer.

For TTK 1.4.0 the observed critical-point arrays are `CellDimension`,
`CellId`, `ScalarField`, `IsOnBoundary`, `ttkVertexScalarField`, and
`ManifoldSize`. On a two-manifold, `CellDimension` is the critical type
(`0=min`, `1=saddle`, `2=max`); the older `CriticalType` name is not emitted.
The 1-separatrix cell arrays include the exact names `SourceId`,
`DestinationId`, `SeparatrixId`, and `SeparatrixType`. They are recorded under
`ttk_arrays` in every JSON report so a runtime upgrade cannot silently change
the schema.

The mandatory colored graph currently requires a closed mesh. The original
`min-saddle-max-saddle` cells are reconstructed from TTK's `QuadVertType` and
`QuadCellId` data, then cut along a new min-max `t` arc. Dual edges retain the
exact `SeparatrixId` or `t`-arc from which they were created.
Validation enforces the Morse-Euler equation, degree three, exactly one
incident `s`, `u`, and `t` edge per node, and equality between graph edges and
shared triangulation arcs. Open facial surfaces fail early until a boundary
closing policy is selected.

## Important scope limits

This is a topology-first baseline, not a face recognition system. A camera-Z
height field is pose-dependent. For biometric use, canonicalize pose or replace
height with an intrinsic scalar such as curvature or geodesic distance, and add
temporal persistence tracking before identity modeling.
