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

Persistence is computed with GUDHI on the same triangulation and with the
same tie-breaking order (simulation of simplicity) as the critical point
classifier, so every critical point carries its own persistence value.
`--persistence-threshold` then reports how many minima, saddles and maxima
survive (`persistent_counts`). This drops low-persistence pairs but does not
reroute separatrices, i.e. it is not yet a full Morse-Smale cancellation;
production simplification should run TTK `TopologicalSimplification` or an
equivalent workflow. On closed surfaces the persistent counts keep the Morse
Euler relation; saddles are always weighted by multiplicity.

## Quick start

```bash
python -m pip install -e '.[dev,topology,signal]'
pytest
python -m morse_lidar.cli --input depth.npy --output descriptor.json --surface open --pixel-size 0.001
python examples/synthetic_demo.py   # end-to-end demo on synthetic iPhone-like scans
```

`--pixel-size` is the grid spacing in depth units; it only matters for the
curvature fields, which are metric.

`depth.npy` must contain a 2D finite NumPy array. The CLI reports topology
violations instead of silently producing a descriptor from an invalid mesh.

Real point clouds are also accepted. ASCII PLY/PCD and XYZ/CSV work with the
base installation; binary PLY/PCD and LAS/LAZ require the optional LiDAR extra:

```bash
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
    --rows 256 --cols 256 --voxel-size 0.002 --persistence-threshold 0.01 \
    --raster-output depth.npy
```

Binary little/big-endian PLY files whose first element is `vertex` (what
scanning apps usually write) are read with NumPy alone; other binary layouts
fall back to Open3D.

The projection is an axis-aligned median depth raster with metric spacing
(`raster_spacing`). `raster_coverage`, `point_count`, `surface`, and
`boundary_vertices` are written to the JSON report. Real scans are open
surfaces; `--surface periodic` (alias `--periodic`) is meant for synthetic
torus data and prints a warning on point clouds, because gluing the raster
edges creates artificial critical points along the seam. Open-surface
descriptors do not report the closed-manifold Euler invariant; separatrices
that run into the masked rim are kept and flagged `ends_on_boundary`.

### Preparing real scans (e.g. iPhone LiDAR)

```bash
morse-lidar --input scan.ply --output scan.json \
    --up-axis y --remove-plane 0.01 --voxel-size 0.01 \
    --geometry delaunay --persistence-threshold 0.01
```

* `--up-axis y` rotates Y-up exports (ARKit/iPhone) so that Z is height.
* `--remove-plane D` drops the dominant plane (floor/table) with RANSAC.
* `--align pca` centres the cloud and rotates its principal axes onto XYZ,
  so the height field does not depend on how the scanner was held.

### Comparing descriptors

Every report contains the persistence diagram of the scalar field
(`persistence_diagram`, GUDHI required); with the heuristic backend every
critical point also carries its own `persistence`. Two reports are compared with the bottleneck distance, which
is stable: it never exceeds the sup-norm difference of the two scalar fields.

```bash
morse-lidar-compare first.json second.json
```

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
`mean-curvature`, and `shape-index`. Curvature fields are pointwise (per unit
area) and therefore independent of mesh density: Gaussian curvature is the
angle deficit divided by the barycentric area, mean curvature is the signed
cotangent-Laplacian estimate (positive on a dome whose normals point towards
+Z), and the shape index is +1 on a dome, 0 on a saddle and -1 on a bowl.
Values on the rim of an open surface are extended in layers from interior
neighbours, including corners without direct interior neighbours. Curvature is noise-sensitive: on raw scans smooth first
(`--sigma` with raster geometry, larger `--voxel-size`) or most critical points will be noise.

Persistence filtering also works on reconstructed meshes:

```bash
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
    --geometry delaunay --scalar gaussian-curvature \
    --persistence-threshold 0.001
```

All geometries use a GUDHI lower-star filtration on the triangulated mesh
with the classifier's tie-breaking, so every pair is created by critical
vertices (dimension 0: minimum-saddle, dimension 1: saddle-maximum,
dimension 2: the global maximum of a closed surface); a k-fold saddle belongs
to k pairs. On open surfaces pairs may also involve rim vertices, which are
masked from the critical point list.

`persistent_minima`/`persistent_maxima` changed meaning in 0.2.0: they now
count classified *interior* extrema that belong to a pair of at least the
threshold (previously: dimension-0 cubical intervals, including extrema on the
rim of the scan). `persistent_saddles` and `persistent_counts` are new.

This is a 2.5D surface: triangles are built in the selected XY projection and
retain the measured Z coordinates. For a full 3D reconstruction, install the
LiDAR extra and use Open3D Poisson reconstruction:

```bash
python -m pip install -e '.[lidar]'
python -m morse_lidar.cli --input scan.ply --output descriptor.json \
    --geometry poisson --poisson-depth 8
```

Poisson trims its lowest-density vertices by default
(`--poisson-density-quantile 0.02`), which opens the mesh; pass `0` to keep
the watertight surface required by the TTK backend.

On macOS, Open3D may additionally require the native `libusb` library. Poisson
reconstruction requires at least 30 points and may produce a closed mesh. Its
Euler characteristic is computed from the reconstructed mesh rather than
assumed to be the torus value.

For persistence, install the optional GUDHI backend and pass a noise
threshold in scalar units:

```bash
python -m pip install -e '.[topology]'
python -m morse_lidar.cli --input depth.npy --output descriptor.json \
    --surface periodic --persistence-threshold 0.02
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

## Reproducible repeat-scan experiment

The versioned JSON contract is documented in [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).
Reports include effective processing parameters; comparison rejects incompatible
settings. Heuristic and TTK arcs both reference critical-point IDs; masked rim
endpoints use a null `destination_id` and a separate `destination_vertex_id`.

`--sigma` (in pixels) and `--median-size` apply only to raster geometry.
Nontrivial smoothing options with Delaunay/Poisson are rejected. Periodic rasters
use wrapped filters. Boundary curvature is extended from interior vertices in
layers, including corners without direct interior neighbours; a boundary
component without interior samples is rejected.

Copy and adapt [examples/repeat_scans.json](examples/repeat_scans.json), then run:

```bash
python -m morse_lidar.experiment examples/repeat_scans.json runs/repeats-001
```

Paths are relative to the manifest. The example assumes segmented point clouds
in metres, with Y up; its parameters are a starting protocol, not calibrated
sensor settings. Use at least two independent scans of one object, preferably
3–5 repeats of each of several objects plus a changed-object control. Assign the
changed object its own label. Keep units, cropping and acquisition procedure
consistent. PCA does not guarantee pose alignment for symmetric objects.

Choose settings before evaluating the repeats and use the same settings for all
inputs. The runner creates a new output directory with the manifest, individual
reports and logs, input hashes, dependency versions and all pairwise distances
in `results.json`. Compare within-object distances with between-object distances
and inspect raster coverage; a small synthetic distance alone is not evidence of
real-scan repeatability. `null` distances mean no finite match and must not be
interpreted as zero. Fixed rows/columns and sigma do not imply a fixed physical
smoothing radius when bounding boxes differ; inspect `raster_spacing` as well.

No real repeat-scan dataset is bundled. The automated experiment test uses
explicitly synthetic fixtures and does not constitute a real-scan experiment.
