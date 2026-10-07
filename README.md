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

### Live processing while scanning (e.g. Revopoint Range)

Scanner drivers are proprietary, so capture stays in the vendor application
(Revo Scan for Revopoint devices) and this package watches its export folder.
OBJ and STL exports are read in addition to PLY (vertices only; faces are
rebuilt by the selected `--geometry`).

```bash
morse-lidar-watch ~/Scans/export -- \
    --geometry delaunay --voxel-size 1 --persistence-threshold 0.5
```

Everything after `--` is passed to `morse-lidar` unchanged, so every scan of a
session is processed with identical parameters. Each new file is processed once
it has stopped changing (`--settle`, seconds), compared with the reference scan
(`--reference`, default: the first scan) and with the previous one, and logged
to `EXPORT/morse-lidar/scans.csv` together with its SHA-256. A draft manifest
for the repeat-scan experiment below is kept up to date in
`manifest.draft.json`; object labels are guessed from file names up to the last
`_` (`vase_01.ply` → `vase`) and must be checked by hand. `--units` (default
`mm`, as exported by Revo Scan) is recorded in the manifest; thresholds and
voxel sizes are in the same units as the exported coordinates.

Descriptors are written as `<file name>.json` (so `scan.ply` and `scan.obj`
do not collide). The session — processed files, reference and previous scan —
is kept in `session.json`, so the watcher can be stopped with Ctrl+C and
restarted on the same folder; a reference computed with different options is
rejected. The `morse-lidar-watch` and `morse-lidar-experiment` commands are
installed by `pip install -e .`.

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

## Face verification across viewing angles

`morse-lidar-face` decides whether two head scans show the same person when the
probe is taken from a different angle. The command:

1. cuts the head out of each scan;
2. registers the probe head rigidly to each enrolled head. Point-to-point ICP
   is started from a grid of yaws, small tilts and back-shifts of the probe
   axis, and the best starts are refined by point-to-plane ICP, so the viewing
   angle does not need to be known;
3. scores each pair.

There are three scores:

- `geometric_mm`: trimmed RMS distance after alignment; this is the decision score;
- `topological_mm`: bottleneck distance between persistence diagrams of the radial
  height field `r(theta, z)` of the two heads, computed over the region both scans see;
- `overlap`: fraction of probe points within 3 mm of the enrolled head.

```bash
pip install -e '.[signal,topology,plot]'
# angle experiment on full head scans + an extra real probe, with plots
morse-lidar-face experiment --person A=biometrics/raw_preview.ply \
    --person B=biometrics/raw_preview_05f4e84d-1b50-4aa0-b895-4d55b7ea554e.ply \
    --probe biometrics/raw_preview_95b40cf3-65cc-4b20-999d-b82a85677891.ply \
    --up-axis y --front-axis +z --out biometrics/face_results
# identify a new scan against enrolled people, with thresholds from the experiment
morse-lidar-face compare probe.ply --gallery A=a.ply --gallery B=b.ply --up-axis y \
    --calibration biometrics/face_results/face_summary.json
```

`--up-axis` is required (Revo Scan and iPhone exports are Y-up). A section
50–120 mm below the top that is wider than 350 mm (a body lying along the wrong
axis), or a head region under 120 mm tall, is rejected as a wrong axis or wrong
units. A tightly cropped scan read along a wrong axis can still pass, which is
why the axis must be given. Negative angles need `=`, for example
`--yaws=-90:90:15`.

The experiment splits every full scan into two disjoint halves. One half is
enrolled. The other half is used to simulate what a scanner would see from each
yaw (default −90…90° in steps of 15°) and pitch (0° and 20°):

- only surfaces that face the camera and are not hidden are kept: normals plus a
  z-buffer whose tolerance grows with the surface slope;
- noise and subsampling are added;
- the view is scrambled by a random rigid motion;
- the head is cut out like a real probe and matched against everyone.

Real `--probe` scans are matched against the whole scans, as `compare` does.
The command writes:

- `face_scores.csv` and `face_summary.json` with the EER, the threshold, the
  genuine/impostor ranges and the settings;
- `face_scores_vs_angle.png`, `face_distributions.png` and `face_registration.png`.

`compare --calibration` refuses a summary made with other settings. A decision
is `same` up to the worst simulated genuine score, `different` from the best
impostor score, and `uncertain` in between. Without `--calibration` or
`--threshold` it only ranks the enrolled people.

Results on the scans in `biometrics/` (two people, 52 genuine and 52 impostor
pairs over all angles):

- geometry separates the two people at every angle: genuine ≤ 1.57 mm, impostor
  ≥ 3.11 mm, EER 0%;
- the topological score overlaps (EER 25%);
- the instant scan `raw_preview_95b40cf3…` is closest to A (2.10 mm against
  2.41 mm to B) and falls into the uncertain zone.

These numbers are optimistic and must not be read as biometric accuracy:

- **one capture:** genuine probes come from the same capture as the gallery,
  with the same hair, expression and scan artefacts;
- **two people:** the impostor distribution comes from a single pair of people;
- **same data:** the threshold is estimated on the data it is then applied to.

Calibrated thresholds need repeat scans of several people, on different occasions
and from different angles. Scale defaults assume an adult head in millimetres
(`--units` converts).

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
