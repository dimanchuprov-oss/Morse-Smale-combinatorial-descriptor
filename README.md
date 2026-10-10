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

`morse-lidar-face` decides whether two head scans show the same person. The
scans may be full heads or single frontal snapshots, and they may be taken from
different angles. For every pair the command:

1. cuts the head out of each scan. It keeps a fixed height under the top of the
   cloud and a fixed radius around the vertical head axis. Scanner noise
   fragments and body parts that do not touch the head (shoulders and chest
   below the chin) are dropped: their shape depends on clothes and posture,
   not on the person;
2. registers the scans rigidly, in both directions. ICP starts from a grid of
   rotations about the vertical axis, small tilts and shifts. Every ICP step
   uses only the point pairs closer than a gate that shrinks from 25 mm to 2 mm.
   With a fixed trimming ratio, two partial views that share a third of their
   surface were pulled towards the parts only one of them has. The pose under
   which the most points lie within 2.5 mm of the other scan wins;
3. scores the registered pair.

There are four scores:

- `geometric_mm`: the decision score. It is the RMS point-to-plane distance
  over the surface where the two scans come within 5 mm of each other,
  averaged over both directions;
- `shared_cm2`: area of that common surface, estimated from the point density
  of each scan. A low score on a small common area is not evidence, because a
  small patch fits many faces;
- `topological_mm`: bottleneck distance between persistence diagrams of the
  radial height field `r(theta, z)` of the two heads, computed over the region
  both scans see;
- `overlap`: fraction of the first scan's points within 3 mm of the second.

```bash
pip install -e '.[signal,topology,plot]'
# every pair of named scans (the person is the file name without the number)
morse-lidar-face matrix biometrics/{Ваня,Влада,Дима,Олег,Сережа}*.ply \
    --up-axis y --out biometrics/face_matrix
# identify a new scan, with the threshold and area limit of the matrix
morse-lidar-face compare probe.ply --gallery A=a.ply --gallery B=b.ply --up-axis y \
    --calibration biometrics/face_matrix/face_matrix_summary.json
# angle experiment: simulated views of full head scans
morse-lidar-face experiment --person A=biometrics/raw_preview.ply \
    --person B=biometrics/raw_preview_05f4e84d-1b50-4aa0-b895-4d55b7ea554e.ply \
    --probe biometrics/raw_preview_95b40cf3-65cc-4b20-999d-b82a85677891.ply \
    --up-axis y --front-axis +z --out biometrics/face_results
```

`--up-axis` is required: Revo Scan and iPhone exports are Y-up. A wrong axis or
wrong units are rejected when either of these holds:

- the section 50–120 mm below the top is wider than 350 mm (a body lying along
  the wrong axis);
- the head region is under 120 mm tall.

A tightly cropped scan read along a wrong axis can still pass, which is why the
axis must be given.

### Real scans of five people

`biometrics/` holds 14 frontal snapshots from a Revopoint Range:

- Ваня, Влада, Олег and Сережа: three each;
- Дима: two.

`matrix` compares all 91 pairs, 13 of the same person and 78 of different
people, in about 40 s on 10 worker processes. It writes:

- `face_pairs.csv`: every pair;
- `face_matrix_summary.json`: EER (all pairs, trusted pairs, by area limit), threshold, identification, decisions and the share of each scan dropped as fragments;
- `face_matrix.png`: heat map of all pairs;
- `face_score_vs_area.png`: score against shared area;
- `face_nearest.png`: the closest same-person and other-person scan of each scan;
- `face_examples.png`: where the surfaces of telling pairs differ.

| pairs | same person | different people | EER |
|---|---|---|---|
| all | 13, median 0.62 mm | 78, median 1.88 mm | 23% |
| sharing ≥ 100 cm² | 6, at most 1.41 mm | 28, at least 1.47 mm | 0% |

The shared area is estimated from the point density of each scan, so sparse
and dense scans measure alike. 100 cm² is about half of a face from the
forehead to the chin. The more surface two scans share, the more their score
can be trusted:

| min shared area, cm² | 0–50 | 60 | 70 | 80–90 | 100 and more |
|---|---|---|---|---|---|
| EER | 23% | 17% | 20% | 11% | 0% |

Under the 100 cm² limit and the calibrated decision rule:

- 6 same-person pairs are accepted;
- 28 different-person pairs are rejected;
- none falls into the uncertain zone;
- 57 pairs are left undecided because they share too little surface.

The closest other scan is the same person for 11 of the 14 scans. Asked the
way `compare` decides, with the area limit:

- 9 scans are identified;
- none is taken for another person;
- 4 are rejected: Влада2, Дима1, Дима2 and Сережа3;
- Ваня3 shares 100 cm² with no other scan.

Олег (0.43–0.47 mm) and Влада (0.49–0.62 mm) are recognised with a wide margin
over everybody else (1.1 mm and more). The failures have visible causes in the data:

- **Дима1 and Дима2** differ by 2.01 mm over 96 cm². This is the one failure
  that a small overlap does not explain. The differences concentrate on the
  nose, the mouth and the chin, while the forehead and the cheeks agree
  (`face_examples.png`). A different expression is the likely cause; the scans
  themselves would have to be checked;
- **Сережа3** misses the eyes and most of the forehead. It shares only
  60–74 cm² with Сережа1 and Сережа2;
- **Ваня3** is a small cut-off scan with 2195 points. It fits other faces
  almost as well as Ваня's: 1.10 mm to Олег3 against 0.86 mm to Ваня1. These
  are the lowest different-person scores in the set;
- **Ваня1–Ваня2** (1.41 mm) is the weakest trusted same-person pair, 0.06 mm
  under the closest trusted impostor;
- **Влада2** is rejected only because its pairs with Влада1 and Влада3 share
  96–97 cm², just under the limit. Their scores are 0.49 and 0.62 mm.

The topological score does not separate people on frontal snapshots (EER 45%).
The radial map of a frontal view covers only 3–8% of the cylinder around the
head, so the persistence diagrams describe a small window.

A known gap in the crop: the head is cut at a fixed height under the top, and on
frontal snapshots the top is the hairline. The neck therefore stays in the crop
when it is connected to the chin (Олег2, Ваня2, Сережа2; the dark band in
`face_examples.png`). It only enters the score where the other scan also has
surface within 5 mm. Cutting at the chin would be cleaner.

These numbers describe this data set and must not be read as biometric accuracy:

- **same data:** the threshold and the 100 cm² limit are chosen on the 14 scans
  they are then applied to;
- **few people:** the 28 trusted different-person pairs come from five people;
- **narrow margin:** the trusted same-person and different-person ranges are
  only 0.06 mm apart.

Scans give reliable results when they:

- are taken facing the camera, within about ±30°;
- have a neutral expression with the mouth closed;
- show the whole face from the forehead to the chin, with the hair off the
  forehead;
- share at least 100 cm² of surface with the enrolled scan.

`compare --calibration` uses the threshold and the area limit of the summary.
It refuses a summary made with other settings or with an older version of the
method: rerun `matrix` or `experiment` after an update. A probe that shares less than
the limit with every enrolled scan gets the decision `insufficient overlap`.
Otherwise a decision is:

- `same` up to the worst trusted same-person score;
- `different` from the best trusted impostor score;
- `uncertain` in between.

Without `--calibration` or `--threshold` the command only ranks the enrolled people.

### Simulated viewing angles

`experiment` checks the angle range on full head scans. It splits every scan
into two disjoint halves of its points and enrolls one half. The other half is
used to simulate what a scanner would see from each yaw (default −90…90° in
steps of 15°) and pitch (0° and 20°):

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

Results on the full scans in `biometrics/` (two people, A and B, 52 genuine and
52 impostor pairs over all angles):

- geometry separates the two people at every angle: genuine ≤ 1.07 mm,
  impostor ≥ 2.05 mm, EER 0%;
- the topological score overlaps less than with the previous registration
  (EER 17%, was 25%);
- the instant scan `raw_preview_95b40cf3…` is closest to B (2.01 mm against
  2.21 mm to A) and falls into the uncertain zone. The previous method put it
  closer to A (2.10 against 2.41 mm), so it cannot be attributed to either.

These numbers are optimistic:

- **one capture:** genuine probes come from the same capture as the gallery,
  with the same hair, expression and scan artefacts;
- **two people:** the impostor distribution comes from a single pair of people;
- **same data:** the threshold is estimated on the data it is then applied to.

The real scans above are the better guide to accuracy.

Negative angles need `=`, for example `--yaws=-90:90:15`. Scale defaults
assume an adult head in millimetres (`--units` converts).

### Expressions: the Multiface benchmark

`morse-lidar-multiface` measures what facial expressions cost the matcher. It
uses the tracked meshes of [Multiface](https://github.com/facebookresearch/multiface)
(Meta, CC-BY-NC 4.0, [arXiv 2207.11243](https://arxiv.org/abs/2207.11243)):
ten people of the v1 script, each with named expressions on one shared mesh
of 7,306 vertices. The data may not be redistributed, so keep the download
folder outside this repository.

```bash
pip install -e '.[signal,plot]'
# ten classes of ten people: about 1 GB of traffic and 50 MB on disk per person
morse-lidar-multiface fetch --dest ~/data/multiface
morse-lidar-multiface bench --data ~/data/multiface --out results/multiface
```

`fetch` is a downloader of its own: the official script passes server file
names to the shell. Each archive is streamed and checked in several ways:

- its length must match the announced length;
- its MD5 sum must match the published one;
- a cut connection is retried;
- redirects are refused.

Only the per-frame vertices without the head pose are kept, plus the head
poses themselves.

The ten classes are:

- neutral;
- eyes closed;
- closed smile, open smile and wide smile;
- open mouth;
- raised brows;
- frown;
- puffed cheeks;
- speech (one sentence).

The probe of a class is its peak frame: the frame that departs most from the
neutral shape over the face.

`bench` turns meshes into scans with a virtual single-shot scanner
(`morse_lidar.virtual_scan`). It models a pinhole camera with a z-buffer:

- samples lie 1.2–2.2 mm apart;
- surfaces seen more obliquely than 75° are lost;
- depth noise is 0.86·z² mm, z in metres;
- depth is rounded to 0.1 mm;
- the eyes, brows and hair are holes, as on the Revopoint.

Each person is enrolled from a frontal shot at 0.5 m. Probes are shot from a
random pose:

- yaw within ±30°;
- pitch within ±10°;
- distance 0.4–0.7 m.

The methods are:

- `b0`: the current matcher;
- `b1`: the matcher on the nose, the bridge of the nose and the forehead only.
  The region comes from the mesh labels, so this is an upper bound for a mask
  found on a real scan;
- `b2`: three templates per person (neutral, open smile, open mouth), and the
  closest one counts.

The command writes:

- `bench_pairs.csv`;
- `bench_summary.json`, with rank-1, EER and the cost of each class, that is
  the growth of the median genuine score over the neutral probes;
- `bench_scores.png`;
- `bench_error_maps.png`, which shows where on the face each expression departs
  from the neutral template.

Results on the ten people (90 expressive probes per method, every probe
against all ten people):

| | b0 | b1 | b2 |
|---|---|---|---|
| rank-1 with expression | 93% | 100% | 99% |
| EER with expression | 11.1% | 3.3% | 6.7% |
| rank-1 at 100 people, rough forecast | ~66% | ~96% | ~84% |
| cost of a smile, mm | +1.3…1.5 | +0.1…0.3 | +1.0…1.4 |

The current matcher fails on smiles: 70% rank-1 on the wide smile, 80% on
the open one, 90% on the closed one. Every other class scores 100%. A smile
moves the cheeks and the mouth by 1.3–1.5 mm, as much as the gap to another
person.

The nose and forehead alone almost remove the effect of the expression, so
comparing face regions is the direction to follow. Narrowing the region costs
something too: the closest other person is 1.04 mm away in b1 against
1.5 mm in b0. In 4 of the 90 probes someone else came closer than the
person's own template.

The forecast assumes independent scores. It counts, for every genuine score,
the fraction of the 810 impostor scores below it.

Limits:

- the b1 region comes from the mesh labels, so 96% is an upper bound;
- all data come from one recording session: the neutral pair gives 0.32 mm,
  against 0.4–0.6 mm between repeat scans of real people;
- there are only ten people.

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
