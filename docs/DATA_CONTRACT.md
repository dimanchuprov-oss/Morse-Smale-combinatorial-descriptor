# Descriptor JSON contract, version 1

`schema_version: 1` identifies reports emitted by the CLI. Unversioned reports
are legacy. IDs are local to one report, not correspondences between scans.
JSON contains no NaN or Infinity literals.

## Critical points and separatrices

- Every `critical_points[].id` is a unique integer. Regular vertices are omitted.
- `vertex_id` is a mesh index for the heuristic backend. TTK's vertex identifiers
  are backend-specific (and may be -1); use `id` for references.
- Every `separatrices[].id` is a unique integer.
- `source_id` references a critical point. `destination_id` references a critical
  point unless `ends_on_boundary` is true; in that case it is null.
- For heuristic arcs, `source_vertex_id` and `destination_vertex_id` identify
  mesh vertices, including masked boundary endpoints. They are null for TTK.
- `vertices` contains heuristic mesh vertex indices; TTK exports `points`
  polylines instead. `points` is available for both backends.
- `kind` is `u` for descent and `s` for ascent.
- Critical-point `persistence: null, essential: true` means infinite persistence;
  `null, false` means unavailable. Finite persistence is a nonnegative number.
- In `persistence_diagram`, dimension keys are strings and a null death means
  an essential class. A null diagram means GUDHI was unavailable.
- `persistent_counts` filters pairs only; arcs and graph are not simplified.
  TTK persistent counts come from the PL classifier, not TTK point annotations.

## Processing metadata and comparison

`parameters` records all parsed processing options, including defaults, excluding
input/output paths and the normalized deprecated `--periodic` alias. Sigma is in
raster pixels. Coordinates and distance parameters use the input's length units;
curvature uses inverse length or inverse squared length. No unit conversion is
performed. Input units must be consistent across scans.

Comparison rejects different schema versions, processing parameters, scalar,
geometry, surface or persistence threshold. Two legacy reports may still be
compared, but their preprocessing compatibility cannot be established. Regenerate
legacy reports to compare with version 1. A null bottleneck distance means infinity
(no finite match), except that `bottleneck_max` is also null for no dimensions.

The experiment manifest additionally records units and object labels; results
record input SHA-256 hashes, resolved paths, effective parameters, Python and
package versions, and hashes of the Python source modules. Labels determine within-object versus between-object pairs;
they are not inferred from filenames. Archive the source revision with a published experiment so that the recorded
source hashes can be reproduced.

## Boundary curvature and smoothing

Curvature at the rim is an extrapolation, not a measured boundary curvature.
Interior values seed synchronous graph-distance layers; each new vertex gets the
mean of already assigned neighbours. Boundary components without an interior
seed raise an error. No raw angle-deficit estimate on the rim is used as a seed.

Raster median and Gaussian filters use `nearest` for open surfaces and `wrap`
for periodic surfaces. Nontrivial `--sigma` or `--median-size` with Delaunay or
Poisson is rejected; those options do not implement mesh smoothing.

`TriMesh` owns read-only vertex, face and value buffers, preventing input mutation
from invalidating cached topology. Construct a new mesh to change its data.
