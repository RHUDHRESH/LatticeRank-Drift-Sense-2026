# Drift-Sense Phase 2 and Phase 3: Findings and Process

**Project:** LatticeRank Drift-Sense  
**Evaluation date:** 18 September 2026  
**Runtime target:** low-spec CPU, offline inference, one native thread by default

## 1. Current status

> **Validated head update (18 September 2026):** the measurements immediately
> below supersede older experimental tables later in this development log.
> Phase 2 official25 is 20 TP / 0 FP / 0 FN and 81.30/85 measured points.
> Two post-degradation-certified industrial hard seeds score 30/0/0 and
> 30/2/0 (TP/FP/FN), respectively. Phase 3 official25 is 24/0/0/1
> (TP/FP/FN/TN), with all positives within 2 px; tough32 is 24/0/0/8 with all
> positives exact. The repository test suite contains 58 passing tests.

Both pipelines run end to end against the supplied organizer generators and
produce the required CSV contract without reading withheld metadata.

| Measurement | Phase 2 official 25 | Phase 3 generated 25 |
|---|---:|---:|
| Completed rows | 25/25 | 25/25 |
| Processing failures | 0 | 0 |
| Present / absent | 20 / 5 | 24 / 1 |
| Present pairs within 1 px | 16/20 | 16/24 |
| Present pairs within 2 px | 17/20 | **24/24** |
| Present pairs within 5 px | 17/20 | **24/24** |
| Median localization error | 0.495 px | 0.709 px |
| Maximum localization error | — | 1.526 px |
| Detection TP / FP / FN | 17 / 0 / 3 | 24 / 1 / 0 |
| Detection F1 | 0.9189 | 0.9796 |
| Typical runtime | about 0.89 s/pair | 0.38–0.51 s/pair |

All 30 focused repository tests pass. The tests include first-layer and
last-layer invisibility, positive and negative SEM contrast, invalid input,
GDS unit handling, optional search-side CAD, metadata isolation, CPU thread
restoration, and output-contract checks.

## 2. Sources and judge data

The supplied material contained two distinct organizer assets.

### Phase 2

The two nested dataset archives form the organizer's coverage-preserving
25-pair Phase 2 cut. It contains:

- `pairs.csv`, which is participant-visible;
- 25 reference images and 25 search images;
- `ground_truth.csv`, withheld during inference;
- `manifest_jury.csv`, withheld during inference;
- the full Phase 2 generator and organizer scoring code.

Composition:

- Set A: 9 nominal present pairs;
- Set B: 9 degraded present pairs;
- Set C: 5 absent-reference pairs;
- Set D: 2 optical RGB bonus pairs.

The set spans all 12 architecture presets, scales 8–12, rotations approximately
−4.9° to +4.9°, all five severity levels, and five true no-match cases.

The full 48-pair generator command is:

```bash
python generate_phase2_dataset_v2.py \
  --output-dir phase2_dataset_v2 \
  --seed 20260916 \
  --crop-attempts 16
```

The exact 25-pair cut is produced with:

```bash
python make_subset_25.py \
  --src phase2_dataset_v2 \
  --dst phase2_dataset_25
```

### Phase 3

The supplied Drift-Sense repository contains the CAD/GDS generator. A fresh
25-pair set was generated with seed 7 using the organizer's defaults:

```bash
python generate_cad_dataset.py \
  --num-samples 25 \
  --architectures dram finfet \
  --split official25 \
  --output-dir phase3-eval25 \
  --seed 7
```

Blind inference received only:

```text
pair_id,reference_gds_path,search_path
```

The manifest, ground-truth coordinates, reference preview, generator parameters,
match label, architecture label, and seed were not exposed to the solver.

The supplied Phase 3 repository does not contain an official dataset-level
scoring script or official point weights. Therefore the Phase 3 measurements
in this report are exact localization and detection measurements, not an
invented official score.

## 3. Phase 2 score

The organizer-compatible measurable Phase 2 subtotal is **72.9838/85**.

| Block | Result |
|---|---:|
| Localization | 33.6/40 |
| Scale | 7.9/10 |
| Rotation | 7.7/10 |
| Presence detection | 13.7838/15 |
| Confidence ranking | 10/10 |
| Total measured | **72.9838/85** |

Additional measurements:

- mean localization credit over present pairs: 0.840;
- confidence AUC against presence: 0.930;
- confidence AUC against correctness: 1.000;
- 17 true positives, zero false positives, three false negatives;
- all five absent pairs correctly rejected;
- false-negative IDs: `p025`, `p028`, and `p033`;
- `p032` was accepted at 1.808 px error.

Compared with the organizer's published naive calibration:

| Metric | Current | Organizer naive baseline |
|---|---:|---:|
| Core localization credit | **0.804** | 0.684 |
| Detection F1 | **0.919** | 0.833 |
| Median localization error | **0.495 px** | approximately 0.67 px on the difficult subset |

The measured subtotal does not by itself prove a final score above 83/100.
Crossing 83 requires at least 10.02 points from the remaining 15 unmeasured
points under the assumed 100-point rubric.

## 4. Common command-line and output contract

Phase 2:

```bash
python register.py --input pairs.csv --output predictions.csv
```

Phase 3:

```bash
python phase3.py --input pairs.csv --output predictions.csv
```

Both commands write exactly:

```csv
pair_id,x,y,theta,scale,found,score
```

Rules enforced by both entry points:

1. Each unique input ID receives exactly one output row.
2. Duplicate IDs keep the first input row.
3. All numeric outputs are finite.
4. `found` is either 0 or 1.
5. `score` is clipped to `[0,1]`.
6. Rejected rows have `x=y=theta=scale=0`.
7. A bad row does not abort the rest of the batch.
8. Output is written to a temporary file and atomically renamed when complete.
9. Native numerical libraries default to one CPU thread.

## 5. Phase 2 input handling

The Phase 2 entry point is `register.py`.

### 5.1 CSV selection

The preferred fields are:

```text
pair_id
reference_path
search_path
```

Sensible alternative ID, reference, and search column names are accepted.
Relative paths are resolved first against the input CSV directory and then
against the project directory.

### 5.2 Image loading

Every image must contain one frame and one usable image plane.

- RGB and other standard color modes are converted to grayscale.
- Integer and floating-point grayscale images are supported.
- Numeric images are mapped from their observed minimum and maximum to 8-bit.
- Constant numeric images become a zero image.
- NaN and infinite working values are replaced before signal processing.
- Missing, malformed, multiframe, or unsupported images produce a rejected row.

## 6. Phase 2 mathematical pipeline

The implementation is in `driftforge/edge_registration.py` and
`driftforge/edge_pose.py`.

### 6.1 Contrast and edge-energy gate

For an input image `I`, robust contrast is measured from the first and
ninety-ninth percentiles:

```text
C = P99(I) - P1(I)
```

The image is Gaussian-smoothed and differentiated with Scharr kernels:

```text
Gx = I * Sx
Gy = I * Sy
G  = sqrt(Gx² + Gy²)
```

Gradient RMS is:

```text
gradient_rms = sqrt(mean(Gx² + Gy²)) / 16
```

The pair abstains when:

- either image dimension is below 64 px;
- robust contrast is below 0.008;
- gradient RMS is below 0.0025.

### 6.2 Edge representation

After Gaussian smoothing with sigma 1, Scharr magnitude is normalized using
its 35th and 99.5th percentiles:

```text
E = clip((G - P35(G)) / (P99.5(G) - P35(G)), 0, 1)
```

This suppresses the noise floor while retaining strong structural boundaries.

### 6.3 Nominal reference scale

The reference is initially area-resized by approximately 10 because the
reference pitch is 1 nm/px and the search pitch is approximately 10 nm/px.
The reported `scale` is Reference pixels per Search pixel and is constrained
to `[8,12]`.

### 6.4 Local descriptor pose proposal

SIFT runs on the edge images with:

- at most 1,400 features;
- three octave layers;
- contrast threshold 0.008;
- edge threshold 12;
- sigma 1.2.

Brute-force L2 KNN matching uses `k=2`. A match survives when:

```text
d1 / d2 < 0.96
```

Matches inconsistent with the permitted 8–12 scale range or ±10° rotation
range are discarded. At least three matches are required.

OpenCV partial-affine RANSAC estimates:

```text
[xs]   [ a -c ][xr]   [tx]
[ys] = [ c  a ][yr] + [ty]
```

RANSAC uses:

- 2.5 px reprojection threshold;
- 1,200 maximum iterations;
- 0.995 confidence;
- 10 refinement iterations.

The similarity parameters are recovered as:

```text
q     = sqrt(a² + c²)
scale = 10 / q
theta = degrees(atan2(c, a))
```

The reference center is mapped through the affine transform to provide a
translation proposal. Pose confidence combines the RANSAC inlier ratio and
the number of supporting inliers.

### 6.5 Spectral pose proposal

A second proposal path uses gradient orientation and directional FFT evidence.

1. Apply a Hann window.
2. Compute the real FFT and log power.
3. Retain spectral radii from approximately 0.018 to 0.42 cycles/pixel.
4. Remove a 64-bin radial median background.
5. Extract up to 32 directional peaks with nonmaximum suppression.
6. Match reference and search spectral directions.

Each peak pair votes:

```text
scale = 10 * search_radius / reference_radius
theta = -wrap(search_angle - reference_angle)
```

Only votes in the legal scale and angle bounds survive. Votes are clustered
with Gaussian density modes, weighted medians, and weighted median absolute
deviation. Spectral confidence is capped because spectral evidence does not
directly prove translation.

### 6.6 Proposal fusion

Descriptor proposals are prioritized. Proposals are merged when they differ
by less than approximately:

```text
0.20 scale units
0.45 degrees
5 pixels when both contain a center
```

At most five distinct pose proposals survive.

### 6.7 Similarity warp

The reference is preblurred before downsampling to avoid aliasing. For each
pose proposal, a centered similarity warp is constructed:

```text
T(p) = (source_scale / scale) * R(theta) * (p - center) + target_center
```

Bilinear interpolation and reflected borders are used. The transformed
template is converted to the same Scharr-edge representation as the search.

### 6.8 Translation correlation

The warped template is searched with normalized zero-mean correlation:

```text
rho(u,v) =
  sum((Suv - mean(Suv)) * (T - mean(T)))
  -------------------------------------------------
  sqrt(sum((Suv - mean(Suv))²) * sum((T - mean(T))²))
```

Four peaks are retained per pose. Nonmaximum suppression uses a radius based
on template size. The candidate center is:

```text
x = left + (template_width  - 1) / 2
y = top  + (template_height - 1) / 2
```

The initial ranking value is:

```text
rank = edge_correlation + 0.16 * pose_confidence
```

At most ten spatially and geometrically distinct candidates enter refinement.

### 6.9 Continuous subpixel refinement

Every selected candidate is optimized continuously in:

```text
(x, y, theta, scale)
```

Bounds are:

- `x,y`: initial value ±4 px, clipped inside the image;
- `theta`: clipped to ±10°;
- `scale`: clipped to 8–12;
- initial scale neighborhood: approximately ±2.

The search patch is sampled with `getRectSubPix`. The reference template is
continuously warped with bilinear interpolation. Powell optimization minimizes:

```text
L(x,y,theta,scale) = -ZNCC(search_edge_patch, warped_reference_edges)
```

Optimization uses at most 20 iterations and 100 function evaluations, with
approximately 0.025 position tolerance. This is the actual subpixel stage;
the refined values are not rounded before output.

### 6.10 Ambiguity and confidence

A runner-up must be spatially remote from the winning site. Let:

```text
gap = max(0, best_correlation - runner_up_correlation)
```

Correlation quality is:

```text
Qc = clip((best_correlation - 0.10) / 0.55, 0, 1)
```

Gap quality is:

```text
Qg = clip(gap / 0.08, 0, 1)
```

Final confidence is:

```text
score = clip(
    0.58 * Qc
  + 0.27 * pose_confidence
  + 0.15 * Qg,
  0, 1)
```

Descriptor solutions require at least three RANSAC inliers and correlation
of at least 0.30. Spectral proposals require strong or isolated spatial
support. The common final gates require approximately:

```text
edge_correlation >= 0.18
pose_confidence  >= 0.08
score            >= 0.30
```

Rejected results retain a finite confidence but publish a zero pose.

## 7. Phase 3 legal inputs and fixed pose

The Phase 3 entry point is `phase3.py`. Preferred fields are:

```text
pair_id
reference_gds_path
search_path
```

An optional `search_gds_path` is accepted when the judge supplies a legal
full-canvas CAD file.

The following training or scoring fields are explicitly excluded from file
selection and are never opened:

```text
params_json_path
reference_sem_path
reference_preview_path
gt_x
gt_y
gt_box_x
gt_box_y
gt_box_w
gt_box_h
match_found
seed
```

Under the supplied Phase 3 specification:

```text
theta = 0 degrees
scale = 10 reference-pixels/search-pixel
```

The unknowns are translation, layer brightness, polarity, and layer visibility.

## 8. Phase 3 mathematical pipeline

The implementation is in `driftforge/cadref.py` and
`driftforge/cad_edges.py`.

### 8.1 GDS parsing and units

All polygons from the top-level GDS cell are grouped by `(layer, datatype)`.
Automatic unit selection compares two interpretations:

- raw numeric coordinates treated as nanometres;
- GDS user units converted through the library unit field.

The interpretation whose extent is physically plausible for the known
1,000 nm reference window is selected. Geometry remains anchored at `(0,0)`;
sparse geometry is not tightly cropped because that would change the reference
coordinate system.

### 8.2 Layer rasterization

The GDS reference covers 1,000 nm. At the fixed 10 nm/search-pixel scale:

```text
template_side = round(1000 / 10) = 100 pixels
```

Each layer becomes an independent 100×100 fractional-coverage mask. Masks
are rasterized at 2× supersampling and averaged down. More than 32 layers is
rejected rather than silently truncated.

### 8.3 Independent edge channels

Each layer is Gaussian-smoothed with sigma 0.65 and differentiated. A pooled
channel is constructed:

```text
Epool = 0.55 * max_layer_edge + 0.45 * mean_layer_edge
```

The proposal bank contains the pooled channel and every independent layer.
This prevents a missing dominant layer from erasing all usable evidence.

### 8.4 Painter-order physical appearance

The repair made after the first Phase 3 judge run adds a bounded physical
appearance hypothesis. Layers are painted in order:

```text
A_next = A_current * (1 - layer_coverage)
       + layer_intensity * layer_coverage
```

The appearance starts from a background level of 0.12. Layer intensity uses
a monotonic trend from approximately 0.20 to 0.85. This channel represents
the fact that an upper material hides a lower material at an overlap.

The appearance channel supplies four reserved translation proposals and an
additional ranking signal. It does not replace the adaptive layer estimator,
so faint, inverted, or absent outer layers remain supported.

### 8.5 Image-based translation proposals

The physical appearance hypothesis is correlated with the normalized search
image using ZNCC. Four spatially distinct peaks are retained.

Every edge channel is correlated against the SEM edge magnitude using
normalized correlation:

- eight peaks from the pooled channel;
- three peaks from each individual layer;
- nonmaximum suppression radius at least 8 px and typically one third of
  the template side;
- candidates within 5 px are merged;
- at most 18 candidates survive.

Periodic aliases remain in the bounded pool so that the evidence stage can
measure ambiguity rather than silently discarding them.

### 8.6 Optional vector CAD proposals

When a legal full-canvas `search_gds_path` is supplied, polygons receive a
translation-invariant signature.

1. Quantize vertices to 0.01 nm.
2. Subtract the polygon centroid.
3. Canonicalize vertex start and traversal direction.
4. Match polygons having the same layer, datatype, and signature.
5. Vote with centroid differences in 0.1 nm bins.

For a translation vote:

```text
delta = search_polygon_centroid - reference_polygon_centroid
```

Geometry confidence is:

```text
geometry_score =
    0.75 * matching_polygon_fraction
  + 0.25 * matching_layer_fraction
```

CAD geometry only proposes positions. The observed SEM still ranks the
positions and determines presence.

### 8.7 Optional raster CAD fallback

If exact polygon voting yields nothing, the full search-side CAD is rasterized
per layer. Each reference layer is correlated with its search-side layer.
Votes within 5 px are clustered, retaining one best vote per layer. Support
combines the number of agreeing layers and their mean correlation.

### 8.8 Patch extraction and illumination removal

At each proposal, a subpixel 100×100 SEM patch is extracted. A least-squares
plane is removed:

```text
P(x,y) = a + b*x + c*y
```

This removes gradual charging, shading, and vignette components before
structural fitting.

### 8.9 Adaptive layer-intensity fit

The smoothed CAD layer masks form the columns of matrix `X`. Flat columns are
removed. A checkerboard half of the pixels is used for fitting, which reduces
same-pixel overfitting.

Ridge strength is:

```text
lambda = max(trace(X'X) / number_of_layers * 1e-3, 1e-8)
```

Layer coefficients are:

```text
beta = inverse(X'X + lambda*I) * X'y
```

The predicted SEM structure is:

```text
y_hat = X * beta
```

Coefficients may be positive, negative, or near zero. Therefore the method
supports bright layers, dark layers, contrast inversion, and invisible layers.

A layer is counted as visible when:

```text
abs(beta_l) * std(feature_l)
    > max(0.025 * std(observed_patch), 1e-4)
```

### 8.10 Candidate evidence

Intensity agreement is cosine correlation:

```text
Ci = max(0, dot(y_hat,y) / (norm(y_hat)*norm(y)))
```

Directional edge agreement is measured on stacked horizontal and vertical
gradients:

```text
Ce = max(0,
    dot([Gx_hat,Gy_hat],[Gx,Gy]) /
    (norm([Gx_hat,Gy_hat])*norm([Gx,Gy])))
```

Adaptive fit is:

```text
F = 0.62 * Ci + 0.38 * Ce
```

The physical appearance correlation is measured independently as `Ca`.

Candidate order is:

```text
rank =
    0.58 * adaptive_fit
  + 0.27 * appearance_correlation
  + 0.05 * proposal_strength
  + 0.10 * legal_geometry_support
```

The appearance term corrected six structural-alias failures without using
IDs, seeds, metadata, or ground truth.

### 8.11 Subpixel translation refinement

The three strongest candidates are evaluated one pixel to each side in `x`
and `y`. Given scores `f(-1)`, `f(0)`, and `f(+1)`, parabolic displacement is:

```text
delta = 0.5 * (f(-1) - f(+1))
        / (f(-1) - 2*f(0) + f(+1))
```

The correction is used only when the fitted parabola is concave and is
clipped to ±0.75 px. The procedure is applied separately to `x` and `y`.
The refined location is retained only when its measured fit improves.

### 8.12 Confidence and acceptance

Let `F1` and `F2` be the best and second-best adaptive fit values:

```text
margin = max(0, F1 - F2)
```

Evidence quality is:

```text
E = clip((F1 - 0.10) / 0.70, 0, 1)
```

Ambiguity quality is:

```text
A = 0.55 + 0.45 * clip(margin / 0.10, 0, 1)
```

Visible-layer support is:

```text
L = 0.75 + 0.25 * min(visible_layers, 3) / 3
```

Final confidence is:

```text
score = clip(E * A * L, 0, 1)
```

Acceptance requires:

```text
adaptive_fit      >= 0.20
edge_correlation  >= 0.06
proposal_strength >= 0.04
```

Successful output is:

```text
x = template_left + 50
y = template_top  + 50
theta = 0
scale = 10
found = 1
```

## 9. Phase 3 repair findings

The first blind run processed all 25 pairs but localized only 18 of 24 present
pairs within 5 px. Six present pairs selected periodic structural aliases.

Before the repair:

```text
within 1/2/3/5 px: 13/18/18/18 of 24
median error:       0.805 px
mean error:         20.683 px
maximum error:      301.925 px
```

Candidate inspection showed that the correct position was usually present in
the bounded candidate pool, but the additive per-layer regression could rank
a periodic alias above it because real SEM layers use painter-style occlusion.

The solution was to add the physical painter-order appearance channel while
keeping independent layer fitting for visibility changes.

After the repair:

```text
within 1/2/3/5 px: 16/24/24/24 of 24
median error:       0.709 px
mean error:         0.760 px
maximum error:      1.526 px
```

Runtime increased by only about 0.05 seconds per pair in the controlled run.

The single generated absent case remains a false positive with confidence
approximately 0.479. The lowest present confidence in this set is approximately
0.55. A threshold between them would make this specific set perfect, but it
was deliberately not hard-coded from one absent example because that would
risk rejecting faint hidden-evaluation matches.

## 10. Invisible-layer behavior

The Phase 3 tests deliberately construct a GDS with a sparse lower layer and
a dominant high layer. They generate SEM searches where only one of those
layers remains visible and test both contrast polarities.

Covered cases:

- first/lower layer visible and last/upper layer invisible;
- last/upper layer visible and first/lower layer invisible;
- bright visible layer;
- dark visible layer;
- optional search-side GDS supplied;
- metadata JSON path present but deliberately invalid and never opened.

Every case must return `found=1`, `theta=0`, `scale=10`, and translation error
below 1 px.

## 11. Failure behavior

Both phases use the same public failure convention:

```csv
pair_id,0.0,0.0,0.0,0.0,0,finite_score
```

Expected input failures include:

- missing paths;
- unreadable images;
- unsupported image shape or mode;
- empty GDS;
- invalid GDS geometry;
- search smaller than the 100×100 CAD footprint;
- more than 32 CAD layers;
- nonfinite internal result.

Unexpected exceptions are written to standard error, but the affected input
still receives a valid output row and subsequent inputs continue.

## 12. Dependencies

Runtime dependencies are deliberately small and CPU-compatible:

```text
numpy==2.4.6
scipy==1.17.1
Pillow==12.3.0
opencv-python-headless==5.0.0.93
gdstk==1.0.1
```

No PyTorch, TensorFlow, scikit-learn, joblib model, GPU runtime, network call,
or external service is used during inference.

## 13. Validation commands

Install:

```bash
python -m pip install -r requirements.txt
```

Run all focused tests:

```bash
python -m pytest -q
```

Expected result:

```text
30 passed
```

Compile-check the public entry points and implementation:

```bash
python -m py_compile \
  register.py \
  phase3.py \
  driftforge/cadref.py \
  driftforge/cad_edges.py \
  driftforge/edge_pose.py \
  driftforge/edge_registration.py
```

Run Phase 2:

```bash
python register.py \
  --input /path/to/phase2/pairs.csv \
  --output /path/to/phase2/predictions.csv
```

Run Phase 3:

```bash
python phase3.py \
  --input /path/to/phase3/pairs.csv \
  --output /path/to/phase3/predictions.csv
```

## 14. Current release stage

The project is at release-candidate validation:

1. The exact supplied Phase 2 25-pair set runs successfully.
2. A fresh 25-pair Phase 3 set from the organizer generator runs successfully.
3. The Phase 3 alias failure was reproduced, diagnosed, repaired, and rerun.
4. Every present Phase 3 pair is now within 2 px.
5. First-layer and last-layer invisibility are tested.
6. Training JSON and ground truth are excluded from inference.
7. Both public commands emit the required schema.
8. All 30 focused tests pass.
9. The inference repository remains CPU-only and compact.

The remaining work before submission is release packaging and, if time allows,
additional seed-based stress testing of Phase 3 absence rejection. Thresholds
should not be tuned from the single absent seed-7 example without a broader
negative set.

## 15. Phase 2 photo and tile validation TODO

The slide and WhatsApp photographs show the practical Phase 2 tile problem:
a small, high-resolution SEM reference tile must be located inside a wider SEM
search field containing repeated device structures and larger mat or separator
boundaries. This photo-derived validation is a separate required workstream.

### 15.1 Inventory the photographs

1. Inspect all supplied images in chronological order.
2. Mark which images contain a Phase 2 reference tile, a search image, a
   reference/search example, a result overlay, or explanation only.
3. Record the visible architecture, scale, rotation, boundary features, layer
   visibility, and image-quality problems.
4. Keep slide screenshots and phone-camera photographs distinct from original
   raster datasets.

### 15.2 Recover usable raster regions

1. Crop away slide chrome, captions, borders, and unrelated panels.
2. Correct camera perspective when a photograph was taken at an angle.
3. Preserve the original aspect ratio and avoid rescaling more than necessary.
4. Convert the recovered regions to one grayscale image plane.
5. Record every crop and geometric correction so the experiment is
   reproducible.

### 15.3 Construct verified tile pairs

For each usable example, create:

```text
pair_id
reference_path
search_path
verified_center_x
verified_center_y
verification_method
source_photo
```

Ground truth should come from an organizer label, an unambiguous visible box,
or careful human verification. Images without trustworthy coordinates may be
used for qualitative overlays but must not be included in numerical scoring.

### 15.4 Run the complete Phase 2 pipeline

Each recovered pair must pass through the same public inference path:

```text
load reference and search
→ normalize image planes
→ extract Scharr edges
→ estimate rotation and scale
→ warp the reference tile
→ correlate against the search field
→ preserve periodic alternatives
→ refine x, y, theta, and scale continuously
→ compute confidence and found/rejected state
```

No photo-specific coordinate, filename, or manually selected search window may
be injected into the solver.

### 15.5 Produce visual verification

For every usable pair, generate a review overlay containing:

- the complete search image;
- the predicted reference footprint;
- the verified footprint when available;
- predicted center and verified center;
- localization error in search pixels;
- predicted scale and rotation;
- confidence and found state;
- the strongest remote runner-up when the layout is periodic.

The overlay is for validation only and must remain outside the inference
submission package.

### 15.6 Photo/tile success gates

The photo-derived tile workstream is complete when:

1. Every supplied photograph has been classified.
2. Every usable Phase 2 pair has a reproducible crop record.
3. All pairs run through `register.py` without manual solver assistance.
4. Quantitatively labelled pairs report localization, scale, and rotation
   errors.
5. Qualitative-only pairs have clear overlays and an explanation of any
   ambiguity.
6. Failures are separated into pose-proposal failure, wrong periodic tile,
   subpixel-refinement failure, and presence-rejection failure.
7. Improvements made for the photographs still pass the frozen official
   25-pair judge set.

### 15.7 Scoring boundary

Photo and slide screenshots are valuable for understanding the real tile
geometry, especially coarse mat and separator boundaries. Their results must
be reported separately from official generator scores unless the original
pixel data and trusted ground truth are available. This prevents screenshot
compression, perspective correction, or manual cropping from inflating or
reducing the reported judge score.

## 16. Adversarial validation findings

These results come from a separate deterministic stress suite. The suite is
deliberately harsher than the frozen 25-pair judge run and is not an official
competition score. It is designed to expose failure modes before submission.

### 16.1 Phase 2 stress suite

The Phase 2 suite contains 30 pairs: 20 present and 10 absent. Every present
search image is degraded, the scale spans the required 8x to 12x range, the
rotation spans -4.9 to +4.9 degrees, all 12 generator architectures appear,
and all five severity levels appear. Four cases are RGB and one case is a
clean 8x control. The blind CSV exposes paths only. A separate audit verified
that every positive has a unique intended match in both intensity and edge
space.

The current solver completed all 30 cases without an execution or schema
failure in 44.37 seconds (1.48 seconds per pair). Its stress score was
**59.91/85**:

- localization: 26.8/40;
- scale: 6.9/10;
- rotation: 7.0/10;
- presence F1: 12.0/15 (14 TP, 1 FP, 6 FN; F1 0.800);
- correctness AUC: 7.205/10.

The score should be read as a diagnostic lower-bound test, not as a forecast
of the official score. The strongest results were RGB handling and pose
accuracy after the correct spatial candidate survived. The main weaknesses
were:

1. Two positives lost the correct spatial peak even though the correct scale
   and rotation proposal existed. This is a candidate-recall and correlation
   ranking failure.
2. Four positives reached an internal location within 0.31 to 2.03 pixels of
   ground truth but were rejected. This is confidence calibration and
   acceptance gating, not subpixel localization.
3. One hard negative was accepted.
4. No severity-4 case was accepted, showing that the current threshold is too
   brittle at the harshest degradation level.

The organizer baseline on the same suite was slower (80 seconds), localized
fewer positives, and achieved F1 0.765. The current solver is therefore a
clear improvement, but it regressed on two specific pairs that the baseline
accepted. Those pairs are the best immediate regression targets.

### 16.2 Phase 3 stress suite

The Phase 3 suite contains 32 zero-rotation, 10x cases: 16 DRAM and 16 FinFET,
24 positives and 8 independently generated hard negatives. It covers first
layer invisible, last layer invisible, both edge layers faint, inverted layer
polarity, non-monotonic layer intensities, boundary placements, and three
degradation tiers. The blind CSV was audited for label leakage and all
positive labels were checked against the requested physical centers.

With optional search GDS enabled, the current solver completed all cases in
81.03 seconds. It produced 22 TP, 8 FP, and 2 FN (F1 0.815). Twelve of the 24
positive cases were within one pixel; every successfully retained true vector
vote was essentially exact. Without search GDS, runtime fell to 12.68 seconds
but only 2 of 24 positives were within five pixels.

The Phase 3 weaknesses are more serious than the Phase 2 weaknesses:

1. All eight hard negatives were accepted. Search CAD currently improves
   localization but provides no useful rejection evidence.
2. The capped vector-voting traversal favors polygons encountered early.
   Bottom placements succeeded on only 2 of 12 positives versus 10 of 12 for
   top placements. This is an ordering bias, not a geometry limitation.
3. Periodic aliases dominate when vector voting misses the true translation.
   Image-only matching is especially vulnerable to invisible,
   non-monotonic, and polarity-inverted layers.
4. FinFET cases succeeded on 4 of 12 positives versus 8 of 12 DRAM cases.
5. The vector path is about 6.4 times slower than the image-only path on this
   suite and regressed one case that image matching placed within 2.38 pixels.

The first Phase 3 fixes should therefore remove vector sampling order bias,
add independent no-match evidence, and fuse vector and image candidates
instead of allowing the vector path to replace a strong image result.

### 16.3 Release priority from the stress tests

The shortest path to a stronger submission is:

1. Phase 3: sample vector features across the whole layout rather than taking
   the first polygons up to the vote cap.
2. Phase 3: require agreement between independent geometry and image evidence
   before accepting a match, then calibrate on the eight hard negatives.
3. Phase 3: retain both vector and image candidates through final scoring.
4. Phase 2: retain more spatial peaks for correct pose proposals and improve
   peak diversity across periodic cells.
5. Phase 2: recalibrate acceptance using the four near-ground-truth false
   rejections and the hard-negative false positive.
6. Re-run the frozen official 25-pair suite after every change so stress-suite
   gains cannot hide an official-set regression.

## 17. Phase 2 frontend tile status

- The intended tile is a frontend feature, not a presentation slide.
- The 40 supplied photographs establish the visual concept: a small reference
  tile shown against a much wider SEM search field. `14.38.30.jpeg` and
  `14.38.31.jpeg` are the clearest examples.
- The current public submission repository contains no frontend code or web
  dependency. It contains only the Phase 2 and Phase 3 command-line solvers,
  their shared modules, tests, and concise documentation.
- The supplied generator archive has a separate Streamlit explorer, but the
  photographs do not define a clickable Phase 2 tile, its text, its state, or
  the action it should perform.
- No guessed frontend framework or tile was added to the submission because
  that would enlarge the judged package without a verified UI contract.
- To implement the tile, the remaining concrete input is the target frontend
  repository or file and the tile's click behavior. The registration result
  data already available for the tile is: found state, x/y location, scale,
  rotation, confidence, runtime, reference image, and search image.
