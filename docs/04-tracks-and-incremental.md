# 04. Track building and the incremental pipeline

Both topics sit in front of the solver. Track building is a real feature. The
incremental pipeline is a test fixture (see document 01).

## Part A: track building

### Input and output

- Input: keypoints per image (pixel coordinates) and pairwise matches
  `(image_a, keypoint_a, image_b, keypoint_b)`. Matches are assumed to be already
  geometrically verified by the caller, but may still contain errors.
- Output: tracks. A track is a list of `(image, keypoint)` observations believed to be
  the same 3D point, with at most one keypoint per image.

Callers who already have tracks skip this helper and pass them straight to the solver.

### How others do it

Read from current sources on 2026-11-14.

| Library | Method | Two keypoints of one image in a track |
|---|---|---|
| OpenMVG | Union-find over (image, keypoint) | Whole track removed |
| OpenSfM | Union-find | Whole track removed |
| Theia | Connected components, with a maximum track length | First keypoint per image kept, others dropped |
| COLMAP global mapper | Union-find | Track removed only if the keypoints are more than 10 px apart; otherwise all kept. Then keep tracks seen in 3 to 100 images. |
| COLMAP incremental mapper | No up-front tracks. Tracks grow from a correspondence graph as images register, and each new track is created by RANSAC triangulation so only geometric inliers join. | Handled implicitly by the inlier test |

Two families: purely combinatorial (union-find, then throw away conflicts), and
geometric (use poses to decide which observations belong). The combinatorial family
loses long tracks, because one bad match fuses two tracks and the conflict rule then
deletes both. The geometric family needs poses.

### Proposal

Our main use case has poses from the start (the priors). That lets the helper use
geometry early, which a from-scratch pipeline cannot do.

**Stage 1, combinatorial (always).** Union-find over all matches. Classify each
component:

- clean: at most one keypoint per image;
- conflicted: some image appears more than once.

Clean components of length 2 or more become tracks. Conflicted components are not
deleted; they go to stage 2 when poses are available, and are dropped otherwise
(the OpenMVG rule, the safe default).

**Stage 2, geometric repair (when poses are available).** For each conflicted
component, and optionally for every track:

1. Triangulate with RANSAC over two-view samples, using the prior poses. Score by
   angular reprojection error so wide-angle cameras are treated fairly.
2. Keep the inlier set as a track if it has at least the minimum number of views and
   a sufficient triangulation angle.
3. Remove those observations and repeat on the remainder, so one fused component can
   split into the two or more real tracks it contains.

The inlier threshold must be loose here, because the prior poses are noisy. It is tied
to the pose-prior uncertainty supplied by the caller, not fixed.

**Stage 3, in-solver robustness (always).** Track building never has to be perfect.
The reprojection residual carries a robust loss, and after optimization observations
with large error are dropped and the solve repeats. Thresholds follow COLMAP's
defaults as a starting point (4 px reprojection error, 1.5 degree minimum
triangulation angle, negative depth rejected).

**Later, if needed.** Track completion and merging as in COLMAP (re-attach dropped
observations that now reproject well; merge two tracks whose merged point reprojects
well in all views). Not in the first version.

### Parameters exposed

| Parameter | Default | Meaning |
|---|---|---|
| `min_views` | 2 with poses, 3 without | Shortest track kept |
| `max_views` | 100 | Longer tracks are truncated or split |
| `conflict_policy` | `repair` with poses, `drop` without | What to do with conflicted components |
| `max_angle_error_deg` | derived from prior noise | Stage 2 inlier threshold |
| `min_tri_angle_deg` | 1.5 | Minimum triangulation angle |

### Where it runs

Track building is CPU code. It runs once, is not the bottleneck, and has no GPU
version.

### As built (milestone 2)

- With poses, every match is first checked against them: the two keypoint rays must
  meet in front of both cameras within the stage 2 angle of the epipolar plane
  (D69). Without this, 10% wrong matches fused most keypoints into one component.
- Stage 2 runs on clean components too by default (`check_clean`).
- Stage 2 runs again after each solve, inside `refine`, from the improved poses with
  the threshold halved (2 degrees first, then 1, never below 0.25; D69, D71).
- Triangulation is least squares on rays rather than DLT (D72).

## Part B: triangulation

Needed by both use cases. Given tracks and poses:

1. Linear (DLT) triangulation on bearing vectors for a starting point.
2. RANSAC over view pairs when the track has outliers (shared with stage 2 above).
3. Reject points behind any camera or with a small triangulation angle.

Points that fail are left out of the problem and can be retried after poses improve.

## Part C: the incremental pipeline (secondary)

Purpose: run on datasets without pose priors, and compare against COLMAP.
Scope: the minimum that gives comparable accuracy on standard small and medium scenes.

### Steps

1. Build the correspondence graph from the caller's verified matches.
2. Pick an initial image pair (many matches, wide baseline), estimate relative pose,
   triangulate.
3. Loop: choose the next image by number of visible triangulated points, register it
   with PnP plus RANSAC, triangulate new points, run a local bundle adjustment.
4. Every so often: global bundle adjustment, filter points, re-triangulate.
5. Finish with a global bundle adjustment.

Steps 3 to 5 call the same solver as the main use case. The gauge here is free, so
this path also exercises gauge fixing (document 02).

### What is deliberately left out

- Geometric verification of matches (the caller's job).
- Image retrieval, loop detection, rig handling.
- COLMAP's global mapper path.

### Parity target

Use COLMAP 4.2.1 with its Ceres backend as the reference. Compare on the same matches
exported from a COLMAP database, so differences come from our pipeline and solver, not
from features. Metrics are in document 05.

### Where it lives

Relative pose and PnP estimation are a sizeable amount of code that is not our core.
They come from poselib-gpu, the owner's sibling library, which only the demo app
links (`demo/`, built with `JBA_BUILD_DEMO`). The library and its wheels never depend
on it (D132). The mapper of steps 1 to 5 lives in the demo too: a native binary for
the COLMAP comparison and a WebAssembly module for the live demo on the documentation
site.
