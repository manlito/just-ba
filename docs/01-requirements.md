# 01. Requirements and scope

Status: draft for review. No code exists yet; this document set is the plan.

## Purpose

just-ba ("just bundle adjustment") is a small library that does one thing: bundle
adjustment for camera reconstructions that arrive with pose priors, as fast and as
correctly as possible. It trades generality for speed. Anything that would make the
core slower or less predictable is left out, even if that makes other uses hard.

Existing options (PyPose `bae`, COLMAP with its GPU backend, Kornia) are general
frameworks. just-ba is not: it ships a closed set of residuals, owns the layout of the
linear system, and talks to the GPU vendor libraries directly.

## Primary use case

Another system already holds a capture session: images, keypoints, matches, camera
intrinsics, and camera pose information from ARCore, ARKit, an unknown SLAM method,
or a pose-predicting network.
Those poses are metric and gravity-aligned but can be very noisy and can drift.
Tracking sessions also break, commonly on Android. Detecting that and removing the
affected priors is done by a separate system before the library is called; the
library stays robust to the uneven quality of what remains (document 02).
Depth maps may be available too.

That system calls just-ba from Python and gets back refined poses, 3D points and
(optionally) refined depth. With global priors the result is in their frame; with
relative priors only it is in the library's own frame, and the caller aligns it to
the AR frame if it wants that.

In this mode the library is a triangulate-and-refine tool. It does not build a
reconstruction from scratch.

Processing is offline: the capture finishes, then the whole session is processed in
one batch. This is not a SLAM system and there is no online or incremental bundle
adjustment during capture.

### Problem size

Keyframe selection happens outside the library, so the input is already a chosen set
of images.

| Range | Status |
|---|---|
| 50 to 2,000 images | The main target. The "fastest" claim is made here. |
| 2,000 to 10,000 images | Supported and must work correctly. Being the fastest is a stretch, not a promise. |
| More than 10,000 images | Not a goal. |

### Scene scale

Single-building scale. Typical subjects:

- the interior of a house, the exterior of a house;
- a large commercial space, a large patio;
- at the upper end, a small to medium park.

City-block scale and beyond is not a goal and nothing is optimized for it. The
library should stay general enough that the park case works.

What this implies:

- Scene extent is tens to a few hundred metres. The working frame is local and
  metric. The only geographic handling is converting optional GPS positions into a
  local frame.
- Single precision on the GPU is comfortable at this extent once the scene is
  re-centred (document 03).
- Interiors are the hard case for geometry, not for size: plain walls give few
  keypoints, tracks are short, and rooms connect only through doorways. Pose priors
  and depth carry more weight there, and the linear system is poorly conditioned
  along those thin connections.
- Parks are the opposite: wide extent, vegetation, and phone depth sensors are of
  little use beyond a few metres, so the depth term mostly drops out.

## Secondary use case

A simple incremental reconstruction path for data without pose priors. It exists for
two reasons only:

- to test the library on standard datasets that have no ARKit or ARCore data;
- to show results on par with COLMAP.

It is not a product feature of the library and gets no optimization effort beyond
what parity needs. The planned desktop app would rely on it for bare photo sets,
which raises the bar on its reliability when the app is scheduled (document 09).

## Functional requirements

### Inputs

The library defines the input contract and documents it. Callers convert their data
to it; a thin layer on top of ARKit and ARCore on the caller's side does that,
including bringing poses and images into one agreed axis and orientation convention.
The library does not read ARKit or ARCore structures itself.

| Input | Required | Notes |
|---|---|---|
| Frames | yes | An unordered set. All frames of a session come from one camera. |
| Keypoints per frame | yes | Pixel coordinates only, computed by the caller from the image. Descriptors never enter the library. |
| Pairwise matches or ready-made tracks | yes | Feature matching is done elsewhere. |
| Camera intrinsics and model | yes | Usually known and fixed (phones report them). |
| Relative pose priors | main use case (trackers) | A list of frame pairs with the relative pose between them and an optional weight. Chosen and cleaned by the caller's system: priors that are completely wrong, such as across a tracking break, are removed before the library is called. The rest may be of uneven quality. |
| Global pose priors | main use case (drift-free sources) | A pose per frame in the source's own frame, with an optional weight. For example poses predicted by a feed-forward network such as VGGT. The frame may have arbitrary scale. |
| Gravity direction per frame | optional | Used as a prior when present. |
| Line segments per frame | optional | Two endpoints in pixels, optional score. Detected by the caller. Not matched between images. Needed for the Manhattan prior. |
| Depth map with confidence per frame | optional | For phones with LiDAR or a depth API. |
| GPS position per frame | deferred | Lower priority. The only planned source of global position. |
| Ground control points | stretch | Not in the first release. |

One camera per session (owner's decision): one intrinsics block, one camera model.
Sessions mixing several lenses are not supported, which keeps the data layout and
kernels simpler and faster.

Explicitly not assumed: capture order, timestamps, velocity, acceleration or other motion data, raw inertial
measurements, tracking-state flags, or pose uncertainty from the tracker. Because no
uncertainty arrives with the poses, the library chooses the prior weights itself
(document 02).

The images themselves are not an input to the core. They would be needed only by the
optional photometric module (document 02, section 4b).

Not supported for now: detector-free (featureless) matching, where there are no stable
keypoints.

### Residuals: a closed set

Users cannot register their own residual types. Every residual is implemented inside
the library so the solver can be specialised for it.

First release:

1. Reprojection error (the core term).
2. Pose priors, relative and global (document 02).
3. Ground control point (stretch; not in the first release).
4. Depth term for joint depth refinement. The first release uses a sparse term at
   keypoints. The photometric method in PyPose `bae` needs image pixels; it is a
   low-priority optional module for later, built only if it fits on top of the core
   without compromising it (document 02).
5. Manhattan prior, a first-class feature. Built from line detections supplied by
   the caller; the library assigns lines to the building's three axes and estimates
   the vanishing directions itself. No line matching is involved (document 02).
6. Gravity (up direction) prior.

Planned, added in-tree when needed:

7. Motion smoothness between consecutive cameras.
8. Constraints on planes or 3D lines.

New residual types are added by changing the library, not by plugging into it. This is
deliberate: block sizes, matrix structure and kernels are fixed at build time.

### Camera models

- Pinhole, with and without lens distortion.
- Wide-angle (fisheye), with distortion.
- Intrinsics can be refined, but can always be held fixed. Fixed is the default.
- Any parameter block (pose, point, intrinsics, ground control point) can be held fixed.

Model names and parameter order follow COLMAP so parity tests need no conversion.

### Track building

A helper turns pairwise matches into tracks and must tolerate wrong matches. The first
version is simple; the method is described in document 04.

### Outputs

The rule: export everything a user would need to continue with their own bundle
adjustment in another tool.

- Refined camera poses.
- The point cloud (3D points).
- Tracks: which keypoint in which image observes which point, with outlier flags.
- Uncertainty of the result: covariance of each camera pose, and of points on request
  (document 02).
- Depth results when depth is used: per-image scale and offset, and refined depth at
  keypoints.
- Intrinsics (refined if freed).
- The priors and weights that were used, so the problem can be reproduced.
- Formats: COLMAP's model format for interchange (poses, intrinsics, points, tracks),
  plus the library's own format for what COLMAP's cannot hold (priors, covariances,
  depth terms). In Python, the same data as arrays.
- Per-iteration history: cost, step quality, damping, linear solver iterations, time.
- Per-residual-type cost breakdown and outlier flags.
- No printing from inside the solver; logging goes through a callback.

## Non-functional requirements

| Requirement | Detail |
|---|---|
| Speed | Aim to be the fastest available option for this problem shape. |
| Correctness | Defined as agreement with Ceres Solver: same problem in, same final cost and parameters out, within stated tolerances. |
| Dependencies | As few as possible. No PyTorch. No intermediate framework between the library and the vendor GPU libraries. CUDA is required only when the CUDA backend is used. |
| Backends | CUDA; Apple Silicon (M1 family) GPU; CPU. |
| Platforms | Linux, macOS (Apple Silicon) and Windows. |
| Later: mobile consumer | A phone app in a separate repository will link the library through its C-level API. The library must stay small, memory-aware and buildable for iOS and Android (document 10). |
| Later: desktop app | A downloadable app for Windows and Mac for non-developers, using the GPU when present and CPU otherwise (document 09). It makes a dependable CPU backend and Windows support requirements of the library. |
| Bindings | Python first. WebAssembly plus WebGPU in the browser is a stretch goal. |
| Deliverables | The library, its Python package, and a sample command-line binary. |
| Regression control | A benchmark suite that fails when accuracy drifts from Ceres or speed regresses as features are added. |
| Distribution | Open source, public after the first working proof of concept. Published to PyPI. |
| Documentation | Generated documentation site, built in CI. |
| CI | Hosted CI for builds, CPU tests, wheels and documentation; GPU tests on separate hardware (document 08). |

### Role of the CPU backend and Ceres

The CPU path matters first for evaluating speed and accuracy. It is also the fallback
for the planned desktop app on machines without a supported GPU, so it must be
dependable and reasonably fast, though it is not where the performance claim is made.
Ceres Solver (latest release) is an optional dependency used as the reference: the
benchmark runs the same problem through both and compares.

## Non-goals

- Feature detection, description or matching. The same rule covers lines: line
  detection is the caller's, and line matching is not used at all.
- A general nonlinear least squares framework or factor graph library.
- User-defined residuals at run time.
- Automatic differentiation exposed to users.
- A full structure-from-motion product (image retrieval, geometric verification,
  loop closure, dense reconstruction).
- Pose graph optimization as a standalone feature.
- Rolling shutter. Each image has one pose. Phone sensors read out row by row, so
  with a moving phone this leaves a small error the solver cannot remove; it is a
  known accuracy limit, reduced by choosing keyframes with little motion. If it
  ever proves to be the limiting factor, the route is an optional per-frame velocity
  input used to correct each keypoint for its row's capture time.

## Open decisions

These are settled in later documents or still need the owner's call.

| Decision | Where |
|---|---|
| C++ or Rust | Decided: C++ (document 06) |
| Whether the Apple backend is GPU (Metal) or CPU (Accelerate) first | Documents 03 and 06 |
| Exact depth formulation | Document 02 |
| Absolute versus relative pose priors as the default | Document 02 |
| Single or double precision on GPU | Document 03 |
