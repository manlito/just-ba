# 02. Problem formulation

What is optimized, and the closed list of residuals. Research cited here was read on
2026-11-14; items marked "unverified" were not confirmed from a primary source.

## Variables

| Variable | Symbol | Size | Usually |
|---|---|---|---|
| Camera pose, world to camera | `T_i = (R_i, t_i)` | 6 degrees of freedom | free |
| 3D point | `X_j` | 3 | free |
| Intrinsics (one camera per session) | `K_c` | model dependent | fixed |
| Manhattan frame, when used | `R_M` | 1 or 3 | free |
| Transform of a global prior set, when it is not the working frame | rigid or similarity | 6 or 7 | free |
| Depth correction of image `i` | `(a_i, b_i)` scale and offset | 2 | fixed for metric sensors, free for network depth |
| Ground control point | `G_k` | 3 | fixed, or free with a prior |

Conventions:

- Internally poses map world to camera, with the computer-vision camera axes
  (x right, y down, z forward). This matches COLMAP, which keeps parity tests direct.
- ARKit and ARCore report camera-to-world poses with a different axis convention.
  The Python layer converts on the way in and out; the exact conversions are to be
  verified when the importer is written.
- Rotations are stored as unit quaternions (w, x, y, z; D47) and updated through a
  3-parameter tangent step. Rotation and translation are updated separately:
  `R <- Exp(dtheta) R`, `t <- t + dt` (D46).

## Objective

```
minimize   sum over all residuals   rho_k( || W_k r_k ||^2 )
```

- `r_k` is a residual from the list below.
- `W_k` whitens it (inverse square root of its covariance), so every term is in units
  of standard deviations and terms of different kinds can be added.
- `rho_k` is a robust loss, chosen per residual type.

Robust losses offered: none, Huber, Cauchy, Geman-McClure, with Ceres's definitions
(D56). Applied by iterative reweighting. As built after milestone 1 there is one
loss setting, for reprojection; one per residual type comes with the priors. Graduated non-convexity (start with a forgiving loss, tighten over
rounds) is offered as an outer schedule, because it is the published remedy for
exactly our situation of gross outliers in priors (see pose priors below).

## Residuals

### 1. Reprojection

```
r = project(K_c, R_i X_j + t_i) - x_ij          (2 values, pixels)
```

`x_ij` is the observed keypoint. Weight: keypoint standard deviation in pixels,
default 1. Loss: Huber or Cauchy by default.

For wide-angle cameras an angular variant (error between bearing vectors) is used in
triangulation and track repair; the solver itself uses pixel error for parity with
the references.

### 2. Pose priors

This is the central feature.

Owner's decision (2026-11-14): the library supports **both relative priors and
global priors**. Either can be used alone, or both together.

**Relative pose prior.** Ties two cameras to a reported motion between them. Suited
to trackers (ARKit, ARCore), whose error is small between nearby frames and drifts
over the session.

```
r = Log( T_rel_prior^-1 * (T_j * T_i^-1) )            (6 values)
```

**Global pose prior.** Ties one camera to a reported pose in some global frame.
Suited to sources whose poses are all expressed in one frame without drift, for
example a feed-forward network such as VGGT that predicts every camera at once.

```
r = [ Log( R_i * R_prior^-1 ) ;  C_i - C_prior ]     (6 values)
```

`C` is the camera centre. Rotation-only and position-only modes exist as well;
position-only is the form COLMAP uses, and is what GPS will use later.

**Gravity prior.** Ties a camera's up direction to the gravity direction reported
for that frame. Optional.

```
r = R_i g_world - g_measured                          (3 values)
```

As built (D82): the chord between the solved and the measured direction, so about
the angle in radians; `g_world` defaults to ARKit's and ARCore's world (y up).
With gravity in a group, the gauge holds the reference's position and heading only
(D83).

#### The frame of a set of global priors

A set of global priors lives in its source's own frame, which need not be the
library's working frame, and need not be metric. Network poses in particular are
consistent with each other but have an arbitrary scale.

Each set of global priors therefore declares how it relates to the working frame:

| Relation | Unknowns added | When |
|---|---|---|
| Is the working frame | none | Global priors are the only source of frame and scale. The result is in that source's frame and scale. The simplest and most common case. |
| Rigid transform | 6 | The set is metric but in a different frame from other metric information |
| Similarity transform (with scale) | 7 | The set has arbitrary scale and is combined with metric information such as relative priors from a tracker or sensor depth |

The transform, when present, is estimated in the solve and returned.

So with network poses alone, the result comes out in the network's frame and scale,
and nothing needs aligning. When network poses are combined with tracker relative
priors, the tracker fixes the metric scale and the library estimates the similarity
that places the network's frame in it.

#### Why relative priors are the default

- A tracker's pose error is mostly drift: small frame to frame, large over the whole
  session. Relative terms are accurate where the tracker is accurate. Absolute terms
  on a drifting trajectory pull the solution toward the drift.
- ARKit does not go back and correct past camera poses when it refines its map, so a
  recorded trajectory keeps its drift and can contain sudden jumps
  (source: Apple developer forum replies; forum-level evidence).
- The clearest published recipe for refining ARKit and HoloLens trajectories (the
  LaMAR benchmark) uses relative tracking terms plus reprojection in bundle
  adjustment, and robust absolute terms with graduated non-convexity in a preceding
  pose-graph step.
- COLMAP's prior is absolute, position-only, with a default of 1 m standard
  deviation. That is a design for GPS, not for phone tracking.

Defaults by source:

| Source | Terms |
|---|---|
| Phone tracker (ARKit, ARCore) | Relative priors, robust loss. Gravity when supplied. |
| Feed-forward network or other drift-free global source | Global priors, robust loss |
| Both | Both, with a similarity transform on the global set |

Which to use is the caller's choice per solve; the library does not guess the
source.

#### Where relative priors come from: the caller

Owner's decision (2026-11-14): cleaning up tracker poses is done in a separate
system, before the library is called. The library does not select pairs, does not
detect broken tracking sessions, and does not validate priors against matches.

Division of work:

| Caller's system (outside this project) | Library |
|---|---|
| Estimates co-visibility between frames | Takes the relative priors it is given |
| Chooses which frame pairs get a relative prior | Uses them as residuals with a robust loss |
| Validates tracker poses and removes the ones that are completely wrong, such as those spanning a tracking break | Tolerates the remaining ones being of uneven quality |
| Makes sure the priors it passes are correct | Builds the starting poses itself from the priors, then refines |
| Aligns the output to another frame afterwards, if it wants one | Returns the result in the frame the priors define (see Gauge) |

What the library receives is therefore a list of relative priors, each naming two
frames and the relative pose between them, with an optional weight. A convenience
helper builds that list from per-frame poses and a list of pairs, for callers whose
poses are already clean.

Capture order, timestamps and co-visibility are not library inputs. They matter to
the caller's system when it chooses and validates pairs.

#### Robustness that stays in the library

The cleanup removes gross failures. What reaches the library is assumed decent but
uneven: some priors will be noticeably worse than others.

- Every relative prior has a robust loss, so one that disagrees with the images and
  with its neighbours is down-weighted during the solve.
- Graduated non-convexity is available as a stronger setting.
- Priors that end up heavily down-weighted are reported in the output, so the
  caller's system can learn what slipped through.
- Frames left with few or no priors and few matches are poorly determined. They are
  kept, flagged, and reported with large uncertainty.

#### Starting poses and connectivity

The library builds its own starting point from whatever priors it has:

- **Global priors present:** each frame with a global prior starts at it.
- **Relative priors only:** treat frames as nodes and priors as links, and chain the
  priors outward from a reference frame along the most trusted links.
- **Both:** global priors place their frames; relative priors fill in the frames
  that have none.
- As built (D70): a group of images linked by relative priors is moved as a whole
  onto its global priors by a robust rigid fit, rather than each image being
  placed at its own global prior, so a few wrong global priors cannot pull single
  images away from their neighbours. Chaining prefers priors that agree with the
  triangles of priors they close.
- Optionally a quick pose-only refinement over all priors follows, before points
  are triangulated.

With relative priors only, the priors must connect all frames. If they fall into
several groups with no prior between them (as happens when the caller drops the
priors across a tracking break), each group is internally consistent but nothing
says where one sits relative to another.

- Global priors remove this problem for the frames they cover, since every such
  frame is placed directly.
- Otherwise, first version: the library reports the groups and solves each on its
  own (or only the largest, by option).
- Later: join groups through the image matches between them.

#### Weights when the caller supplies none

Relative priors may arrive without weights, since trackers report no uncertainty.

- The default covariance grows with the motion between the two frames (the LaMAR
  heuristic), so frames far apart get a looser tie than frames close together. The
  values were measured on ARCore captures (D77): 0.001 rad + 1.2% of the rotation,
  5 mm + 2% of the distance.
- Callers who have better information pass a weight per prior and override it.
- An option to estimate the overall prior weight from the data (how well priors and
  images agree after a first solve) is worth testing, since a fixed default cannot
  suit every tracker.

No published typical covariances for ARKit or ARCore poses were found. Default
values will be set from our own captures and from the synthetic tests.

#### GPS position prior (deferred, lower priority)

Not in the first release. It is the one planned source of fully global information,
and a common one, so the formulation is kept here for when it is taken up.

```
r = C_i - ( R_geo * g_i + t_geo )            (3 values)
```

`g_i` is the GPS reading converted to a local east-north-up frame in metres.
`R_geo, t_geo` place that frame in the library's working frame: a rotation about the
vertical and a translation, 4 unknowns, estimated once for the session.

- The working frame is unchanged. GPS does not move the result into
  geographic coordinates; the estimated transform is returned so the caller can
  georeference the output.
- Weight: metres, far weaker than the pose priors. Phone GPS helps against slow
  drift over a large outdoor scene (the park case) and is useless indoors, where it
  should be left out or will be down-weighted by its robust loss.
- This is the form COLMAP's position prior takes, so parity with it is tested here.

#### A warning from recent work

A preprint from August 2026 (arXiv 2608.21008; abstract only read, not peer reviewed)
reports that triangulating points from prior poses and then running bundle adjustment
made accurate ARKit poses slightly worse on every room tested, because the structure
triangulated from the priors carries their error. It proposes skipping triangulation
and optimizing a depth along each keypoint's ray instead.

Consequences for this plan:

- The validation set must include a "priors already good" case, with the pass
  condition that we do not make them worse.
- Triangulation and track repair are repeated after the first solve with improved
  poses (document 04).
- A per-keypoint depth parameterization is noted as a research alternative to free
  3D points. It fits naturally when depth maps are supplied. Not in the first
  release.

### 3. Ground control point (stretch, not in the first release)

A marker with a known world position `G_k`, observed in several images.

```
r_obs   = project(K_c, R_i G_k + t_i) - x_ik          (2 values per observation)
r_prior = G_k - G_k_known                              (3 values, only if G_k is free)
```

Two modes: the point is held fixed at its known position, or left free with a prior
weighted by its survey accuracy (separate horizontal and vertical values, as mapping
tools do). Ground control observations are trusted more than ordinary keypoints and
get their own weight and loss.

### 4. Depth

#### What the reference does, and why it does not transfer directly

The RGB-D refinement example in PyPose `bae` (on its `rgbd` branch) was read in
full. It is not a depth residual:

- It optimizes camera poses only. Depth is a constant input. There is no per-image
  depth scale or offset and there are no free 3D points.
- The residual is photometric: pixel brightness differences between an image and
  another image warped through the depth map.
- It needs the images themselves, sampled every iteration, and no keypoints.

That is a different kind of method from the rest of this library, which takes
keypoints and never sees pixels. So "depth refinement like bae" splits into two
separate features.

#### 4a. Sparse depth term (first release)

At each keypoint that has a depth reading `d_ij`:

```
r = ( depth_i(X_j) - (a_i * d_ij + b_i) ) / sigma_ij          (1 value)
```

- `depth_i(X_j)` is the depth of point `X_j` in camera `i` (along the optical axis,
  or along the ray if the sensor reports range; a flag selects which).
- `a_i, b_i` are the per-image scale and offset. For metric sensors (LiDAR, ARCore
  depth) they are fixed at 1 and 0. For network depth, which is often only correct up
  to scale, they are free with a prior toward 1 and 0.
- `sigma_ij` grows with depth and uses the sensor's confidence value when given.
- An inverse-depth form `1/depth - 1/d` is offered for far or network depth, as used
  by InstantSfM.
- Loss: Cauchy. Phone depth has long-tailed errors at object edges.

As built (D87): the comparison is made in the reading's units,
`r = ((depth_i(X_j) - b_i) / a_i - d_ij) / sigma_ij`, and `a_i / (depth - b_i) - 1 / d_ij`
for the inverse form. The two forms agree with the ones above when a = 1, b = 0;
with a free scale the form above is biased (the scene shrinks to shrink the
residuals), this one is not. The report gives the refined depth at each keypoint and
the corrected reading.

What "refinement" means here: the points get depths consistent with all views and
the depth readings; each image gets a corrected scale and offset; the caller can
apply that correction to the full depth map.

This is the formulation other keypoint-based systems use: kornia-rs (linear and
log-depth forms with a per-view scale), MP-SfM (per-point depth with a Cauchy loss
and a per-image scale), InstantSfM (inverse depth).

#### 4b. Dense photometric refinement (low priority, optional, later)

Owner's decision (2026-11-14): the keypoint depth term 4a is the first-release
feature. Photometric refinement is wanted eventually but is low priority, and only
if it fits what the core already does. Core bundle adjustment stays first class and
is never slowed or complicated to make room for it.

The bae-style method, for reference:

```
r = I_i(p) - I_j( project(K_j, T_j * T_i^-1 * backproject(K_i, p, D_i(p))) )     (1 value)
```

A pixel `p` of image `i` is lifted with its depth, moved into image `j`, and the
brightness there is compared. Only the two poses are free.

How well it fits:

| Aspect | Fit |
|---|---|
| Linear system | Good. Each residual couples two camera poses and no points, the same shape as a relative pose prior. It adds pose-to-pose blocks to the reduced system and nothing new. |
| Outer loop, damping, robust loss, backends' vector operations | Reused unchanged |
| Joint solve with reprojection, priors and depth | Possible in one system, since everything shares the pose blocks |
| Inputs | Does not fit. Needs the images and full depth maps, not keypoints. An optional second input path. |
| Kernels | New. Image and gradient sampling on the device, per-pixel visibility checks, an image pyramid. |
| Linearization | Different rhythm. Images are re-sampled at each relinearization and held fixed between, so the residual is not a pure function of the parameters the way the others are. |
| Validation | Harder. A lower photometric error does not guarantee better poses; the bae example itself says so and guards each step with a held-out check. |

Conclusion: it can be an optional module on top of the core, entering the solver as
one more pose-to-pose residual type, without changing the core's design. It is its
own milestone, after the GPU backends, and is skipped if 4a proves sufficient on
real captures.

What the core does now to keep that door open, at no cost:

- The residual table allows pose-to-pose terms between arbitrary image pairs, which
  relative priors already require.
- The backend interface allows per-image buffers to be uploaded (depth maps need
  this for sampling anyway).
- Linearization is a separate step from the linear solve, so a residual type may
  refresh its own data when it is relinearized.

#### Sensor notes

- ARKit LiDAR depth is 256 x 192 with a three-level confidence map. Studies report
  usable accuracy to roughly 5 m and long-tailed noise at edges.
- ARCore depth comes from motion (plus a time-of-flight sensor when present), is
  best between 0.5 m and 5 m, and its raw form carries a per-pixel confidence.
- No authoritative noise-versus-range curve was found for either. `sigma` defaults
  will come from our own data.

### 5. Manhattan prior from line detections (first class)

Owner's decision (2026-11-14): the Manhattan prior is a first-class feature, for
interiors above all and also for house exteriors.

#### The idea

In a built scene most straight edges run along three perpendicular directions: one
vertical and two horizontal. Every image that sees such edges therefore carries
direct evidence of its own rotation relative to the building. That evidence does not
drift, which is exactly what tracker poses lack.

#### Input: line detections, not line matches

The caller passes line segments per frame, each as two endpoints in pixels, with an
optional score. Detection is done outside the library, like keypoint detection.

No matching of lines between images is needed, and none is asked for. A vanishing
direction is a property of the scene, shared by every image, so each line only has
to be assigned to one of the three axes or to "none". That is a classification the
library does itself from the geometry. Line matching, which is not a solved problem,
never enters.

#### Variables

One Manhattan frame per scene: the rotation `R_M` from the building's axes to the
working frame.

- When the working frame is gravity-aligned (tracker poses or a gravity input), the
  building's vertical is the frame's vertical, and `R_M` reduces to a single angle:
  the heading of the walls about the vertical axis. This is the common case, and
  matches the owner's example of correcting a small rotation about the vertical.
- Otherwise `R_M` is a full rotation with 3 degrees of freedom.

#### Residual

For a line segment in image `i` assigned to axis `k`:

```
n   = unit normal of the plane through the camera centre and the segment
      (cross product of the two endpoint bearing vectors)
d_k = R_M * e_k                      the axis direction in the working frame
r   = n . ( R_i * d_k ) / sigma      (1 value)
```

If the segment really runs along axis `k`, that axis direction lies in the plane and
`r` is zero. `r` is the sine of the angle by which it misses.

- Endpoints are converted to bearing vectors through the camera model, so lens
  distortion and fisheye cameras are handled with no special case.
- `sigma` shrinks with segment length: long segments fix a direction precisely,
  short ones do not.
- Loss: robust, always. Some lines will be wrongly assigned.

The same residual with the vertical axis gives a gravity constraint from the image
itself, which complements the gravity input or stands in for it when absent.

#### What it constrains, and what it costs

- It constrains each camera's rotation, and the Manhattan frame. Not translations,
  not points.
- In the linear system each camera gains a term on its own block plus a coupling to
  one tiny shared block (1 or 3 unknowns). That is the same shape as shared
  intrinsics and adds almost nothing to the solve. No new elimination step, no
  change to the Schur complement.
- Per-image vanishing points are not separate unknowns. They follow from `R_i` and
  `R_M`, and can be reported for display or checking.

#### Assignment and initialization (inside the library)

1. Using the prior poses, express every segment's possible direction in the working
   frame. Vertical candidates are those consistent with the vertical axis.
2. Estimate the wall heading by voting: each roughly horizontal segment votes for a
   heading modulo 90 degrees; the strongest peak wins.
3. Assign each segment to the axis it fits within a threshold, if exactly one fits.
   Otherwise mark it "none".
4. Solve. Then repeat the assignment with the improved poses and a tighter
   threshold, a small number of rounds.

#### When the scene is not Manhattan

This prior is strong, so it must not be applied where it is false.

- It is opt-in per solve, and never on by default.
- A support test runs first: the share of segments explained by the best frame. Below
  a threshold the prior is refused with a clear message.
- Lines that fit no axis (sloped roofs, stairs, furniture, vegetation) are left out,
  not forced.
- Buildings with wings at different angles need several horizontal headings sharing
  one vertical. The data layout allows more than one heading; the first version
  supports one, with the others treated as "none".
- A park has little use for it. Indoors and around houses it is expected to be one
  of the most effective terms in the library.

#### Interaction with other terms

- With relative pose priors, drift in heading is the dominant error over a long
  indoor walk. The Manhattan term anchors heading in every image that sees walls, so
  the two are complementary: relative priors for local accuracy, Manhattan for
  global rotation.
- The gauge is unaffected. `R_M` is free and is returned. The caller can use it to axis-align the model, which is
  useful for floor plans.
- If intrinsics are being refined, endpoint bearings change with them. First
  version: the Manhattan term requires fixed intrinsics, the normal case for phones.

#### Not included

- 3D line reconstruction, which would need line matches.
- Constraints on planes or on the point cloud (walls being flat or perpendicular).
  Those need plane detection and belong to a later discussion.

Prior art read: LIMAP uses the same sine-of-angle residual between a line and a
vanishing direction, with a soft perpendicularity term between vanishing-point pairs.
The single shared frame used here is the stricter, classical Manhattan form.

#### As built (milestone 3)

- Residual and weights: D84. Assignment, the vote and the support test: D85
  (`assign_lines` in `manhattan.h`). The gauge: the frame is held and each group
  with segments holds only its reference's position (D86).
- In `refine` (opt-in, `use_manhattan`), segments are assigned after the prior-only
  solve, the priors are solved again with them, then assigned again each round
  with a tighter angle.
- Synthetic results (document 07): heading drift in a room and in two rooms joined
  by a doorway is removed far better than by the images alone; random segments are
  refused; a mixed scene is not made worse.
- At the pose-only stage the relative priors still pull, since their default
  weights (D77) trust a tracker more than the synthetic one deserves: there the
  drift is about halved, and the images finish the job.

### 6. Planned residuals

Not in the first release; formulations recorded so the data layout leaves room.

- **Motion smoothness.** Constant velocity over three consecutive cameras. Couples
  three poses, so it widens the band in the reduced system.

## Camera models

Names and parameter order follow COLMAP.

First release:

| Model | Parameters | Use |
|---|---|---|
| SIMPLE_PINHOLE | f, cx, cy | Pinhole |
| PINHOLE | fx, fy, cx, cy | Pinhole |
| SIMPLE_RADIAL | f, cx, cy, k | Pinhole, one radial term |
| RADIAL | f, cx, cy, k1, k2 | Pinhole, two radial terms |
| OPENCV | fx, fy, cx, cy, k1, k2, p1, p2 | Pinhole, radial and tangential distortion |
| OPENCV_FISHEYE | fx, fy, cx, cy, k1, k2, k3, k4 | Wide angle |

OPENCV, with `u = X/Z`, `v = Y/Z`, `r2 = u^2 + v^2`:

```
radial = k1*r2 + k2*r2^2
du = u*radial + 2*p1*u*v + p2*(r2 + 2*u^2)
dv = v*radial + 2*p2*u*v + p1*(r2 + 2*v^2)
x = fx*(u + du) + cx
y = fy*(v + dv) + cy
```

OPENCV_FISHEYE, with `r = sqrt(u^2 + v^2)`, `theta = atan(r)`:

```
theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
x = fx * theta_d * u / r + cx
y = fy * theta_d * v / r + cy
```

Limit: this fisheye model requires the point in front of the camera, so the field of
view is below 180 degrees. That covers phone ultra-wide lenses. Lenses beyond 180
degrees need a different model.

Candidates for later: FULL_OPENCV, THIN_PRISM_FISHEYE, EUCM.

### Intrinsics

- Fixed by default.
- When refined, three groups can be freed independently: focal length, principal
  point, distortion. (COLMAP's GPU backend cannot free focal length and distortion
  separately; ours should.)
- Images sharing a physical camera share one intrinsics block.
- An optional prior keeps refined intrinsics near their starting values.

## Gauge

Which global freedoms remain, and how they are removed:

| Situation | Free directions | Handling |
|---|---|---|
| Global priors used as the working frame | none | Nothing to fix. The result is in the priors' frame and scale. |
| Relative priors only | position and rotation of the whole scene; scale is fixed because the priors are metric | The library fixes them by convention |
| Relative priors with gravity (input, or vertical lines) | position, and heading about the vertical | Vertical axis follows gravity; the rest by convention |
| Relative priors plus a global set with its own transform | as for relative priors; the transform absorbs the difference | By convention; the transform is returned |
| No priors (incremental path) | position, rotation and scale | Hold one camera fixed plus one more constraint for scale, as COLMAP does |

Convention when the frame is free: the reference frame's camera defines the origin,
and the vertical axis follows gravity when it is known. The exact rule is documented
with the output.

**Output frame.** With global priors as the working frame, the output is already in
that frame. With relative priors only, the output is in the library's own frame, and
aligning it to the AR session's frame is the caller's job: a single rigid transform,
for which a helper can be provided.

## Uncertainty of the result

The library reports how certain the refined values are, as covariances taken from
the linear system at the solution.

| Output | How | Cost |
|---|---|---|
| Covariance of each camera pose (6 x 6) | From the inverse of the reduced camera system | One extra factorization and partial inverse at the end. Affordable in the main size range, where that system is at most about 12,000 unknowns. |
| Covariance of selected points (3 x 3) | Back-substitution through the point blocks | On request, for chosen points or all |
| Full joint covariance | Not offered | Too large to be useful |

Points to settle when it is implemented:

- Covariances are only meaningful when the input weights are meaningful. With
  default or guessed prior covariances the output is relative, not calibrated. The
  documentation must say so.
- Covariance depends on the gauge. With global priors the values are in the priors'
  frame. Otherwise they are relative to the reference frame chosen by the gauge
  convention, which must be stated with the output.
- The reference for testing is Ceres's covariance estimation on the same problem.
- This is a reason, besides robustness, to have the direct factorization path: the
  iterative solver alone cannot produce covariances cheaply.

## Conditioning

Before solving, the scene is shifted and scaled to unit size and pixel coordinates
are taken relative to the principal point. Covariances are transformed to match, and
the result is mapped back afterwards. This matters for single precision
(document 03).

## Open decisions

| Decision | Recommendation |
|---|---|
| Whether to build photometric refinement (4b) at all | Decided: 4a first; 4b is optional and low priority. Revisit after seeing 4a on real data. |
| Default prior mix | Decided: relative and global priors both supported; the caller chooses per solve |
| Pose step parameterization | Decided: rotation and translation separately, left rotation-vector step (D46) |
| Graduated non-convexity on by default for priors | Decide from the synthetic jump tests |
| Per-keypoint depth along the ray instead of free 3D points | Research item, not in the first release |
