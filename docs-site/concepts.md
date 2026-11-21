# Concepts

## Conventions

Poses map world to camera and are stored as `qw qx qy qz tx ty tz`: a unit
quaternion (Hamilton, w first) and a translation. Camera axes are x right, y down,
z forward, as in OpenCV and COLMAP. ARKit and ARCore use OpenGL camera axes (y up,
looking down -z): `poses_from_camera_to_world(T, "arkit")` converts their
camera-to-world matrices. The world frame is kept as it is.

Camera models and their parameter order follow COLMAP: SIMPLE_PINHOLE, PINHOLE,
SIMPLE_RADIAL, RADIAL, OPENCV and OPENCV_FISHEYE (see
[Camera models](_generated/cameras.md)). One camera per session is the normal case;
several intrinsics blocks of one model are accepted (`image_intrinsics`).

## The problem

A `Problem` holds poses (N, 7), points (M, 3), observations (`obs_image`,
`obs_point`, `obs_xy`, and `obs_sigma` in pixels), the intrinsics, and any priors.
Array attributes are copied in and out. `Problem.validate()` says what is wrong
with one; `read_problem` and `write_problem` use the [problem file format](file-format.md).

Blocks can be held: `image_fixed` holds a pose, `point_fixed` a point, and
`intrinsics_free` frees groups of intrinsics per block (1 focal length, 2
principal point, 4 distortion); intrinsics are held unless freed.

A freed group can be fully free or regularized: with `intrinsics_prior_sigma` (per
parameter, in its units; +inf for none) it is pulled toward `intrinsics_prior`, by
default the values it started from. This is for refining another process's
calibration without letting it wander:

```python
p.intrinsics_free = [1 | 4]                  # focal length and distortion
jba.set_intrinsics_prior(p, 0.01,            # focal within about 1 %
                         distortion_sigma=0.005)  # principal point: held anyway
jba.solve(p)                                 # writes the target used back to p.intrinsics_prior
```

Blocks of one camera can instead share a mean: with `set_intrinsics_groups` each
block of a group is tied to the group's mean (focal lengths a little apart, as
autofocus leaves them), and the mean only loosely to the calibration, so a poor
calibration is corrected by all the group's images together while one image that
sees little cannot drift off:

```python
p.intrinsics_free = [1] * p.intrinsics.shape[0]   # focal length of every block
jba.set_intrinsics_groups(p, [0] * p.intrinsics.shape[0],
                          0.005,                   # each block within about 0.5 % of the mean
                          group_focal_fraction=0.05)  # the mean within about 5 % of the calibration
jba.solve(p)                                      # writes the mean to p.intrinsics_group_mean
```

## Pose priors

- **Relative priors** (`rel_images`, `rel_pose`, `rel_sigma`) constrain the motion
  between two images, `T_j T_i^-1`. This is what a tracker such as ARKit or ARCore
  gives reliably over short spans; `add_relative_priors_from_poses` makes them from
  per-image poses. Their default standard deviations grow with the motion
  (`PriorDefaults`).
- **Global priors** (`glob_image`, `glob_pose`, `glob_set`, `glob_sigma`) constrain an
  image's pose in a prior set's frame, for drift-free sources. A set relates to the
  working frame by the identity, a rigid motion or a similarity
  (`set_relation`, `set_transform`), estimated when free.

Priors are always robust (a Cauchy loss of scale 3, in standard deviations, by
default), so a prior that disagrees with the images loses weight instead of bending
the solution; `ResidualReport` lists those. The caller chooses the pairs and removes
priors that are completely wrong.

## Gravity and the Manhattan prior

A gravity prior (`grav_image`, `grav_dir`, `grav_sigma`) gives the direction of
gravity measured in an image's camera axes; `gravity_world` is its direction in the
working frame, (0, -1, 0) by default, ARKit's and ARCore's world.

The Manhattan prior uses line segments detected in the images (`line_image`,
`line_xy`): `refine` assigns them to the axes of a Manhattan frame and, when the
scene supports it, constrains the poses' headings. It is opt-in
(`RefineOptions.use_manhattan`), since it is wrong where the scene is not Manhattan.
By default the frame is estimated from the segments. When the working frame is
already aligned with the building, set `RefineOptions.estimate_manhattan_frame =
False`: the segments are assigned to the axes of `scene.manhattan_frame`, or of
`aligned_frame(scene)` (axis 2 up, axes 0 and 1 along the working frame's
horizontal axes) when that is empty.

## Depth

A depth reading per keypoint (`refine(..., keypoint_depth=...)`, or `obs_depth` in a
problem) adds a residual between the point's depth and the reading. `depth_kind`
says whether readings are z or range, `depth_form` whether the residual is linear or
inverse, and a per-image correction `a d + b` (`depth_correction`) can be estimated
(`depth_correction_free`: `DEPTH_SCALE`, `DEPTH_OFFSET`) for sensors with a scale or
offset error.

## Gauge

Bundle adjustment leaves the global frame, and the scale when nothing metric sets
it, free. With `GaugeMode.AUTO` (the default), when no image and no point is held fixed,
the solver holds a reference pose in every group of connected images whose frame
nothing else defines, and the scale when nothing metric sets it. Where gravity sets
the vertical, only that pose's translation and its heading about the vertical are
held; with the Manhattan prior, the frame sets the rotation. `GaugeMode.NONE` leaves
the freedoms to the damping. `SolverSummary.gauge` says what was held.

## Robust losses

`SolverOptions.loss` applies to reprojection residuals (none by default; `refine`
uses a Cauchy loss of scale 2), and each prior type has its own. The losses are
Ceres's, with the same definitions and scales, so costs compare directly; the
objective is half the sum of the losses of the squared whitened residuals.
