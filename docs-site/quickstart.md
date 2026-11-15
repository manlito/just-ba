# Quick start: refining a phone capture

The usual input is a capture from ARKit or ARCore: a camera-to-world matrix per
image from the tracker, keypoints in each image, and matches between images from
your own matcher. `refine` turns the tracker's motion into relative pose priors,
builds tracks from the matches, triangulates, and solves, with rounds of outlier
removal.

```python
import numpy as np
import just_ba as jba

# From the capture (your code):
T = ...          # (N, 4, 4) camera-to-world matrices from ARKit / ARCore
keypoints = ...  # N arrays (n_i, 2): keypoints of image i, pixels
matches = ...    # (M, 4) int32: image_a, keypoint_a, image_b, keypoint_b
K = [fx, fy, cx, cy]  # PINHOLE intrinsics, pixels

# The tracker's poses, in the library's conventions (world to camera, OpenCV axes).
tracked = jba.poses_from_camera_to_world(T, "arkit")

# The camera, and relative priors between each image and the next three.
scene = jba.make_problem("PINHOLE", K, poses=tracked)
n = len(tracked)
pairs = np.array([(i, i + d) for i in range(n) for d in (1, 2, 3) if i + d < n], dtype=np.int32)
jba.add_relative_priors_from_poses(scene, tracked, pairs)

result = jba.refine(scene, keypoints, matches=matches)

refined = result.problem               # poses, points and inlier observations
print(result.summaries[-1])            # the last solve
print(int(result.refined.sum()), "of", n, "images refined")
camera_to_world = jba.camera_to_world_from_poses(refined.poses, "arkit")
```

What to look at afterwards:

- `result.refined`: per image, whether it has enough inlier observations (30 by
  default, `RefineOptions.min_observations`) for its pose to come from the images
  rather than from the priors alone.
- `result.report`: priors the robust losses down-weighted, which points at places
  where the tracker jumped.
- `jba.covariance(result.problem)`: pose uncertainties.
- `jba.export_colmap(result.problem, "model/")` writes a COLMAP text model;
  `jba.write_result` writes the problem file with the results.

The priors' weights come from `SolverOptions.prior_defaults` unless the problem gives
its own (`rel_sigma`). Gravity (`grav_image`, `grav_dir`), line segments for the
Manhattan prior (`line_image`, `line_xy`, with `RefineOptions.use_manhattan`) and a
depth reading per keypoint (`refine(..., keypoint_depth=...)`) are optional inputs;
see [Concepts](concepts.md).

## A problem that is already built

When the points and observations already exist (a BAL file, or your own
triangulation), `solve` runs the solver on the problem in place:

```python
problem = jba.read_bal("problem-49-7776-pre.txt")
options = jba.SolverOptions()
options.backend = jba.Backend.CPU
summary = jba.solve(problem, options)
print(summary.termination, summary.initial_cost, "->", summary.final_cost)
```
