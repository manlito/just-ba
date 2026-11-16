# just-ba

Minimal, fast bundle adjustment, with Python bindings. Its main job is refining
reconstructions that come with noisy camera poses from ARKit, ARCore or another
tracker: the tracker's poses become pose priors, and the solver refines poses,
points and intrinsics against the keypoints.

- **Backends**: the CPU (double precision, the reference), Metal on Apple Silicon and
  CUDA on NVIDIA GPUs (single precision, within tolerance of the CPU).
- **Correct means agreeing with Ceres Solver** on the same problem: every residual,
  Jacobian and solve is checked against it.
- **Priors**: relative and global pose priors, gravity, a Manhattan prior from line
  segments, and depth at keypoints, all robust.
- **Not in scope**: feature detection, matching, and cleaning up the tracker's poses
  belong to the caller.

```{toctree}
:maxdepth: 2
:caption: Guide

install
quickstart
concepts
backends
cli
file-format
```

```{toctree}
:maxdepth: 2
:caption: Reference

_generated/api
_generated/residuals
_generated/cameras
_generated/benchmarks
references
```
