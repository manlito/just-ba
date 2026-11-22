# just-ba

Minimal, fast bundle adjustment, with Python bindings. Its main job is refining
reconstructions that come with noisy camera poses from ARKit, ARCore or another
tracker: the tracker's poses become pose priors, and the solver refines poses,
points and intrinsics against the keypoints.

:::{important}
**Look at [bae](https://github.com/pypose/bae) first.** just-ba follows the direction that bae, "Bundle
Adjustment in the Eager Mode" (Zitong Zhan, Huan Xu, Zihang Fang, Xinpeng Wei, Yaoyu Hu and Chen Wang, IEEE
T-RO 2026), showed: fast bundle adjustment on the GPU, as a library you call. just-ba is a separate and much
narrower implementation, in C++ with hand-written kernels and no PyTorch, for one job (refining a tracker's
poses); none of bae's code is used, and its solver follows Ceres Solver, its correctness reference. The idea,
and the credit for it, are bae's authors'. For bundle adjustment inside PyTorch, with automatic
differentiation and the general problems bae handles, bae is the library to depend on.
See [References and acknowledgements](references.md).
:::

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
demo
python-pipeline
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
