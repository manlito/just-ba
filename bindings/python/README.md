# just-ba

Minimal, fast bundle adjustment, with Python bindings. Its main job is refining
reconstructions that come with noisy camera poses from ARKit, ARCore or another
tracker: the tracker's poses become relative or global pose priors, and the
solver refines poses, points and intrinsics against the keypoint observations.

- Backends: CPU (double precision, the reference), Metal on Apple Silicon, CUDA on
  NVIDIA GPUs (single precision on the GPUs).
- Correctness means agreeing with Ceres Solver on the same problem.
- Camera models SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV and
  OPENCV_FISHEYE, with COLMAP's parameter order.
- Priors: relative and global pose priors, gravity, a Manhattan prior from line
  segments, and depth at keypoints.

## Install

| Package | Platforms | Backends |
|---|---|---|
| `pip install just-ba` | Linux x86_64, macOS arm64 (13 or later), Windows x86_64 | CPU; Metal on macOS |
| `pip install just-ba-cu12` | Linux x86_64, Windows x86_64 | CPU and CUDA 12 |
| `pip install just-ba-cu13` | Linux x86_64, Windows x86_64 | CPU and CUDA 13 |

Install one of them: all three provide the `just_ba` module. The CUDA packages
install NVIDIA's cuDSS, cuBLAS and NVRTC from pip and need an NVIDIA driver.
Python 3.12 or later.

```python
import just_ba
print(just_ba.__version__, just_ba.compiled_backends())
print(just_ba.backend_available(just_ba.Backend.CUDA))
```

## Use

`just_ba.refine(scene, keypoints, matches=...)` is the main entry point: a
`Problem` holding the camera and the pose priors, one `(n, 2)` keypoint array per
image, and pairwise matches `(m, 4)`. `just_ba.poses_from_camera_to_world(T,
"arkit")` converts ARKit or ARCore camera-to-world matrices to the library's
poses. `just_ba.solve(problem, options)` solves a `Problem` that already has its
points and observations.

Licence: Apache-2.0.
