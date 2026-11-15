# just-ba

A minimal, high-performance bundle adjustment library in C++ with Python bindings,
for refining reconstructions that come with noisy camera poses from ARKit, ARCore
or another tracker. Early development: milestones 0 to 4 done, the main use case (`jba::refine`)
with gravity, the Manhattan prior from line segments, depth at keypoints, covariances
and export to COLMAP, checked against Ceres and synthetic ground truth, on the CPU and
on Apple GPUs (Metal) and NVIDIA GPUs (CUDA, milestone 5: met on Linux, Windows check outstanding).
The plan is in [docs/](docs/README.md).

## Build

Needs CMake 3.24 or later and a C++23 compiler. On macOS the Metal backend is built
by default.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build --parallel
```

```bash
ctest --test-dir build --output-on-failure
```

## Python bindings

The bindings use nanobind, needed at build time only, and the stable ABI from
Python 3.12. Set up a virtual environment with the build and test requirements:

```bash
uv venv --python 3.12 bindings/python/.venv && uv pip install --python bindings/python/.venv/bin/python -r bindings/python/requirements.txt
```

```bash
cmake -S . -B build -DJBA_BUILD_PYTHON=ON -DPython_EXECUTABLE=$PWD/bindings/python/.venv/bin/python
```

The package is assembled in `build/bindings/python/just_ba` and tested by `ctest`
(label `python`). To use it from that build:

```bash
PYTHONPATH=build/bindings/python bindings/python/.venv/bin/python -c "import just_ba; print(just_ba.CameraModel.OPENCV)"
```

Or install it like a released package, built through `pyproject.toml`
(scikit-build-core; CPU, and Metal on macOS):

```bash
uv pip install --python bindings/python/.venv/bin/python .
```

`uv build --wheel` writes the wheel to `dist/`; set `MACOSX_DEPLOYMENT_TARGET=13.0`
on macOS for a wheel that runs on macOS 13 and later.

The main entry point is `just_ba.refine(scene, keypoints, matches=...)`: a `Problem`
holding the camera and the pose priors, one `(n, 2)` keypoint array per image, and
pairwise matches `(m, 4)`. `just_ba.poses_from_camera_to_world(T, "arkit")` converts
ARKit or ARCore camera-to-world matrices to the library's poses. Optional inputs:
gravity priors (`grav_image`, `grav_dir`), line segments for the Manhattan prior
(`line_image`, `line_xy`, with `RefineOptions.use_manhattan`), and a depth reading per
keypoint (`refine(..., keypoint_depth=...)`). `just_ba.covariance(problem)` gives pose
and point covariances; `export_colmap` and `write_result` write the result.

## Reference solver (test-only)

Ceres 2.2.0 is built separately, with Eigen only, and never linked into the library.

```bash
cmake -DPREFIX=$PWD/build/ceres-2.2.0 -P third_party/ceres/build.cmake
```

```bash
python3 tests/data/fetch_bal.py
```

```bash
cmake -S . -B build -DJBA_WITH_CERES=ON -DCMAKE_PREFIX_PATH=$PWD/build/ceres-2.2.0
```

## Generated residuals

Residuals are written once in `residuals/symbolic/` and emitted for each target
into `residuals/generated/`, which is checked in. To regenerate:

```bash
uv venv --python 3.12 residuals/symbolic/.venv && uv pip install --python residuals/symbolic/.venv/bin/python -r residuals/symbolic/requirements.txt
```

```bash
residuals/symbolic/.venv/bin/python residuals/symbolic/generate.py
```

## Command line

```bash
build/cli/just-ba convert-bal problem.txt problem.jba
```

```bash
build/cli/just-ba info problem.jba
```

```bash
build/cli/just-ba solve problem.jba --loss huber:2 --output solved.jba --history history.json
```

Run `build/cli/just-ba` without arguments for every option.

## GPU backends (Metal, CUDA)

The solve can run on the GPU: `SolverOptions::backend = BackendKind::Metal` (Apple
GPUs) or `BackendKind::Cuda` (NVIDIA GPUs) in C++, `options.backend =
just_ba.Backend.METAL` or `CUDA` in Python, or:

```bash
build/cli/just-ba solve problem.jba --backend metal
```

Both work in single precision on the GPU and take every decision in double on the
host; their answers agree with the CPU's to single-precision tolerance, not bit for
bit (document 03, D91). They share one kernel source and one host side (D96). Their
`sparse_schur` forms the reduced camera system and factorizes it, with Accelerate
on Metal (its default, D92) and with NVIDIA cuDSS on CUDA when built with it (D98).
`ctest -L gpu` compares each GPU backend built with the CPU backend on every parity
problem. `python3 bench/gpu_check.py` runs the whole GPU check by hand: build, GPU and
Python tests, and the benchmark against this machine's baselines (`bench/README.md`).

Metal is built by default on macOS. CUDA needs the CUDA toolkit, 12 or later (the
CUDA sources are C++20, D113), and is built for the GPU of the build machine unless
`CMAKE_CUDA_ARCHITECTURES` is set:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJBA_WITH_CUDA=ON
```

For `sparse_schur` on CUDA, add `-DJBA_WITH_CUDSS=ON -Dcudss_DIR=<cuDSS>/lib/cmake/cudss`
(cuDSS 1.0 or later, NVIDIA's redistributable).

## Parity with Ceres

With the reference solver enabled, `build/tests/ceres/jba_parity <problem>` solves
the same problem with both solvers and checks the agreement (document 05).

## Benchmark

`bench/run.py` solves the BAL problems and a few synthetic ones and compares
iteration counts, final costs and time per iteration with a stored baseline of the
same machine (`bench/README.md`, D81):

```bash
python3 bench/run.py --build build --compare bench/baseline/apple-m5pro.json
```

`--backend metal` benchmarks the Metal backend, and `--ceres` adds Ceres's times
on the same problems.

## Real captures

`tests/real` runs `refine` from Python on every capture file it finds in `tests/data`
or `tests/data/real` (both git-ignored), or in `JBA_REAL_CAPTURES`, and skips when
there are none (D78). With the Python bindings built:

```bash
ctest --test-dir build -L real -V
```

## Licence

Apache-2.0 (the licence file is added before the repository goes public).
