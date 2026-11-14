# 07. Roadmap

Milestones are ordered so that correctness is established before speed, and the main
use case (pose priors from a phone) is reached as early as possible. No dates: each
milestone has an exit condition instead.

## Milestone 0: decisions and spikes

- Language is decided: C++ (document 06). Set up the build (C++20, CMake) on Linux,
  macOS and Windows.
- Spike the symbolic tool on the reprojection residual: derive, emit for CPU and one
  GPU target, check against finite differences.
- Build Ceres 2.2.0 minimally and run it on one BAL problem from our harness.
- Fix the on-disk problem format used by the CLI and tests.

Exit: language chosen, one generated residual verified, Ceres runs in CI.

Results (2026-11-14, measured on the owner's M5 Pro):

- Build: C++23 (D52), CMake, Ninja or the platform default. First CI run
  (2026-11-14, manual trigger, D49) passed on Ubuntu 24.04 (GCC), macOS 15
  (Apple clang) and Windows Server 2025 (MSVC 14.51), with no compiler warnings.
- Symbolic tool: wrenfold (D45). Reprojection residual for SIMPLE_PINHOLE,
  PINHOLE, SIMPLE_RADIAL and RADIAL, emitted for the CPU and for Metal. Jacobians
  match central finite differences on 864 cases including edge cases (worst
  relative error 6e-10). 119 to 241 operations per observation with all
  Jacobians.
- Metal: the kernels give bit-identical results to the CPU code instantiated in
  float. Against double, Jacobians agree to about 3e-7 relative, residuals to
  about 1e-4 px. Moving the principal point to the origin only lowers that to
  about 0.8e-4 px: the floor is float rounding of pixel values of a few hundred.
  Milestone 4 should not expect conditioning alone to fix residual precision.
- Ceres 2.2.0 (minimal, Eigen only) solves BAL ladybug-49 (49 images, 31,843
  observations) to convergence in 31 iterations, 0.48 s. Its initial cost agrees
  with just-ba's CPU evaluation to 5e-15, which also checks the BAL conversion and
  the residual against an independent Ceres functor.
- Problem file format: document 11 (D48).

- Ceres in CI: the Linux job builds Ceres 2.2.0 (about 8 minutes, cached
  afterwards) and solves ladybug-49. Exit condition met.
- The Metal test also passed on the hosted macOS 15 Apple Silicon runner. One run is
  not yet proof of reliability (document 08), so it stays informational for now.
- One full CI run used about 24 minutes of the private-repository quota (macOS
  counts 10x, Windows 2x); with the Ceres cache warm, about 15.

## Milestone 1: CPU reference, plain bundle adjustment

- Problem data structures, block fixing, ordering.
- Reprojection residual; camera models SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL.
- Several intrinsics blocks of one model, one per image at most (D53), so BAL runs
  as-is; one block stays the fast path.
- Levenberg-Marquardt loop, implicit Schur with conjugate gradients, block-Jacobi
  preconditioner.
- Robust losses.
- Gauge fixing for problems without priors.
- CLI: load BAL, solve, write history as JSON.
- Builds and tests on Linux, macOS and Windows from the start.
- Progress and cancellation callback in the solver API.
- Level 1 and Level 2 tests against Ceres (document 05).

Exit: matches Ceres final cost on the BAL test set in double precision.

## Milestone 2: the main use case on CPU

- Relative and global pose prior residuals; transform for a global prior set.
- Starting poses built from the priors; reporting of unconnected groups.
- Relative priors taken as a caller-supplied list; helper to build the list from
  poses and pairs; reporting of down-weighted priors.
- Remaining camera models: OPENCV, OPENCV_FISHEYE.
- Triangulation from prior poses.
- Track building, stages 1 and 2 (document 04).
- Outlier filtering and re-solve loop.
- Synthetic generator with drift, jumps and outliers.
- Python bindings, first version.

Exit: a real ARKit or ARCore session goes from keypoints, matches and noisy poses to
a refined reconstruction through the Python API, and the synthetic tests recover
ground truth within tolerance.

This is the proof of concept. The repository goes public after it, with the licence
and contribution files in place and hosted CI switched on (document 08).

## Milestone 3: Manhattan prior and depth

- Manhattan prior from line detections: line input, axis assignment, heading
  initialization, the residual, the support test (document 02, section 5).
- Gravity prior.
- Depth residual and per-image depth correction (document 02).
- Per-residual-type cost reporting.
- Result covariances for poses and selected points.
- Export: COLMAP model format plus the library's own format, complete enough for a
  user to continue with their own bundle adjustment.

Exit: on synthetic interiors with heading drift, the Manhattan prior removes the
drift and is refused on non-Manhattan scenes; depth-assisted refinement beats
pose-and-reprojection-only refinement on synthetic data with known truth; all match
Ceres on the same formulation.

## Milestone 4: Apple backend (Metal)

- Developed on the owner's Apple Silicon machine, the main development machine, so
  the first GPU backend has a fast edit-run-compare loop.
- Backend interface frozen from the CPU experience.
- Metal kernels from the symbolic source, single precision; host-side double for
  decisions. Metal has no double precision, so this milestone is where the
  precision design is proven.
- Deterministic gather-based accumulation.
- Accelerate for the optional direct solve.
- macOS wheels.

Exit: same answers as the CPU reference within tolerance on an M1-class machine,
faster than Ceres with Accelerate on the same machine.

## Milestone 5: CUDA backend

- Developed on the owner's laptop (NVIDIA GPU, 8 GB), which has both Linux and
  Windows. Linux is the primary environment for CUDA work and for building the
  competitors; Windows is used to confirm the native Windows build and tests.
- A Google Cloud VM with an NVIDIA L4 (24 GB) can be started by the owner for
  development sessions: interactive work and manual benchmark runs, including the
  large size class. It is not used for CI.
- Mostly a port: the kernels, precision handling and backend interface already work
  on Metal.
- All kernels in single precision by default; double as an option.
- Speed benchmark against Ceres, Caspar and `bae`.
- Explicit reduced system with a direct solver as an option; full-system conjugate
  gradients as an experiment.

Exit: same answers as the CPU reference within single-precision tolerance, and faster
than every competitor at equal accuracy on the small and medium size classes
(50 to 2,000 images). If it is not faster there, this milestone is not finished.
The large class (up to 10,000 images) must be correct and competitive.

## Milestone 6: release to PyPI

Details in document 08.

- Wheels for Linux (CPU and CUDA) and macOS, published to PyPI from CI.
- Benchmark results stored per commit; gates on accuracy and time per iteration.
- Generated documentation site: user guide, Python API reference, residual and camera
  model reference, benchmark tables.
- A one-command GPU test and benchmark run, done by hand before each release. No
  scheduled or paid CI jobs.

Exit: `pip install` works on the three platforms, the documentation site is live, and
CI blocks regressions.

## Milestone 7: incremental pipeline and COLMAP parity

- Minimal incremental reconstruction (document 04, part C).
- Level 3 comparison with COLMAP 4.2.1.

Exit: within the agreed margins of COLMAP on the chosen scenes.

This sits late on purpose. It is a validation tool, and the solver it validates must
exist first. It can be pulled earlier if parity evidence is needed sooner.

## Later

- GPS position prior (deferred; the only planned global information).
- Ground control points (stretch).
- Joining unconnected groups of frames through image matches.
- Desktop app for Windows and Mac (document 09). Depends on milestones 4 to 7. For
  bare photo sets it also needs a bundled feature matching component and a more
  robust incremental pipeline than parity testing alone requires.
- Photometric depth-based pose refinement as an optional module (document 02,
  section 4b). Low priority. Built only if the keypoint depth term proves
  insufficient on real captures, and only as an add-on to the core.
- Headline benchmark numbers on a datacenter GPU, since laptop timings are noisy
  under thermal throttling. Run by hand on the development VM (NVIDIA L4).
- Several Manhattan headings sharing one vertical, for buildings with angled wings.
- Plane and 3D line constraints.
- Stronger preconditioners.
- Track completion and merging.
- WebAssembly and WebGPU backend.

## Decisions needed from the owner

| # | Decision | Recommendation | Document |
|---|---|---|---|
| 1 | Language | Decided: C++ | 06 |
| 2 | Pose prior forms | Decided: relative and global priors both supported, supplied and pre-cleaned by the caller; robust loss in the library | 02 |
| 3 | Depth formulation | Decided: per-point depth residual with per-image scale and offset first; photometric refinement optional and later | 02 |
| 4 | Order of GPU backends | Decided: CPU, then Metal, then CUDA | 03 |
| 5 | Is the incremental pipeline needed before the GPU backends | No | this document |
| 6 | Can the existing system export real sessions for a private test set | Needed for milestone 2 | 05 |
| 7 | Open source licence | Decided: Apache-2.0 | 08 |
| 8 | GPU test hardware | Decided: the owner's Mac for Metal; laptop (Linux and Windows) for CUDA development; a Google Cloud L4 VM for development sessions. All GPU testing is by hand; CI stays free, with no nightly jobs | 08 |

## Main risks

| Risk | Effect | Response |
|---|---|---|
| Single precision is not accurate enough on large or badly scaled scenes | GPU results miss tolerance | Local parameterization, compensated sums, host-side double decisions; double on CUDA as fallback |
| Conjugate gradients converge slowly with very noisy priors | Slow or stalled solves | Direct solve option; better preconditioner |
| Three kernel targets drift apart | Backend-specific bugs | One symbolic source, cross-backend equality tests on every commit |
| "Fastest" is not reached against Caspar | Core promise missed, and found late because CUDA now comes after Metal | Profile against it at the start of milestone 5; adopt full-system conjugate gradients if it wins |
| Scope growth through new residual types | Core path slows down | The regression rule in document 05: no new residual without proof the existing benchmarks held |
