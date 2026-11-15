# 07. Roadmap

Milestones are ordered so that correctness is established before speed, and the main
use case (pose priors from a phone) is reached as early as possible. No dates: each
milestone has an exit condition instead.

## Milestone 0: decisions and spikes

- Language is decided: C++ (document 06). Set up the build (C++23, D52; CMake) on Linux,
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

Results (2026-11-14, M5 Pro, 18 threads unless stated):

- Built: problem setup with fixed blocks and partially freed intrinsics, the
  Levenberg-Marquardt loop (D54), implicit Schur with conjugate gradients and the
  Schur-Jacobi preconditioner (D55), four robust losses (D56), automatic gauge (D57),
  progress and cancellation callback, `just-ba solve` with JSON history.
- Level 1 against Ceres: residuals and Jacobians agree with Ceres's automatic
  differentiation to 1.3e-14 for all four models; losses to 1e-15.
- Level 2, synthetic (14 cases: every model, fixed blocks, sigma, shared and
  per-image intrinsics, three losses with 5% outliers), both solvers run to tight
  tolerances: final costs agree to 1e-15 to 2e-12, camera centres to 1e-15 (2e-8
  with all shared intrinsics free), rotations exactly.
- Level 2, BAL test set (the smallest problem of each family, no gauge, D60):

  | Problem | Final cost, relative difference | Iterations, just-ba / Ceres | Time, just-ba / Ceres |
  |---|---|---|---|
  | ladybug-49 | 6.0e-9 | 35 / 35 | 0.76 s / 1.51 s |
  | trafalgar-21 | 1.0e-8 | 8 / 8 | 0.34 s / 0.70 s |
  | dubrovnik-16 | 7.4e-11 | 12 / 12 | 0.79 s / 1.35 s |
  | final-93 | 3.8e-10 | 6 / 6 | 1.77 s / 2.24 s |
  | venice-52 | 1.6e-4 at 300 iterations, 2.7e-7 at 2,000 | 232 / 190 at 300 | |

  Times with 3 threads each, three runs in parallel; indicative only. The cost
  differences are those with Ceres on one thread, which is how the parity tool now
  runs it (D63); they were measured again on 2026-11-14 and are repeatable. With
  several threads Ceres's end point varies from run to run, on ladybug-49 between
  1e-9 and 5e-7 (document 12).
- Histories match to 12 digits until the first inner solve long enough for rounding
  to change its stopping iteration (iteration 3 on trafalgar). Our own solver built
  without fused multiply-add diverges from itself at the same point, so this is
  rounding sensitivity, not a difference in the method.
- Results are bit-identical for 1 to 18 threads (D59).

Status: done. Exit met on four of the five problems at better than 1e-8, venice-52
with the longer run (D60). Tested on macOS (owner's Mac) and, in a quick CI run
(D61) on 2026-11-14, on Ubuntu 24.04 and Windows Server 2025, with no compiler
warnings. A quick CI run takes about 1.5 minutes and bills about 6 minutes.

## Milestone 2: the main use case on CPU

Before the features, from the review of milestones 0 and 1 (document 12). The
review and the first features were done in parallel and merged on 2026-11-14; the
status of each item after the merge:

- Done (D81): a stored baseline of times and iteration counts (`bench/`), so the
  regression rule of document 05 can be applied to every residual added from here.
- Done (D75): the CPU backend extended for several residual types. Iteration
  histories of the four fast BAL problems are bit-identical to the review's code.
- Done (D67, D80): one loss per residual type in the options, and the `loss.*` file
  sections.
- Done (D66): the automatic gauge made aware of priors.
- Done (D74, D79): the Python boundary copies arrays and normalises quaternions on
  the way in; public headers are separated from internal ones.
- Done (D62, document 12): the two small differences from Ceres in the trust-region
  loop, and the small input and output fixes.
- Done: Ceres harness cost functors for both prior types and the two new camera
  models (Level 1 and Level 2 results below).

Then:

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

Results so far (2026-11-14, M5 Pro):

- Built: relative and global pose priors with identity, rigid and similarity set
  transforms (D64, D65, D67); the gauge chosen per group of connected images (D66);
  OPENCV and OPENCV_FISHEYE (D68); unprojection for every model; triangulation on
  rays (D72); track building with an epipolar check of matches, stage 1 and stage 2
  (D69); starting poses from priors with groups reported (D70); the per-residual
  report with down-weighted priors; `refine`, the main use case end to end (D71); the
  synthetic generator (D73).
- Level 1: prior residuals agree with finite differences to 6e-10 and with Ceres's
  automatic differentiation to 2e-15; OPENCV and OPENCV_FISHEYE with Ceres to 2e-14.
- Level 2: 17 new synthetic cases against Ceres (relative priors, outliers in them,
  Huber and Cauchy and no loss on priors, global priors of each relation, both kinds
  together, a pose graph without points, fixed blocks, the new models with and
  without free distortion). Final costs and parameters agree to about 1e-15 with
  identical iteration counts, except with all eight parameters of one shared OPENCV
  or fisheye block free: there the minimum is a flat valley, costs agree to 1e-11
  and parameters to a few 1e-6, about as much as Ceres's own end point moves
  between runs.
- Synthetic scenes through `refine`, median camera centre and rotation error after
  alignment to the truth, against the starting poses built from the priors:

  | Scene | Start | Refined |
  |---|---|---|
  | Room, tracker drift | 4.6 cm, 0.73 deg | 7.6 mm, 0.12 deg |
  | Room, accurate priors (must not get worse) | 5.3 mm, 0.068 deg | 1.2 mm, 0.022 deg |
  | Room, 10% wrong matches | 2.0 cm, 0.30 deg | 2.8 mm, 0.056 deg |
  | Room, 5% wrong relative priors left in | 1.8 cm, 0.35 deg | 3.8 mm, 0.063 deg; all 8 reported, none wrongly |
  | Room, tracking break, priors across it removed | | two groups, 5.6 and 5.8 mm |
  | Room, relative priors plus network priors in a similarity frame, 5% outliers | 2.6 cm, 0.73 deg | 5.5 mm, 0.11 deg; scale within 2.1%; all 4 outliers reported |
  | Room, network priors only | 4.8 cm, 0.87 deg | 2.2 mm, 0.029 deg |
  | Room, frames shuffled | 2.3 cm, 1.1 deg | 6.0 mm, 0.096 deg |
  | Two rooms joined by a doorway | 3.4 cm, 0.36 deg | 6.1 mm, 0.23 deg |
  | Building walked around outside, loop not closed | 16 cm, 1.0 deg | 3.3 mm, 0.010 deg |
  | Room, OPENCV / OPENCV_FISHEYE | 2.4 / 1.5 cm | 4.4 / 0.4 mm |

  With the default relative prior weights measured on ARCore (D77). The synthetic
  trackers drift more in heading than those captures, so the tighter rotation weight
  costs a little on the drifting scenes: room with drift 5.1 mm and 0.090 deg before
  D77.
  Keypoint noise is 0.5 px. With noise-free keypoints the images alone recover the
  truth exactly, so these figures are the scenes' own information limit, not a
  bias. The whole set runs in about 7 seconds.
- Python bindings, first version (D74): problems, solve with callbacks, the report,
  starting poses, track building and `refine`, plus conversion of ARKit and ARCore
  camera-to-world matrices. Tested from Python on a scene built in NumPy: through
  `refine`, a drifting tracker's 7.9 mm and 0.32 deg become 1.2 mm and 0.039 deg.
  The problem files match document 11's reference reader. Built and tested on the
  owner's Mac; added to the CI build on Linux and Windows, not yet run there.
- Real sessions (D78): three ARCore captures from a Pixel 9a, 37 to 58 keyframes,
  keypoints and matches from the owner's matching tool, the tracker's poses and
  relative priors, through `refine` from Python. Results without ground truth:

  | Capture | Refined images | Points | Reprojection, median / p90 | Priors down-weighted | Consecutive poses moved (median) |
  |---|---|---|---|---|---|
  | 20261114-154702 | 37 of 37 | 24,635 | 0.89 / 2.48 px | 0 of 144 | 0.22 deg, 5.9 mm |
  | 20261114-155058 | 51 of 51 | 27,470 | 1.02 / 2.82 px | 2 of 291 | 0.28 deg, 9.6 mm |
  | 20261114-163628, tracking lost three times | 52 of 58, 4 groups | 17,783 | 0.89 / 2.60 px | 21 of 152 | 0.53 deg, 16 mm |

  On the third capture ARCore is up to 13 degrees off within a segment where it was
  about to lose tracking; those priors are the ones down-weighted. Six of its images
  have too few observations to count as refined (D76). The captures also set the
  default relative prior weights (D77).
- Exit met and milestone finished on 2026-11-14, with every item from the review
  done. Baseline (`bench/baseline/apple-m5pro.json`): ladybug-49 31 iterations at
  13 ms each, venice-52 50 at 142 ms, final-93 15 at 112 ms; repeat runs on the
  same machine vary by up to 5% per iteration. CI passes on Linux and Windows, Ceres parity
  included (quick mode).

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

Results (2026-11-14, M5 Pro):

- Built: the gravity prior (D82) with a gauge that holds only the reference's
  position and heading when gravity is known (D83); line segments, their assignment
  and the support test (D85), the Manhattan residual (D84) with the frame as the
  rotation gauge of every group it ties (D86); depth at keypoints with per-image
  scale and offset, compared in the reading's units (D87); cost by residual type in
  every solve summary; covariances of poses and points (D88); export to COLMAP's
  text model and `result.*` sections in the own format (D89). Python bindings for
  all of it. The CPU backend takes the new camera-side residuals as factor sets and
  the depth rows next to the reprojection rows.
- Level 1: the gravity, line and depth residuals agree with finite differences to
  6e-11; gravity and depth also with Ceres's automatic differentiation of
  independent functors, to 2e-15. The line residual meets Ceres at Level 2 only
  (document 13).
- Level 2: 26 new synthetic cases against Ceres (gravity with outliers and the
  heading-held gauge; lines with one and three freedoms, outliers, gravity,
  priors, two groups tied only by the frame; depth along the axis and the ray,
  linear and inverse, free scale and offset, outliers, priors, fixed blocks). Final
  costs agree to about 1e-15, 1e-13 for pose graphs without points, whose
  histories differ from Ceres's from the first inner solve already in milestone 2.
- Covariances agree with Ceres's dense SVD covariance to 2e-11 or better, and with
  the spread of 300 noisy runs (observed over predicted variance: median 0.92,
  range 0.78 to 1.19).
- Line assignment on a synthetic room seen from the true poses: frame within 0.05
  degrees, no segment assigned to a wrong axis, 8% left out (nearly horizontal
  planes, which hold both horizontal axes). Support: 1.0 in a Manhattan room, 0.52
  in a mixed one (40% off-axis segments, a wing 30 degrees off), 0.12 for random
  directions, which is refused (threshold 0.35).
- Through `refine`, median camera centre and rotation error after alignment to the
  truth:

  | Scene | Without | With |
  |---|---|---|
  | Room, strong heading drift, Manhattan | 8.9 mm, 0.161 deg | 5.7 mm, 0.062 deg |
  | Two rooms joined by a doorway, strong heading drift, Manhattan | 12 cm, 1.73 deg | 3.6 cm, 0.22 deg |
  | Room, mixed segments, Manhattan | 8.9 mm, 0.161 deg | 5.7 mm, 0.073 deg |
  | Room, random segments, Manhattan | 8.9 mm, 0.161 deg | refused; identical result |
  | Room, drift, metric depth on half the keypoints | 7.6 mm, 0.115 deg | 4.3 mm, 0.092 deg |
  | The same with 5% gross depth errors | 7.6 mm, 0.115 deg | 4.3 mm, 0.084 deg |
  | The same with network depth (10% scale, 5 cm offset per image, both estimated) | 7.6 mm, 0.115 deg | 5.1 mm, 0.108 deg |

  At the pose-only stage, before anything is triangulated, the segments about halve
  the heading drift; the relative priors, weighted for ARCore (D77), still pull
  toward this tracker's larger drift. The images finish the job.
- Iteration histories of the milestone 2 cases (four BAL problems, seven prior
  cases) are bit-identical to milestone 2's. Benchmark: counts and costs unchanged;
  time per iteration, best of ten interleaved runs against a milestone 2 build on
  the same machine, from 12% faster to 2% slower. The machine ran about 10% slower
  than when the baseline was taken, so the baseline file was not replaced.

Status: done. Exit conditions met on synthetic data. Tested on the owner's Mac and, in
a quick CI run (D61) on 2026-11-15, on Ubuntu 24.04 and Windows Server 2025, Ceres
parity included. The first CI run found a covariance test whose problem was
singular but passed on the Mac by rounding; singular systems are now refused on
every platform (D88). Not yet tried: real captures with segments or depth (the
exported ARCore captures carry neither; their `refine` test still passes).
Reviewed in document 13: every figure above reproduces. Fixed after the review:
the covariance of a pose whose heading the gauge holds (D83) was reported in the
gauge's basis, not the pose tangent; the Ceres covariance test now has a gravity
case that compares that pose.

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

Results (2026-11-14, owner's M5 Pro; the only Apple machine available, so "M1-class"
means Apple Silicon here):

- Built: the Metal backend (D90 to D93). The GPU linearizes the reprojection
  residual from the generated Metal code, scales the Jacobian, forms column norms,
  gradient, point and camera blocks, and solves the step in single precision; the
  host keeps the state and every decision in double, shares its whole setup with
  the CPU backend (`cpu::System`), and linearizes the few camera-side factors and
  the depth rows in double. Both linear strategies: iterative_schur, with the
  conjugate gradients loop and Ceres's stopping test running in batches on the
  device, and sparse_schur, the reduced camera system formed on the GPU by tiles
  and factorized in double by Accelerate's sparse Cholesky (D92), the default on
  Metal. `SolverOptions::backend`, `--backend metal` in the CLI, `Backend.METAL` in
  Python; `refine` runs on it unchanged.
- The precision design held, with three additions the measurements forced (D91):
  residuals from the host's double evaluation, a pivot floor, and a cap on the
  trust-region radius. Conditioning alone does not fix residual precision, as
  milestone 0 predicted; taking the residual from the host does.
- Same answers: the problems of the 55 Ceres parity cases, solved on both backends
  to tight tolerances, with each Metal solver (110 tests): final costs agree with
  the CPU to 1.6e-9 or better (median 1e-11), median camera centres to 8e-5 of the
  scene scale or better (median over the cases 5e-7; the largest are the gravity
  cases, whose heading gauge is path-dependent, D86), rotations to 2e-5 rad. Every Metal
  solve is bit-identical when repeated (D93). The three ARCore captures through
  `refine` pass the same checks on every configuration, with the same reprojection
  medians (0.89, 1.02, 0.89 px).
- Faster than Ceres with Accelerate (D94: SPARSE_SCHUR, 18 threads, the same gauge,
  losses and problem), times best of three for the solve alone:

  | Problem | Ceres + Accelerate | Metal, sparse_schur | Speed-up | Metal, iterative_schur | CPU (double) | Final cost vs Ceres |
  |---|---|---|---|---|---|---|
  | ladybug-49 | 30 it, 0.363 s | 44 it, 0.141 s | 2.6x | 46 it, 0.413 s | 31 it, 0.411 s | 2.9e-5 |
  | trafalgar-21 | 9 it, 0.171 s | 11 it, 0.041 s | 4.2x | 11 it, 0.121 s | 10 it, 0.193 s | 5.2e-8 |
  | dubrovnik-16 | 12 it, 0.564 s | 12 it, 0.053 s | 10.6x | 14 it, 0.156 s | 14 it, 0.283 s | 1.2e-7 |
  | venice-52 | 50 it, 6.224 s | 50 it, 1.016 s | 6.1x | 50 it, 0.801 s | 50 it, 7.103 s | 9.6e-2 (another basin) |
  | final-93 | 8 it, 1.132 s | 10 it, 0.227 s | 5.0x | 20 it, 1.296 s | 15 it, 1.675 s | 5.2e-7 |
  | synthetic pinhole, 50 images | 4 it, 0.360 s | 4 it, 0.054 s | 6.7x | 6 it, 0.039 s | 6 it, 0.049 s | 2.0e-13 |
  | synthetic radial, free shared intrinsics, 30 | 14 it, 0.727 s | 14 it, 0.226 s | 3.2x | 23 it, 0.478 s | 36 it, 0.586 s | 5.3e-8 |
  | synthetic relative priors, 50 | 3 it, 0.320 s | 3 it, 0.046 s | 7.0x | 4 it, 0.031 s | 4 it, 0.029 s | 8.0e-10 |
  | synthetic global similarity, 50 | 3 it, 0.324 s | 3 it, 0.046 s | 7.0x | 4 it, 0.032 s | 4 it, 0.032 s | 1.2e-9 |
  | synthetic OPENCV, 30 | 2 it, 0.107 s | 2 it, 0.019 s | 5.6x | 4 it, 0.025 s | 4 it, 0.017 s | 3.7e-12 |
  | walk, 500 images, 480k observations | 5 it, 0.342 s | 5 it, 0.058 s | 5.9x | 8 it, 0.532 s | 7 it, 1.276 s | 1e-10 |
  | walk, 2,000 images, 1.9M observations | 7 it, 1.670 s | 7 it, 0.279 s | 6.0x | 9 it, 3.283 s | 9 it, 11.47 s | 1e-9 |

  The walks are the main size range (D5): each point seen by 8 neighbouring images,
  relative priors (`bench/run.py --large`, single runs). venice-52 with the gauge
  held is the problem whose end point depends on rounding (D60); at 50 iterations
  neither solver has converged and they sit in different basins. On the three
  captures, refine's solves take 0.41 to 0.89 s on Metal against 1.95 to 4.43 s on
  the CPU. At the small end the CPU backend is as fast or faster (the dense
  synthetic problems), as document 03 expected; choosing the backend by size is
  for later.
- Benchmark baselines for both Metal solvers (`bench/baseline/apple-m5pro-metal-*`),
  with Ceres's figures alongside. The CPU backend's counts and costs are unchanged
  from milestone 3 throughout.
- Not done: macOS wheels. Packaging for PyPI is milestone 6's subject (document 08)
  and none exists yet; the wheel for macOS is built there with the others. The
  Metal backend needs nothing beyond the system frameworks (Metal, Foundation,
  Accelerate), so the wheel needs nothing extra.
- Also not done: strategy B on the CPU, and depth rows on the GPU (they are
  linearized on the host; no capture with depth exists yet to measure against).

Status: done. Exit met on the owner's M5 Pro: same answers as the CPU reference
within single-precision tolerance, and faster than Ceres with Accelerate on every
problem measured, 2.6 to 10.6 times. No M1 was available, so "M1-class" is shown on
newer Apple Silicon only. macOS wheels moved to milestone 6 with the other wheels.
The Metal tests run by hand where a Metal device is present (label `gpu`). The
shared host side, the trust-region changes and the rest pass a quick CI run (D61)
on 2026-11-15 on Ubuntu 24.04 and Windows Server 2025, Ceres parity included; that
mode has no macOS job, and the full mode runs the Metal tests on the hosted macOS
runner as an informational step (document 08).
Reviewed in document 13: the figures above reproduce (2.5 to 10.8 times when
re-measured). Fixed after the review: sparse_schur stopped the process when no
camera-side block was free, and its conjugate-gradients fallback, which the real
captures need, had no test outside them; both now have Metal cases. "Same answers"
is shown on the parity problems; through `refine` on the captures the backends
agree to about 1e-3 of the scene scale, and no test compares them there (open,
document 00).

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
| 6 | Can the existing system export real sessions for a private test set | Decided: yes, through the owner's matching tool; the files stay outside the repository (D78) | 05 |
| 7 | Open source licence | Decided: Apache-2.0 | 08 |
| 8 | GPU test hardware | Decided: the owner's Mac for Metal; laptop (Linux and Windows) for CUDA development; a Google Cloud L4 VM for development sessions. All GPU testing is by hand; CI stays free, with no nightly jobs | 08 |

## Main risks

| Risk | Effect | Response |
|---|---|---|
| Single precision is not accurate enough on large or badly scaled scenes | GPU results miss tolerance | Local parameterization, compensated sums, host-side double decisions; double on CUDA as fallback |
| Conjugate gradients converge slowly with very noisy priors | Slow or stalled solves | Direct solve option; better preconditioner |
| Three kernel targets drift apart | Backend-specific bugs | One symbolic source; cross-backend equality tests (`ctest -L gpu`), run by hand; CI's full mode runs them without failing on them (document 08) |
| "Fastest" is not reached against Caspar | Core promise missed, and found late because CUDA now comes after Metal | Profile against it at the start of milestone 5; adopt full-system conjugate gradients if it wins |
| Scope growth through new residual types | Core path slows down | The regression rule in document 05: no new residual without proof the existing benchmarks held |
