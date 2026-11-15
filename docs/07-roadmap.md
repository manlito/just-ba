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
  same machine vary by up to 5% per iteration.

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
| 6 | Can the existing system export real sessions for a private test set | Decided: yes, through the owner's matching tool; the files stay outside the repository (D78) | 05 |
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
