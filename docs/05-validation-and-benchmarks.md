# 05. Validation and benchmarks

Two promises have to be checked continuously: results match a trusted solver, and
speed does not degrade as features are added. This document defines how.

## What "correct" means

Owner's decision (2026-11-14): Ceres Solver is the definition of correct. The
correctness claim is "given the same problem, just-ba reaches the same answer as
Ceres", measured by the Level 1 and Level 2 checks below.

Comparison with ground truth is kept, but as a diagnostic for modelling choices
(prior weights, robust losses, defaults), not as part of the correctness claim. A
result can match Ceres exactly and still be far from the truth if the problem was
set up with poor weights; that is a tuning issue, tracked separately.

## References

| Reference | Version | Used for |
|---|---|---|
| Ceres Solver | 2.2.0 (tagged 2023-10-13) pinned; `master` tracked separately | Accuracy and CPU speed baseline for every residual type |
| COLMAP | 4.2.1 (2026-09-29), Ceres backend | End-to-end parity of the incremental path (and of GPS priors, when those are added) |
| COLMAP Caspar backend | same release, built with `CASPAR_ENABLED` | GPU speed comparison on the problems it supports |

Notes on the references, as of 2026-11-14:

- Ceres has no release after 2.2.0. Its cuDSS sparse GPU backend and the move to
  Abseil exist only on `master` (informally called 2.3). Pin 2.2.0 for accuracy
  checks, because a tagged version is reproducible; benchmark `master` as a second
  column when GPU numbers are wanted.
- A minimal Ceres 2.2.0 build needs only Eigen (with its bundled mini logging).
  That keeps the reference cheap to build in CI.
- Caspar supports only the reprojection residual on two camera models
  (SIMPLE_RADIAL, PINHOLE), no pose priors and no robust loss, and is float32 by
  default. It is a speed reference for plain bundle adjustment only.

Ceres is a development and test dependency. It is never linked into the shipped
library or the Python wheel.

## Level 1: residual and Jacobian checks

For every residual type and every camera model:

- Residual values agree with an independent Ceres cost functor on random inputs.
- Analytic Jacobians agree with central finite differences, and with Ceres automatic
  differentiation. (As built: the differences are taken in double, which gives
  about 1e-9 relative; the comparison with Ceres, at 1e-14, is the tight one.)
- Edge cases: point near the image centre (fisheye `r -> 0`), point near the
  principal plane, identity rotation, large rotations, fixed blocks.
- Every backend (CPU, CUDA, Apple) produces the same residuals and Jacobians as the
  CPU reference within a stated tolerance per precision.

These run on every commit once the repository is public; until then CI is started
by hand (D49). They need no dataset.

## Level 2: solver parity with Ceres

Same problem, same starting point, same loss, into both solvers.

### What is compared

| Quantity | Pass condition (starting values, to be tuned) |
|---|---|
| Final cost | Relative difference below 1e-6 in double precision, 1e-3 in single |
| Camera centres | Median difference below 1e-6 of scene scale (double) |
| Rotations | Median angular difference below 1e-6 rad (double) |
| Iterations to converge | Within 1.5x of Ceres |
| Cost after each iteration | Recorded; ours must never be worse than Ceres by more than a set margin at convergence |

Bundle adjustment problems can have several local minima that are equally good, so
parameter comparison is done only on problems that converge to the same basin; final
cost is the primary criterion.

Measured in milestone 1 (results in document 07): with an inexact inner solve,
per-iteration histories agree only until rounding changes a conjugate gradients
stopping decision, typically after two or three iterations. On well-conditioned
problems the end points still agree to 1e-8 or better. On venice-52 with the gauge
held, rounding alone selects different basins, so BAL parity runs without gauge
fixing (decision D60). Ceres itself is not repeatable from run to run with several
threads; on ladybug-49 that alone moves the final cost difference between 1e-9 and
5e-7. The parity tool therefore runs Ceres on one thread (D63).

### Problem sets

| Set | Source | Exercises |
|---|---|---|
| BAL | Bundle Adjustment in the Large (ladybug, trafalgar, dubrovnik, venice, final) | Plain reprojection at scale. The standard benchmark; Ceres `bundle_adjuster`, MegBA, DeepLM, Caspar and bae all report on it. |
| Synthetic | Generated in-repo with known ground truth | Every residual type, every camera model, controlled noise on priors, outlier matches, fixed blocks |
| COLMAP exports | Reconstructions from public scenes | Real camera models and track statistics |
| Phone captures | ARKit and ARCore sessions with poses and depth | The main use case |

BAL uses its own camera (angle-axis rotation, focal length and two radial terms,
negative z forward). It is supported in the test harness as a conversion into our
RADIAL model, not as a camera model in the library.

### Synthetic generator

The generator is the main tool for the features BAL cannot test:

- Pose-prior noise: white noise, slow drift, and sudden jumps (as when a tracking
  session relocalises).
- Global priors from a simulated network: consistent poses with noise and an
  arbitrary scale. Cases: alone, and combined with metric relative priors, where the
  estimated similarity transform must recover the true scale.
- Uneven priors: after a simulated cleanup, a share of the relative priors is
  still noticeably wrong. Pass condition: the result stays within tolerance of the
  clean case, and the bad priors are reported as down-weighted.
- Frames delivered in shuffled order, to prove nothing depends on order.
- Weakly connected frames (few priors, few matches). Pass condition: they are
  flagged and reported with large uncertainty, not silently trusted.
- Outlier fraction in matches and in priors.
- Depth noise that grows with distance, plus a per-image scale and offset error.

Scene types, matching the intended use (document 01):

| Scene | Stresses |
|---|---|
| Single room | Short baselines, depth available everywhere |
| Multi-room house | Rooms joined only through doorways: thin connections, poor conditioning, drift between rooms |
| House exterior, walked around | Loop that the tracker has not closed |
| Large commercial space or patio | Longer distances, depth sensor out of range in places |
| Small to medium park | Hundreds of metres, no useful depth, the upper size class |

For the Manhattan prior the generator also produces line segments: wall, floor and
ceiling edges along the three axes, plus distractors (sloped, random, and a second
wing at another heading), with endpoint noise. Required cases:

- Manhattan scene with heading drift in the priors: the prior must remove the drift.
- Non-Manhattan scene (the park): the support test must refuse the prior.
- Mixed scene: off-axis lines must end up unassigned, and the result must not be
  worse than without the prior.

Ground truth is known, so accuracy is measured against truth as well as against
Ceres. Reported covariances are checked the same way: against Ceres's covariance
estimation, and against the actual spread of errors over repeated noisy runs.

One required case is "priors already accurate". Pass condition: the refined poses are
not worse than the input. A recent preprint reports that triangulating from priors
and then adjusting can degrade good ARKit poses (document 02), so this is tested
explicitly.

## Level 3: end-to-end parity with COLMAP

Applies to the incremental path.

- Feed both systems the same verified matches, exported from one COLMAP database.
- Compare after aligning the two reconstructions with a similarity transform:
  registered image count, camera centre error, rotation error, mean reprojection
  error, mean track length, number of points.
- For datasets with ground-truth poses, report the standard pose accuracy curves
  against truth for both systems.

Pass condition: within a few percent of COLMAP on each metric for the chosen scenes.
The exact margins are set after the first measurements.

COLMAP's own pose-prior adjustment uses position-only absolute priors with a
covariance weight and an optional Cauchy loss. Our global prior in position-only
mode is the same term, so parity is tested with it configured that way. Relative
priors have no COLMAP counterpart and are checked against Ceres.

## Speed benchmark

### What is measured

- Wall time to convergence, and time per outer iteration.
- Split by phase: residual and Jacobian evaluation, building the linear system,
  linear solve, parameter update, host-to-device transfer.
- Peak memory on host and device.
- Linear solver iterations per outer iteration.

Time is reported at equal accuracy: time to reach a cost within a fixed margin of the
best known cost, not time for a fixed number of iterations.

### Size classes

The headline speed claim is made on the main target range. Each class has its own
table in the results.

| Class | Images | Claim |
|---|---|---|
| Small | 50 to 200 | Fastest, including set-up and transfer time, since overhead dominates here |
| Medium | 200 to 2,000 | Fastest |
| Large | 2,000 to 10,000 | Correct and competitive; fastest is a stretch |

Time is measured end to end from the solver call to the result, not only the inner
solve, because for small problems set-up is a large share. As built (D101):
`SolverSummary::time_seconds` is the whole call to `solve()` and `setup_seconds` the
part before the iterations; the iteration history's times count from the call too.
`bench/run.py` records the best of several warm calls (the device already started,
as in a process that solves more than once) and one cold call, the first call to
`solve()` in a fresh process (the device's start-up included; a whole command-line
run adds about 0.15 s more on CUDA for the process and libraries, D107). Competitors are timed the
same way: Ceres from building its problem to the end of `Solve()`. BAL problems fall mostly
in the medium and large classes and have more points per camera than phone sessions,
so real and synthetic phone-shaped problems carry the headline numbers.

### Against whom

| Competitor | Hardware | Problems |
|---|---|---|
| Ceres 2.2.0, sparse Schur and iterative Schur, multithreaded | CPU | All |
| Ceres `master` with CUDA and cuDSS | NVIDIA GPU | All |
| COLMAP Caspar | NVIDIA GPU | Reprojection-only problems |
| PyPose `bae` | NVIDIA GPU | BAL |
| Ceres with Accelerate sparse | Apple Silicon CPU | All |
| kornia-rs bundle adjustment (CPU, Rust) | CPU | Problems with depth and position priors. It is the closest existing match to our residual set: depth with a per-view scale, camera-centre prior, up-vector prior, motion prior. |
| GTSAM CUDA optimizers | NVIDIA GPU | BAL, if they are in a tagged release (unverified) |

Published numbers to sanity-check our own measurements against, not to quote as ours:
the `bae` paper reports about 9.8 s total on its BAL set against 84 s for Ceres with
CUDA on an RTX 4090; the Caspar paper reports 5 to 20 times faster than the best of
Ceres, DeepLM and MegBA on six BAL problems.

As run in milestone 5 (D99; how, in `bench/competitors/README.md`; results in
document 07): Ceres 2.2.0 with SuiteSparse and Ceres master with cuDSS, COLMAP
4.2.1's Caspar and bae 0.2, each on the same problem as just-ba on the owner's
laptop. Caspar needs the problem reduced to its residual and camera models
(`jba_to_colmap`); it runs a fixed 200 iterations in single precision and its
results differ from run to run. "Equal accuracy" is the time to reach the worse of
the two final costs. Kornia-rs and GTSAM were not run.

### Regression gate

- A fixed set of small and medium problems runs on every pull request on CPU.
  Fail if accuracy leaves tolerance, or if time per iteration regresses by more than
  a set percentage against the stored baseline for that machine class.
- GPU runs are done by hand, before releases and after changes to GPU code, with
  one command that runs the suite and writes the results. They are not CI jobs:
  CI uses free hosted runners only, with nothing scheduled.
- Results are stored as JSON per commit so trends can be plotted.
- Built for the CPU (D81): `bench/run.py` and a baseline for the owner's Mac in
  `bench/baseline/`, taken at the end of milestone 2. Since milestone 6 CI runs it
  too (D117): the job "benchmark gate (CPU)" on the hosted Linux runner, venice-52
  left out, against `bench/baseline/github-ubuntu-24.04.json` with loose tolerances
  (final cost 1e-6 relative, counts within 10%, time per iteration +50%), since the
  runner's CPU varies; CI is still started by hand (D110). By-hand runs are stored
  per commit with `run.py --store <machine>` under `bench/results/`.
- Every new residual type must add: Level 1 checks, a synthetic Level 2 problem, and
  a line in the speed benchmark showing existing problems did not slow down.

That last rule is the guard against feature growth hurting the core path.

## The sample binary's role

The sample command-line binary is also the benchmark driver:

- load a problem (BAL file, COLMAP model, or our own format);
- choose backend, precision and options;
- run, then write results and the per-iteration history as JSON;
- optionally run Ceres on the same problem when built with the reference enabled.

## Open questions

- Tolerances for single precision on GPU. Caspar's authors list float32 summation
  error as a limitation, so realistic thresholds need measurement. Measured on Metal
  (milestone 4, D95): on the parity cases' problems run to tight tolerances, final
  costs agree with the CPU to 1.6e-9 or better and median camera centres to 8e-5 of
  the scene scale; the Metal tests check 1e-6 and 1e-4, the latter on the median
  over images (document 13 lists what they do not compare).
- Which public phone-capture datasets to adopt (ARKitScenes is the obvious first
  candidate). The private test set exists (D78): ARCore captures exported by the
  owner's matching tool, run by tests/real when present.
