# 13. Review of milestones 3 and 4

Review of the code and documents as of commit `a31193d` (2026-11-15): milestone 3
(`04746e3..f708339`: gravity, the Manhattan prior, depth, cost by type, covariances,
export) and milestone 4 (`f708339..a31193d`: the Metal backend, both linear solvers,
`cpu::System`, the radius cap, the precision-limit termination, the speed
reference). It records what was checked, what was found, and what should change.
The review changed no code; it corrected statements in the other documents that no
longer matched the code or the measurements.

## What was checked

All on the owner's Mac (Apple M5 Pro, macOS 26.6, Apple clang 21), in a build of
this commit.

| Check | Result |
|---|---|
| Build (Release), with the Ceres harness and the Python bindings | No warnings |
| Tests without the slow one: unit, Python, the three real captures on three configurations, 111 Metal tests, 60 parity tests (four BAL problems included) | 298 of 298 pass |
| The slow test (venice-52 parity) | Passes |
| Address and undefined-behaviour sanitizers: every C++ test including the 111 Metal tests; ladybug-49 for 5 iterations on the CPU and on each Metal solver, and 12 iterations with the gauge free, where sparse_schur takes its fallback | Clean |
| Thread sanitizer: same runs | Clean |
| Build with `-ffp-contract=off` (document 12's advice) | 177 of 178 pass; `test_metal_reprojection` fails (finding 14) |
| Generated residual code (`generate.py --check`) | Current |
| CPU benchmark against `apple-m5pro.json` | Counts, terminations and final costs identical. Time per iteration +2% to +12%; two small problems are over the 10% limit (+11.3%, +11.7%, at 8 ms per iteration) |
| Metal benchmarks against both Metal baselines | Counts, terminations and final costs identical; times within the limit |
| Speed against Ceres with Accelerate, re-measured | 2.5 to 10.8 times on the ten baseline problems (recorded: 2.6 to 10.6); the walks 5.1 and 6.3 times (recorded: 5.9 and 6.0, single runs) |
| Iteration histories of builds of `04746e3` (milestone 2), `f708339` (milestone 3) and `a31193d`, CPU backend: four BAL problems with and without the gauge, nine synthetic cases with priors, free distortion and a Huber loss | Bit-identical in every field, 17 of 17 |
| Milestone 3's recorded figures, read from the tests' output | Reproduced: Jacobians 6.2e-11 against finite differences; 26 new parity cases, final costs within 2.3e-13 (median 1.5e-15); covariance within 2.3e-11 of Ceres; observed over predicted variance 0.78 to 1.19, median 0.92; line assignment and support; every row of the `refine` table |
| Metal against the CPU on the 110 cases | Final costs within 1.6e-9 (median 7.6e-12); where parameters are checked, median centres within 7.8e-5 of the scene scale and rotations within 1.8e-5 rad; every case repeats bit for bit |
| Metal by hand beyond the tests: the five BAL problems on both solvers; nine edge cases (all points fixed, all images fixed, a free gauge, two images, mixed residual types) | Repeatable, and identical with one host thread, everywhere. One crash (finding 1) |
| The three captures through `refine` on every configuration | Reprojection medians 0.89, 1.02, 0.89 px on all; time in the solves 0.44 to 0.89 s on Metal with sparse_schur against 1.95 to 4.61 s on the CPU. Agreement between backends: finding 5 |
| Code read: the Metal backend and kernels, `cpu::System`, the trust-region loop, the covariance, export, report, Manhattan assignment, problem file and validation, `refine`'s changes, the symbolic sources of the new residuals, the Metal and covariance tests, the CI workflow | Findings below |
| Standing rules (CLAUDE.md) | Hold, with one question about CI (finding 6) |

Not read line by line: the CPU backend's inner solve (`backends/cpu/src/backend.cpp`,
lines 330 to 700). The parity tests and the bit-identical histories cover it.

Not run: CI itself and anything on Linux or Windows; the Metal tests on a hosted
macOS runner; COLMAP reading the exported model (no COLMAP here; milestone 7);
captures with segments or depth (none exist); an M1-class machine.

## Verdict

Both milestones did what document 07 says. Milestone 3's exit conditions hold on
synthetic data and every recorded figure reproduces. Milestone 4's exit holds on this
machine: the Metal backend gives the CPU's answers within single-precision tolerance
on the 110 test cases, repeats bit for bit, and is faster than Ceres with Accelerate
on every problem measured. The move of the host side into `cpu::System` changed no
result: histories are bit-identical back to milestone 2.

Three things need fixing, none of them in the solve path the tests exercise:

- The default Metal solver kills the process when no camera-side block is free
  (finding 1).
- The covariance of the one pose per group whose heading the gauge holds is
  reported in the wrong coordinates (finding 2).
- The fallback of sparse_schur, which real captures need, has no test that runs
  without the owner's private data (finding 3).

The standing rules hold: CI uses free hosted runners and no schedule, the residual
set is closed, the only new link dependency is Apple's own Accelerate framework, and
Ceres is linked by test targets only.

## Findings

Ranked by severity. "Measured" means reproduced in this review; "by reading" means
not reproduced.

### High

**1. sparse_schur on Metal traps when there is no camera-side unknown.**
`backends/metal/src/backend.mm:572` returns from `setup_direct` before the symbolic
factorization when `nf == 0`, and `direct_solve` (`backend.mm:599`) then hands
Accelerate the empty structure, which stops the process (SIGTRAP, exit 133). Measured:
`jba_make_problem p.jba --model PINHOLE --fix-images 12`, then
`just-ba solve p.jba --backend metal`. sparse_schur is the default on Metal (D92), so
any solve with every pose and intrinsics block fixed (refining points only) dies,
and from Python it takes the interpreter with it. iterative_schur on Metal and the
CPU solve the same problem. No test has a problem without camera-side unknowns.

### Medium

**2. Covariance of a heading-held pose is in the gauge's basis, not the pose
tangent.** For an image in `Gauge::heading_images` (the reference of a group with
gravity priors, D83) the backend linearizes the rotation in the basis (b0, b1, a) of
`heading_basis` (`backends/cpu/src/backend.cpp:113`), and `covariance` copies that
block out unchanged (`backend.cpp:913`). `covariance.h` promises the tangent of D46.
Measured on the parity case `gravity`: the block's third row and column are zero,
and the variance along a = R up, which the gauge holds, comes out as 7.6e-6 rad^2
instead of zero. The camera-centre covariance of that image is computed from it
(`api/src/covariance.cpp:48`) and is wrong as well. Other images are unaffected.
The Ceres covariance test has no case with gravity or segments, so nothing compares
this block.

**3. The conjugate-gradients fallback of sparse_schur is not exercised by any test
in the repository.** Measured by solving the 55 Metal test problems with
sparse_schur and reading the linear iterations of each step: no step falls back.
Nor does any step on the five BAL problems in the benchmark's configuration,
ladybug-49 included, although D92 names it. The fallback does run, and works, with
the gauge free (ladybug-49: 36 of 44 steps), at tight tolerances (2 steps), and on
all three real captures; those runs are repeatable and clean under the sanitizers.
So the path is covered only by `tests/real`, which needs the owner's private files,
and by hand. A change that breaks `backend.mm:875` to `897` would pass `ctest -L gpu`.

**4. The Metal tests compare less than they appear to.** `tests/metal_check.cpp`:

- Parameters are compared by the median over images of the centre and rotation
  differences (`metal_check.cpp:150`): up to half the images can be arbitrarily
  wrong. Points, intrinsics, set transforms, the Manhattan frame and depth
  corrections are compared only through the cost.
- The repeat check covers poses, points and intrinsics (`metal_check.cpp:157`), not
  set transforms, the frame or depth corrections.
- All 110 cases end on the precision-limit termination. Outside `tests/real`, no
  Metal test ends on the function or parameter tolerance, which is how a solve at
  the default tolerances ends.
- BAL problems are not Metal tests. Their agreement with the CPU (ladybug-49 3e-5 in
  cost at default tolerances) is seen only in the benchmark.

The limits themselves are sound: costs are checked at 1e-6 and reach 1.6e-9.

**5. On real captures the backends agree a thousand times less closely than on the
test problems, and nothing checks it.** Measured through `refine`: median camera
centre difference between the CPU and Metal 2e-4 to 1.5e-3 of the scene scale,
worst image 1.8% of the scene scale and 0.05 rad (the capture where tracking was
lost; the number of refined images differs there, 52 against 51). Point counts
differ by tens. `tests/real/test_captures.py` runs every configuration through the
same loose checks and compares none with another. The cause is not the Metal
arithmetic alone: on these captures one to three of the five solves in each run
stop on the 50-iteration limit, on the CPU as on Metal, so the backends are compared
part-way down, and `refine` then makes discrete choices (tracks, outlier removal)
from slightly different poses. Document 07's "same answers" is shown for the parity
problems, not for the captures.

### Low

**6. CI's full mode runs the GPU tests, and cannot fail on them.**
`.github/workflows/ci.yml:59` runs `ctest -L gpu` on the hosted macOS runner with
`continue-on-error`. The runner is free, so the cost rule holds. But CLAUDE.md and
document 08's decision say GPU tests are run by hand, never in CI, while document
08's plan table says hosted macOS "if it proves reliable"; and the step, written for
one spike test, now covers 111 tests whose failure would not show as a failed run.
The owner should pick one (open item in document 00).

**7. Statements in the documents the measurements do not support.** Corrected in
this change:

- Document 07 said the line residual agrees with Ceres's automatic differentiation
  at Level 1. `test_ceres_level1` covers gravity and depth, not the line residual,
  which is checked against finite differences and at Level 2.
- D92 said the fallback is needed on ladybug-49. In the benchmark's configuration it
  is not taken (finding 3).
- D92 said sparse_schur has Ceres's SPARSE_SCHUR iteration counts. On the synthetic
  problems, yes; on BAL it takes 44 against 30 (ladybug-49), 11 against 9, 10
  against 8.
- The open item on venice-52 gave 0.27 million for Metal's sparse_schur against
  Ceres's 0.24. Measured at 50 iterations with the gauge held: Metal sparse_schur
  0.237 million, Metal iterative_schur 0.253, the CPU 0.250, Ceres with Accelerate
  0.262.
- Document 03's backend table gave the CPU a direct solver; it has none (D92). Its
  "CPU fallback when the device runs out of memory" is not built: a failed buffer
  allocation is a thrown error. The interface list lacked `cost_by_type`.
- Document 07's risk table answered kernel drift with cross-backend tests on every
  commit; they run by hand.
- Document 08 described CI as after milestone 1.
- `docs/README.md` still listed real sessions as an open decision (D78 settled it).

**8. Covariance treats a free point whose block is not positive definite as a fixed
point.** `backend.cpp:822` inverts the point's 3 x 3 block with no relative pivot
limit, and when that fails the point's observations stay in the reduced system
without the Schur term (`backend.cpp:828`), as if the point were known exactly: the
poses come out too certain, silently, and only the point's own covariance is NaN.
By reading. A point seen from two images that share a camera centre came out with an
honest large variance instead (102 m^2 along the ray), because rounding left the
pivot positive; the wrong branch needs it at or below zero.

**9. Export: unchecked inputs, and a test that would miss a wrong coordinate.**
`api/src/export.cpp:94`: an `obs_keypoint` index outside its image throws
`std::out_of_range` without saying which observation, not the documented
`std::runtime_error`; two observations naming one keypoint overwrite each other and
leave a track pointing at a 2-D point of another 3-D point. Write errors after
opening are not checked (`export.cpp:103` to `165`). With a principal point at or
below zero the derived image size is not positive. `tests/test_export.cpp` never
compares the written x, y with the observations, and covers one PINHOLE camera
only: the single-focal branch for the principal point (`export.cpp:110`) and
several cameras are untested. No reader other than the test has read the files.

**10. Strategy B's set-up grows with the square of the track length.**
`backend.mm:481` to `502` lists every ordered pair of a point's observations for
every pair of block roles, 12 bytes each, in a map of vectors on the host and again
on the GPU. The 2,000-image walk (track length 8) needs about 100 MB; two million
observations at track length 30 would need about 0.7 GB twice. Indices are 32-bit.
Nothing limits or reports it. By reading and arithmetic.

**11. Threadgroup sizes are checked for one kernel of three.** `backend.mm:169`
checks that `reduce` may run 256 threads per group; `block_gather` (256) and
`schur_tiles` (64) are dispatched at fixed sizes without the check
(`backend.mm:666`, `673`). Fine on this GPU.

**12. The host and kernel structures are compared by size only.**
`backend.mm:183`. Two fields of one size swapped in `Args` on one side would pass.

**13. `refine` can drop the Manhattan prior part-way, and can end worse than it
started.** By reading: each round re-runs the support test at half the angle with
the same 0.35 threshold (`api/src/refine.cpp:251` to `277`); if it fails there, the
term is cleared and the result reports the prior as refused although earlier solves
used it. No test reaches that. Measured in `test_refine`: on the two-room scene with
strong heading drift, `refine` without the prior ends at 12 cm and 1.73 degrees from
a start of 10 cm and 1.31 degrees. Document 07's table shows that row; nothing says
it is worse than the input.

**14. `test_metal_reprojection` depends on how the host compiler rounds.** Built
with `-ffp-contract=off`, the CPU's float evaluation differs from Metal's by 6.1e-5
px against the test's 1e-5 (`tests/test_metal_reprojection.cpp:98`). The 110
comparison tests pass in that build. It matters only if the Mac build ever stops
fusing multiply-adds.

**15. Comments that no longer match the code** (left as they are; no code was
changed):

- `backends/cpu/internal/jba/cpu/system.h:239`: `obs_range` is "empty for the
  others", but a depth block uses its image's list.
- `core/include/jba/problem.h:138`: the loss order stops at the Manhattan line;
  depth is index 5.
- `core/include/jba/manhattan.h:50`: "about 0.07" for random segments; the tests
  measure 0.11 and 0.12.
- D79's list of public headers predates `covariance.h`, `export.h` and
  `manhattan.h`.

### Smaller points, no action needed now

- The radius is clamped to the cap before the first step, on the CPU too when the
  caller's maximum is below the initial radius; Ceres clamps only after a step.
  No effect at the defaults.
- A trust-region radius below about 1e-30 would overflow the damping in single
  precision; the solve would then end as a failure after five invalid steps.
- With a Huber loss the BAL problems crawl at tight tolerances (200 iterations, the
  actual decrease twice the predicted one) on the CPU as on Metal. Not new and not
  specific to a backend.
- Float robust weights on the GPU can fall on the other side of Huber's threshold
  from the host's double cost for a residual exactly at the scale. The cost that
  decides is the host's.

## Recommended changes

In order. Items 1 to 3 before milestone 5 ports this backend to CUDA, since the port
would copy all three.

| # | Change | Why |
|---|---|---|
| 1 | With no camera-side unknown, skip the factorization in `direct_solve` (the step is back substitution alone), and add a Metal case with every image fixed, for both solvers. | Finding 1: a crash on the default path. |
| 2 | Rotate the covariance block of a heading-held pose back to the pose tangent (B Sigma B^T) before the camera centre is derived, and add a Ceres covariance case with gravity priors. | Finding 2. |
| 3 | Add a Metal case that takes the fallback and assert that it did (the summary's linear iterations exceed its steps): a synthetic problem with far points and a free gauge, since BAL data is not in the repository. | Finding 3. |
| 4 | In `jba_metal_check`: compare the worst image as well as the median, compare points and the other parameter blocks where the gauge fixes them, extend the repeat check to every block, and add a case at default tolerances. | Finding 4. |
| 5 | Decide what agreement between backends `refine` should give on captures, and whether its solves should run to convergence; then compare the configurations in `tests/real`. | Finding 5. |
| 6 | Owner: keep or remove the Metal step in CI's full mode; if kept, let it fail the run. | Finding 6. |
| 7 | Covariance: leave out, or refuse, a point whose block fails to invert, with a relative pivot limit as for the reduced system. | Finding 8. |
| 8 | Export: validate `obs_keypoint`, check the streams after writing, compare coordinates in the test, and test a single-focal model and two cameras. | Finding 9. |
| 9 | Check every fixed threadgroup size against its pipeline, and compare a few field offsets of `Args` as well as sizes. | Findings 11 and 12. Cheap. |
| 10 | Bring the four comments of finding 15 in line. | |

Not recommended now: reducing the memory of strategy B's set-up (finding 10) before
a problem that needs it is measured.

## Questions for the owner

1. Should CI's full mode run the Metal tests at all (finding 6)?
2. Is agreement between backends on real captures a requirement, and to what
   tolerance (finding 5)?

## Follow-up (2026-11-15)

Done at the owner's request after the review, changes 1 to 3:

- **Finding 1.** `direct_solve` returns at once when there is no camera-side
  unknown; the step is back substitution alone. New Metal case `points_only` (every
  image and the intrinsics fixed), with both solvers. Before the change it stopped
  the process; it now agrees with the CPU to 3e-12 in cost.
- **Finding 2.** The covariance of a heading-held pose is rotated back to the pose
  tangent, P <- T P T^T with T = diag(B, I), before the camera centre is derived.
  The Ceres reference now reports that pose too, mapping HeadingHeldPose's
  two-coordinate tangent through the same basis, and the Ceres covariance test has a
  case with gravity priors: 10 poses compared, 3.8e-14 relative. With the change
  undone the case fails at 0.99.
- **Finding 3.** `jba_metal_check --expect-fallback` fails unless some sparse_schur
  step took more than one linear iteration. A free gauge alone did not make the
  synthetic problems fall back (0 or 1 steps on the parity problems and on tracks
  of three); the generator has no far points. Points seen by two neighbouring images
  each did, since every link of the chain then has a free scale. New
  Metal case `fallback` (12 images, 300 points, track length 2) takes the fallback
  on 13 steps. Costs are compared (7e-11) and parameters are not, as in `no_gauge`.
  With the fallback replaced by an invalid step, the case fails.

`ctest -L gpu` now has 114 tests. All 303 tests without the slow one pass,
including Python and the three captures. Changes 4 to 10 and the two questions are
still open.
