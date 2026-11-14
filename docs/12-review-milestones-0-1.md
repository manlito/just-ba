# 12. Review of milestones 0 and 1

Review of the code and documents as of commit `96e9e54` (2026-11-14), before
milestone 2 starts. It records what was checked, what was found, and what should
change first. The review itself changed no code. The owner then asked for the two
differences from Ceres and the small fixes to be made; see "Follow-up" at the end.

## What was checked

| Check | Result |
|---|---|
| Build and tests on the owner's Mac (Release, Apple clang 21) | 5 of 5 pass, Metal test included, no warnings |
| Ceres parity, quick set (Level 1, 14 synthetic cases, four BAL problems; venice-52 not run) | 37 of 37 pass |
| Address and undefined-behaviour sanitizers: unit tests, and five iterations of ladybug-49 | Clean |
| Thread sanitizer: same runs | Clean |
| Code read in full: core, CPU backend, api, CLI, problem file reader, Ceres harness, CI workflow | Findings below |
| Trust-region loop compared line by line with Ceres 2.2.0 `trust_region_minimizer.cc` | Two small differences, below |
| Documents 00 to 11 compared with the code | Stale statements corrected in this change |

The sanitizer runs were done by hand; they are not part of CI.

## Verdict

Milestones 0 and 1 did what they set out to do, and the exit conditions recorded in
document 07 hold. The solver is small (about 3,000 lines without generated code and
tests), follows Ceres closely enough to compare iteration by iteration, and is
deterministic across thread counts. No correctness defect was found in the solve
path.

The main limit is structural, not a defect: the CPU backend is written for exactly
one residual type. Milestone 2 adds residuals that do not involve points, so the
backend has to be reshaped before they are added, not while.

## Findings

### Differences from Ceres in the trust-region loop

Decision D54 says the loop follows Ceres exactly. Two details do not:

| Where | just-ba | Ceres 2.2.0 | Effect |
|---|---|---|---|
| Step norm used by the parameter tolerance (`trust_region.cpp`, `backend.cpp` `compute_step`) | Norm of the tangent step | Norm of `x - candidate` in stored coordinates, so a rotation step counts about half (quaternion) | Our parameter-tolerance stop fires slightly later. No effect on the parity results, which stop on the function tolerance or run with a 1e-12 tolerance. |
| Consecutive invalid steps before failure | Fails on the sixth (`>`) | Fails on the fifth (`>=`) | Only on numerically failing problems. |

Both were fixed in the follow-up (D62).

### Backend shape

- `core/include/jba/backend.h` has seven calls, and `compute_step` does everything
  between linearization and the candidate point: block building, the conjugate
  gradients loop, back substitution, the predicted decrease and the retraction. The
  fine-grained operation list in document 03 was not built. This is a reasonable
  interface for the host loop, but it means the conjugate gradients loop lives in
  the CPU backend and a GPU backend would write its own. Document 03 now describes
  both levels and leaves the freeze to milestone 4, as planned.
- `backends/cpu/src/backend.cpp` is one 760-line class tied to the reprojection
  residual: every product, the preconditioner, the gradient and the predicted
  decrease assume that each residual touches one pose, one intrinsics block and one
  point.
- Whole fixed blocks are removed from the unknowns at setup. Single fixed
  coordinates (the gauge's scale coordinate, intrinsics groups that stay fixed) are
  kept as unknowns with zero Jacobian columns. The result is the same as removing
  them; document 03 said they were removed.
- Observations are sorted once, by point then image, with index lists per image and
  per intrinsics block. Document 03 said "sorted twice".

### Input and output

- `validate` requires quaternion norms within 1e-9 of one. Poses that come from
  single-precision data (ARKit and ARCore matrices are float) will fail unless the
  caller normalises in double. The Python layer should normalise on the way in.
- `validate` does not check for non-finite values in points, intrinsics, translations
  or observations. They are caught later as "initial cost is not finite", without
  saying which input is bad.
- The CLI and the parity tool print paths and messages into JSON without escaping.
  A Windows path with backslashes gives invalid JSON.
- The problem file reader accepts a section name twice and keeps the last one; it
  should reject the file. A section with a data type the reader does not know fails
  the read even when the section itself is unknown; document 11 now says so.
- `loss.*` sections were listed for milestone 1 in document 11 and were not built.
  The loss is a solver option only. Moved to milestone 2.
- All headers sit in one include directory. `backend.h` and `trust_region.h` are
  internal but look public.

### Tests

- Good coverage of the happy path: Jacobians against finite differences (in double,
  not extended precision as document 05 said), four camera models, losses, fixed
  blocks, gauge, shared and per-image intrinsics, determinism, cancellation, file
  round trips, BAL conversion.
- Not covered: the failure paths (`Termination::Failure`, a non-finite start, an
  indefinite block), images or points without observations, the same point observed
  twice in one image, an empty problem. Reading the code, each of these is
  handled; none is exercised by a test.
- Determinism across thread counts is tested on 12 images in CI and was checked on
  BAL by hand.
- There is no stored speed baseline yet. The numbers in document 07 were taken with
  three runs in parallel and are marked indicative.

### Smaller points, no action needed now

- Vector updates and the preconditioner product in the conjugate gradients loop run
  on one thread; work vectors are allocated on every solve step; the candidate
  parameters are a full copy per step. None of it matters at the current sizes.
  Leave it until the benchmark exists.
- The scene is not re-centred or re-scaled before solving (document 02,
  "Conditioning"). In double on the CPU this is not needed; it belongs to milestone 4.
- The Ceres reference functor treats any model that is not one of the first three as
  RADIAL. A guard in `make_reprojection_cost` prevents misuse today; the functor
  needs explicit cases when OPENCV and OPENCV_FISHEYE are added.

## Changes before milestone 2

In order. Items 1 and 2 decide whether the milestone can meet its exit condition at
all; item 3 should be done before any new residual is added.

| # | Change | Why now |
|---|---|---|
| 1 | Owner: confirm that real ARKit or ARCore sessions can be exported for a private test set (document 07, owner decision 6). | The exit condition of milestone 2 is a real session going through the Python API. |
| 2 | Store a speed and iteration-count baseline for the current solver: the five BAL problems and a few synthetic ones, written as JSON by the CLI, under `bench/`. | Document 05 requires every new residual type to show that existing problems did not slow down. Without a baseline taken now, that cannot be shown. Deterministic counts (iterations, operator applications) work on hosted runners; times are taken by hand. |
| 3 | Reshape the CPU backend for several residual types, with no change in behaviour. Pose priors are residuals on camera-side blocks only: a relative prior touches two poses, a global prior one pose and, when the set has its own transform, one shared block. The camera-side products, the preconditioner blocks, the gradient, the cost and the predicted decrease each need a second source of terms, and the list of camera-side block kinds needs a third entry. | Adding this inside the present class would double its size. Done first, the refactor can be checked exactly: iteration histories must stay bit-identical on the parity suite. |
| 4 | One loss per residual type in `SolverOptions` (today one `loss` for reprojection), and the `loss.*` file sections. | Document 02 gives priors their own loss. |
| 5 | Make the automatic gauge aware of priors, following the table in document 02: nothing held with global priors as the frame; the reference pose only, without the scale coordinate, with relative priors. | Today `choose_gauge` looks only at fixed images and points. |
| 6 | Settle the two differences from Ceres above. Done (D62). | Cheap now; confusing later, once prior terms make histories harder to compare. |
| 7 | Decide the Python boundary: copy arrays in and out, or view them. `Problem` owns its arrays and the backend copies them again, so "without copying" (document 03) is not what the code does. Recommendation: copy. For 10^6 observations it is about 30 MB and a few milliseconds, and it keeps ownership simple. Normalise quaternions at the same boundary. | Has to be settled before the bindings are written. |
| 8 | Separate public headers from internal ones. | Before the bindings and the C-level API fix the surface. |
| 9 | Small fixes: JSON escaping in the CLI and parity tool, duplicate section names rejected, non-finite inputs reported by `validate`, tests for the failure paths and for unobserved blocks. Done in the follow-up. | Easier before more code depends on them. |
| 10 | Extend the Ceres harness: cost functors for the two prior types, and the two remaining camera models. | Level 1 and Level 2 checks for milestone 2 need them. |

Optional: add the sanitizer builds as a documented manual step, or to CI's full
mode. They cost nothing on hosted Linux runners, and the thread pool is written by
hand.

## Questions for the owner

1. Can real sessions be exported (item 1)?
2. Is copying at the Python boundary acceptable (item 7)?

## Follow-up (2026-11-14)

Done at the owner's request after the review:

- The step norm is now the norm of candidate minus current in stored coordinates,
  and the solve fails on the fifth consecutive invalid step, both as in Ceres (D62).
- `validate` names the first value that is not finite (intrinsics, poses, points,
  observations) and refuses a sigma that is not finite.
- The problem file reader refuses a section name that appears twice, and the writer
  refuses to produce one.
- The CLI, the parity tool and the Ceres tool escape strings in their JSON output.
  The CLI summary also reports the gauge's scale coordinate.
- New tests: a start with no finite cost ends in `Failure` with the parameters
  untouched; an image and a point without observations are left alone; a point
  observed twice in one image; a problem with no observations; the step norm
  convention; the duplicate-section and non-finite cases.

Checked afterwards: unit tests, the full parity set including venice-52 (38 of 38),
and the three sanitizers. The iteration histories of all five BAL problems (cost,
inner iterations and radius at every iteration) are bit-identical to those of the
previous commit, so the fixes changed no result.

Two things learned while checking:

- **Ceres is not repeatable from run to run with several threads.** Three runs of
  ladybug-49 with identical settings gave Ceres final costs of 13344.31821,
  13344.31895 and 13344.32537 (the last in 33 iterations, not 35), against our
  13344.31823 every time. The relative difference therefore moved between 1e-9 and
  5e-7, and the test's limit is 1e-6. The test has not failed, but the margin was
  thinner than the table in document 07 suggests. The owner chose to run the Ceres side on one thread (D63). Measured
  afterwards: eight runs of ladybug-49 give one Ceres result and a difference of
  5.96e-9 every time; final-93 alternates between two results 7e-12 apart; the four
  fast BAL problems reproduce the table in document 07; the venice-52 test takes
  about four minutes instead of two. Full parity set: 38 of 38.
- **A harmless-looking edit changed our own results.** Writing the point update as
  one expression let the compiler fuse the multiply and the add, and ladybug-49
  ended at 13344.318211 where it had ended at 13344.318229. The statement was put
  back in two parts, with a comment. This is the rounding sensitivity document 07
  already describes; the practical rule is to compare histories with the previous
  commit after any change to arithmetic in the backend, which is what the stored
  baseline of item 2 is for.
- **A new test passed on the Mac and failed on Linux and Windows.** The first
  quick CI run of this work failed in one place: the test for a start with no
  finite cost put a point at a camera centre, which gives a depth of exactly zero
  only with fused multiply-add. The test now uses a zero translation and a point at
  the origin, which is exact everywhere. A local build with `-ffp-contract=off`
  reproduces what the other platforms compute and is worth running before CI.
