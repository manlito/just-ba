# GPU speed comparison (milestone 5)

How the competitors of document 05 were built and run against the CUDA backend on the
owner's laptop (Intel i7-9750H, 12 threads; NVIDIA RTX 2070 Max-Q, 8 GB; CachyOS,
CUDA 13.4, GCC 16), on 2026-11-15, and again after the review of milestone 5 (document 14) with times of
the whole call (D101). Results are in document 07 (milestone 5), in
`bench/baseline/rtx2070maxq-cuda-sparse.json` (Ceres) and in
`results/rtx2070maxq.json` (Caspar and bae: per problem every time, final cost and
cost history, written by `compare.py` from the raw outputs). Everything was built in user space, in
a temporary directory that was removed afterwards; none of it is a dependency of the
library.

## What was built

| Competitor | Version | Build |
|---|---|---|
| Ceres (CPU) | 2.2.0 | SuiteSparse 7.14.1 (CHOLMOD) with OpenBLAS 0.3.34 and LAPACK, Eigen 3.4.0, glog 0.7.1, gflags 2.2.2, `SCHUR_SPECIALIZATIONS=ON`, static |
| Ceres (GPU) | master `e831bfef` (2.3.0) | as above, plus `USE_CUDA`, cuDSS 1.0.0 (`CUDA_SPARSE`), architecture 75 |
| COLMAP Caspar | 4.2.1 | `CUDA_ENABLED=ON CASPAR_ENABLED=ON`, single precision (Caspar's default), architecture 75, GUI, OpenGL and MVS off; against the Ceres 2.2.0 above |
| bae | tag 0.2 (`c9be7bf`) | PyPose 0.9.5, PyTorch 2.14.1 for CUDA 13.0, warp-lang 1.19.0, extensions built for sm_75; without cuDSS (0.2's cuDSS code predates cuDSS 1.0; its BAL example uses PCG) |

just-ba was built with `-DJBA_WITH_CUDA=ON -DJBA_WITH_CUDSS=ON` and, for the Ceres
columns, `-DJBA_WITH_CERES=ON` (with `-DJBA_CERES_VERSION=2.3.0` against master).

## How each was run

On the laptop, one GPU job at a time. just-ba's times are of the
whole call to `solve()`, set-up included (D101): cold, the first call of a fresh
process (device start-up included), and warm, the best of three calls with the device
already started (`--warm-up`). Both are reported.

- **Ceres:** `bench/run.py --huge --backend cuda --ceres <2.2.0 jba_ceres_solve>
  --ceres-sparse-library suite_sparse --ceres-gpu <master jba_ceres_solve>`: the same
  problem files, gauge and losses, SPARSE_SCHUR, all threads. Ceres's time runs from
  building its problem to the end of `Solve()`, and its per-iteration times are on
  the same clock.
- **Caspar:** `caspar.sh` per problem. `jba_to_colmap` writes one problem in both
  forms. Caspar handles only the reprojection residual on SIMPLE_RADIAL or PINHOLE
  cameras, so BAL is reduced to SIMPLE_RADIAL (`--simple-radial`, BAL's k2 dropped),
  and the walks (500 and 2,000 images) are generated without pose priors. COLMAP's
  `bundle_adjuster` runs `--BundleAdjustment.backend CASPAR` with the same intrinsics
  freed as in the problem file. Caspar always runs 200 iterations
  (`BundleAdjustmentCaspar.solver_iter_max`), three times per problem; the fastest
  run is used. Its time is taken from glog's timestamps, from "Creating Caspar bundle
  adjuster" to its termination line ("Caspar: MAX_ITERATIONS" or another status),
  which is its set-up and solve, after COLMAP has read the model and started CUDA.
  Its final cost is recomputed in double from the model it writes
  (`colmap_cost.py`). Caspar's results differ from run to run.
- **bae:** `bae_bal.py`, the BAL example with its own settings: 20 LM steps, PCG
  (tolerance 1e-4, 250 iterations at most), float64. Its time runs from its first
  step and includes that step's set-up, about 1.2 s, after the problem is loaded on
  the GPU. just-ba solves the same BAL file (its RADIAL model is bae's f, k1, k2).

`compare.py <results> results/rtx2070maxq.json` reads the raw outputs (caspar.sh's
and bae_bal.py's, and just-ba's JSON and histories next to them) and writes, per
problem, every time, final cost and cost history, and the ratios below.

## Equal accuracy

"Faster at equal accuracy" (document 07) is measured as the time each solver needs to
reach the worse of the two final costs, within a relative 1e-6, using per-iteration
histories: just-ba's `--history` (times since the call), Ceres's per-iteration times,
Caspar's per-iteration `score_best` and `dt_tot` (its own iteration clock, which
starts after its set-up, so Caspar is credited as if it had stopped as soon as it
reached its final cost, and against just-ba's cold call, whose clock includes the
device's start-up, the comparison leans towards Caspar), and bae's per-step costs.
