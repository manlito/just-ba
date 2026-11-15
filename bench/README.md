# Benchmark

`run.py` solves a fixed set of problems with the CLI and records, per problem, the
iteration count, accepted steps, linear iterations, termination, initial and final
cost, and the best time of three runs (D81). `--compare` checks a run against a
stored baseline:

- counts, termination and final cost (to 1e-9 relative) must not change. For a given
  platform they are deterministic (D59), so a change means the arithmetic changed;
  document 12 shows how easily that happens.
- time per iteration may grow by at most 10% (`--max-slowdown`).

Problems: the five BAL problems (`python3 tests/data/fetch_bal.py`) and five
synthetic ones written by `jba_make_problem` (built with the tests): pinhole, radial
with free intrinsics, relative priors with outliers, global priors in a similarity
frame, and OPENCV.

```bash
python3 bench/run.py --build build/dev --compare bench/baseline/apple-m5pro.json
```

```bash
python3 bench/run.py --build build/dev --out bench/baseline/apple-m5pro.json
```

The first command checks the current build against the baseline; the second replaces
the baseline. Do that only when a change is meant to alter results or speed, and say
why in the commit.

Baselines are per machine class, under `baseline/`. Counts differ between platforms
(fused multiply-add on Apple and ARM, not on x86-64 without it), and times differ
everywhere, so a run is compared only with the baseline of its own machine. Times
are taken by hand on an otherwise idle machine; CI does not run this.

| Baseline | Machine and configuration |
|---|---|
| `apple-m5pro.json` | The owner's Mac (Apple M5 Pro), all threads; CPU backend |
| `apple-m5pro-metal-sparse.json` | The same Mac; Metal backend, sparse_schur (its default), with Ceres with Accelerate alongside |
| `apple-m5pro-metal-iterative.json` | The same Mac; Metal backend, iterative_schur, with Ceres with Accelerate alongside |

`--backend metal|cuda` and `--linear-solver` select the configuration (default: the CPU;
the linear solver defaults to auto, the library's rule). A baseline records its
configuration and `--compare` refuses one of another. `--ceres <jba_ceres_solve>`
solves every problem with Ceres as well (SPARSE_SCHUR, the same gauge and losses)
and records its iterations, cost and time next to ours (`--ceres-sparse-library`:
`eigen`, `accelerate`, `suite_sparse` or `cuda`, as that Ceres was built); the milestone 4 speed
reference is a Ceres built with Accelerate:

```bash
cmake -DPREFIX=$PWD/build/ceres-2.2.0-accelerate -DACCELERATE=ON -P third_party/ceres/build.cmake
```

```bash
python3 bench/run.py --build build/dev --backend metal --ceres build/accel/tests/ceres/jba_ceres_solve --compare bench/baseline/apple-m5pro-metal-sparse.json
```

where `build/accel` is a just-ba build configured with
`-DJBA_WITH_CERES=ON -DCMAKE_PREFIX_PATH=$PWD/build/ceres-2.2.0-accelerate`. `--large` adds two
problems in the main size range (walks of 500 and 2,000 images); they are not in the
baselines.
