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

| Baseline | Machine |
|---|---|
| `apple-m5pro.json` | The owner's Mac (Apple M5 Pro), all threads |
