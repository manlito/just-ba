# 15. Review of the fixes after the review of milestone 5

Review of the two commits that answered document 14 (`2a5308e..e1194d1`, 2026-11-15:
`6eb33fb`, code and tests; `e1194d1`, benchmark tooling, recorded results and
documents), and of the decisions they record, D101 to D104. It covers that change
only, not milestone 5 again. It records what was checked, what was found, and what
should change. The review changed no code; it corrected statements in the other
documents that did not match the code or the measurements: the test count on the Mac
and the exception against Caspar in document 07, a note on D102 and three open items
in document 00, and two sentences of `bench/competitors/README.md`.

## What was checked

On the same two machines as document 14: the owner's Mac (Apple M5 Pro, macOS 26.6)
and the owner's laptop (i7-9750H, RTX 2070 Max-Q 8 GB, CachyOS, GCC 16.2, CUDA 13.4,
cuDSS 1.0.0.35 from NVIDIA's redistributable archive, checksum verified against its
manifest). On the laptop everything was built in a temporary directory in user space,
removed afterwards; `/tmp`, `~/.nv` and `~/.cmake` were the same before and after.

| Check | Result |
|---|---|
| Mac: build (Release), with the Ceres harness (Ceres 2.2.0 with Accelerate) and the Python bindings | No warnings |
| Mac: every test, the slow one included | 312 of 312 pass (document 07 said 304; corrected) |
| Mac: address and undefined-behaviour sanitizers, every C++ test but the slow one | 309 of 309 pass, the 121 tests labelled `gpu` among them. The out-of-bounds read of document 14 (finding 2) is gone |
| Mac: Metal bit-identical to before the fixes. Builds of `2a5308e` and `e1194d1`; the 58 GPU test problems at the tests' tolerances and the five BAL problems at the defaults, each with both solvers | 125 of 126 solves identical: the summary, every field of every iteration but the time, and the solved problem file byte for byte. The one that differs is venice-52 with the iterative solver, which D103 says it changes |
| Mac: the problem generator before and after (`--far-points` added) | 9 of 9 files identical, the benchmark's and the GPU cases' options among them |
| Mac: `bench/run.py --compare` against the three baselines | CPU: counts, terminations and costs identical, time per iteration -4% to +2%. Metal sparse_schur: identical, -15% to +8%. Metal iterative_schur: identical, but +8% to +15% on the BAL problems in each of three runs, 4 to 6 problems over the 10% limit (finding 9) |
| Mac: Metal's speed-up against Ceres, run again with `--ceres` | Warm 2.7 to 11.8 times, at equal cost 2.1 to 28; the recorded 2.9 to 11.9 and 2.2 to 27 reproduce |
| Mac: wall time around `solve()` from Python against `time_seconds`, three problems, both solvers | Within 1.5 ms on every solve: the reported time is the call (D101) |
| Mac: venice-52 before and after D103, 50 and 300 iterations | D103's figures reproduce: with the iterative solver and 300 iterations allowed, invalid steps 36 to 23, ending at 250,211 instead of 249,523 (finding 7) |
| Laptop: build (Release) with `-DJBA_WITH_CUDA=ON -DJBA_WITH_CUDSS=ON` and the Python bindings | No warnings |
| Laptop: tests without the slow one, without the Ceres harness | 192 of 192 pass with `ctest -j 6`: the 120 tests labelled `gpu`, the Python tests with the new CUDA test, the rest. With the 118 tests of the Ceres harness, which was not rebuilt, that is the 310 of document 07 |
| Laptop: `bench/run.py --backend cuda --large --compare` on seven problems of the sparse baseline and five of the iterative one | Counts, terminations and final costs identical on all twelve; time per iteration -20% to +7% (sparse_schur), -4% to 0% (iterative_schur) |
| Laptop: D102's figures by hand: 500 images with tracks of 100; 10,000 images with tracks of 16 | Reproduce. Tracks of 100: `Auto` (sparse_schur) 2.29 s warm, set-up 0.74 s; iterative_schur 5.10 s. Tracks of 16: 4,591 MiB needed, 4,935 free; `Auto` runs with iterative_schur and says why in the message, at most 2.9 GB; sparse_schur asked for stops with that message as its error |
| Laptop: an allocation failure inside cuDSS's factorization, which the test hooks only imitate. A scratch copy with the two memory checks switched off, the same 10,000-image problem | cuDSS returns status 2 from the first factorization at 7.6 GB. `Auto`: the solve continues with iterative_schur, converges on, and reports the change. sparse_schur asked for: "ran out of device memory in the factorization; iterative_schur needs less", exit 1. cuDSS leaves no CUDA error behind, so the next `finish()` does not misreport it |
| Laptop: cuDSS's memory estimates | Index 1 is the peak device memory: cuDSS's own example in the archive (`examples/cpp/cudss_get_set.cpp`) prints index 0 as "device (stable)" and index 1 as "device (peak)". On the problem above the estimate (293 MB) was above what was free (195 MiB), so the check after the analysis would have caught it too |
| Laptop: wall time of the whole `just-ba` process against the cold time it reports | 0.14 to 0.16 s outside the call on three small problems (finding 5) |
| Laptop: venice-52 on CUDA | 17 invalid steps of 50 with sparse_schur, 271,831; as recorded |
| Recorded figures recomputed from `bench/baseline/*.json` and `bench/competitors/results/rtx2070maxq.json` | Every range in document 07's milestone 5 results, D101 and the status lines follows from the files, and `compare.py`'s ratios recompute from its own histories. D102's statement about the small dense problems does not follow (finding 2), and the equal-cost figures against Caspar rest on two different cost scales (finding 3) |
| The five baselines before and after | Counts and costs of every problem identical to the baselines of `2a5308e`, except venice-52 on Metal's iterative solver and on CUDA's sparse_schur; times now of the whole call |
| Code read: `backends/gpu/src/backend.cpp` against the former map-of-lists code, the device interface, `backends/cuda/src/backend.cu`, the Metal device, `solver.inc`, `api/src/solve.cpp`, `core/src/trust_region.cpp`, the CLI, the bindings, `tests/gpu_check.cpp`, `tests/gpu_cases.cmake`, `tests/synthetic.h`, the Ceres harness's timing, `bench/run.py`, `bench/competitors/compare.py` and `caspar.sh`, the CI workflow | Findings below |
| CI's new job, by reading and against the laptop's archive | The cuDSS archive's address and its `lib/cmake/cudss` are right (the same file was fetched here). The CUDA version it installs is not (finding 1) |
| Standing rules (CLAUDE.md) | Hold. The new job is on the free hosted Linux runner, with no GPU; the workflow is still started by hand only, with no schedule. No new dependency: the test hooks and the CI job add none |

Not run: the CI workflow itself (it runs on pushed commits; nothing was pushed); MSVC
on the shared GPU code; the Ceres harness, Caspar and bae on the laptop (their
recorded numbers are used as recorded); the slow test on the laptop and under the
sanitizers; the thread sanitizer; `compute-sanitizer`; two solves at once in one
process; a second CUDA device.

## Verdict

The fixes hold. Each finding of document 14 was looked at again:

| Finding of document 14 | State |
|---|---|
| 1. Reported time left out the set-up | Fixed (D101). Measured: `time_seconds` is the call to within 1.5 ms; the history's times count from the call; Ceres is timed from building its problem. What "cold" leaves out is in finding 5 |
| 2. Short buffers read past their end | Fixed. The sanitizer build passes |
| 3. Set-up of sparse_schur: quadratic lists on one thread, `Auto` not weighing it | The lists: fixed. Same entries in the same order (by reading, and 125 of 126 Metal solves bit-identical), in parallel with no shared writes; tracks of 100 on CUDA 11.7 s to 2.3 s. `Auto` not weighing it: answered by measurement (D102), but the statement of what was measured is not what the baselines show (finding 2) |
| 4. `Auto` failing for lack of memory | Fixed (D102), on all three paths: before the set-up (measured), a set-up that does not fit (forced by the test hook), a factorization that runs out part way (forced, and measured with a real cuDSS failure). After the switch every buffer the iterative path needs exists: they are allocated for both strategies before strategy B's own, and strategy B's stay allocated |
| 5. venice-52: invalid steps | In part (D103), as the documents say. Where no diagonal entry is lost `spd_inverse` does what it did: by reading, and every solve but venice-52's bit-identical. Finding 7 |
| 6. Comparison not recorded, ranges misquoted | Fixed. The runs are in the repository, `compare.py` writes the ratios, `run.py` writes both Ceres columns and the large problems, and equal cost is computed for Ceres as for the others. Finding 3 is about how it is computed for Caspar |
| 7. CI compiles nothing of the milestone | The shared layer is now built on every runner (GCC measured here, no warnings; MSVC not until the workflow runs). The CUDA job is written but, as written, cannot build (finding 1) |
| 8. 32-bit quantities | Fixed for the entries, values, E and G, and the non-zeros, with a fall-back under `Auto`. One index is left (smaller points) |
| 9. Tests | Python runs `Auto` on CUDA; a BAL-shaped case; each fallback forced. Gaps in finding 8 |
| 10. The CUDA device and the thread's state | Fixed as far as reading shows: the caller's device, restored after each call; `available()` runs a kernel. Not measured with a second device |
| 11. `run.py` leaves its problems | Fixed. Measured: no directory left; `--only` with `--compare` checks only what ran |
| 12, 13. Documents and comments | Brought in line |

Nothing found is a defect in the solver. The three findings that matter are a CI job
that will fail when it first runs, one sentence of D102 that the recorded baselines
contradict, and the way "equal cost" is computed against Caspar, which leans towards
Caspar and so leaves the claims standing.

On the fairness of the comparison as it is now described:

- **Ceres.** Fair. Both are timed from the call, both histories are on that clock,
  and equal cost is the time to the worse of the two final costs. Building the Ceres
  problem is counted, which is what D101 asks of both sides; it is 1 to 18 percent of
  Ceres's time (7.1 s of 39.4 s on the 10,000-image walk), so by Ceres's own
  `total_time_in_seconds` the warm range would be 3.3 to 11.3 instead of 3.4 to 13.8.
- **Caspar.** Fair, and on the careful side. End to end: Caspar's clock runs from
  the creation of its adjuster, after COLMAP has started; if that is after CUDA's
  start-up, as `bench/competitors/README.md` says, the warm figure is like for like
  and the cold one counts a start-up against just-ba that Caspar does not pay. The
  review could not check this, the raw logs not being in the repository; with the
  0.12 to 0.5 s of Caspar's own set-up that the two clocks differ by, no conclusion
  changes either way. At equal cost: finding 3.
- **bae.** Fair. bae's clock starts at its first step, after its problem is on the
  GPU, and just-ba's at the call; both ratios are quoted, and the smallest (2.7
  times, venice-52, where bae goes on to a lower cost than just-ba reaches) is
  reported as the exception it is.

The standing rules hold.

## Findings

Ranked by severity. "Measured" means reproduced in this review; "by reading" means
not reproduced.

### Medium

**1. The `cuda-compile` job installs a CUDA that cannot compile the project.** By
reading. `.github/workflows/ci.yml:107` sets `CUDA_VERSION: "13-0"` and line 116
installs `cuda-nvcc-13-0`. The project compiles its CUDA sources as C++23, required
(`CMakeLists.txt:34`, `35`), and D52 records that nvcc accepts `-std=c++23` from
CUDA 13.3; the option's own help says "13.3 or later" (`CMakeLists.txt:26`). So the
first run of the job stops in the configure or the build step, before anything of
finding 7 of document 14 is compiled. D104 says the job has not run; this is why it
matters that it has not. The rest of the job is plausible by reading and could not
be tried here: the package names follow NVIDIA's scheme (`cuda-nvcc`,
`cuda-cudart-dev`, `libcublas-dev`, `cuda-nvrtc`, each with the version suffix);
`libcudss.so` needs `libcublas.so.13` and `libnvrtc.so.13` when any test starts
(measured with `readelf` on the laptop), which the last two packages bring, provided
their directory is on the loader's path on the runner; and `/usr/local/cuda/bin`
exists only if the packages installed set that link up, where
`/usr/local/cuda-13.x/bin` is the path they own. Every test binary links the CUDA
backend in that job, so a library the loader does not find fails all of them, not
only the GPU ones.

**2. D102 says sparse_schur is the faster solver end to end "except the small dense
50-image problem"; the recorded baselines say otherwise for CUDA.** Measured from
`rtx2070maxq-cuda-sparse.json` and `rtx2070maxq-cuda-iterative.json`, and again on
the laptop:

| Problem | `Auto` (sparse_schur): warm, cold | iterative_schur: warm, cold | sparse over iterative, warm |
|---|---|---|---|
| synthetic pinhole, 50 | 0.096 s, 0.317 s | 0.072 s, 0.180 s | 1.3x |
| synthetic relative priors, 50 | 0.093 s, 0.308 s | 0.049 s, 0.160 s | 1.9x |
| synthetic global similarity, 50 | 0.096 s, 0.307 s | 0.056 s, 0.160 s | 1.7x |
| synthetic OPENCV, 30 | 0.049 s, 0.248 s | 0.035 s, 0.164 s | 1.4x |

On four of the five small dense synthetic problems `Auto` on CUDA is the slower
choice, by 1.3 to 1.9 times warm and 1.5 to 1.9 times cold, not on one problem by
1.25 times; this review's runs give the same (0.088 s against 0.071 s, 0.045 s
against 0.033 s). The Metal figures quoted in D102, 0.062 s against 0.048 s, are in
no file: the Metal baselines have 0.037 s against 0.038 s for that problem, and
sparse_schur slower on two others (0.038 s against 0.030 s, 0.043 s against
0.037 s). The amounts are 20 to 45 ms warm and 0.08 to 0.15 s cold, the cold part
being mostly cuDSS's start-up, and against Ceres all of these are still 6 to 12 times
faster warm. But these are the small class, where document 05 asks for the fastest
"including set-up and transfer time", and D102's reason for adding no rule ("no
rule by track length was added" because only one case lost) is weaker than written.
The iterative solver ends 4e-7 to 3e-5 above sparse_schur's cost there, within the
same tolerances.

**3. "At Caspar's final cost" compares two different cost scales, and the one
exception it reports is an effect of that.** Measured from
`bench/competitors/results/rtx2070maxq.json`. `compare.py:109` replaces the last
entry of Caspar's history by the cost recomputed in double from the model Caspar
wrote, takes that as the target, and looks for the first earlier entry at or below
it (`compare.py:36`); the earlier entries are Caspar's own single-precision
`score_best`. On the walks the two agree (1e-6 and 3e-5). On BAL Caspar's score is
below the recomputed cost of the same model:

| Problem | Caspar's score, last iteration but one | Its final model recomputed in double | Caspar "reaches its final cost" at iteration |
|---|---|---|---|
| dubrovnik-16 | 26,657 | 27,022 (+1.4%) | 137 of 199 |
| trafalgar-21 | 34,613 | 34,646 (+0.1%) | 186 |
| ladybug-49 | 13,596 | 13,603 (+0.05%) | 175 |
| venice-52 | 286,188 | 368,625 (+29%) | 35 |
| final-93 | 158,862 | 158,967 (+0.07%) | 123 |

So Caspar is credited with its final cost at the iteration where its own,
lower-reading score passes that number, which is earlier than its model was that
good; on venice-52 at iteration 35, 0.40 s, when its score had 164 iterations still
to fall. Every equal-cost ratio against Caspar is therefore a lower bound on
just-ba's side, and `bench/competitors/README.md`'s "credited as if it had stopped
as soon as it reached its final cost" understates the credit. The claims stand: the
end-to-end ratios do not use the history, and just-ba's final cost is below Caspar's
on both scales on every problem. What does not stand as written is the exception in
document 07 and its status line, "venice-52, where Caspar reaches its cost in 0.93
of just-ba's cold time": the file does not show when Caspar's model reached 368,625,
only when its score did. Also, the README says Caspar always runs 200 iterations;
the recorded history of the 2,000-image walk has 77.

### Low

**4. A cuDSS failure that is neither success nor an allocation failure is taken as
"not positive definite", every step, without a word.** By reading,
`backends/cuda/src/backend.cu:186` and `196`. This is D102's choice and right for a
matrix problem. But a failure that is not about the matrix (an internal error, an
execution failure, a library that stopped working) now gives a solve that forms the
tiles, fails the factorization and runs conjugate gradients at every step, reports
`linear_solver: sparse_schur` and an empty note, and is slower than either solver.
Nothing in `SolverSummary` counts the steps that fell back; only a history with more
than one linear iteration per step shows it.

**5. "Cold" is the first call of a process, not a command-line run.** Measured on
the laptop, the median of five runs: the `just-ba` process takes 0.14 to 0.16 s
longer than the cold `time_seconds` it prints (0.45 s against 0.29 s on the 50-image
pinhole problem, 0.39 s against 0.24 s on the 30-image OPENCV one). About 0.05 s is
starting the process and reading the file (`just-ba info` alone); the rest was not
split further, and loading the CUDA libraries before `main` and releasing the device
after the solve are both outside the call. Document 07 and D101 describe cold as what "a
command-line run pays". For the library's claim the call is the right boundary, and
Ceres's column leaves its process and file reading out as well; but the smallest
cold ratio, 1.2 times on the 30-image OPENCV problem (0.248 s against Ceres's
0.298 s), is smaller than what is outside the call, so as two command lines that
problem is not shown to be faster.

**6. Smaller points in the CUDA device.** By reading, `backends/cuda/src/backend.cu`:

- `available()` (line 326) keeps its first answer for the life of the process,
  for the device that was current in that call. A device that was full or busy at
  that moment, or a second device of another architecture, is not asked again.
- A pinned host allocation that fails (line 244) is reported as "out of device
  memory" and, under `Auto`, starts the whole set-up again with the iterative
  solver, which needs the same host mirrors.
- Under `Auto`, any `DoesNotFit` restarts the set-up (`backend.cpp:949`), also one
  from a buffer of strategy A itself; the second attempt then fails the same way
  after building the host system twice. Correct, only slow to say no.

**7. D103 changes venice-52 and nothing else, and where it changes it the cost ends
higher.** Measured. Metal, iterative solver: 50 iterations, 14 invalid steps before
and after, 253,423 to 254,246; 300 allowed, 36 to 23 invalid, 249,523 to 250,211.
CUDA, sparse_schur: 50 iterations, 19 to 17 invalid, 270,448 to 271,831; 300
allowed, 135 to 123 invalid, 263,801 (document 14) to 264,155. D103 states its
figures honestly and the documents keep venice-52 open; this only adds that the
change has not yet lowered a cost anywhere it was measured, so the case for it is
the fewer discarded steps alone. On CUDA at 300 iterations 123 of 254 steps are
still invalid.

**8. Tests: what the new ones do not hold.** By reading:

- The Python CUDA test (`bindings/python/tests/test_bindings.py:404`) runs `Auto`
  but does not look at `s.linear_solver`, so a build with cuDSS that fell back to
  the iterative solver for every solve would pass.
- The forced fallbacks run on one small pinhole case. The real path was measured
  once in this review, by hand; nothing in the repository reaches a real allocation
  failure, and it cannot without a full device.
- The test hooks (`backends/gpu/src/backend.cpp:55`, `jba/gpu/backend.h:72`) are
  process-wide variables compiled into the library that ships. They are only read
  during a solve, so two solves at once are safe as long as no test arms them, and
  the header says so.
- Nothing tests `--warm-up`, or `setup_seconds` and `linear_solver` in the summary
  on the CPU and Metal; `run.py` checks on every problem that the warm and the cold
  solve give the same counts and cost, which is the check that the warm-up does not
  disturb the timed solve (it holds on every problem run here), but `run.py` is not
  a test.

**9. The Metal iterative baseline did not reproduce within its limit.** Measured,
three runs: counts and costs identical, time per iteration +8% to +15% on the five
BAL problems, 4 to 6 problems over 10% each time, while the CPU and the Metal
sparse_schur baselines reproduce in the same session. The Mac was in use (the window
server at about 45 percent of a core), which a solver that is all GPU feels most, so this
may be the session and not the baseline. Against that: the build of `2a5308e`,
measured in the same minutes, gives ladybug-49's iterations in 0.41 s, which is its
old baseline (0.413 s), and the fixes do not touch that path but for one branch; the
new baseline has 0.370 s for the whole call. All five baselines carry the commit
`2a5308e-dirty`: they were taken from a working tree before the fix commit existed,
and only the identical counts and costs measured here tie them to `e1194d1`.

### Smaller points, no action needed now

- One 32-bit index of strategy B is not in the new check: the tile kernels address
  `tile_partial` and the tiles' values by `64 * chunk` and `64 * tile` in 32 bits
  (`backends/gpu/kernels/solver.inc:696`, `703`), which passes 2^32 at 67 million
  tile chunks. The memory check stops this first on any device under about 20 GB,
  and the non-zero check on tiles of full size; small blocks on a large device
  could pass both. The checks at `backend.cpp:509` also admit exactly 2^32.
- `run.py:197` writes the run's `linear_solver` from the last problem solved. Under
  `Auto` with a problem that falls back, the file's header then names the solver of
  that problem, and `--compare` accepts or refuses the baseline by it.
- `run.py:224` and `compare.py` test the two times with `if a and b`: a time of
  exactly zero would drop the ratio. It does not occur in the recorded files.
- Document 07's Metal range "at equal cost 2.2 to 27" has venice-52 as its upper
  end, the problem the same paragraph sets apart; without it the range is 2.2 to
  11.6.
- `SolverSummary::linear_solver` after a change part way names the solver of the
  last steps; the message says that the first steps were sparse_schur.
- After a change part way strategy B's buffers stay allocated until the solve ends
  (4.6 GB in the case measured). Nothing needs the memory then.
- The warm-up solves a copy of the same problem, so the timed call also finds the
  allocator and the device's memory pool warm for exactly its sizes. That is what a
  second solve of the same session sees; a first solve of a different size in a
  started process would lie between warm and cold.
- MSVC: nothing was found by reading that it should refuse in
  `backends/gpu/src/backend.cpp` or the headers now that the column starts are
  64-bit; this stays unverified until the workflow runs on Windows.

## Recommended changes

In order.

| # | Change | Why |
|---|---|---|
| 1 | Set the CI job's CUDA to a version that D52 allows (13.3 or later; the laptop's is 13.4), put the versioned `bin` directory on the path, and run the workflow once by hand before relying on it. | Finding 1. One line, and the only way to know the rest of the job. |
| 2 | Correct D102's sentence from the baselines (done in this review as a note), and decide whether `Auto` on CUDA should take the iterative solver for small dense problems: the baselines suggest a rule on how full and how small the reduced system is. Measure before adding it. | Finding 2. The small class is where set-up counts. |
| 3 | In `compare.py`, compare like with like: either recompute Caspar's cost in double at more than the last iteration (the model at a few iteration limits), or scale its history so that its last score equals the recomputed cost, and say which. Until then quote the equal-cost figures against Caspar as lower bounds and drop the 0.93. | Finding 3. |
| 4 | Count the steps that fell back in `SolverSummary` (or in the message), and treat a cuDSS status that is not about the matrix as an error after it repeats. | Finding 4. |
| 5 | Say in documents 05 and 07 that cold is the first call of a process, and what a command-line run adds on CUDA (about 0.1 s besides reading the file). | Finding 5. |
| 6 | Python CUDA test: assert `linear_solver` is sparse_schur when the build has cuDSS. A ctest case for `--warm-up` and the summary's new fields on the CPU. | Finding 8. |
| 7 | Record the Metal iterative baseline again on an idle machine, from a clean commit. | Finding 9. |
| 8 | Add `64 * tile chunks` to the 32-bit check. | Smaller points. |

## Questions for the owner

1. Should `Auto` on CUDA choose the iterative solver for small dense problems
   (finding 2), or is 20 to 45 ms there accepted for the simpler rule?
2. Is the equal-cost comparison with Caspar to be redone with its costs recomputed
   in double along the way (finding 3), or quoted as a lower bound?
