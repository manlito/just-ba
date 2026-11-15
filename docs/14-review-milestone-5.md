# 14. Review of milestone 5

Review of the code and documents as of commit `66ad963` (2026-11-15): milestone 5
(`53bb968..66ad963`, twelve commits: the shared GPU layer and Metal moved onto it,
the CUDA device, sparse_schur with cuDSS, the rework of strategy B, the speed
comparison and its tooling). It records what was checked, what was found, and what
should change. The review changed no code; it corrected statements in the other
documents that did not match the code or the measurements.

## What was checked

On two machines: the owner's Mac (Apple M5 Pro, macOS 26.6, Apple clang 21) and the
owner's laptop (i7-9750H, RTX 2070 Max-Q 8 GB, CachyOS, GCC 16.2, CUDA 13.4, cuDSS
1.0.0.35 from NVIDIA's redistributable archive, in user space). On the laptop
everything was built in a temporary directory, removed afterwards.

| Check | Result |
|---|---|
| Mac: build (Release), with the Ceres harness and the Python bindings | No warnings |
| Mac: tests without the slow one: unit, Python, the three real captures on three configurations, 114 Metal tests, 60 parity tests | 303 of 303 pass |
| Mac: the slow test (venice-52 parity) | Passes |
| Mac: address and undefined-behaviour sanitizers, every C++ test; ladybug-49 for 5 iterations on the CPU and on each Metal solver, and 12 iterations with the gauge free | 113 of 297 fail: every `metal_*` test, on one out-of-bounds read (finding 2). With that one line corrected in a scratch copy, 297 of 297 pass. The ladybug runs are clean (49 intrinsics blocks, so the read stays in bounds) |
| Mac: thread sanitizer, same runs | Clean, 297 of 297 |
| Mac: generated residual code (`generate.py --check`) | Current, the new CUDA file included |
| Mac: CPU benchmark against `apple-m5pro.json` | Counts, terminations and final costs identical. Time per iteration: first run +2% to +26%, three small problems over the limit; two more runs of those, -3% to +9%, and the build of `53bb968` gives the same. Noise at 2 to 4 iterations of 7 ms |
| Mac: Metal benchmarks against both Metal baselines | Counts, terminations and final costs identical; times -35% to +7% (sparse_schur), -29% to 0% (iterative_schur) |
| Mac: Metal bit-identical to before the milestone. Builds of `53bb968` and `66ad963`; the 57 GPU test problems at the tests' tolerances and the five BAL problems at the defaults, each with both solvers (the fallback case with sparse_schur only) | 123 of 123 solves identical: the summary, every field of every iteration but the time, and the solved problem file byte for byte. (The documents say 124 solves.) |
| Laptop: build (Release) with `-DJBA_WITH_CUDA=ON -DJBA_WITH_CUDSS=ON`, the Ceres harness (Ceres 2.2.0, Eigen only) and the Python bindings | No warnings |
| Laptop: tests without the slow one: unit, Python, 113 CUDA tests, 60 parity tests | 302 of 302 pass serially; 302 of 302 in each of three runs with `ctest -j 12`; the 113 CUDA tests with `-j 24` (300 without the Python bindings, the number in document 07) |
| Laptop: the slow test | Passes (675 s) |
| Laptop: every CUDA solve of the list above (123), run twice in separate processes | Bit-identical, 123 of 123 |
| Laptop: `compute-sanitizer` on CUDA solves, both solvers: memcheck (three cases), racecheck and initcheck (three cases, the fallback case included) | No errors, no hazards |
| Laptop: `bench/run.py --backend cuda --large --compare rtx2070maxq-cuda-sparse.json` | Counts, terminations and final costs identical on all twelve problems; time per iteration -21% to +2% |
| Laptop: wall time of `solve()` against the time the benchmark reports, every baseline problem (also on the Mac for Metal) | The reported time leaves out the set-up: finding 1 |
| Laptop: the large class by hand: the 10,000-image walk of document 07; the same with tracks of 16; 500 images with tracks of 100 | Document 07's 1.6 s and 3.6 GB reproduce (1.70 s, 3,627 MiB) as the reported time; findings 1, 3 and 4 |
| Laptop: venice-52 on CUDA and Metal, both solvers, 50 and 300 iterations; an instrumented scratch copy that also factorizes each reduced system with a dense Cholesky on the host | Finding 5 |
| Metal against CUDA on the 123 solves | Same iteration counts and costs to 1e-8 on 101; the rest differ in counts by a few iterations and agree in cost to 1.5e-9 on the test problems and 1e-6 on BAL, venice-52 apart. As expected: not bit for bit between backends |
| Recorded speed figures against the baseline file | The Ceres columns of document 07 follow from `rtx2070maxq-cuda-sparse.json`, two ranges misquoted (finding 6). Caspar's and bae's figures have no recorded data to check against |
| Code read: `backends/gpu` (layout, device interface, host logic, kernels), `backends/cuda`, the Metal device, the changes to `api`, `core`, the bindings, the CLI, the generator, `tests/gpu_check.cpp`, `tests/to_colmap.cpp`, `bench/run.py`, `bench/competitors`, the CI workflow | Findings below. No remaining use of the legacy default stream: every copy, memset and kernel is on the device's stream, and every upload from a pinned mirror is waited for before the host writes that mirror again |
| Standing rules (CLAUDE.md) | Hold. The CI workflow is unchanged by the milestone: manual trigger, free hosted runners, no schedule, no GPU job. The residual set is closed. New link dependencies are optional and CUDA-only: the CUDA runtime (static) and cuDSS, which itself loads cuBLAS and NVRTC from the toolkit. Ceres, SuiteSparse, COLMAP and bae are test or benchmark tools only; PyTorch appears only in bae's own environment |

Not run: Ceres with SuiteSparse, Ceres master with cuDSS, Caspar and bae (their
recorded times are used as recorded); the Windows build and tests, which the owner
has left pending and which is not a finding; two solves at once in one process on
CUDA; the real captures on CUDA (they are not on the laptop); a GPU of another
architecture than the build machine's; the address sanitizer on the laptop.

## Verdict

The engineering of the milestone holds. Metal on the shared layer gives the results
it gave before, bit for bit. The CUDA backend agrees with the CPU on every GPU case
within the Metal tests' limits, repeats bit for bit across processes, is clean under
NVIDIA's memory, race and initialization checkers, and reproduces its baseline. The
earlier stream-ordering bug has no remaining instance.

The speed claim does not hold as written. The times compared with the competitors
start after just-ba's backend is built; document 05 requires them to include the
set-up, and for sparse_schur the set-up is longer than the iterations on seven of
the twelve baseline problems (finding 1). With the set-up counted, just-ba is still
faster than Ceres on every problem measured, by 1.7 to 7.4 times instead of 3.5 to
17.6, once the device is started. Against Caspar and bae nothing can be re-derived,
because their runs were not recorded (finding 6); the margins "to the competitor's
cost" are large, but the end-to-end minimum against Caspar, 1.7 times, is smaller
than the factor the set-up adds on several problems. Whether the exit condition is
met against Caspar is therefore open until the comparison is repeated with times
taken around `solve()`.

Besides that, five things need attention:

- An out-of-bounds read on every GPU solve with one camera, new in this milestone,
  which makes the sanitizer build fail (finding 2).
- The set-up of sparse_schur itself: it grows with the square of the track length,
  runs on one host thread, and makes `Auto` the slower choice end to end on dense
  problems (finding 3).
- `Auto` on an 8 GB card fails with an error inside the supported size range, where
  the iterative solver runs (finding 4).
- venice-52, the recorded exception: on CUDA nearly half of sparse_schur's steps are
  thrown away, and the reason given in D99 is not the one measured (finding 5).
- CI compiles nothing of this milestone (finding 7).

The standing rules hold.

## Findings

Ranked by severity. "Measured" means reproduced in this review; "by reading" means
not reproduced.

### High

**1. The reported solve time leaves out the backend's set-up, and the speed
comparison is built on it.** `SolverSummary::time_seconds` starts in
`run_trust_region` (`core/src/trust_region.cpp:23`), after `solve()` has built the
backend (`api/src/solve.cpp:118`, `124`). For a GPU backend the constructor does
everything that is not an iteration: the host layout, every allocation and upload,
and for sparse_schur the tile lists and the analysis by cuDSS or Accelerate
(`backends/gpu/src/backend.cpp:103` to `459`). `bench/run.py` and the histories
used for "equal accuracy" read that field. Document 05 says the opposite is
required: "Time is measured end to end from the solver call to the result, not only
the inner solve", and the small class is to be fastest "including set-up and
transfer time". Measured on the laptop, wall time around `solve()` from Python,
CUDA with sparse_schur, the median of the second and third solve in a process:

| Problem | Reported | Around `solve()` | Set-up | Ceres 2.2.0 (recorded) | Speed-up recorded | Speed-up with set-up |
|---|---|---|---|---|---|---|
| ladybug-49 | 0.267 s | 0.318 s | 0.051 s | 1.110 s | 4.0x | 3.5x |
| trafalgar-21 | 0.039 s | 0.067 s | 0.028 s | 0.372 s | 5.1x | 5.6x |
| dubrovnik-16 | 0.069 s | 0.123 s | 0.054 s | 0.910 s | 9.6x | 7.4x |
| final-93 | 0.320 s | 1.006 s | 0.686 s | 2.688 s | 7.9x | 2.7x |
| synthetic pinhole, 50 | 0.038 s | 0.440 s | 0.402 s | 0.945 s | 13.9x | 2.1x |
| synthetic radial, free intrinsics, 30 | 0.343 s | 0.583 s | 0.240 s | 1.244 s | 3.5x | 2.1x |
| synthetic relative priors, 50 | 0.031 s | 0.439 s | 0.408 s | 0.829 s | 14.0x | 1.9x |
| synthetic global similarity, 50 | 0.041 s | 0.457 s | 0.416 s | 0.797 s | 13.9x | 1.7x |
| synthetic OPENCV, 30 | 0.015 s | 0.125 s | 0.110 s | 0.267 s | 8.9x | 2.1x |
| walk, 500 images | 0.088 s | 0.276 s | 0.188 s | 1.500 s | 13.2x | 5.4x |
| walk, 2,000 images | 0.380 s | 1.139 s | 0.759 s | 6.871 s | 17.6x | 6.0x |
| walk, 10,000 images | 1.70 s | 6.7 s | 5.0 s | 32.2 s | 20x | 4.8x |

(The recorded speed-up divides Ceres's time by the baseline's CLI time, which is a
first solve in its process and so a little longer than the "reported" column here;
trafalgar-21's ratio rises for that reason.) Against Ceres master on the GPU the
range becomes 1.9 to 6.8 times instead of 2.6 to 14.9. The first solve of a process
pays the device's start-up on top, also outside the reported time: 0.12 to 0.33 s
more on these problems. Counted that way, as a command-line run is, the range
against Ceres on the CPU is 0.8 to 5.1 times, and the 30-image OPENCV problem is
slower than Ceres (0.33 s against 0.27 s). With the iterative solver the set-up is
5 to 160 ms; it is the tile lists of strategy B that cost (finding 3). The same
holds on Metal (M5 Pro): set-up 8 to 235 ms against reported times of 7 to 500 ms.
Milestone 4's table says "for the solve alone", which is accurate, but it is then a
table of iterations, not of calls.

What this does to the claims:

- Against Ceres the exit condition still holds on every problem, by smaller margins,
  once the device is started; in a fresh process on all but the 30-image OPENCV
  problem. Ceres's own time, `total_time_in_seconds`, includes its preprocessing.
  Ceres was not re-run: its times are the baseline's.
- Against Caspar, `bench/competitors/README.md` said Caspar's time "is set-up and
  solve, as just-ba's `time_seconds` is". It is not: Caspar was timed with its
  set-up and just-ba without. The end-to-end ratios (1.7 to 18.5) are biased by the
  set-up column above, and the smallest of them cannot be assumed to stay above 1.
  The "time to Caspar's cost" ratios (6.8 to 41) compare histories, where Caspar's
  `dt_tot` may or may not include its set-up; not checked.
- Against bae the margins (26 to 67 times) are far larger than the set-up.
- D98 chose `Auto` from reported times (finding 3).

### Medium

**2. Every initialized buffer shorter than 16 bytes is read past its end.**
`backends/gpu/src/backend.cpp:116` rounds the size up to 16 bytes and passes that
size with the caller's pointer, and the devices copy that many bytes from it
(`backends/metal/src/backend.mm:175`; `backends/cuda/src/backend.cu:166`, `172`).
Before the milestone the Metal backend copied `count * sizeof(T)`. Measured with the
address sanitizer: `heap-buffer-overflow`, a read of 16 bytes from the 4-byte
`intr_off` of a problem with one intrinsics block (`backend.cpp:180`); the
per-block masks and any other table of fewer than four entries do the same. So
every GPU solve with one camera, the normal case, reads up to 14 bytes beyond a heap
block: measured on Metal, and by reading the same on CUDA, whose device copies the
same size. The bytes are never used and no Release test notices, but it is
undefined behaviour, and the sanitizer build, which document 13 found clean, now
fails 113 of its 114 GPU tests at the first solve and checks nothing after it.

**3. The set-up of sparse_schur grows with the square of the track length, on one
host thread, and `Auto` does not weigh it.** `setup_direct`
(`backends/gpu/src/backend.cpp:355` to `417`) lists every ordered pair of a point's
observations for every pair of block roles into a `std::map` of vectors, copies
them into one array and uploads them; document 13's finding 10 described the memory,
and the time is the larger cost. Measured on the laptop:

| Problem | sparse_schur: reported, around `solve()`, GPU memory | iterative_schur: reported, around `solve()`, GPU memory |
|---|---|---|
| synthetic pinhole, 50 images, every point in every image | 0.038 s, 0.44 s | 0.058 s, 0.073 s |
| 500 images, 20,000 points, tracks of 100 (2.0 million observations) | 1.59 s, 11.7 s, 2.5 GB | 4.95 s, 5.1 s, 0.4 GB |
| final-93 | 0.32 s, 1.0 s | 4.08 s, 4.1 s |
| walk, 2,000 images, tracks of 8 | 0.38 s, 1.14 s | 8.9 s, 9.1 s |

On the first two `Auto` picks the solver that is slower for the caller, by 6 and by
2.3 times, while the reported times say it is faster. On the walks and on BAL it is
still the right choice. D98's "slower only on the small dense synthetic problems, by
20 to 40 ms" is a statement about reported times. The same on Metal: 0.108 s
against 0.036 s on the 50-image problem. A capture that circles an object, with
tracks of a hundred images, is not exotic.

**4. Inside the supported range, `Auto` on the 8 GB card ends in an error where the
iterative solver runs.** Measured: a walk of 10,000 images and 1.2 million points
with tracks of 16 (19.2 million observations). `just-ba solve --backend cuda` stops
after 20 s with `cudss: factorization failed (status 2)`, GPU memory at 7.6 GB;
status 2 is cuDSS's allocation failure. `--linear-solver iterative_schur` solves
it in 264 s with 2.9 GB. Two things meet here. `resolve_linear_solver`
(`api/src/solve.cpp:68`) picks sparse_schur whatever the problem's size and the
device's memory, and strategy B needs several times the memory of strategy A (E, G
and the tile entries). And a cuDSS call that does not return success is thrown
(`backends/cuda/src/backend.cu:48`, `111`): D98's fallback to conjugate gradients
covers what cuDSS reports through its info value, a failed pivot, and not a
factorization that could not run. Nothing tests either path, and the message does
not say that memory ran out or that another solver would work. Document 03 records that the CPU
fallback is not built; this is the case before it: a GPU solver that would have
worked.

**5. venice-52: nearly half of CUDA's sparse_schur steps are thrown away, and D99's
explanation is not what is measured.** D99 says exact steps take venice-52 into
worse regions than inexact ones. Measured, gauge held, default tolerances:

| Configuration | 50 iterations | 300 allowed |
|---|---|---|
| Metal, sparse_schur | 236,658; 0 invalid steps, all 50 factorizations succeed | 236,560 after 168; 0 invalid; 2.0 s |
| Metal, iterative_schur | 253,423 | 249,523 after 92; 36 invalid; 1.3 s |
| CUDA, sparse_schur | 270,448; 19 invalid | 263,801 after 281; 135 invalid; 8.7 s |
| CUDA, iterative_schur | 251,841 | 251,831 after 61; 17 invalid; 3.3 s |
| CPU, double: the laptop; the Mac | | 238,466 after 235, 213 s (D99's figure); 248,409 after 143, 40 s |
| Ceres 2.2.0, SuiteSparse (recorded) | 261,813 | |

Metal's exact steps reach the lowest cost of all, below bae's 245,656. What goes
wrong on CUDA is something else. An instrumented scratch copy shows that at 28 of
the 50 factorizations the reduced system, formed in single precision, is not
positive definite: cuDSS says so, and a dense Cholesky of the same values in double
on the host agrees every time, failing almost always in the rows of one camera,
whose diagonal is at times not even positive. So cuDSS is right and the fallback is taken. But on 14
of the 28 the fallback fails at once: the same camera's preconditioner block cannot
be inverted (`backends/gpu/kernels/solver.inc:436`), `pcg_init` ends the loop as
failed (`solver.inc:969`), and the step is invalid with no linear iteration
(`backends/gpu/src/backend.cpp:741`). The trust region then halves and quarters the
radius until the damping covers the rounding, takes a step, grows the radius again,
and repeats. Each thrown-away step costs a full step. The iterative solver has the
same failure on both backends (36 of 92 steps on Metal). It is a property of the
shared layer since milestone 4, not of cuDSS; venice-52 on CUDA happens to walk into
it and on Metal's sparse_schur path does not. The exception is real, but it is a
defect to look at (one camera block losing definiteness in single precision), not a
law about exact steps, and it is the mechanism real captures with far points would
hit.

**6. The comparison with Caspar and bae is not recorded, and two recorded ranges
are misquoted.** `rtx2070maxq-cuda-sparse.json` holds just-ba's and Ceres's
numbers only. Caspar's and bae's times, costs and histories, from which "6.8 to 41
times", "1.7 to 18.5 times" and "26 to 67 times" were computed, exist only as those
sentences in document 07; the temporary directory they were in is gone. Caspar's
results differ from run to run and were run three times; which run was used is not
stated. Nobody, this review included, can check those figures or redo the
arithmetic with set-up times (finding 1). The baseline's `ceres_gpu` entries and
the two build descriptions were merged in by hand: `bench/run.py` writes one Ceres
column, so the file cannot be regenerated by the command in `bench/README.md`. The
10,000-image walk is in no file. Checked against the file, document 07's Ceres
figures hold except: "BAL 3.5 to 9.6" (the 3.5 is the synthetic radial problem; BAL
is 4.0 to 9.6), and "2.6 to 14.6" against Ceres master (2.6 to 14.9). Both
corrected. And by D99's own method ladybug-49 is not an equal-cost comparison with
Ceres: just-ba stops 2.8e-5 above Ceres's cost, outside the 1e-6 that D99 uses for
Caspar and bae; Ceres passes just-ba's final cost at iteration 22 of 30, about 0.81 s
of its 1.11 s, which gives 2.9 times, not 4.0.

**7. CI compiles nothing of this milestone.** `backends/gpu` is added only when a
GPU backend is built (`CMakeLists.txt:73`), Metal is built on macOS only and CUDA is
off by default, so the quick mode (Linux and Windows) compiles neither the shared
host logic nor the kernels, and the full mode compiles them with Apple clang alone.
Document 08's plan has a free job, "CUDA kernels compiled (not run)", on every pull
request; it does not exist. The shared host logic (`backends/gpu/src/backend.cpp`)
is plain C++ with no GPU dependency and could be compiled, and run against a device
written for the host, on every runner at no cost. As it stands MSVC has never
compiled any of it, which is also what the pending Windows check will meet first:
`std::vector<long>` column starts (`jba/gpu/device.h:85`) are 32-bit there.

### Low

**8. 32-bit quantities in strategy B are narrowed without a check.** By reading:

- `backends/cuda/src/backend.cu:83` copies the column starts from `long` to `int`
  for cuDSS. The lower triangle of a dense reduced system passes 2^31 entries near
  11,000 images; long before that the values alone need more memory than any
  consumer card has, so memory fails first today.
- `backends/gpu/src/backend.cpp:408` to `411` and `449`: entry indices, the tile
  value offset and `csc_src` are `uint32_t` casts of `size_t`. The entry count
  passes 2^32 at 51 GB of entries.
- The one check that exists (`backend.cpp:428`, E and G offsets) throws "too many
  observations for sparse_schur" under `Auto` instead of choosing the iterative
  solver.

**9. Tests of the CUDA backend leave out what is new in it.** The Python test
(`bindings/python/tests/test_bindings.py:393`) uses the iterative solver only, so
the default path on CUDA is not run from Python. No test reaches a cuDSS status
other than success or an allocation failure. BAL problems are not GPU tests
(document 13, finding 4, still open, as are its other points about `gpu_check.cpp`:
medians, the repeat check's blocks). Nothing compares the two GPU backends with
each other, which one kernel source makes cheap to state: same counts on 101 of 123
solves here.

**10. The CUDA device assumes it owns the thread's CUDA state.** By reading,
`backends/cuda/src/backend.cu`: `cudaSetDevice(0)` (line 144) changes the calling
thread's device and is not restored, so a caller working on device 1 continues on
device 0 after a solve; the device is not selectable. `cudaGetLastError()` (line
205) returns the last error of any CUDA call on the thread, so an error the caller
left unread before `solve()` is reported as "kernel launch". `available()` (line
225) counts devices; the header says it also checks that the driver can run this
build's kernels (`jba/cuda/backend.h:11`), and with the default
`CMAKE_CUDA_ARCHITECTURES=native` a GPU of another generation would pass
`available()` and fail at the first launch. That last point decides how wheels are
built in milestone 6.

**11. `bench/run.py` leaves its problems behind.** `run.py:135` makes a temporary
directory and never removes it. Measured on the Mac: 52 `jba_bench_*` directories
from earlier runs, 1.3 GB; a run with `--large` leaves 81 MB. With `--only`,
`--compare` reports every other problem as "not run" and exits 1.

**12. The run-time dependencies of a cuDSS build are more than cuDSS.** Measured
(`readelf -d`): the binary needs `libcudss.so.1`, which needs `libcublas.so.13` and
`libnvrtc.so.13`. Document 03 said cuDSS only; corrected. It matters for the size of
a CUDA wheel (document 08).

**13. Comments and help text that no longer match** (left as they are; no code was
changed):

- `cli/main.cpp:39`: `--linear-solver` "default auto: sparse_schur on metal,
  iterative_schur on cpu (sparse_schur needs metal)".
- `core/include/jba/solver.h:84`: `Auto` "Measured from 16 to 2,000 images" (D98
  measured 10,000 on CUDA; and see finding 3).
- `bench/run.py:3`: "on the CPU or the Metal backend".
- `backends/metal/kernels/layout_sizes.metal:2`: checked by `device.mm`; the file is
  `backend.mm`.
- `jba/gpu/layout.h:8`: "one definition ... the layouts agree by construction", but
  `uint_pair` is `uint2` for Metal and nvcc and a struct of its own on the host
  (same size and use, so harmless).
- D98 quotes times from three stages of the work (the 2,000-image walk at 0.78,
  0.44 and, in the baseline, 0.39 s; the 10,000-image walk at 2.0 s against document
  07's 1.6 s).

### Smaller points, no action needed now

- `Device::to_device` returns before a copy from a pinned mirror is done on CUDA.
  Every caller reaches `finish()` or `to_host()` before the host writes that mirror
  again, so it is correct today; the interface (`jba/gpu/device.h:101`) does not
  say that callers must.
- Metal still compares the structures by size only (document 13, finding 12). On
  CUDA host and kernels are one compiler, so there is nothing to compare.
- The first solve of a process on CUDA takes 0.12 to 0.33 s longer than later ones
  (context and cuDSS start-up). A command-line run pays it every time.
- `jba_make_problem` takes minutes for 25,000 images; the generator, not the
  solver, limited how large a walk this review could try.
- `third_party/ceres/build.cmake` leaves an entry in the user's CMake package
  registry (`~/.cmake/packages/Eigen3`, written by Eigen's build) that points into
  its work directory. Found while cleaning the laptop.
- A cuDSS handle and its buffers are leaked if `CudssSolver`'s constructor throws
  half-way; `analyze` called twice would leak the first buffers. Neither happens
  in the code as it is.

## Recommended changes

In order. Items 1 to 3 before the comparison is quoted anywhere outside the
repository.

| # | Change | Why |
|---|---|---|
| 1 | Start `time_seconds` (or add a second field) at the call to `solve()`, backend construction included; have `bench/run.py` report both set-up and iterations; say in document 05 which one each table shows. | Finding 1. Document 05 already requires it. |
| 2 | Copy `count * sizeof(T)` and zero the rest in `Impl::buffer` (or give `Device::alloc` the initialized size). Run the sanitizer build again. | Finding 2. One line. |
| 3 | Repeat the Caspar and bae comparison with times around the call, and commit what was measured: per competitor and problem the time, final cost and history, next to the baseline; let `run.py` write the second Ceres column and the large problems. | Findings 1 and 6: the exit condition against Caspar is open until then. |
| 4 | Build the tile lists without the map of vectors (count, then fill; in parallel), and measure again. Then choose `Auto` from the problem (tile entries, or mean squared track length, against observations) and from the device's free memory, falling back to the iterative solver instead of throwing. | Findings 3 and 4. |
| 5 | Treat any cuDSS status other than success in the factorization as "fall back to conjugate gradients", and say in the error what ran out when nothing can be done. Add a test that forces it. | Finding 4; it is what D98 intends. |
| 6 | Study the block that loses definiteness on venice-52: form the camera diagonal blocks of the reduced system more carefully (compensated or double sums for the Schur term), or keep a step when only the preconditioner block fails by flooring it as `spd_inverse` floors pivots. Add venice-52's failing step as a case. | Finding 5. It also decides the open item on venice-52 in document 00. |
| 7 | CI: compile `backends/gpu` everywhere with a host device (plain loops over the kernel bodies would also run the kernels' logic on the CPU), or at least compile it. Decide on document 08's "CUDA kernels compiled" job. | Finding 7. Free runners. |
| 8 | Check the 32-bit narrowings and throw a clear error, or fall back. | Finding 8. |
| 9 | Python test of CUDA's default solver; a GPU case that needs no BAL download but has BAL's shape (an intrinsics block per image, far points); remove `run.py`'s temporary directory. | Findings 9 and 11. |
| 10 | Save and restore the thread's device, clear the error state before the first launch, make `available()` try a kernel. Before milestone 6 builds wheels. | Finding 10. |
| 11 | Bring the comments of finding 13 in line. | |

## Questions for the owner

1. Which time is the claim about: the call (document 05 as written) or the
   iterations? If the call, milestone 5's exit is shown against Ceres and open
   against Caspar until change 3 is done.
2. Should `Auto` ever throw for lack of memory when another solver of the same
   backend would run (finding 4)?
3. venice-52 stays an exception in the documents. Is finding 5 worth fixing before
   milestone 6, given that real captures take the same fallback?
