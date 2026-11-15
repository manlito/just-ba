# 16. WebGPU and WebAssembly: an exploration

Status: proposed for main as an opt-in backend (`JBA_WITH_WEBGPU`, off by default;
D127), 2026-11-15. Publishing it (wheels, a hosted demo) is separate. The question from the owner: can just-ba run on WebGPU, natively and as
WebAssembly in a browser, at a speed close enough to Metal and CUDA that a desktop
app could use WebGPU alone, or at least good enough for a browser-only demo?

## Answer

- **It runs, natively and in the browser, and agrees with the CPU reference.** All
  122 GPU test cases (the 55 Ceres parity problems with both linear solvers, plus
  the fallback and out-of-memory cases) pass against the CPU at the Metal backend's
  tolerances. The same library compiles to a 597 KB WebAssembly module that solves
  in Chrome on WebGPU.
- **Speed on the same Mac** (the owner's M5 Pro, the same algorithm on both
  backends): with the iterative solver, native WebGPU takes 1.0 to
  1.4 times Metal's time, and in the browser 1.1 to 1.6 times, 2.1 on the smallest
  problem. On the 2,000-image walk (1.9 million observations): Metal 3.5 s, native
  WebGPU 3.6 s, Chrome 4.0 to 4.3 s. With sparse_schur, Metal's default, native WebGPU takes 1.2 to 2.2 times
  Metal's time (1.4 on the 2,000-image walk; the high ratios are the small problems,
  where fixed costs dominate). The browser has no direct factorization yet (below), so
  it runs the iterative solver, which on BAL takes 1.6 to 9 times as long as Metal's
  default.
- **Repeatability: a fault on Apple's side, worked around.** On BAL problems
  (one free intrinsics block per image), repeated WebGPU solves used to differ,
  breaking the library's rule that the same input gives the same result bit for bit
  (D59, D93). It is not a defect of this code: it reproduces in a small Metal program
  without Dawn or the library, which points to Apple's shader compiler or GPU. A
  reordered loop in one kernel avoids it at no measurable cost (below); every
  problem tried now repeats.
- **For the desktop app:** WebGPU is good enough on Apple GPUs to be the only GPU
  path on a Mac if the determinism defect is fixed, at a cost of up to 40 percent
  with the iterative solver and 20 to 120 percent with sparse_schur (40 at
  2,000 images). How it compares with CUDA on an NVIDIA GPU, through Dawn's Direct3D 12 or Vulkan
  backends, was not measured: that needs the laptop.
- **For a browser demo:** good enough today. The 2,000-image walk, the top of the
  main size range (D5), solves in about 4 s in Chrome; a single-threaded WebAssembly
  CPU solve of the 500-image walk takes 9.1 s against 0.66 s on WebGPU.

## What was built

| Piece | Where | Notes |
|---|---|---|
| WGSL kernels | `backends/webgpu/kernels/solver.wgsl`, `linearize.wgsl` | A hand port of `backends/gpu/kernels/solver.inc`, kernel by kernel, the same arithmetic and order of additions. `linearize.wgsl` is instantiated per camera model by the host. |
| Generated residuals in WGSL | `residuals/generated/wgsl/reprojection.wgsl` | A `WgslGenerator` in `residuals/symbolic/emit.py`. Every model takes intrinsics as `array<f32, 8>`, so one kernel body calls any of them. The other generated files are unchanged byte for byte. |
| The device | `backends/webgpu/src/backend.cpp` | `gpu::Device` over `webgpu.h`: the same source natively (Dawn) and in the browser (Emscripten's emdawnwebgpu). About 620 lines. |
| API | `BackendKind::WebGpu`, name `webgpu` | CLI, Python enum, `jba_gpu_check` and `bench/run.py` take it. |
| WebAssembly module and page | `bindings/wasm` | `just_ba.js` + `just_ba.wasm` and a demo page that generates the benchmark's synthetic problems in the browser (the same generator and seeds) or reads a BAL or `.jba` file. |

### The heap

WGSL has no pointers, and the shared host logic (D96) passes the kernels an `Args`
structure of 69 device pointers. Rather than change that logic, the WebGPU device
gives every buffer a range of one storage buffer, the heap, and hands out byte
offsets in it as device addresses. Every allocation of the host logic happens before
`bind()`, so the heap is created there, at its final size, with every initial value
written while mapped. `bind()` turns `Args`' pointers into word indices for a
uniform `Args` in WGSL; each dispatch's `Op` goes into a ring of uniform slots read
at a dynamic offset. Floats are bitcast from the heap's 32-bit words; structures
(`Chunk`, `Factor`, `Tile`) are read at their C layout, field by field. The host
logic in `backends/gpu/src/backend.cpp` is unchanged.

Consequences:

- One storage binding, so no WebGPU per-stage binding limit applies. Its size is the
  limit: 4 GiB on the Mac natively and in Chrome, enough for strategy A at 10,000
  images (3 GB at most on CUDA, D102). Past it, `alloc` throws `DoesNotFit`, and
  `Auto` falls back as on the other backends.
- Every kernel reads and writes through one `read_write` binding, so the driver
  cannot use a read-only path for the inputs. Not measured.

### What WGSL changed in the kernels

- Finiteness is tested on the bits: WGSL lets an implementation assume there are no
  NaNs or infinities, so `isfinite` could be folded away.
- Every thread of a group must reach a barrier, so kernels that stop early once the
  conjugate gradients are done (`done`) run their barriers anyway.
- No templates: the 6 x 6 tile sums of strategy B are a function of their own
  (`tile_sums66`), as `solver.inc` specializes `tile_sums<6, 6>`. Without it,
  sparse_schur on the 2,000-image walk took 0.47 s instead of 0.40 s.
- WebGPU allows 65,535 groups per dimension: dispatches above that use a second
  dimension, and the group count rides in `Op` so surplus groups return.
- Group memory is 16 KiB, exactly WebGPU's default limit, as in `solver.inc`.

## Correctness

`jba_gpu_check --backend webgpu` on every GPU case (`webgpu_<case>`,
`webgpu_sparse_<case>`, label `gpu`): 122 tests, all pass, with the tolerances of
D95. The first case solved agreed with the CPU to 4e-11 in cost. A solve in Chrome
gives the native WebGPU result to every printed digit on the synthetic problems; on
the larger walks the counts differ, because WebAssembly does not fuse multiply-adds
on the host and the host's double arithmetic rounds differently (the reason D81
keeps baselines per platform).

### Repeated solves differed on BAL problems (Apple's fault, worked around)

Before the workaround below, repeated WebGPU solves of ladybug-49, venice-52 and
final-93 (in a later run also trafalgar-21) gave different iteration counts and
final costs (for example 47 iterations and a final cost of
13344.703 in one run, 49 and 13344.687 in another). Dubrovnik-16, trafalgar-21 and
every synthetic problem, including the 2,000-image walk, repeat bit for bit. Metal,
running the same algorithm, repeats on all of them. So the library's determinism
(D59, D93) does not hold on WebGPU. `webgpu_bal_shape` failed its in-process repeat
check once in four `ctest -j4` runs, never when run alone.

What goes wrong: in `block_gather` (the camera-side gather of the Schur product,
mode 1), a thread's private sum for a per-image intrinsics block now and then comes
out as the correct sum minus exactly one product, `j1 * r1` of the thread's first
observation, in the first pass of the loop over the block's columns (checked against
a host recomputation for ten failing threads). The affected steps are therefore
slightly wrong, not only reordered; every step is still checked against the cost in
double on the host, which is why the answers stay within tolerance.

Why it is not this code (a review on a second model, then checked by hand):

1. With every buffer hashed after every dispatch, the first difference is in
   `block_gather`, with every input identical. Waiting on the host after every
   dispatch (`JBA_WEBGPU_SYNC=1`) does not remove it.
2. Re-running that dispatch on unchanged memory differs in 26 of 4,300 dispatches on
   ladybug-49 and 113 of 680 on final-93; on dubrovnik-16 and trafalgar-21, none.
3. Dawn's compiled MSL, run by a small Metal program (no Dawn, none of our host
   code) on a snapshot of final-93's buffers, still fails, in every Metal math mode.
4. Reduced to a kernel in which each thread only reads memory nothing writes, adds
   in registers and stores one word of its own, with the group memory and barriers
   compiled out (16 bytes of group memory), it still gives different sums in 130 of
   20,000 dispatches. Removing Tint's bounds clamps and loop guards by hand does not
   help.
5. Reversing the column loop moves the fault from column 0 to column 4: it follows
   the loop's first pass, not a memory location.

A kernel with no shared state between threads, run on constant inputs, that returns
different values from run to run is a fault of Apple's shader compiler or GPU on
this code shape, not of `solver.wgsl`, `backend.cpp`, Dawn's runtime or Tint's
additions. The WGSL was also reviewed kernel by kernel against `solver.inc`
(indexing, structure layouts, State offsets, barriers, early returns): no race.

The workaround (D126): `block_gather` takes the intrinsics columns in the order 1,
2, ..., np - 1, 0 (`c = (cc + 1) % np`), the same sums per column. The repeat test
then fails in 0 of 4,300 and 0 of 680 dispatches. Five solves of each of the five BAL
problems, with each linear solver, repeat bit for bit, and the WebGPU tests pass
three times out of three under `ctest -j4`, where `webgpu_bal_shape` had failed
once. Cost: none measurable. Time per conjugate-gradients iteration moved by 0.91 to
1.09 times on the BAL problems, the radial problem and the 500-image walk, which
never runs the changed loop and moved by 1.01, the size of the noise.

It sidesteps the trigger without explaining it. `jf_times`, `rows_of` and
`scale_observations` have loops of the same shape and were not tested on their own;
every problem tried repeats, but a new problem shape could meet the fault elsewhere.
The reproduction (the Metal program, the kernels, the buffer snapshot and the
reordered kernel) was saved under `build/metal-repro` in the exploration's worktree,
for a report to Apple. Still to try: a later Dawn release, and the same build on
Vulkan and Direct3D 12 on the laptop. `bench/run.py --allow-nondeterministic` stays,
so a regression shows as a warning instead of stopping the run.

## Speed

The owner's M5 Pro (20 cores), macOS 26.6, Dawn's release v20261113.214538, Chrome
156 in the desktop app's browser pane. Native times are `bench/run.py`'s: the
whole call, best of three warm calls (D101). Browser times are one warm call each
(`jba_solve` with warm-up). Problems marked * do not repeat on WebGPU (above); their
times are from one of the differing runs.

### Native: WebGPU against Metal

| Problem | Metal sparse | WebGPU sparse | Metal iterative | WebGPU iterative | Browser iterative |
|---|---|---|---|---|---|
| ladybug-49 * | 0.140 | 0.285 | 0.410 | 0.552 | 0.629 |
| trafalgar-21 | 0.054 | 0.046 | 0.125 | 0.160 | 0.170 |
| dubrovnik-16 | 0.057 | 0.077 | 0.159 | 0.203 | 0.200 |
| venice-52 * | 0.574 | 0.921 | 0.805 | 0.997 | 0.918 |
| final-93 * | 0.180 | 0.237 | 1.333 | 1.466 | 1.621 |
| pinhole-50 | 0.042 | 0.068 | 0.041 | 0.046 | 0.087 |
| radial, free intrinsics, 30 | 0.170 | 0.267 | 0.494 | 0.682 | 0.724 |
| relative priors, 50 | 0.034 | 0.065 | 0.031 | 0.036 | 0.049 |
| global similarity, 50 | 0.042 | 0.094 | 0.038 | 0.044 | 0.055 |
| OPENCV, 30 | 0.023 | 0.031 | 0.025 | 0.024 | 0.033 |
| walk, 500 images | 0.084 | 0.103 | 0.561 | 0.706 | 0.662 |
| walk, 2,000 images | 0.287 | 0.396 | 3.512 | 3.630 | 3.97 to 4.32 |

Seconds. Iteration counts of the two GPU backends are close but not equal (WebGPU's
rounding differs: no control of fused multiply-adds in WGSL), so part of each gap is
a different path: on the 500-image walk WebGPU's conjugate gradients took 1,156
iterations against Metal's 1,043.

Per conjugate-gradients iteration on the 2,000-image walk, where both take nine
steps: Metal 2.0 ms, native WebGPU 2.2 ms, Chrome 2.3 ms.

Where the time goes beyond Metal's:

- **Copies to the host.** On Apple's unified memory the Metal backend reads the
  reduced camera system, the step and the state in place; WebGPU copies each through
  a read-back buffer. For strategy B that is the whole reduced system every step
  (tens of MB at 2,000 images). This is the largest part of sparse_schur's gap and
  is inherent in WebGPU; it would be smaller on a discrete GPU, where CUDA copies
  too.
- **Round trips.** Every `finish()` maps a buffer and waits. Natively this costs tens
  of microseconds; in the browser, where the GPU runs in another process, more
  (pinhole-50: 0.087 s in Chrome against 0.046 s natively, nine steps).
- **Kernel compilation:** 1.45 s the first time on this Mac (33 pipelines, WGSL to
  MSL to Metal), about 0.3 s once the system's shader cache has them. Cold calls in
  the benchmark are 0.3 s slower than Metal's for that reason. Chrome caches compiled
  pipelines too: from loading the page to a finished cold solve of pinhole-50,
  0.54 s.

### The CPU, for scale

| Problem | Native CPU, 20 threads (baseline) | WebAssembly CPU, one thread | Browser WebGPU |
|---|---|---|---|
| pinhole-50 | 0.057 | 0.389 | 0.087 |
| walk, 500 images | not in the baseline | 9.09 | 0.66 |

The WebAssembly build is single-threaded (no `SharedArrayBuffer`), so the host side
also runs on one thread: the cost in double at every step, the depth rows and
factors, the retraction. On the 2,000-image walk that host work is a few hundred
milliseconds of the four seconds.

## The browser build

- Emscripten 7.0.0 (Homebrew) with emdawnwebgpu from the same Dawn release; CMake
  with `emcmake`, `-DJBA_WITH_WEBGPU=ON -DJBA_BUILD_WASM=ON`. Emscripten needs
  Python 3.10 or later: set `EMSDK_PYTHON` where the default `python3` is older
  (Xcode's is 3.9).
- The solver is synchronous C++ and WebGPU's waits are asynchronous in a browser.
  JSPI (JavaScript Promise Integration) suspends the module at each wait, so the
  page stays responsive and no code changes. Chrome and Edge ship JSPI; Safari and
  Firefox were not checked. Where JSPI is missing, Emscripten's Asyncify does the
  same at a cost in size and speed, and is incompatible with native wasm exceptions,
  which the build uses.
- Single-threaded host, so the page needs no special headers and can be served from
  any static host. Threads would need `SharedArrayBuffer` and the
  cross-origin-isolation headers.
- Memory: wasm32 caps the module's memory at 4 GB. The heap lives in GPU memory;
  the module holds the problem, the host's double state and the host mirrors. The
  2,000-image walk ran without trouble; the 10,000-image walk was not tried.
- No strategy B in the browser: the reduced system is formed on the GPU but nothing
  factorizes it. A sparse Cholesky in the WebAssembly module (none in the library
  yet; the CPU backend has no strategy B either, document 03), or a dense one for
  small problems, would give the browser the solver that is 1.6 to 9 times faster on
  BAL.

Running it:

```bash
EMSDK_PYTHON=/opt/homebrew/opt/python@3.15/bin/python3.15 emcmake cmake -S . -B build/wasm -G Ninja -DJBA_WITH_WEBGPU=ON -DJBA_BUILD_WASM=ON
```

```bash
EMSDK_PYTHON=/opt/homebrew/opt/python@3.15/bin/python3.15 cmake --build build/wasm
```

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory build/wasm/bindings/wasm
```

Then open http://127.0.0.1:8765 in Chrome. The native build:

```bash
cmake -S . -B build/native -G Ninja -DJBA_WITH_WEBGPU=ON
```

with Dawn's release unpacked under `third_party/webgpu/dawn` and emdawnwebgpu under
`third_party/webgpu/emdawnwebgpu` (both ignored by git; `JBA_DAWN_DIR` and
`JBA_EMDAWNWEBGPU_DIR` point elsewhere).

## Before merging, and after

Done before proposing it for main (D127): the merge with milestone 6, the fixes of
the review (the Op ring's flush; the read-back buffer unmapped before an error is
raised; objects of a batch left open by an exception released; a WebGPU solve
started from another solve's iteration callback no longer waits on itself), and the
whole test suite with and without WebGPU. Left for later:

1. The Metal fault behind the non-repeatable solves (above) is worked around in
   the one kernel where it showed. Check the other kernels with the same loop shape,
   and report it to Apple with the saved reproduction.
2. Measure on the laptop: Dawn on Direct3D 12 and Vulkan against CUDA, on Windows
   and Linux. This decides whether the desktop app can use WebGPU alone on Windows,
   and covers the Intel and AMD machines document 09 worries about.
3. Publishing (separate work): how Dawn is obtained for wheels or the app (a pinned
   prebuilt release per platform, as here, or built from source), and hosting the
   browser demo.
4. A host-side factorization for the browser (strategy B), if the demo should match
   Metal's default.
5. CI: a compile check of the WGSL (Tint's command-line tool from the same release)
   would catch a kernel that no longer compiles at no cost. The GPU tests stay
   manual (D40).
6. Keep two kernel sources in step: every change to `solver.inc` now needs the same
   change in `solver.wgsl`. D96 avoided exactly this for Metal and CUDA; WGSL cannot
   be produced by their preludes, since it has no pointers. A generator that writes
   both from one description is the alternative, and an intermediate layer of the
   kind D31 rules out.
