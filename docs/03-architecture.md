# 03. Architecture

How the library is laid out so that it can be the fastest option for its narrow job,
on three kinds of hardware, with almost no dependencies. The language is C++
(document 06). Sections marked "as built" describe the code after the milestone they
name; the rest is still the plan.

## Design principles

1. **The problem shape is known at build time.** A closed residual set means fixed
   block sizes, fixed sparsity patterns and hand-specialised kernels. No virtual
   dispatch, no dynamic graph, no run-time differentiation.
2. **The library owns the linear system.** It builds and solves the normal equations
   itself. Vendor libraries are used only for what they do best (a direct factorization
   of a small matrix), and are called directly.
3. **Kernels are written once per shading language, from one description.** Residuals
   and Jacobians are derived symbolically at development time and emitted as plain
   expressions for each target.
4. **Single precision on the GPU is the baseline, not an afterthought.** Two of the
   three GPU targets cannot do double precision at all (see below).
5. **Deterministic by default.** Same input, same backend, same result, bit for bit.

## Layers

```
 Python package        sample CLI binary        (later) JS/wasm package
        \                    |                          /
         +------------- C-level API ------------------+
                             |
                    core (host, CPU)
      problem data . ordering . LM loop . damping strategy
      track building . triangulation . robust reweighting
                             |
                    backend interface
        linearize . build blocks . operator apply . vector ops
           /            |               |              \
     CPU backend   CUDA backend   Apple backend   WebGPU backend
     (reference)                    (Metal)         (stretch)
```

- **Core** runs on the host and is identical for every backend. It holds the problem,
  runs the outer Levenberg-Marquardt loop, and makes every accept or reject decision.
- **Backend interface** is a small set of operations (listed below). A backend is the
  implementation of those operations on one kind of device.
- **Bindings** are thin. Python passes NumPy arrays without copying; results come back
  the same way. (As built: arrays are copied in and out, and quaternions normalised
  on the way in, D74; the Problem owns its arrays and the backend copies them again.)

## Problem data

Structure-of-arrays, contiguous, fixed-size blocks.

| Block | Size | Notes |
|---|---|---|
| Camera pose | 7 stored (quaternion + translation), 6 in the tangent space | One per image |
| Intrinsics | Depends on the model (document 02) | One block per session (one camera per session) normally; several blocks of the same model are accepted, mainly for BAL (D53). Usually fixed. The camera model is therefore a per-solve constant, so each kernel runs one model with no per-observation branching. |
| Point | 3 | One per track |
| Depth correction | 2 per image (scale, offset) when used | See document 02 |
| Ground control point | 3 | Fixed, or free with a prior |

Residual instances are stored as index arrays into those blocks, one array set per
residual type. Every accumulation is a gather: each output block is written by one
task that walks its own observations in a fixed order.

As built (CPU): observations are sorted once, by point then image, so the
observations of a point are a contiguous range. Per image and per intrinsics block
there is an index list into that order; within a list the observations of one point
are still consecutive, which the preconditioner uses.

A block with far more observations than the others, as a shared intrinsics block,
which sees all of them, is gathered in units of at most 16,384 observations (on the
CPU, extended to the end of a point's observations) that run in parallel, and its
units' sums are added in order (D145). The explicit reduced camera system takes a
shared intrinsics block's elimination terms once per point: all of a point's
observations in the block share it, so the sums over their pairs factor (D145).

Any block can be marked fixed. Fixed blocks are removed from the system at setup, not
masked at run time. As built: single fixed coordinates inside a free block (the
scale coordinate held by the gauge, intrinsics groups that stay fixed) remain as
unknowns with zero Jacobian columns; their step is exactly zero.

## The solver

### Outer loop

Levenberg-Marquardt with a trust-region update of the damping term, on the host.
Robust losses are applied by iterative reweighting (as built: Ceres's residual and
Jacobian correction, which is the same thing for the losses offered; D56). Per
iteration:

1. Linearize: residuals and Jacobian blocks (device).
2. Build the blocks of the normal equations (device).
3. Solve the linear system (device, or device plus host for a direct factorization).
4. Apply the step on the manifold (device), evaluate the new cost.
5. Accept or reject, update damping (host).

### Linear system structure

With points eliminated by the Schur complement, the unknowns left are cameras
(and the few shared blocks). In the main use case this reduced system is small:
a phone session has hundreds to a few thousand frames against hundreds of thousands
of points.

What each residual type adds to the reduced system:

| Residual | Effect on structure |
|---|---|
| Reprojection | Couples cameras that see a common point (the usual pattern) |
| Global pose prior | Diagonal blocks only; plus a border to one small block when the prior set has its own transform |
| Relative pose prior | Couples the two cameras of each caller-supplied pair. Such pairs normally share points already, so it rarely adds new blocks to the reduced system. |
| Depth term | Camera and point terms like reprojection, plus the per-image depth block |
| Ground control point | Like reprojection against a fixed or lightly free point |
| Shared intrinsics (when free) | A dense border: every image of that camera couples to one block |
| Manhattan prior | Camera diagonal blocks, plus a border to one tiny shared block (1 or 3 unknowns). No points involved. |

Every one of these is known in advance. This is the benefit of the closed set: the
pattern is computed once at setup and reused every iteration.

### Linear solver options

Two strategies, both planned, chosen by problem size and backend:

**A. Implicit Schur with preconditioned conjugate gradients.** The reduced matrix is
never formed. Each iteration applies it as a sequence of block products. Preconditioner:
block-Jacobi on the camera blocks to start, with a stronger option later.
- Needs no vendor solver at all. Works on every backend, including the browser.
- Lowest memory.
- Built first, because it needs nothing external. Whether it or strategy B is the
  default in the main size range is decided by the benchmark (see the size analysis
  below).

**B. Explicit reduced camera system with a direct factorization.** Form the reduced
matrix and factorize it.
- Dense Cholesky when there are few cameras; sparse Cholesky otherwise.
- Uses a vendor solver: cuDSS or cuSOLVER on CUDA, Accelerate on Apple (CPU).
- Usually more robust on ill-conditioned problems, which noisy priors can produce.

Reference points from current systems (2026-11-14):

- COLMAP's Caspar backend and the Graphite solver both skip the Schur complement and
  run conjugate gradients on the full system with a per-block preconditioner. Caspar's
  authors report an earlier cuDSS-based version was slower.
- PyPose `bae` uses the Schur complement with conjugate gradients, with matrix-free
  and explicit variants.
- Ceres uses explicit Schur with direct solvers for small and medium problems and
  iterative Schur for large ones.

### What the target size implies

The main target is 50 to 2,000 images (document 01). With 6 unknowns per camera the
reduced camera system is then at most about 12,000 by 12,000.

| Images | Reduced system | Dense storage, single precision | Consequence |
|---|---|---|---|
| 50 | 300 | under 1 MB | Trivial. Solve on the CPU even when the rest runs on the GPU. |
| 500 | 3,000 | 36 MB | Dense Cholesky is fast on any backend |
| 2,000 | 12,000 | about 580 MB | Dense Cholesky still fits an 8 GB GPU and takes on the order of a second; sparse is better when the scene is not densely connected |
| 10,000 | 60,000 | about 14 GB | Dense does not fit. Sparse direct or conjugate gradients only. |

These are estimates from matrix sizes, not measurements.

So in the main range the linear solve is small, and a direct factorization
(strategy B) is a strong candidate for the default: it needs no iteration count
tuning and is more robust when noisy priors make the system ill-conditioned. Most of
the time then goes into linearization and building the Schur complement, which scale
with the number of observations, so that is where kernel optimization effort belongs.
The iterative strategies matter mainly for the 2,000 to 10,000 range and for
memory-limited devices.

At the small end (tens of images) GPU launch and transfer overhead can exceed the
work itself. The CPU backend may be the fastest choice there, and backend selection
should take problem size into account.

A third strategy, **C. conjugate gradients on the full system without Schur**, is
a real contender and cheap to try once A exists, since it reuses the same kernels.
The benchmark (document 05) decides the default per problem size; this is not settled
by argument.

### Backend operations

A backend implements roughly these, and nothing else:

| Operation | Purpose |
|---|---|
| `upload`, `download` | Move problem data and results |
| `linearize(type)` | Residuals and Jacobian blocks for one residual type |
| `cost()` | Robustified total cost, accumulated in high precision |
| `build_diagonal_blocks()` | Camera and point diagonal blocks, gradient |
| `invert_point_blocks()` | 3x3 inverses with damping |
| `apply_schur(x) -> y` | Implicit reduced-system product |
| `build_reduced_system()` | Explicit variant, optional per backend |
| `factor_and_solve()` | Direct solve, optional per backend |
| vector ops | dot, axpy, scale, copy on block vectors |
| `back_substitute()` | Recover the point step |
| `retract(step)` | Apply the step on the manifold |

Keeping this list short is what makes a new backend affordable.

As built after milestone 1 (`core/internal/jba/backend.h`), the interface the host
loop sees is coarser than the table: `cost`, `linearize`, `gradient_max_norm`,
`parameter_norm`, `compute_step`, `candidate_cost`, `accept_candidate`. One call,
`compute_step`, covers everything from the diagonal blocks to the retracted
candidate, including the conjugate gradients loop. The operations in the table
exist as private functions of the CPU backend, for the reprojection residual and
the pose priors (D75).
As frozen in milestone 4 (D90): the coarse interface stays, plus `cost_by_type`
(milestone 3) and `max_trust_region_radius()`, which lets a single-precision
backend cap the radius where its damping is still resolved (D91). The inner solve belongs to each backend:
on Metal the conjugate gradients loop, stopping test included, runs in batches on
the device and never returns to the host per dot product. What every backend needs
on the host (variable layout, observations in point-major order, the camera-side
factors and their evaluation in double, the cost in double, the retraction) is one
shared class, `cpu::System`; each backend adds its own linear algebra.
The two GPU backends share more (milestone 5, D96): one kernel source,
`backends/gpu/kernels/solver.inc`, compiled by Metal and by nvcc behind a prelude of
macros each, and one host side, `gpu::GpuBackend`, above a small device interface
(`jba/gpu/device.h`: buffers with a host side, dispatches of a kernel over threads
or groups, and the direct factorization of option B). The structures the kernels
share with the host are defined once (`jba/gpu/layout.h`). A GPU backend is that
interface: Metal's is about 250 lines with Accelerate, CUDA's about 240 with cuDSS.

## Precision

Confirmed constraints:

- Metal has no double-precision type in its shading language.
- WebGPU's shading language has only 32-bit and 16-bit floats.
- CUDA supports double.

So single precision on the GPU is mandatory for two targets. The design assumes it
everywhere and treats double on CUDA as an option:

- **State in double on the host.** Poses and points are stored in double. The device
  receives values re-centred and re-scaled so that single precision is spent on
  the part that matters.
- **Local parameterization.** Points relative to a scene centre, scene scaled to unit
  size, pixel coordinates relative to the principal point.
- **Careful accumulation.** Costs and gradients are summed with compensated or
  pairwise summation, in a fixed order. Caspar's authors name float32 summation error
  as a known limitation, and COLMAP reorders factors to reduce it.
- **Accept or reject in double.** The cost used for the step decision is evaluated in
  double on the host when the single-precision value is too close to call.
- **Iterative refinement** for direct solves done in single precision.

The CPU backend runs in double and is the accuracy reference for the others.

As built on Metal (D91): the state and every decision in double on the host; the
device gets points minus the scene's centre and translations t + R c, the rotation
Jacobian corrected back to the library's tangent. Re-centring the observations on the
principal point buys little (milestone 0); what fixes residual precision is taking
the whitened residuals from the host's double cost evaluation, rounded after the
subtraction, so the gradient is accurate to float relative to the residual rather
than to the pixel value. Sums are fixed-order gathers and trees (D93), not
compensated sums. Two limits of single precision are explicit: Cholesky pivots are
floored at 1e-6 of their diagonal entry, and the trust-region radius is capped at
1e6, both about eight float epsilons. A predicted decrease too small for the float
sums ends the solve as converged at the backend's precision. The direct solve
factorizes in double a reduced system formed in single precision; where that system
is not positive definite the step falls back to conjugate gradients instead of
iterative refinement.

## Accumulation without float atomics

Building the normal equations means adding many small blocks into shared rows. On
CUDA this is commonly done with atomic float adds. WebGPU has no atomic float add, and
atomic adds make results depend on thread timing.

The library uses a gather formulation on every backend: because observations are
pre-sorted by camera and by point, each output block is the sum over a contiguous
range, computed by one thread or by a tree reduction. This is deterministic and
portable. If CUDA atomics prove faster, they can be an opt-in fast mode.

## Kernel generation

- Each residual is written once in a symbolic tool, with its manifold handling.
- The tool derives the Jacobians, simplifies common subexpressions, and emits flat
  arithmetic.
- Emit targets: the host language (CPU), CUDA C, Metal Shading Language, and WGSL.
- Generated code is checked into the repository. The generator is a development tool,
  not a build or run-time dependency.

The tool is wrenfold, pinned, with small custom emitters for C++, Metal and CUDA
(D45; the CUDA one since milestone 5). The solver kernels themselves are written by
hand, once for both GPU backends (D96).
SymForce was the other candidate; its C++ output depends on Eigen and its own
runtime types. CI checks that the checked-in generated files are current.

## Backends

| Backend | Kernels | Precision | Direct solver (option B) | Runtime dependencies |
|---|---|---|---|---|
| CPU (reference) | Host language, threads, SIMD | double | Not built yet (D92). Planned: Accelerate on Apple; a small built-in dense Cholesky elsewhere | None beyond the system |
| CUDA (as built, milestone 5) | The shared kernel source, compiled by nvcc for the build machine's GPU unless architectures are given | single, with host double for decisions; double kernels not built | cuDSS's sparse Cholesky in double on the GPU, optional (`JBA_WITH_CUDSS`, D98); then the default | CUDA runtime, linked statically; when built with cuDSS, cuDSS and the cuBLAS and NVRTC it loads |
| Apple (as built, milestone 4) | Metal Shading Language: the shared kernel source behind the Metal prelude (D96), compiled at run time from embedded source | single, with host double for decisions | Accelerate's sparse Cholesky in double (the default there, D92) | System frameworks only (Metal, Foundation, Accelerate) |
| WebGPU (stretch) | WGSL | single | none | Browser |

Notes:

- There is no vendor GPU sparse direct solver on Apple. Option B on Apple means the
  reduced system is built on the GPU and factorized on the CPU by Accelerate, which is
  reasonable because that system is small.
- The CUDA backend with option A needs only the CUDA runtime library, the same
  footprint as Caspar.
- CUDA libraries are not bundled into the Python wheel. The CUDA packages
  (just-ba-cu12, just-ba-cu13, D114) link the CUDA runtime statically and depend on
  NVIDIA's own pip packages for cuDSS, cuBLAS and NVRTC, which `just_ba` loads from
  those packages before the compiled module (D115).
- WGSL could also serve as the Apple kernel source through a WebGPU implementation,
  giving two kernel sets instead of three. That adds a dependency between the library
  and Metal, which conflicts with the "no intermediary" requirement, so it is not the
  plan; it remains a fallback if maintaining three sets proves too costly.

## Build order of backends

1. CPU reference in double. Everything is validated against Ceres here first.
2. Apple (Metal). Built first among the GPU backends because the main development
   machine is Apple Silicon, and because Metal's single-precision-only constraint
   tests the precision design early.
3. CUDA, on Linux and Windows. The speed comparison with competitors happens here.
4. WebGPU. Serves the browser, and run natively it also covers Windows machines
   with Intel or AMD GPUs (desktop app, document 09) and Android phones
   (document 10).

All backends are compiled into one binary where the platform allows and selected at
run time. Running out of device memory is a reported error with a CPU fallback; an
8 GB GPU is the reference consumer device. As built on Metal and CUDA (D102): with
`Auto`, option B is not taken where its reduced system would not fit the device's
free memory or its 32-bit indices, a set-up of B that still runs out starts again
with option A, and a factorization that runs out part way hands the rest of the solve
to A; asked for explicitly, B reports that it does not fit. A buffer that cannot be
allocated for option A is a reported error; the fallback to the CPU is not built.

## Repository layout

Directories marked `*` do not exist yet.

Each library keeps its public headers in `include/jba` and its internal ones in
`internal/jba` (D79); only the public directories reach users of the library.

```
docs/            this plan
core/            problem data, problem file, LM loop, gauge choice, priors, starting poses
residuals/       symbolic definitions and generated code per target
backends/
  cpu/           the solver backend
  gpu/           what the GPU backends share: kernel source, structures, host side (D96)
  metal/         Apple GPU backend (milestone 4); the milestone 0 spike kernel too
  cuda/          NVIDIA GPU backend (milestone 5)
  webgpu/        WebGPU backend, WGSL kernels, opt-in (document 16, D127)
tracks/          track building, triangulation
pipeline/      * incremental reconstruction (test fixture)
bindings/
  python/
  wasm/          WebAssembly module and demo page (exploration, document 16)
api/             public entry points (solve, evaluate, refine), later the C-level API
cli/             sample binary and benchmark driver
bench/           benchmark driver and stored baselines (D81)
tests/           unit tests; tests/ceres holds the reference harness, tests/real runs private captures when present
third_party/     reference solver build (Ceres), test-only
```

## Open decisions

| Decision | How it gets settled |
|---|---|
| Default linear strategy (A, B or C) per problem size | Metal and CUDA with cuDSS: B wherever it fits the device, measured end to end from 16 to 10,000 images and on tracks of 8 to 100 images (D92, D98, D102). CPU: benchmark once it has B |
| Preconditioner beyond block-Jacobi | Benchmark on ill-conditioned prior-heavy problems |
| Symbolic tool | Decided: wrenfold (D45) |
| Whether CUDA double is worth shipping | Accuracy results in single precision |
| Atomics fast mode on CUDA | Benchmark |
