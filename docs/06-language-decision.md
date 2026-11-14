# 06. Language decision: C++ or Rust

Status: **decided, C++** (owner, 2026-11-14). The comparison below is kept as the
record of why.
Facts below were checked against primary sources on 2026-11-14.

## What the choice does and does not affect

The hot code is not written in either language. Kernels are CUDA C on NVIDIA, Metal
Shading Language on Apple, and WGSL in the browser, generated from one symbolic
description (document 03). Those files are identical whichever host language is
chosen.

The host language covers: problem setup, the outer loop, the CPU reference backend,
track building, the device plumbing, bindings and the build. So the decision is about
integration cost and risk, not about run-time speed.

## Constraints that are the same in both languages

- No double precision on Metal or in WGSL. Single precision on the GPU with
  double on the host is forced either way.
- No atomic float add in WGSL.
- No vendor GPU sparse direct solver on Apple.
- Shipping CUDA-dependent Python wheels is the same problem: depend on NVIDIA's pip
  packages and load them at run time; do not bundle them.
- The browser backend is a separate kernel set.

## Comparison against the requirements

| Requirement | C++ | Rust |
|---|---|---|
| Python bindings (top priority) | nanobind 3.1.0: stable ABI from Python 3.12, zero-copy arrays including GPU arrays. Wheels with cibuildwheel. More build configuration per platform. | PyO3 0.30 with rust-numpy and maturin: one tool builds wheels, stable ABI from Python 3.9. Zero-copy for CPU arrays; GPU array exchange not verified. |
| CUDA | Native. Compiler, cuSPARSE, cuSOLVER and cuDSS 1.0.0 are C and C++ libraries. CMake treats CUDA as a first-class language. | `cudarc` 0.19 is an active, thin binding with run-time loading and covers the driver, cuSPARSE and cuSOLVER. No cuDSS binding exists; we would generate one. Writing kernels in Rust itself is alpha (Rust-CUDA, NVIDIA's cuda-oxide), so kernels stay in CUDA C. |
| Apple GPU | Objective-C++ or Apple's metal-cpp headers. Accelerate called directly. | `objc2-metal` 0.3 (the older `metal` crate is deprecated). Accelerate through C FFI. |
| CPU | Eigen, CHOLMOD and Accelerate are long established. | `faer` 0.24 has sparse Cholesky and is what the Rust solvers use. No published comparison with CHOLMOD found. |
| Browser (stretch) | Emscripten 7 with the Dawn WebGPU port. Works; Dawn has no tagged releases, so a commit is pinned. | wasm-bindgen with `wgpu` 30: one API for native and web. The smoother path. Threading in wasm needs extra tooling; status not verified. |
| Minimum dependencies, vendor libraries called directly | Direct by construction. No binding layer at all. | Direct through thin binding crates or hand-written FFI. Each crate is one more dependency, though a small one. |
| Ceres as reference | Linked natively. Our residuals can be compared with Ceres cost functors in one test process. | The `ceres-solver` crate (0.5.1) describes itself as incomplete. Practical route is to run Ceres as a separate program and compare files. |
| Generated Jacobians | SymForce's C++ output is its primary target; CUDA output is experimental. | SymForce's Rust output is minimal. wrenfold emits Rust. |
| Build and packaging | CMake across CUDA, Metal and wasm. Powerful, tedious. | Cargo for everything Rust; CUDA C kernels need a custom build step or run-time compilation. |

## Evidence from existing projects

- Every GPU bundle adjustment implementation found is C++ with CUDA: Ceres, COLMAP's
  Caspar, MegBA, Graphite, GTSAM's CUDA solver, and Sprux (which also has a Metal
  backend).
- Rust has several young CPU-only solvers (apex-solver, factrs, tiny-solver), all on
  the same sparse backend, none with a GPU path.
- kornia-rs (v0.2.0, September 2026) is the strongest Rust evidence. It ships a CPU
  bundle adjustment with a Schur complement, `faer` Cholesky, depth residuals with a
  per-view scale, and position, up-vector and motion priors, plus an incremental
  reconstruction. Its CUDA support (through `cudarc`, with kernels written in Rust
  and compiled by NVIDIA's cuda-oxide) covers image processing and stereo, not
  bundle adjustment.

Meaning: in C++ every problem we will hit has been solved in public code we can read
and benchmark against in-process. In Rust the CPU side is proven and close to our
residual set, and we would be first on the GPU side.

## Risks

| | C++ | Rust |
|---|---|---|
| Main risk | Build and packaging effort across three platforms and wasm; memory-safety bugs in host code. | Being first: no cuDSS binding, incomplete Ceres binding, immature GPU-kernel tooling, less prior art to copy. |
| Mitigation | Keep the host small; sanitizers and fuzzing in CI; cibuildwheel. | Keep kernels in CUDA C; write the missing bindings; run Ceres out of process. |
| What it costs if wrong | Slower iteration on tooling. | Time spent on infrastructure that is not bundle adjustment. |

## Recommendation: C++

Reasons, in order of weight:

1. **"No intermediary" is literal in C++.** CUDA, cuDSS, Metal and Accelerate are
   called with nothing in between. In Rust every one of them sits behind a binding
   crate or FFI we maintain, and one binding (cuDSS) does not exist.
2. **Correctness is checked against Ceres, and Ceres is C++.** Linking it into the
   test binary lets us compare residuals, Jacobians and whole solves in one process,
   at every level of document 05. That is the cheapest route to "most correct".
3. **All the competitors are C++ and CUDA.** To claim "fastest" we must run them
   side by side and read their kernels. Same language, same build, same profiler.
4. **The Python customer is served equally well.** nanobind gives zero-copy arrays
   and a stable ABI; the user never sees the host language.
5. **The host language does not affect speed**, so Rust's safety advantage applies to
   a small, non-critical part of the code.

Suggested specifics if C++ is chosen: C++20 (since superseded: C++23, D52), CMake, nanobind, no Eigen in the shipped
library (fixed-size block math is generated or hand-written), Ceres and Eigen as
test-only dependencies.

## What would change the recommendation

- **The browser becomes a primary target.** Rust with `wgpu` gives one code path for
  native and web, and could serve Apple too. If WebAssembly plus WebGPU moves from
  stretch to core, Rust becomes the better choice.
- **The native WebGPU backend becomes important.** The planned desktop app
  (document 09) would use it for Windows machines with Intel or AMD GPUs. That is
  the same kernel set as the browser, so it adds weight to this side of the scale
  without tipping it: CUDA and Metal remain the primary backends.
- **The owner or team is markedly more productive in Rust.** Since run-time speed is
  unaffected, maintainer velocity is a legitimate deciding factor, and the Rust risks
  above are extra work, not blockers.
- **A single kernel source is valued over direct vendor access.** A portable layer
  (`wgpu`, or CubeCL, which is still alpha) is more natural from Rust, at the cost of
  the "no intermediary" requirement.

## A hybrid that is not recommended

Rust host with CUDA C, Metal and WGSL kernels is workable, and is what a Rust choice
would look like. A C++ core with a Rust shell on top (or the reverse) adds a language
boundary for no gain and is not proposed.

## Items not verified

- Whether cuDSS 1.0.0 is formally out of preview, and its licence text.
- Performance of portable layers against native CUDA on this kind of workload.
- `faer` sparse Cholesky against CHOLMOD.
- Metal compute reliability on hosted macOS CI runners.
- Rust WebAssembly threading toolchain status in late 2026.
