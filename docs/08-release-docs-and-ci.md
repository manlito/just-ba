# 08. Release, documentation and CI

The library will be open source and published to PyPI. It goes public after the first
working proof of concept. This document covers what that requires.

## Open source

| Item | Plan |
|---|---|
| Timing | Private until the proof of concept works (end of milestone 2), then public |
| Licence | Decided: Apache-2.0 (owner, 2026-11-14). Chosen over MIT for its explicit patent grant, patent retaliation clause and stated rule for contributions. |
| Reference code | PyPose `bae` is Apache-2.0 and Ceres is BSD-3-Clause. Neither is copied into the library; both are reference or test-only. |
| NVIDIA libraries | Not redistributed by us. The wheel depends on NVIDIA's own pip packages. |
| Before going public | Licence file, contribution guide, code of conduct, security contact, and a check that no private data sets or capture sessions are in the history |

## PyPI release

- Package name: `just-ba` was not registered on PyPI when checked on 2026-11-14
  (neither was `justba`). The name is not reserved until something is published
  under it.
- Publishing: PyPI trusted publishing from the release workflow, so no long-lived
  token is stored. The workflow runs on every push to main and publishes (D152); it
  uses no GitHub environment (D111); the pending publishers on PyPI are registered for `release.yml`. PyPI takes
  one pending publisher per workflow at a time, so the first release goes up one
  package at a time (D129).
- Versioning: semantic versions; the Python API is the public contract. Pre-1.0 while
  the API settles. The release workflow chooses the version: the last release's
  patch bumped, or `project(VERSION)` in CMakeLists.txt when that is higher, so a
  minor or major release is an edit there (D114, D152). Each release is tagged
  `vX.Y.Z`.
- Wheels and the sdist: `.github/workflows/release.yml`, on every push to main, or by
  hand to build without publishing (D116, D152).

### Release checklist

Every merge to main is a release (D152), so the checks come before the merge:

1. A GPU change: the GPU check by hand, `python bench/gpu_check.py` on the Mac (Metal)
   and on the laptop under Linux and Windows (CUDA); see document 05. To check the
   wheels themselves, `gh workflow run release --ref <branch>` builds them without
   publishing, and `python bench/gpu_check.py --wheel <file>` takes one.
2. A minor or major release: raise `project(VERSION)` in CMakeLists.txt in the same
   pull request. Otherwise the patch is bumped.
3. Merge. The release workflow builds and tests every wheel, publishes them, and
   tags `vX.Y.Z`; the docs workflow deploys the site.
4. A new package only (D129): its first upload fails until its pending publisher is
   added on PyPI; add it and re-run the failed publish job (`gh run rerun <run>
   --failed`; uploaded files are skipped).

### Wheels

| Wheel | Platforms | Contents |
|---|---|---|
| Default | Linux x86_64, macOS arm64, Windows x86_64 (Linux arm64 not in the first release, D108) | CPU backend. On macOS also the Metal backend, since it needs only system frameworks. |
| CUDA | Linux x86_64, Windows x86_64 | CPU plus CUDA backend. Depends on NVIDIA's pip packages for the CUDA runtime (and solver libraries if the direct-solve option is enabled). |
| Source distribution | all | For anyone building themselves |

Open points:

- Decided (D114): separate CUDA packages, `just-ba-cu12` and `just-ba-cu13`, beside
  `just-ba`; all three provide the module `just_ba`. They load NVIDIA's libraries
  from NVIDIA's pip packages (D115).
  Both CUDA 12 and CUDA 13 are supported, with the CUDA sources built as C++20
  (D113).
- Windows is a target (required by the desktop app, document 09): CPU and CUDA
  wheels for Windows x86_64, built and tested on hosted Windows runners, with GPU
  tests on the owner's Windows laptop.
- PyPI has a per-file size limit (100 MB by default). Not bundling NVIDIA libraries
  keeps the wheels far below it.
- Use the stable Python ABI so one wheel per platform covers many Python versions.

## Documentation

Built by `docs-site/` (Sphinx, D119) and the `docs` workflow, which runs on every push
to main and by hand (D148): it builds the live demo from poselib-gpu's sources, builds
the site with warnings as errors, runs the docstring examples, and publishes to
Cloudflare Pages (D112; by hand only with `deploy=true`). No previews on pull requests.

| Part | Content | Source |
|---|---|---|
| User guide | Install, quick start for the phone-capture use case, concepts (priors, depth, fixed blocks), backends | Hand-written pages |
| Python API reference | Every public function and type | Generated from docstrings and type stubs |
| Residual reference | Each residual: formula, parameters, Jacobian conventions | Generated from the same symbolic definitions as the kernels, so it cannot drift |
| Camera models | Formulas and parameter order | Same source |
| Benchmarks | Current accuracy and speed tables against the references | Generated from stored benchmark results |
| Core API reference | For contributors | Deferred (D119): the headers' comments are plain `//`, which Doxygen does not read |
| Design notes | This `docs/` folder | Not on the site for now (D119); the problem file format (document 11) is |

Tooling follows the language choice: Sphinx or MkDocs for the Python side, with
Doxygen (C++) or rustdoc (Rust) for the core. Hosted on Cloudflare Pages (D112),
which is free and serves a site built from a private repository.

Docstring examples are executed in CI so the documentation cannot go stale.

## Continuous integration

### What is free for a public repository

- **Standard GitHub-hosted runners are free for public repositories**: Linux, Windows
  and macOS, including Apple Silicon macOS runners.
- **GPU runners are not free, and are not used.** GitHub's GPU machines are "larger runners", which are
  billed per minute even on public repositories and require a paid plan (Team or
  Enterprise). The current offering is a single NVIDIA T4. The exact per-minute price
  was not confirmed; sources disagreed.
- Whether Metal compute works reliably on the hosted Apple Silicon runners is not
  yet established. It worked in the first run (2026-11-14, macos-15): the Metal
  test ran on the device and passed. Some projects use their own Macs for Metal
  testing.

### Plan

Current state (after milestone 1): the workflow is started by hand while the
repository is private (D49), in a quick mode (Linux and Windows, Ceres parity
without the slow test) or a full mode that adds macOS and venice-52 (D61). A third
job checks that the generated residual code is current. Since milestone 4 the full
mode's macOS job also runs the 114 Metal tests (`ctest -L gpu`) as an informational
step that cannot fail the run. That sits between the table below (hosted macOS "if
it proves reliable") and the decision at the end of this document (by hand, never
in CI). D110 settles it: the step is removed and the Metal tests are run by hand.
CI stays manual and light through milestone 6, and the release workflow is started
by hand too (D111). Since D148 (2026-11-17) CI's quick mode and the docs workflow run
on every push to main as well, never on pull requests. The table below is the plan for the public repository.

| Job | Runner | When | Cost |
|---|---|---|---|
| Build, unit tests, Level 1 and Level 2 checks on CPU (document 05) | Hosted Linux, Windows and macOS | Every pull request | Free |
| CUDA kernels compiled (not run) | Hosted Linux with the CUDA toolkit installed | Every pull request | Free |
| WebGPU backend compiled (not run), and the WebAssembly module built (D128) | Hosted Linux with Dawn's prebuilt release; Emscripten 7.0.0 with emdawnwebgpu | Every pull request | Free |
| Kernel logic on CPU: the generated kernels built as host code and compared with the reference | Hosted Linux | Every pull request | Free |
| Metal backend tests | By hand on the owner's Mac (D110) | Before releases and after GPU changes | Free |
| CUDA backend tests and speed benchmark | Not a CI job. Run by hand on the owner's laptop, or on the development VM when it is up | Before releases and after GPU changes | None beyond development |
| CPU speed regression gate | Hosted Linux | Every pull request | Free |
| Wheels and PyPI publishing | Hosted, all platforms | Every push to main (`release.yml`, D152) | Free |
| Documentation build and publish | Hosted Linux | Pull request preview; publish on release | Free |

Points to note:

- **Generated kernels make most GPU testing possible without a GPU.** Because each
  kernel is emitted from one symbolic source for several targets, the CPU emission
  exercises the same arithmetic. What still needs real hardware is the device
  plumbing, precision behaviour and speed.
- **CI must cost nothing (owner's rule).** Only free hosted runners are used. There
  are no scheduled or nightly jobs, no paid GPU runners, no cloud machines started
  by CI, and no self-hosted runners. Anything that needs a GPU is run by hand.
- **GPU checks are a manual step with a script.** One command runs the GPU test
  suite and the benchmark and writes the results file, so a manual run is quick and
  repeatable. The release checklist requires it.
- **Hosted runners give noisy timings.** The speed gate on hosted machines uses
  generous thresholds or deterministic counts (iterations, operator applications);
  precise timing comparisons are done by hand on a GPU machine.
- **Development VM.** The owner can start a Google Cloud VM with an NVIDIA L4
  (24 GB) for development sessions. It is for interactive development and manual
  benchmark runs only, and is never wired into CI.

## Open decisions

| Decision | Recommendation |
|---|---|
| Licence | Decided: Apache-2.0 |
| One wheel with run-time CUDA loading, or separate CUDA wheels | Decided: separate packages (D114) |
| GPU testing | Decided: by hand, never in CI. The owner's Mac for Metal; the owner's laptop (Linux and Windows) or the development VM for CUDA. |
| Documentation host | Decided: Cloudflare Pages (D112) |
