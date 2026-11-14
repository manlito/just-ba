# just-ba planning documents

just-ba ("just bundle adjustment") is a planned minimal, high-performance bundle
adjustment library with Python bindings and a sample command-line binary. Its main
job is refining reconstructions that come with noisy camera poses from ARKit, ARCore
or another tracker.

Status: milestone 1 (CPU solver, parity with Ceres) done; milestone 2 (the main use
case on the CPU) in progress. Results in [07-roadmap.md](07-roadmap.md). Milestones 0
and 1 were reviewed before milestone 2; the findings are in
[12-review-milestones-0-1.md](12-review-milestones-0-1.md). Build instructions:
[../README.md](../README.md).

## Documents

| # | Document | What it settles |
|---|---|---|
| 00 | [Decision log](00-decisions.md) | Every decision made so far, with reasons and rejected options. Read first. |
| 01 | [Requirements and scope](01-requirements.md) | Use cases, features, non-goals |
| 02 | [Problem formulation](02-problem-formulation.md) | Variables, the closed residual list, camera models, priors, depth |
| 03 | [Architecture](03-architecture.md) | Layers, solver, precision, backends, repository layout |
| 04 | [Tracks and incremental pipeline](04-tracks-and-incremental.md) | Track building, triangulation, the test-only reconstruction path |
| 05 | [Validation and benchmarks](05-validation-and-benchmarks.md) | Parity with Ceres and COLMAP, speed benchmark, regression gates |
| 06 | [Language decision](06-language-decision.md) | C++ or Rust, with a recommendation |
| 07 | [Roadmap](07-roadmap.md) | Milestones, owner decisions, risks |
| 08 | [Release, documentation and CI](08-release-docs-and-ci.md) | Open source, PyPI, docs site, CI and GPU testing |
| 09 | [Desktop app](09-desktop-app.md) | The later Windows and Mac app, and what it requires of the library now |
| 10 | [Mobile app as a consumer](10-mobile-consumer.md) | A separate-repository phone app that will link this library, and the constraints that keeps open |
| 11 | [Problem file format](11-problem-file-format.md) | The on-disk format used by the CLI and tests |
| 12 | [Review of milestones 0 and 1](12-review-milestones-0-1.md) | What was checked, what was found, what changes before milestone 2 |

## Decisions waiting on the owner

Decided so far:

- Language: C++, standard C++23 (D52).
- Correctness means agreeing with Ceres Solver.
- The speed claim is made for 50 to 2,000 images; up to 10,000 is supported.
- Depth starts as a term at keypoints; photometric refinement (as in PyPose `bae`)
  is an optional, low-priority module for later.
- Rolling shutter is out of scope.
- The Manhattan prior is a first-class feature, built from caller-supplied line
  detections with no line matching.

- One camera per session normally; several intrinsics blocks of one model are
  accepted, for BAL (D53). Frames are an unordered set. Processing is offline batch.
- Pose priors come in two forms, both supported: relative (for trackers such as
  ARKit and ARCore) and global (for drift-free sources such as a pose-predicting
  network). The caller chooses the pairs and removes priors that are completely
  wrong; the library is robust to the rest.
- GPS is deferred; ground control points are a stretch.
- Licence: Apache-2.0.

Still open (full list in [00-decisions.md](00-decisions.md)):

1. **Default prior weights.** The values need real captures to set.
2. **Real sessions for a private test set.** Milestone 2 cannot meet its exit
   condition without them.

## Reference material

- PyPose `bae`: approach reference only. Not a dependency and not copied.
- Ceres Solver 2.2.0: accuracy and CPU speed baseline, test-only.
- COLMAP 4.2.1: end-to-end parity reference; its Caspar backend is the GPU speed
  reference.

Facts about external projects were checked on 2026-11-14 and will age. Each document
lists what could not be verified.
