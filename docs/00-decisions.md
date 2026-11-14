# 00. Decision log

Every decision made with the project owner during planning (2026-11-14), with the
reason and what was rejected. **Read this before asking the owner a question: if it
is answered here, do not ask again.** To change a decision, add a new dated entry
that supersedes the old one; do not silently edit history.

## Purpose and scope

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D1 | just-ba is a minimal bundle adjustment library, aiming to be the fastest and most correct for its narrow job, at the cost of generality. | Existing tools (PyPose `bae`, COLMAP with Caspar, Kornia) are general frameworks. The owner wants something minimal and as low-level as needed. | A general nonlinear least squares or factor graph framework. |
| D2 | PyPose `bae` is a reference for the approach only. | The owner pointed to it to show the direction. | Forking, porting or depending on it. |
| D3 | Residuals are a closed, built-in set. New ones are added by changing the library. | Fixed block sizes and matrix structure allow full specialisation. | User-defined residuals at run time. |
| D4 | "Most correct" means agreeing with Ceres Solver on the same problem. Ground truth is only a diagnostic for tuning. | Ceres is the trusted implementation. | Defining correctness by closeness to ground truth. |
| D5 | "Fastest" is claimed for 50 to 2,000 images. Up to 10,000 is supported without that promise. More than 10,000 is not a goal. | That is the real workload; keyframe selection happens outside. | Optimizing for city-scale problems. |
| D6 | Scene scale is a single building: house interiors and exteriors, a large commercial space or patio, up to a small or medium park. | That is what will be reconstructed. | City-block scale and beyond. |
| D7 | Processing is offline batch: capture finishes, then the session is processed. | "This is not a SLAM." | Online or incremental bundle adjustment during capture. |
| D8 | Deliverables: the library, Python bindings, a sample command-line binary. | Python is the main customer. | |

## Inputs

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D9 | Feature detection and matching are outside the library. It receives keypoints (pixel coordinates) and matches or tracks. | Matching is done separately; the owner plans a state-of-the-art matcher on the caller's side. | Descriptors as input (not needed). Detector-free matching (not for now). |
| D10 | The library defines the input contract; callers convert to it. A layer over ARKit and ARCore on the caller's side does the conversion, including axis and orientation conventions. | Keeps the library free of platform formats. | Reading ARKit or ARCore structures in the library. |
| D11 | One camera per session: one intrinsics block, one camera model. | Simpler and faster; multi-lens sessions are uncommon. | Several cameras or lenses in one session. |
| D12 | Frames are an unordered set. Capture order and timestamps are not inputs. | Photos may be dragged in with no order; revisits make order unreliable as a guide. | Assuming consecutive frames. |
| D13 | Not assumed as inputs: velocity, inertial data, tracking-state flags, pose uncertainty from the tracker. | "Assume it's just poses." | Motion data of any kind. |
| D14 | Optional inputs: gravity per frame, depth map with confidence, line segments per frame. | Available on capable phones; never required. | Making any of them mandatory. |
| D15 | Rolling shutter is out of scope. One pose per image. | It needs motion input or many extra unknowns; keyframe selection reduces it at the source. | Estimating it, or consuming velocity to correct for it. Revisit only if real captures show it limits accuracy. |

## Pose priors

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D16 | Both relative and global pose priors are supported, alone or together. The caller chooses per solve. | Relative suits trackers (accurate locally, drifting globally). Global suits drift-free sources, for example poses predicted by a network such as VGGT in its own frame. | Relative only (decided briefly, then reversed the same day). |
| D17 | Cleaning tracker priors is the caller's job. The caller's system estimates co-visibility, chooses the pairs, and removes priors that are completely wrong, such as across a tracking break. | Sessions break often, especially on Android; the owner prefers that validation live in a separate system. | Pair selection, break detection and validation against matches inside the library (designed, then removed). |
| D18 | The library stays robust to uneven priors: robust loss on every prior, a stronger graduated option, and a report of priors that were heavily down-weighted. | Some priors that survive cleanup will still be worse than others. | Trusting priors blindly. |
| D19 | A set of global priors declares its relation to the working frame: it is the frame, or it has a rigid transform, or a similarity transform with scale. | Network poses have their own frame and arbitrary scale. | Assuming every prior source is metric and in one frame. |
| D20 | With relative priors only, the output is in the library's own frame and the caller aligns it to the AR frame. | No global pose information is passed in for that case. | The library aligning to the AR frame itself. |
| D21 | Default prior weights are set by the library when none are given. The values are to be set from real captures. | Trackers report no uncertainty. | Open item; see the end of this document. |
| D22 | GPS priors are deferred, lower priority. Ground control points are a stretch. | GPS is common and will be the only fully global source, but not needed first. | Including either in the first release. |

## Other residuals and features

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D23 | The Manhattan prior is a first-class feature, built from caller-supplied line detections. The library assigns lines to the three axes and estimates the vanishing directions itself. Opt-in per solve. | A very strong prior for interiors and house exteriors; corrects heading about the vertical. | Line matching between images (not a solved problem, and not needed). |
| D24 | Depth: the first release uses a depth term at keypoints with a per-image scale and offset. | Fits a keypoint-based library. | |
| D25 | Photometric depth refinement (as in the `bae` example) is an optional, low-priority module for later, only if it fits on top of the core. | It needs image pixels and separate machinery; core bundle adjustment must stay first class. | Making it a requirement or letting it shape the core. |
| D26 | Camera models: pinhole and wide-angle, with distortion for both. Names and parameter order follow COLMAP. Intrinsics are fixed by default and can be refined. | Phones supply intrinsics; refinement matters only without prior data. | |
| D27 | Track building is a helper in the library, simple at first, tolerant of bad matches. | The owner wants it provided; the method was left to research. | |
| D28 | An incremental reconstruction path exists only to test without priors and to show parity with COLMAP. | Validation, not a product feature. | A full structure-from-motion product. |
| D29 | Outputs: poses, point cloud, tracks, uncertainty, depth results, and the priors used. Enough for a user to continue with their own bundle adjustment. | The owner's export rule. | Meshes (never planned). |

## Technology

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D30 | Language: C++. | Vendor libraries called with nothing in between; Ceres links into tests natively; all GPU competitors are C++. | Rust. Would be reconsidered only if the browser became a primary target. |
| D31 | No PyTorch and no intermediate framework. Minimum dependencies. CUDA only when the CUDA backend is used. | Performance and control. | |
| D32 | Backends: CPU, Apple Silicon GPU (Metal), CUDA. Build order: CPU, then Metal, then CUDA. | The development machine is an M5 Mac; Metal's single-precision limit tests the precision design early. | CUDA before Metal (the original order). |
| D33 | Ceres Solver is an optional, test-only reference for accuracy and CPU speed. | It defines correctness (D4). | Shipping Ceres or using it as the CPU backend. |
| D34 | Platforms: Linux, macOS on Apple Silicon, Windows. | A desktop app for Windows and Mac is planned. | |
| D35 | The CPU backend must be dependable, not only a reference. | It is the fallback on machines without a supported GPU. | |
| D36 | WebAssembly with WebGPU in the browser is a stretch goal. | Nice to have; Python is the real customer. | |

## Release, CI and hardware

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D37 | Open source under Apache-2.0, made public after the first working proof of concept. Published to PyPI. | Patent grant and contribution terms. | MIT (simpler, but no patent terms). |
| D38 | Generated documentation site, built in CI. | Needed for a public PyPI release. | |
| D39 | CI must cost nothing: free hosted runners only. No nightly or scheduled jobs, no paid GPU runners, no cloud machines started by CI, no self-hosted runners. | "I don't want to have costs." | A nightly GPU job (proposed, then rejected). |
| D40 | GPU testing and benchmarks are run by hand. | Follows from D39. | |
| D41 | Hardware: the owner's M5 Mac (Metal); the owner's laptop with an 8 GB NVIDIA GPU, with both Linux and Windows (CUDA); a Google Cloud VM with an L4 GPU that the owner starts for development sessions only. | What is available. | Using the VM for CI or scheduled jobs. |
| D42 | The name `just-ba` was free on PyPI when checked on 2026-11-14. | | |

## Downstream consumers

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D43 | A desktop app for Windows and Mac, for non-developers, is planned for later. UI details are not decided now. | One more way to try the library. | Deciding UI technology during planning. |
| D44 | A mobile app is a further stretch goal, in a separate repository that links this library. Eventually Gaussian splats; no meshes. | The owner's direction. | Building anything for mobile inside this project. |

## Milestone 0 (2026-11-14)

Settled during the milestone 0 work. D45 to D51 were made in the work and recorded
for the owner to review; D52 and D53 are the owner's.

| # | Decision | Why | Rejected or ruled out |
|---|---|---|---|
| D45 | Symbolic tool: wrenfold, pinned (0.6.0), with small custom emitters for C++ (templated, raw pointers) and Metal. Generated files are checked in; CI checks they are current. | Its output can be reshaped by overriding formatters, so both emitters are about 100 lines and the generated code needs no header. Output was identical across runs. | SymForce: its C++ output uses Eigen and its own runtime types, against "no Eigen in the shipped library"; its CUDA output is experimental. Hand derivation: not needed. |
| D46 | Pose tangent: rotation and translation updated separately. R <- Exp(dtheta) R with dtheta a rotation vector in radians (left perturbation), t <- t + dt. Order (dtheta, dt). | Simple Jacobians; covariances in radians and metres; a matching Ceres manifold makes parity direct. | The SE(3) exponential: couples rotation and translation for no gain in bundle adjustment. |
| D47 | Quaternions are stored (w, x, y, z), Hamilton convention. | Same as COLMAP's text files and Ceres's rotation functions. | Eigen's (x, y, z, w) memory order. |
| D48 | Problem file format: a binary container of named, typed little-endian arrays (document 11). Unknown sections are kept, so new optional sections need no version change. | Fast for 10^6 observations, readable from NumPy in a few lines, no dependency in C++. | NPZ (needs zip and zlib in C++). Text (slow and large). HDF5 (a dependency). COLMAP's model format (cannot hold priors, weights or fixed flags; it stays the interchange format). |
| D49 | The CI workflow is manual-trigger only while the repository is private. Push and pull request triggers are switched on when it goes public. | Hosted minutes on private repositories count against a quota, macOS at 10x. Follows D39. | Running CI on every push now. |
| D50 | Metal kernels are compiled at run time from source embedded in the library. | Needs no Metal offline compiler (a separate Xcode download). Revisit in milestone 4 if compile time matters; pipelines can be cached. | Precompiled `.metallib` files for now. |
| D51 | Ceres for tests is built from the 2.2.0 tag archive on GitHub, with a pinned hash, by `third_party/ceres/build.cmake`. | ceres-solver.org did not resolve on 2026-11-14; the tag archive is the same release. | Package managers (versions and options vary by platform). |
| D52 | C++ standard: C++23 everywhere, CUDA sources included. Requires CUDA 13.3 or later. | The owner chose "whatever CUDA supports". nvcc accepts `-std=c++23` from CUDA 13.3 (13.5 documentation checked on 2026-11-14). Brings `std::expected` for the C-level API and `std::mdspan` for Jacobian blocks. | C++20 (the suggestion in document 06; no reason to stay). C++26 (compiler support still partial). |
| D53 | The solver accepts several intrinsics blocks, one per image at most, all of the same camera model. One block per session stays the normal case and the fast path. Refines D11. | BAL has one camera per image, and milestone 1 parity with Ceres runs on BAL. The file format already holds several blocks (D48). One model per problem keeps kernels free of per-observation branching. | BAL parity only after averaging or fixing intrinsics (changes the problem, so not a parity test). Several camera models in one problem. |

## Still open

| Item | When it gets decided |
|---|---|
| Default prior weight values (D21) | Milestone 2, from real captures |
| Which linear solver is the default at each problem size | From the benchmark |
| Whether the caller's system or the library joins groups of frames left unconnected after a tracking break | When real broken sessions are available |
| Desktop app: UI technology, bundled matcher, bare photos versus pose-carrying captures first | When the app is scheduled |

## Things discussed that turned out not to matter

Recorded so they are not raised again.

- **Capture order.** It would only help trust consecutive pairs in weakly matched
  places such as doorways. It does not help with revisits, and it is not an input
  (D12, D17).
- **Free GPU runners for open source.** They do not exist on GitHub. Moot under D39.
- **Whether the library needs images.** The core never reads pixels. Only the
  optional photometric module would (D25).
