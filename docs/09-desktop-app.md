# 09. Desktop app (later)

A downloadable application for Windows and macOS, for people who are not developers.
They drag in photos, it reconstructs, they see the result. It is one more way to try
the library, added after the library itself works. This document records what it
requires so that earlier work does not block it.

## What the owner asked for

- Runs locally on Windows and on Mac. No install steps beyond downloading it.
- Uses the GPU when there is one: NVIDIA on Windows, Apple Silicon (M1 to M5) on Mac.
- Still works on CPU only. Not the fast path, but it must run.
- Aimed at people who want a solution, not at developers.
- Test hardware available: a Windows laptop with an 8 GB NVIDIA GPU.

## What it changes in the rest of the plan

### 1. Windows becomes a supported platform

Earlier documents treated Windows as "only if asked". It is now a target:

- The CPU and CUDA backends must build and pass tests on Windows.
- Windows wheels on PyPI follow for free once that is true.
- Hosted Windows CI runners are free for public repositories; GPU testing on Windows
  uses the owner's laptop.

### 2. "Drag in photos" needs steps the library deliberately does not have

Photos alone carry no keypoints, no matches and no pose priors. Going from photos to
a reconstruction needs, in order:

| Step | Where it lives |
|---|---|
| Read images and their metadata (focal length from EXIF) | App |
| Detect keypoints and match them between images | App, using an existing third-party component |
| Verify matches geometrically | App, or the pipeline module |
| Build tracks | Library helper (document 04) |
| Incremental reconstruction, since there are no pose priors | Pipeline module (document 04, part C) |
| Bundle adjustment | Library core |
| Show the result in 3D | App |

Two consequences:

- **Feature detection and matching stay out of the library.** The app bundles a
  separate component for them. The library's scope and dependency list do not change.
- **The incremental pipeline stops being only a test fixture.** The app depends on
  it, so it needs to be reliable on ordinary photo sets, not just comparable to
  COLMAP on benchmark scenes. It still lives outside the core and gets no claim on
  the core's design. This is the largest cost of the app and should be weighed when
  the app is scheduled.

An alternative that avoids that cost: the app's first version accepts captures that
already contain poses (for example exported from a phone capture app) and only later
accepts bare photos. Owner's call when the app is planned.

### 3. GPUs that are not NVIDIA or Apple

Many Windows laptops have only an Intel or AMD GPU. CUDA does not run on them, so
with the current backend list they fall back to CPU.

The WebGPU kernel set planned for the browser (document 03) also runs natively on
Windows through Direct3D or Vulkan, and would cover those machines. So the app raises
the value of that backend: it would serve the browser and non-NVIDIA Windows GPUs
with one kernel set. It stays after CUDA and Metal in the build order, but is no
longer only a stretch goal for the web.

### 4. Memory budget

An 8 GB GPU is the reference consumer device. The implicit Schur strategy
(document 03) has the lowest memory use of the options and is the default for the
app. The benchmark (document 05) gains a memory ceiling: the standard problem sets
must fit in 8 GB of device memory, and problems that do not fit must fail with a
clear message or fall back to CPU, not crash.

### 5. CPU-only must be dependable

For this audience the CPU backend is the safety net on any machine. That raises it
from "accuracy reference" to "must be reasonably fast and never fail". It already
exists first in the roadmap; the change is that its speed on a typical laptop gets
measured and tracked.

## Packaging and distribution

| Topic | Note |
|---|---|
| Backend selection | Automatic at start-up: CUDA if a suitable NVIDIA GPU and driver are found, Metal on Apple Silicon, otherwise CPU. Shown to the user, with a manual override. |
| NVIDIA libraries | The app needs only the CUDA runtime if the default solver strategy is used. NVIDIA's licence allows shipping its runtime libraries inside an application under conditions; the exact terms must be read before release. The user still needs a recent NVIDIA driver. |
| Mac | Apple Silicon only is sufficient (M1 and later). Needs code signing and notarization, which requires a paid Apple developer account. |
| Windows | Needs code signing to avoid security warnings, which requires a certificate. |
| Download size | Kept small by not bundling optional solver libraries. |
| Updates | Decide when the app is planned. |

## UI technology

Not decided now. The app sits on the same C-level API or Python package as every
other user of the library, so the choice does not affect the library. Options to
weigh when the time comes: a Python application bundled with the wheel (fastest to
build, larger download), or a native or web-technology shell over the C-level API.
A 3D viewer for cameras and points is the one essential component.

## What the library does now to keep this possible

- Windows is in the build and CI matrix from the start (milestone 1), so it never
  becomes a port.
- Backends are selected at run time, not at build time, in one binary.
- Out-of-memory on the device is a reported error with a CPU fallback path.
- Progress and cancellation are part of the solver API (a callback per iteration),
  which a UI needs and the CLI benefits from.
- The pipeline module keeps a clean boundary: matches in, reconstruction out.

## Open decisions (for when the app is scheduled)

| Decision | Note |
|---|---|
| Bare photos from the first version, or pose-carrying captures first | Decides how much the incremental pipeline must mature |
| Which feature detection and matching component to bundle | Licence and size matter; it is outside the library |
| UI technology | See above |
| Whether the native WebGPU backend is built for non-NVIDIA Windows GPUs | Depends on how common those users are |
