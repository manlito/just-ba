# 10. Mobile app as a downstream consumer

A further stretch goal, beyond the desktop app: a phone app that lets people capture
and reconstruct on the device and get a 3D model, eventually as Gaussian splats
(no meshes planned). Everything runs locally on the phone.

**The app is not part of this project.** It lives in its own repository and links
this library. This document only records what being linked from a phone requires, so
the library stays friendly to that use.

## Why it fits

The phone is where the library's main use case originates: ARKit or ARCore supplies
the poses, intrinsics and depth that the library is built to refine. On the device
the same inputs arrive directly, without an export step.

## What the library must allow

| Requirement | Consequence for the library |
|---|---|
| Linked from Swift and Kotlin, no Python | A stable C-level API is the primary interface; Python is one binding on top of it, not the only way in (already the layering in document 03) |
| iOS | The Metal backend is the same kernel set as on Mac. The library must build for iOS on arm64 as a static library or framework. |
| Android | No CUDA and no Metal. GPU use means the WebGPU kernel set run natively over Vulkan; otherwise CPU on arm64. A third reason for that backend, after the browser and non-NVIDIA Windows machines. |
| Small download | No heavy dependencies in the core, optional parts excluded at build time (reference solver, incremental pipeline, unused backends) |
| Limited memory, shared with the rest of the app | Lowest-memory solver strategy available everywhere; a caller-set memory budget; clear failure instead of a crash |
| Heat and battery | Solves must be interruptible and resumable; iteration and time budgets exposed; nothing assumes it may use every core |
| Runs after capture | Batch only. The capture finishes, then the session is processed. Online or incremental adjustment during capture is not a goal (owner's decision). |
| No console | Logging only through the callback (already a requirement) |

## Output for Gaussian splatting

Splat training takes camera poses, intrinsics and a sparse point cloud as its
starting point, most commonly in COLMAP's model format. The library already plans to
read and write that format for parity testing (document 05), so export for a splat
trainer needs nothing extra. Training the splats themselves is outside this project.

## Effect on the language decision

Neutral. Both C++ and Rust build for iOS and Android and expose a C-level API. C++
is the more common path into Xcode and the Android NDK; Rust needs an extra binding
layer but is routinely shipped on both. The recommendation in document 06 stands.

## What changes now

Nothing is built for mobile in this project. Three constraints are adopted so the
option stays open:

1. The C-level API is the primary interface, kept free of Python-only concepts.
2. The core and CPU backend avoid anything that does not exist on iOS or Android.
3. Optional components are separable at build time.
