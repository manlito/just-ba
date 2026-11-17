# Demo app

A minimal incremental reconstruction (document 04, part C): poselib-gpu verifies the
matches of every pair, gives the first two poses and registers each further image by
PnP; this library builds the tracks, triangulates and adjusts. It is the only part of
this repository that links poselib-gpu (D132): the library, the wheels and the CLI do
not, and a default build does not look for it.

| File | What it is |
|---|---|
| `src/mapper.h`, `src/mapper.cpp` | The mapper: pair verification, tracks (connected components of the verified matches, split by triangulation where a wrong match joined two points; D138), first pair, registration, triangulation, adjustment and filtering. Ported from poselib-gpu's incremental test, using only its public header `jpo.h`. |
| `src/scenes.h`, `src/scenes.cpp` | Generated captures with ground truth: orbit (16 images), wall walk (14), camera grid (13), with keypoint noise, wrong matches and pairs that do not overlap. |
| `src/evaluate.h`, `src/evaluate.cpp` | A reconstruction against its ground truth. |
| `src/capture_io.h`, `src/capture_io.cpp`, `src/container.*` | The capture file (below), in the container of document 11. |
| `src/compare.h`, `src/compare.cpp`, `src/align.*` | A reconstruction against another of the same capture: COLMAP's, or the truth (document 05, level 3). |
| `src/main.cpp` | `jba_demo`, the command line. |
| `tools/colmap_capture.py` | A COLMAP database, and its model, to a capture file. Standard library only. |
| `tools/colmap_compare.sh` | One scene from its images: COLMAP, then the capture file, then `jba_demo run`. |
| `tools/colmap_fixture.py`, `tools/test_colmap_roundtrip.py` | The COLMAP path tested without COLMAP. |
| `web/demo_wasm.cpp`, `web/index.html` | The live demo: the WebAssembly entry points and the page. |

## Build

poselib-gpu's sources are expected next to this repository (`../poselib-gpu`);
`-DJBA_POSELIB_GPU_DIR=<path>` points elsewhere. Its GPU backends follow this
build's: Metal on Apple, CUDA with `JBA_WITH_CUDA`, WebGPU with `JBA_WITH_WEBGPU` and
the same Dawn.

```bash
cmake -S . -B build/demo -G Ninja -DJBA_BUILD_DEMO=ON
cmake --build build/demo --target jba_demo
```

```bash
build/demo/demo/jba_demo synth
```

Each generated scene is reconstructed and compared with its ground truth. Options:
`--scene orbit|wall|grid|all`, `--seed n`, `--noise px`, `--wrong fraction`,
`--false-pairs n`, `--poses auto|cpu|metal|cuda|webgpu` (poselib-gpu's backend),
`--ba cpu|metal|cuda|webgpu` (the solver's), `--export dir` (COLMAP's text model per
scene), `--write dir` (each scene as a capture file, its truth as the reference), `-v`
(each step). With `JBA_BUILD_TESTS`, `ctest -R demo_` runs it and the COLMAP round trip.

## Against COLMAP

The comparison of milestone 7 (document 05, level 3): both systems start from the same
matches of one COLMAP database. With COLMAP 4.2.1, after its feature extraction,
matching and `colmap mapper`:

```bash
colmap model_converter --input_path sparse/0 --output_path sparse/0-txt --output_type TXT
python3 demo/tools/colmap_capture.py database.db --model sparse/0-txt -o scene.jbc
build/demo/demo/jba_demo run scene.jbc
```

Or all of it for one scene, from its images, with `demo/tools/colmap_compare.sh
<images> <work_dir>` (`COLMAP=` and `JBA_DEMO=` name the executables): feature
extraction with one SIMPLE_RADIAL camera, exhaustive matching (`MATCHER=sequential_matcher`
for long sequences), the mapper, the conversion, the capture file and `jba_demo run`,
whose output lands in `<work_dir>/compare.txt` and our model in `<work_dir>/ours`.
COLMAP is a reference here, never a dependency: install it somewhere disposable and
remove it afterwards, as for milestone 5 (`bench/competitors/README.md`).

`colmap_capture.py` takes COLMAP's verified matches (`two_view_geometries`); `--raw`
takes the matcher's output instead, for the demo's own verification. Pairs with fewer
than 15 matches are left out (`--min-matches`). The cameras of the images must share
one of the six models the library has.

`jba_demo run` reconstructs the capture with COLMAP's mapper thresholds (RANSAC at 4 px
for pairs and 12 px for PnP, 30 PnP inliers, filter at 4 px; `--thresholds demo` for
the generated scenes' own) and the reference's refined intrinsics (`--intrinsics
capture` for the database's), and prints both sides: images registered, points,
observations, mean track length, mean reprojection error; then, on the images both
registered, rotation error after the best common rotation and camera-centre error
after the best similarity, as a share of the extent of COLMAP's camera centres; then
which keypoints each side groups into which point (points split or joined, and why
the reference's observations are missing from ours). `--two-view-tracks` and
`--tri-angle deg` undo two of COLMAP's rules, for experiments.
`--export dir` writes our model in COLMAP's text format next to it. The margins of
the pass condition are set after the first measurements (document 05).

Run against COLMAP 4.2.1 on South Building and Gerrard Hall on 2026-11-16 (results in
document 07, milestone 7). `ctest -R demo_colmap_roundtrip` runs the path without COLMAP: a generated
scene written as a COLMAP database and text model by `colmap_fixture.py` (COLMAP's
schema, ids reversed with gaps, keypoints with six columns, a camera per image for the
orbit), read back by `colmap_capture.py`, checked to hold the same capture, and
reconstructed against its truth.

## The capture file (.jbc)

Document 11's container (its header, section layout and numpy snippet apply) with
these sections. Image and keypoint indices start at 0; keypoints are in the pixel
convention of the intrinsics (COLMAP's: the centre of the first pixel at 0.5, 0.5).

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `camera.model` | i32 | [1] | yes | COLMAP model id, 0 to 5 |
| `intrinsics` | f64 | [B, P] | yes | One block per camera |
| `camera.size` | i32 | [B, 2] | no | Width, height |
| `image.intrinsics` | i32 | [N] | if B > 1 | Block of each image |
| `image.names` | u8 | [bytes] | no | N names, separated by `\n` |
| `keypoint.begin` | i64 | [N + 1] | yes | Image i owns keypoints begin[i] to begin[i + 1] - 1 |
| `keypoint.xy` | f64 | [K, 2] | yes | Pixels |
| `pair.images` | i32 | [R, 2] | yes | Images a, b of each pair |
| `pair.begin` | i64 | [R + 1] | yes | Pair r owns matches begin[r] to begin[r + 1] - 1 |
| `pair.match` | i32 | [M, 2] | yes | Keypoint in a, keypoint in b (indices within each image) |
| `reference.intrinsics` | f64 | [B, P] | no | The reference's own (refined) intrinsics |
| `reference.registered` | u8 | [N] | with a reference | 1 where the reference registered the image |
| `reference.pose` | f64 | [N, 7] | with a reference | World to camera, qw qx qy qz tx ty tz |
| `reference.point.xyz` | f64 | [Q, 3] | with a reference | Points |
| `reference.obs.image`, `.point`, `.keypoint` | i32 | [O] | with a reference | Each observation's image, point and keypoint |

## The live demo

Emscripten 7.0.0 and emdawnwebgpu from Dawn's release (D122). Emscripten needs Python
3.10 or later (`EMSDK_PYTHON` if the first `python3` on the path is older).

```bash
emcmake cmake -S . -B build/demo-wasm -G Ninja -DJBA_BUILD_DEMO=ON -DJBA_WITH_WEBGPU=ON \
  -DJBA_EMDAWNWEBGPU_DIR=<emdawnwebgpu_pkg>
cmake --build build/demo-wasm
```

`build/demo-wasm/demo/web` then holds `index.html`, `jba_demo.js` and
`jba_demo.wasm`, ready for any static server. The documentation site includes them
when built with `JBA_DEMO_WEB=build/demo-wasm/demo/web` (`docs-site/README.md`).

## What it checks

A scene passes when every image is registered, the median rotation error is below
0.5 degrees after the best common rotation, the median camera-centre error is below
1 % of the scene's extent after the similarity that best aligns the centres, and the
mean reprojection error is below 1.5 px or twice the keypoint noise. RANSAC
thresholds follow the noise (`thresholds_for_noise`): poselib-gpu's defaults up to
half a pixel, larger above.

Measured on 2026-11-16 on the owner's M5 Pro: the three scenes pass at seeds 0 to 8
natively, on the CPU and with Metal for both libraries, and at seed 0 in the browser
pane (Chrome) on WebGPU and on the CPU. With the default backends they pass up to 2 px
of noise with half the matches wrong and 30 unrelated pairs, except the orbit at half the matches
wrong, where no pair keeps the 100 inliers the first-pair rule asks for.
