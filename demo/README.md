# Demo app

A minimal incremental reconstruction (document 04, part C): poselib-gpu verifies the
matches of every pair, gives the first two poses and registers each further image by
PnP; this library builds the tracks, triangulates and adjusts. It is the only part of
this repository that links poselib-gpu (D132): the library, the wheels and the CLI do
not, and a default build does not look for it.

| File | What it is |
|---|---|
| `src/mapper.h`, `src/mapper.cpp` | The mapper: pair verification, first pair, registration, triangulation, adjustment and filtering. Ported from poselib-gpu's incremental test, using only its public header `jpo.h`. |
| `src/scenes.h`, `src/scenes.cpp` | Generated captures with ground truth: orbit (16 images), wall walk (14), camera grid (13), with keypoint noise, wrong matches and pairs that do not overlap. |
| `src/evaluate.h`, `src/evaluate.cpp` | A reconstruction against its ground truth. |
| `src/main.cpp` | `jba_demo`, the command line. |
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
scene), `-v` (each step). With `JBA_BUILD_TESTS`, `ctest -R demo_synth` runs it.

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
pane (Chrome) on WebGPU and on the CPU. On the CPU they pass up to 2 px of noise with
half the matches wrong and 30 unrelated pairs, except the orbit at half the matches
wrong, where no pair keeps the 100 inliers the first-pair rule asks for.
