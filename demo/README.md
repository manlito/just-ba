# Demo app

A minimal incremental reconstruction (document 04, part C): poselib-gpu verifies the
matches of every pair, gives the first two poses and registers each further image by
PnP; this library builds the tracks, triangulates and adjusts. It is the only part of
this repository that links poselib-gpu (D134): the library, the wheels and the CLI do
not, and a default build does not look for it.

| File | What it is |
|---|---|
| `src/mapper.h`, `src/mapper.cpp` | The mapper: pair verification, tracks (connected components of the verified matches, split by triangulation where a wrong match joined two points; D140), first pair, registration, triangulation, adjustment and filtering. Ported from poselib-gpu's incremental test, using only its public header `jpo.h`. |
| `src/scenes.h`, `src/scenes.cpp` | Generated captures with ground truth: orbit (16 images), wall walk (14), camera grid (13), with keypoint noise, wrong matches and pairs that do not overlap. |
| `src/evaluate.h`, `src/evaluate.cpp` | A reconstruction against its ground truth. |
| `src/capture_io.h`, `src/capture_io.cpp`, `src/container.*` | The capture file (below), in the container of document 11. |
| `src/compare.h`, `src/compare.cpp`, `src/align.*` | A reconstruction against another of the same capture: COLMAP's, or the truth (document 05, level 3). |
| `src/main.cpp` | `jba_demo`, the command line. |
| `tests/test_cancel.cpp` | `demo_cancel`: the mapper's progress callback (below). |
| `tools/colmap_capture.py` | A COLMAP database, and its model, to a capture file. Standard library only. |
| `tools/colmap_compare.sh` | One scene from its images: COLMAP, then the capture file, then `jba_demo run`. |
| `tools/colmap_table.py` | The comparison over several capture files, as Markdown tables. |
| `tools/colmap_fixture.py`, `tools/test_colmap_roundtrip.py` | The COLMAP path tested without COLMAP. |
| `web/demo_wasm.cpp`, `web/index.html`, `web/demo.js` | The live demo (D156): the WebAssembly entry points, the page and its script. |
| `web/samples/` | The live demo's sample set: 44 images of ETH3D's *delivery_area* and `samples.json` (their focal lengths). |
| `tools/fetch_match_web.sh` | The live demo's keypoints and matcher modules, from just-keypoints' and just-matching's sites. |
| `web/capture.html`, `web/capture.js`, `web/capture_worker.js`, `web/weights_worker.js` | The live capture (D157): the page, its camera loop, the mapper session in a worker, FlexGlue's weights from the visitor's checkpoint. |
| `web/gates.js`, `web/gates.test.mjs` | The live capture's keep rules, ported from Like Recon, and their tests (`node --test demo/web/gates.test.mjs`). |
| `web/view.js` | The live capture's 3D view. |

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
`--false-pairs n`, `--focal-error f` (the mapper is given focal lengths that much too
large, 1.2 being 2.2 times the truth, marked as a guess: it searches for them and
refines them), `--poses auto|cpu|metal|cuda|webgpu` (poselib-gpu's backend),
`--ba cpu|metal|cuda|webgpu` (the solver's), `--refine none|focal|focal+distortion`
(the intrinsics the adjustments refine; none by default here), `--export dir`
(COLMAP's text model per scene), `--write dir` (each scene as a capture file, its truth
as the reference), `-v` (each step). With `JBA_BUILD_TESTS`, `ctest -R demo_` runs it, the COLMAP round trip
and the progress callback's test.

## Calling the mapper

`demo::Mapper::run(capture, on_step, on_progress)`; both callbacks are optional and
neither changes the result.

- `on_step(const Snapshot&)` gets the kept model's steps (first pair, each
  registration, final) after the run, for drawing the reconstruction as it grew.
- `on_progress(const Progress&) -> bool` is called live, as the run goes (D146):
  after the focal search, pair verification and track building, after each step of
  every model with a `Snapshot` of it in `live` (valid during the call only; the
  model being grown, which may not be the one kept), and after every solver
  iteration of every adjustment (`stage` "adjust", with `iteration` and `cost`).
  `Progress` also carries the model's index, its registered images, the capture's
  images and the adjustments so far. Returning false stops the run as soon as it
  can, within one solver iteration: `Reconstruction::cancelled` is then true,
  `error` is "cancelled", the rest of the result is incomplete, and `on_step` is not
  called. This is what a desktop app needs for live progress and a Stop button.

With position priors in the capture (`Capture::prior_position`, D150), a model moves into
their frame once three of its images with a prior are registered and spread across a
line by more than five standard deviations, and every later adjustment holds those
images to their priors: the result is in the priors' frame, units and axes
(`Reconstruction::aligned`). `jba_demo synth --priors sigma` checks it on the generated
scenes; `jba_demo run --no-priors` ignores them.

## Sessions: a map that grows while the capture is taken

`demo::Session` (D153) grows a map image by image, for a capture that reconstructs as
it is taken. `add(SessionImage)` takes an image's keypoints, its matches to earlier
images of the session and an optional position prior. The matches are verified and
added to the tracks; once the map has started (a pair reaching 16 degrees of median
triangulation angle, or 8, 4 after `start_patience` images each without one), the
image is placed by PnP, its new points triangulated and a window of the latest
`window` images adjusted with the older ones that see their points held. An image
that can't be placed is tried again as the map grows; after more than
`restart_after` of them in a row the map is put aside as a piece and a new one
started from images in none (an image whose pairs link to an earlier piece goes back to it, which becomes the current one again); a young piece that started from a narrow pair is started
again once a wider one comes (`rebuild_within`). `snapshot()` is the map now (every
point with its track, stable as the map grows), `piece(k)` and `largest_piece()` the
pieces, `keypoint_points(i)` the landmarks of image `i`, `retry(pairs, new_piece)` tries the last image again with more pairs (added with
`defer_split`, no new piece starts until then), and `finish(capture)` reconstructs the whole capture (any pairs, normally
more of them, matched more thoroughly) from the live map with
`Mapper::run_from(capture, StartFrom)`. `run_from` keeps the start's poses, but leaves
out start images whose relative rotations disagree with their verified pairs by a
median of over 5 degrees, and registers them again like any other image.

A session can also start from a model (D154): `Session(mapper, model, intrinsics,
width, height, SessionSeed, options)` takes images already placed (their keypoints
and poses) and the model's points with the keypoints that observe them, one track per
point. The map has started with them; they and the intrinsics are held while new
images join, so a capture goes on into a finished reconstruction without moving it;
`num_seeded()` says how many there are.

`jba_demo run capture.jbc --resume` runs the mapper, then `run_from` its own result
(and again from that), and prints what the resume kept: points, observations, error,
the agreement with the run (D155).

```sh
build/demo/demo/jba_demo session live.jbc --finish all.jbc --window 8
```

feeds `live.jbc`'s images in order with their pairs to earlier images, as a capture in
progress would, finishes with `all.jbc`'s pairs (the same images), and compares the
largest piece and the finish with a batch run of `all.jbc` (and two batch runs with
different seeds, for the spread to expect). `demo_session` checks it on the generated
scenes.

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
capture` for the database's). As COLMAP's mapper, it refines focal length and
distortion in every adjustment (`--refine` to change that), and a focal length the
database marks as a guess (no `prior_focal_length`: no focal length in the image's
metadata) is first searched for: the pairs with the most matches are verified under 61
focal lengths from a quarter to four times the guess, in one poselib-gpu call, and the
one with the lowest total MSAC score is the start (D144). It prints both sides: images registered, points,
observations, mean track length, mean reprojection error; then, on the images both
registered, rotation error after the best common rotation and camera-centre error
after the best similarity, as a share of the extent of COLMAP's camera centres; then
which keypoints each side groups into which point (points split or joined, and why
the reference's observations are missing from ours). `--two-view-tracks` and
`--tri-angle deg` undo two of COLMAP's rules, for experiments.
`--min-pair-inliers n` and `--min-pair-coverage share` set the pair rules (D151): a
pair's inliers go to the tracks from n of them, and only if their convex hull covers
that share of each image (0, the default, turns the coverage rule off; the capture file
needs `camera.size`). With more than one model grown, every model is listed.
`--export dir` writes our model in COLMAP's text format next to it, and `--json file`
the figures, which `demo/tools/colmap_table.py` gathers over several scenes. With a
ground truth in the capture file (`colmap_capture.py --truth`, as ETH3D's
`dslr_calibration_undistorted`), both reconstructions are also compared with it. The margins of
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
| `camera.focal_unknown` | u8 | [B] | no | 1: the block's focal length is only a guess. Without the section, a focal length of exactly 1.2 times the larger side of `camera.size` (COLMAP's guess) counts as one |
| `image.intrinsics` | i32 | [N] | if B > 1 | Block of each image |
| `image.names` | u8 | [bytes] | no | N names, separated by `\n` |
| `image.prior_position` | f64 | [N, 3] | no | Position priors (D150): each image's measured camera centre in a metric frame whose third axis is up (east, north, up); NaN for an image without one |
| `image.prior_sigma` | f64 | [N, 2] | no | Their standard deviations, horizontal and vertical; NaN: the mapper's defaults (3 and 5) |
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
| `truth.registered` | u8 | [N] | no | 1 where the ground truth has a pose |
| `truth.pose` | f64 | [N, 7] | with `truth.registered` | Ground-truth world to camera, in the truth's frame and units |

## The live demo

Structure from motion on real photos in the browser (D156). The page loads the sample
set or the visitor's own images, finds ALIKED keypoints with just-keypoints' module and
matches every pair with LightGlue through just-matching's (both on WebGPU), then calls
`demo_reconstruct` with the keypoints and matches: the mapper with COLMAP's thresholds,
SIMPLE_RADIAL cameras whose focal length and distortion it refines, on the CPU or
WebGPU. By default it runs once on each and shows the mapper's timings side by side;
keypoints and matches are kept, so reconstructing again costs only the reconstruction.
URL parameters: `run=1` runs once the sample set is loaded, `n=k` keeps its first k
images, `backends=cpu|webgpu|both`, `focal=guess` drops the samples' focal lengths (as
for the visitor's own photos: searched for).

Emscripten 7.0.0 and emdawnwebgpu from Dawn's release (D122). Emscripten needs Python
3.10 or later (`EMSDK_PYTHON` if the first `python3` on the path is older).

```bash
demo/tools/fetch_match_web.sh
```

```bash
emcmake cmake -S . -B build/demo-wasm -G Ninja -DJBA_BUILD_DEMO=ON -DJBA_WITH_WEBGPU=ON \
  -DJBA_EMDAWNWEBGPU_DIR=<emdawnwebgpu_pkg>
cmake --build build/demo-wasm
```

The first fetches the keypoints and matcher modules and LightGlue's float16 weights
(Apache-2.0, 24 MB) into `third_party/match-web`, checked against pinned hashes.
`build/demo-wasm/demo/web` then holds the page, `demo.js`, `jba_demo.js`,
`jba_demo.wasm`, `jkp/`, `jmt/` and `samples/`, ready for any static server
(`python3 -m http.server -d build/demo-wasm/demo/web`). The documentation site includes
the folder when built with `JBA_DEMO_WEB=build/demo-wasm/demo/web`
(`docs-site/README.md`).

The `docs` workflow builds the same way, from poselib-gpu's `main`, and deploys the
site with the demo on every push to main (D148).

The sample images are ETH3D's (Schöps et al., CVPR 2017), CC BY-NC-SA 4.0
(non-commercial): the undistorted DSLR images of *delivery_area*, scaled to 1280 px and
re-encoded as JPEG; `samples.json` gives each one's focal length from ETH3D's calibration,
scaled the same way.

## The live capture

The visitor's camera, captured and reconstructed as it goes (D157): `web/capture.html`,
in the same folder as the live demo. Per camera frame, on the page's thread: ALIKED
keypoints (just-keypoints), the matcher's features (FlexGlue's dense features, or
LightGlue's of the keypoints), the matches to the last keyframe at the live depth
(FlexGlue's layer-0 head), checked against a relative pose on the CPU
(`demo_pair_inliers`), and Like Recon's keep rules (`web/gates.js`: overlap, support,
redundancy, confident keypoints, blur against a decaying sharp level, the best frame of
the band, re-localization against held keyframes). A keyframe is matched to the three
before it at the online depth (FlexGlue at 3 layers, the strongest 1,024 matches) and
added to a `Session` in a worker (`demo_session_new`, `demo_session_add`), whose map the
page draws. With FlexGlue, a keyframe the map can't place is matched to four more chosen
by appearance (the shortlist) and tried again (`demo_session_retry`). Finish matches the
shortlist over every keyframe (LightGlue: each with the five before it) and runs
`demo_session_finish` (`Session::finish`, `Mapper::run_from` from the live map), with
progress and Stop; the points are coloured from the keyframes and can be downloaded as
PLY. The session takes Like Recon's settings: a window of 8, a new piece after 10
keyframes in a row not placed, 15 verified matches for an online pair, the map starting
from the widest pair it can.

FlexGlue's weights come from the visitor: `flexglue.pth` from FlexGlue's release, checked
against its SHA-256, converted to float16 by just-matching's module (`jmt.convert`) in a
worker that ends afterwards, and kept in the origin's private file system
(`flexglue-fp16.jmtw`, 284 MB). Phones default to LightGlue, 1,024 keypoints and a
960 px working image. URL parameters: `matcher=flexglue|lightglue`, `keypoints=1024|2048`,
`side=<px>`, `debug=1`, and for testing without a camera `video=<url>` (a recorded video
instead, `rate=` its playback rate).

Tried in Chrome 156 on an M5 Pro with house-04's 41 keyframes made into a video
(ffmpeg's motion interpolation, 12 frames a second, played at half speed): FlexGlue kept
27 keyframes, 26 placed live, and the finish registered all 27 (4,040 points, 1.32 px);
LightGlue kept 23 and registered 12. FlexGlue's live frames took 150–850 ms (the
interpolated frames are not a camera's; try the page with a real one).

## What it checks

A scene passes when every image is registered, the median rotation error is below
0.5 degrees after the best common rotation, the median camera-centre error is below
1 % of the scene's extent after the similarity that best aligns the centres, and the
mean reprojection error is below 1.5 px or twice the keypoint noise. RANSAC
thresholds follow the noise (`thresholds_for_noise`): poselib-gpu's defaults up to
half a pixel, larger above.

Measured on 2026-11-16 on the owner's M5 Pro: the three scenes pass at seeds 0 to 8
natively, on the CPU and with Metal for both libraries. With the default backends they pass up to 2 px
of noise with half the matches wrong and 30 unrelated pairs, except the orbit at half the matches
wrong, where no pair keeps the 100 inliers the first-pair rule asks for.
