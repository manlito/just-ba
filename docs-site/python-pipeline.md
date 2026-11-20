# From photos in Python

The [live demo](demo.md)'s whole pipeline, from Python: photos in, cameras and points out.
Four packages, each on the CPU, on Apple GPUs (Metal) or on NVIDIA GPUs (CUDA):

| Step | Package | What it does |
|---|---|---|
| Keypoints | just-keypoints | ALIKED keypoints and descriptors, 2048 per image |
| Matches | just-matching | LightGlue on every pair of images |
| Poses | poselib-gpu | Each pair verified by relative pose; images registered by PnP |
| Bundle adjustment | just-ba | Cameras, points, focal length and distortion adjusted together |

Between them sits an incremental mapper of about 550 lines of Python and NumPy, below,
with the live demo's rules (COLMAP's thresholds): it builds the tracks, triangulates the
points, chooses the images to register and filters the model after each adjustment. The test suite runs this on the demo's 44 sample
images, ETH3D's *delivery_area*, on each backend (`tests/pipeline`).

## Install

```bash
pip install just-ba just-keypoints just-matching poselib-gpu pillow
```

On Linux or Windows with an NVIDIA GPU, install the `-cu12` or `-cu13` package of each
instead (`just-ba-cu13`, `just-keypoints-cu13`, ...). LightGlue's weights for ALIKED are
not in just-matching's package: download `aliked_lightglue.pth` from
[LightGlue's v0.1_arxiv release](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv)
(Apache-2.0).

## 1. Keypoints

just-keypoints and just-matching cannot yet be imported into one Python process (both
bind a type named `Features`, and the second import fails), so the keypoints are found
by a script of their own and saved:

```python
# keypoints.py photos/*.jpg
import sys
import numpy as np
from PIL import Image
import just_keypoints as jkp

backend = "cuda" if jkp.backend_available("cuda") else "metal" if jkp.backend_available("metal") else "cpu"
extractor = jkp.Extractor(backend=backend, top_k=2048)  # ALIKED (aliked-n16), weights built in
out = {}
for i, path in enumerate(sys.argv[1:]):
    image = np.asarray(Image.open(path).convert("RGB"))
    f = extractor.extract(image)
    out[f"xy{i}"], out[f"desc{i}"] = f.xy, f.desc
    out[f"size{i}"] = [image.shape[1], image.shape[0]]
np.savez("keypoints.npz", **out)
```

## 2. Matches and reconstruction

```python
# reconstruct.py, with mapper.py (below) next to it
import numpy as np
import just_matching as jmt
from mapper import Mapper

backend = "cpu"                       # or "metal", "cuda": every step on it
f = np.load("keypoints.npz")
n = len([k for k in f if k.startswith("xy")])
xy = [f[f"xy{i}"] for i in range(n)]
sizes = np.array([f[f"size{i}"] for i in range(n)])

m = jmt.Matcher("aliked_lightglue.pth", backend=backend)
features = [m.describe_keypoints(int(sizes[i, 0]), int(sizes[i, 1]), xy[i], f[f"desc{i}"]) for i in range(n)]
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        idx = np.asarray(m.match(features[i], features[j]).idx)  # (k, 2): keypoint of i, of j
        if len(idx) >= 15:
            pairs.append((i, j, idx))

focal = 702.83                        # pixels; the demo's samples are calibrated
r = Mapper(xy, sizes, focal, pairs, backend=backend).run()
print(f"{r.registered.sum()} of {n} images, {len(r.points)} points, "
      f"{np.median(r.reprojection_px):.2f} px median, focal {r.intrinsics[0]:.1f} px")
```

`r.poses` are world-to-camera poses (`qw qx qy qz tx ty tz`, OpenCV axes) of the
registered images, `r.problem` the final model as a {py:class}`just_ba.Problem` (the last adjustment's values, after its filter). It can be
written with {py:func}`just_ba.write_problem` or exported for COLMAP with
{py:func}`just_ba.export_colmap`.

With the focal length unknown, start from 1.2 times the larger side of the image. The
mapper refines it, but unlike the live demo it does not search for it first, so a guess
far off may register fewer images.

## Results

The 44 sample images at 1280 px, 2048 keypoints each: 946 pairs matched, 835 with 15
matches or more, 408 verified. The live demo's C++ mapper, given the same keypoints and
matches (`jba_demo run`), registers the same 40 images with 9,547 points (0.58 px mean
reprojection error).

| Backend | Machine | Registered | Points | Median error | Keypoints | Matching | Reconstruction |
|---|---|---|---|---|---|---|---|
| CPU | M5 Pro | 40 of 44 | 9,568 | 0.37 px | 10 s | 389 s | 4.0 s |
| Metal | M5 Pro | 40 of 44 | 9,561 | 0.37 px | 1.2 s | 25 s | 1.6 s |
| CUDA | RTX 2070 Max-Q | 40 of 44 | 9,557 | 0.37 px | 2.6 s | 28 s | 5.4 s |

The refined focal length is within 0.1 % of ETH3D's calibration on each. Most of the
reconstruction's time is its 15 bundle adjustments; matching every pair is most of the
whole, and on the CPU nearly all of it.

## The mapper

`tests/pipeline/mapper.py`, the demo's mapper (`demo/src/mapper.cpp`) in Python without
what the test does not need: one model, no focal length search, no position priors.

```{literalinclude} ../tests/pipeline/mapper.py
:language: python
```
