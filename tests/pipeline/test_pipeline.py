"""The live demo's whole pipeline from Python, on the live demo's sample set.

The images of demo/web/samples (ETH3D's delivery_area, 1280 px, focal lengths from
ETH3D's calibration) go through the same steps as in the browser, with the Python
packages in place of the WebAssembly modules:

  keypoints     just-keypoints, ALIKED (aliked-n16), 2048 per image
  matches       just-matching, LightGlue for ALIKED, every pair; pairs with 15 matches
                or more go on (the live demo's MIN_MATCHES)
  reconstruction  mapper.py: poselib-gpu verifies the pairs and registers the images,
                just-ba builds the tracks and adjusts (the live demo's mapper rules)

Once on the CPU, and once on each of Metal and CUDA where all four packages have it:
every step on that backend.

Needs, besides just_ba: just-keypoints, just-matching and poselib-gpu (pip, with Metal
on macOS, or their -cu12 / -cu13 packages for CUDA), Pillow, and LightGlue's weights for ALIKED
(tests/pipeline/fetch_weights.py, or JBA_LIGHTGLUE_WEIGHTS naming the file). Skipped
without them, and on Windows. JBA_PIPELINE_IMAGES=n keeps the first n images (all 44
by default); JBA_PIPELINE_REQUIRE=cpu,cuda fails instead of skipping those backends
(CI requires the CPU).

The checks are loose, as for the real captures: they catch a pipeline that breaks.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import just_ba as jba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SAMPLES = os.path.join(ROOT, "demo", "web", "samples")
WEIGHTS = os.environ.get("JBA_LIGHTGLUE_WEIGHTS") or os.path.join(ROOT, "tests", "data", "weights",
                                                                  "aliked_lightglue.pth")
TOP_K = 2048        # the live demo's default
MIN_MATCHES = 15    # the live demo's MIN_MATCHES

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="not run on Windows")


def missing():
    for module in ("just_keypoints", "just_matching", "poselib_gpu", "PIL"):
        if importlib.util.find_spec(module) is None:
            return f"{module} is not installed"
    if not os.path.isfile(WEIGHTS):
        return f"no LightGlue weights at {WEIGHTS} (tests/pipeline/fetch_weights.py)"
    return None


def extract(args):
    """just-keypoints in its own process (extract.py says why)."""
    return subprocess.run([sys.executable, os.path.join(HERE, "extract.py"), *args],
                          check=True, capture_output=True, text=True).stdout


def available(backend):
    import just_matching as jmt
    import poselib_gpu as pg
    sys.path.insert(0, HERE)
    from mapper import BACKENDS
    return (jba.backend_available(BACKENDS[backend]) and jmt.backend_available(backend) and
            pg.backend_available(backend) and extract(["--available", backend]).strip() == "1")


@pytest.mark.parametrize("backend", ["cpu", "metal", "cuda"])
def test_demo_pipeline(backend, tmp_path):
    required = backend in os.environ.get("JBA_PIPELINE_REQUIRE", "").split(",")
    reason = missing()
    if not reason and not available(backend):
        reason = f"no {backend} backend in every package here"
    if reason:
        (pytest.fail if required else pytest.skip)(reason)
    import just_matching as jmt

    sys.path.insert(0, HERE)
    from mapper import Mapper

    samples = json.load(open(os.path.join(SAMPLES, "samples.json")))["images"]
    if os.environ.get("JBA_PIPELINE_IMAGES"):
        samples = samples[:int(os.environ["JBA_PIPELINE_IMAGES"])]
    n = len(samples)
    focal = samples[0]["focal"]
    assert all(s["focal"] == focal for s in samples)

    # Keypoints.
    t0 = time.perf_counter()
    out = tmp_path / "features.npz"
    extract(["--backend", backend, "--top-k", str(TOP_K), "--out", str(out),
             *[os.path.join(SAMPLES, s["file"]) for s in samples]])
    f = np.load(out)
    xy = [f[f"xy{i}"] for i in range(n)]
    desc = [f[f"desc{i}"] for i in range(n)]
    sizes = np.array([f[f"size{i}"] for i in range(n)])
    t_kp = time.perf_counter() - t0

    # Matches: every pair, as the live demo.
    t0 = time.perf_counter()
    m = jmt.Matcher(WEIGHTS, backend=backend)
    feats = [m.describe_keypoints(int(sizes[i, 0]), int(sizes[i, 1]), xy[i], desc[i]) for i in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = np.asarray(m.match(feats[i], feats[j]).idx)
            if len(idx) >= MIN_MATCHES:
                pairs.append((i, j, idx))
    t_match = time.perf_counter() - t0

    # Reconstruction.
    r = Mapper(xy, sizes, focal, pairs, backend=backend).run()

    registered = int(r.registered.sum())
    err = r.reprojection_px
    s = r.seconds
    print(f"\n[{backend}] {n} images, {sum(len(k) for k in xy)} keypoints in {t_kp:.1f} s; "
          f"{len(pairs)} of {n * (n - 1) // 2} pairs with {MIN_MATCHES}+ matches in {t_match:.1f} s; "
          f"{r.pairs_kept} pairs verified, {r.tracks} tracks, first pair {r.first_pair}\n"
          f"[{backend}] {registered}/{n} registered, {len(r.points)} points, {len(err)} observations, "
          f"reprojection mean {err.mean():.2f} px, median {np.median(err):.2f} px; "
          f"f {r.intrinsics[0]:.1f} px (calibrated {focal}), k {r.intrinsics[3]:.4f}; "
          f"{r.adjustments} adjustments; {s['total']:.1f} s (verify {s['verify']:.1f}, pnp {s['pnp']:.1f}, "
          f"triangulate {s['triangulate']:.1f}, adjust {s['adjust']:.1f})")

    assert r.problem.validate() == ""
    assert registered >= 0.9 * n
    assert len(r.points) >= 100 * registered
    assert np.median(err) < 1.0 and err.mean() < 1.5
    # The calibrated focal length holds; the images are undistorted.
    assert abs(r.intrinsics[0] / focal - 1) < 0.02
    assert abs(r.intrinsics[3]) < 0.02
    # Every registered camera sees the model in front of it.
    p = r.problem
    R = jba.matrix_from_quat(p.poses[:, :4])
    z = np.einsum("kj,kj->k", R[p.obs_image, 2], p.points[p.obs_point]) + p.poses[p.obs_image, 6]
    assert (z > 0).all()
