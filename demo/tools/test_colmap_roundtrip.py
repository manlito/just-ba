#!/usr/bin/env python3
"""The COLMAP path end to end, without COLMAP: generated scene -> capture file ->
COLMAP database and text model (colmap_fixture.py) -> capture file (colmap_capture.py)
-> jba_demo run --check against the scene's truth.

    test_colmap_roundtrip.py path/to/jba_demo work_dir

Checks on the way that the second capture file holds what the first did, image by image
(names, cameras, keypoints to float32, matches, the reference), whatever the ids.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jbc  # noqa: E402


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="")
        sys.exit(f"failed ({r.returncode}): {' '.join(cmd)}")
    return r.stdout


def by_name(path):
    s = jbc.read(path)
    names = bytes(s["image.names"][2]).decode().split("\n")
    n = len(names)
    np_ = s["intrinsics"][1][1]
    intr = s["intrinsics"][2]
    block = s["image.intrinsics"][2] if "image.intrinsics" in s else [0] * n
    kb, kxy = s["keypoint.begin"][2], s["keypoint.xy"][2]
    ref_intr = s["reference.intrinsics"][2] if "reference.intrinsics" in s else intr
    images = {}
    for i, name in enumerate(names):
        images[name] = {
            "camera": tuple(intr[np_ * block[i]:np_ * block[i] + np_]),
            "ref_camera": tuple(ref_intr[np_ * block[i]:np_ * block[i] + np_]),
            "keypoints": list(kxy[2 * kb[i]:2 * kb[i + 1]]),
            "registered": s["reference.registered"][2][i],
            "pose": tuple(s["reference.pose"][2][7 * i:7 * i + 7]),
        }
    pairs = {}
    pi, pb, pm = s["pair.images"][2], s["pair.begin"][2], s["pair.match"][2]
    for r in range(len(pi) // 2):
        a, b = names[pi[2 * r]], names[pi[2 * r + 1]]
        m = [(pm[2 * k], pm[2 * k + 1]) for k in range(pb[r], pb[r + 1])]
        if a > b:
            a, b, m = b, a, [(y, x) for x, y in m]
        pairs[(a, b)] = sorted(m)
    obs = set()
    oi, op, ok = (s[f"reference.obs.{t}"][2] for t in ("image", "point", "keypoint"))
    xyz = s["reference.point.xyz"][2]
    for k in range(len(oi)):
        obs.add((names[oi[k]], ok[k], tuple(round(v, 9) for v in xyz[3 * op[k]:3 * op[k] + 3])))
    return images, pairs, obs


def f32(v):
    import struct
    return struct.unpack("f", struct.pack("f", v))[0]


def same(first, second):
    i1, p1, o1 = by_name(first)
    i2, p2, o2 = by_name(second)
    assert i1.keys() == i2.keys(), "image names differ"
    for name, a in i1.items():
        b = i2[name]
        assert a["camera"] == b["camera"], f"{name}: camera differs"
        assert a["ref_camera"] == b["ref_camera"], f"{name}: reference camera differs"
        assert [f32(v) for v in a["keypoints"]] == b["keypoints"], f"{name}: keypoints differ"
        assert a["registered"] == b["registered"] and a["pose"] == b["pose"], f"{name}: reference pose differs"
    assert p1 == p2, "matches differ"
    assert o1 == o2, "reference observations differ"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    demo, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    run([demo, "synth", "--write", work])
    for scene, extra in (("orbit", ["--camera-per-image"]), ("wall", []), ("grid", [])):
        first = os.path.join(work, f"{scene}.jbc")
        colmap = os.path.join(work, f"{scene}-colmap")
        second = os.path.join(work, f"{scene}-from-colmap.jbc")
        run([sys.executable, os.path.join(HERE, "colmap_fixture.py"), first, colmap] + extra)
        run([sys.executable, os.path.join(HERE, "colmap_capture.py"), os.path.join(colmap, "database.db"),
             "--model", os.path.join(colmap, "sparse"), "--min-matches", "0", "-o", second])
        same(first, second)
        print(f"{scene}: the capture file survives the COLMAP database and model")
        run([demo, "run", second, "--check", "--thresholds", "demo"])
    print("PASS")


if __name__ == "__main__":
    main()
