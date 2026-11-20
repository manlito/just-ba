"""Keypoints of images with just-keypoints (ALIKED, aliked-n16), in a process of their own.

    python extract.py --backend cpu --top-k 2048 --out features.npz image...
    python extract.py --available cuda      # prints 1 or 0

The end-to-end test runs this as a subprocess because just_keypoints and just_matching
(0.1.1 and 0.1.4, the releases of November 2026) cannot be imported into one Python
process: both bind a C++
type named Features in an anonymous namespace, and nanobind's registry, shared by
modules built with the same nanobind, takes them for one type ("type 'Features' was
already registered"), so the second import fails.
"""

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--available")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--out")
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()

    import just_keypoints as jkp
    if a.available:
        print(int(jkp.backend_available(a.available)))
        return
    from PIL import Image

    extractor = jkp.Extractor(backend=a.backend, top_k=a.top_k)
    out = {}
    for i, path in enumerate(a.images):
        image = np.asarray(Image.open(path).convert("RGB"))
        f = extractor.extract(image)
        out[f"xy{i}"], out[f"desc{i}"] = f.xy, f.desc
        out[f"size{i}"] = np.array([image.shape[1], image.shape[0]])
    np.savez(a.out, **out)


if __name__ == "__main__":
    sys.exit(main())
