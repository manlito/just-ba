#!/usr/bin/env python3
"""Downloads LightGlue's weights for ALIKED, which the end-to-end test matches with, and
checks their hash.

    python3 tests/pipeline/fetch_weights.py [--dest tests/data/weights]

The weights are the Computer Vision and Geometry Lab's (Apache-2.0), from LightGlue's
v0.1_arxiv release; just-matching reads the checkpoint as released. Not checked in.
Uses only the Python standard library.
"""

import argparse
import hashlib
import pathlib
import sys
import urllib.request

URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth"
SHA256 = "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503"
DEFAULT_DEST = pathlib.Path(__file__).resolve().parent.parent / "data" / "weights"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=pathlib.Path, default=DEFAULT_DEST)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    path = a.dest / "aliked_lightglue.pth"
    if path.exists() and sha256(path) == SHA256:
        print(f"{path}: present")
        return 0
    tmp = path.with_suffix(".part")
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    got = sha256(tmp)
    if got != SHA256:
        tmp.unlink()
        print(f"{URL}: sha256 {got}, expected {SHA256}", file=sys.stderr)
        return 1
    tmp.replace(path)
    print(f"{path}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
