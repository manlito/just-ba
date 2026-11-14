#!/usr/bin/env python3
"""Downloads BAL problems used by the tests, checks their hashes, and decompresses them.

    python3 tests/data/fetch_bal.py [--dest tests/data/bal] [names...]

BAL data is not checked in. Source: https://grail.cs.washington.edu/projects/bal/
Uses only the Python standard library.
"""

import argparse
import bz2
import hashlib
import pathlib
import sys
import urllib.request

BASE = "https://grail.cs.washington.edu/projects/bal/data"

# name -> (path under BASE, sha256 of the .bz2 file)
PROBLEMS = {
    "problem-49-7776-pre": (
        "ladybug/problem-49-7776-pre.txt.bz2",
        "1ccb15701a92a8ad909d30860a0108cd3f2d7916c1ecf2851e59a6198b9de6b0",
    ),
}


def fetch(name: str, dest: pathlib.Path) -> None:
    rel, sha = PROBLEMS[name]
    out = dest / f"{name}.txt"
    if out.exists():
        print(f"{out} already present")
        return
    url = f"{BASE}/{rel}"
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"{name}: sha256 mismatch: expected {sha}, got {got}")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(bz2.decompress(data))
    tmp.replace(out)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(pathlib.Path(__file__).parent / "bal"))
    parser.add_argument("names", nargs="*", default=list(PROBLEMS))
    args = parser.parse_args()
    for name in args.names:
        if name not in PROBLEMS:
            raise SystemExit(f"unknown problem {name}; known: {', '.join(PROBLEMS)}")
        fetch(name, pathlib.Path(args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
