"""Turns pyproject.toml into one of the CUDA packages (D114), in place.

    python packaging/variant.py cu12 --root /opt/jba-cuda --platform linux
    python packaging/variant.py cu13 --root C:/jba-cuda --platform windows

Run by the release workflow only, before cibuildwheel; never commit the result.
``--root`` is where packaging/fetch_cuda.py put the CUDA toolkit (``cuda/``) and
cuDSS (``cudss/``), as the build sees it (inside the manylinux container on Linux).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# GPU architectures per package (document 08): real code for each, plus PTX for the
# newest so later GPUs can still run it. CUDA 12 keeps Pascal and Volta, which CUDA
# 13 dropped (D113); below 7.5 nothing is tested (no such GPU at hand), and the
# test kernel in available() turns a mismatch into "unavailable" (D104).
ARCHITECTURES = {
    "cu12": "60-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120",
    "cu13": "75-real;80-real;86-real;89-real;90-real;100-real;120",
}
# NVIDIA's pip package for cuDSS; it depends on cuBLAS and NVRTC itself. 1.x: the
# soname the build links (libcudss.so.1), same as the CI's cuDSS 1.0.0.35.
DEPENDENCY = {"cu12": "nvidia-cudss-cu12==1.*", "cu13": "nvidia-cudss-cu13==1.*"}


def sub_once(text: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"variant.py: pattern not found in pyproject.toml: {pattern}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(ARCHITECTURES))
    ap.add_argument("--root", required=True)
    ap.add_argument("--platform", choices=["linux", "windows"], required=True)
    args = ap.parse_args()
    v, root = args.variant, args.root.rstrip("/\\")
    nvcc = f"{root}/cuda/bin/nvcc" + (".exe" if args.platform == "windows" else "")

    text = PYPROJECT.read_text()
    text = sub_once(text, r'^name = "just-ba"$', f'name = "just-ba-{v}"')
    text = sub_once(text, r'^(description = ".*)"$', rf'\1 (CUDA {v[2:]} package)"')
    text = sub_once(text, r'^dependencies = \["numpy>=1.26"\]$',
                    f'dependencies = ["numpy>=1.26", "{DEPENDENCY[v]}"]')
    defines = "\n".join([
        f'JBA_PACKAGE_VARIANT = "{v}"',
        'JBA_WITH_CUDA = "ON"',
        'JBA_WITH_CUDSS = "ON"',
        f'CMAKE_CUDA_ARCHITECTURES = "{ARCHITECTURES[v]}"',
        f'CMAKE_CUDA_COMPILER = "{nvcc}"',
        f'CUDAToolkit_ROOT = "{root}/cuda"',
        f'CMAKE_PREFIX_PATH = "{root}/cudss"',
    ])
    text = sub_once(text, r'^JBA_PACKAGE_VARIANT = "cpu"$', defines)
    PYPROJECT.write_text(text)
    print(f"pyproject.toml is now just-ba-{v} ({args.platform}, CUDA under {root})")


if __name__ == "__main__":
    main()
