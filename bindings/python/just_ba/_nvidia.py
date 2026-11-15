"""Finds the NVIDIA libraries the CUDA backend links, before ``_core`` is loaded.

The CUDA packages (just-ba-cu12, just-ba-cu13) link the CUDA runtime statically
and cuDSS as a shared library; cuDSS needs cuBLAS and NVRTC. None of them is in
the wheel: they come from NVIDIA's own pip packages (nvidia-cudss-cuXX, ...),
which install under the ``nvidia`` namespace package. They are loaded here by
full path, so the dynamic loader finds them by name when ``_core`` is loaded
(Linux), or their directories are added to the DLL search path (Windows).

A CUDA toolkit and cuDSS installed elsewhere also work: on Linux through the
loader's usual search (LD_LIBRARY_PATH), on Windows through the directories in
JUST_BA_DLL_PATH (separated by ``;``).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Load order: each library's dependencies first.
_LINUX = ("libcublasLt.so*", "libcublas.so*", "libnvrtc.so*", "libcudss.so*")
_WINDOWS = ("cublasLt64_*.dll", "cublas64_*.dll", "nvrtc64_*.dll", "cudss64_*.dll")

_handles: list = []  # keeps the loaded libraries and DLL directories alive


def _nvidia_roots() -> list[Path]:
    try:
        import nvidia  # namespace package of NVIDIA's pip wheels
    except ImportError:
        return []
    return [Path(p) for p in getattr(nvidia, "__path__", [])]


def _find(roots: list[Path], pattern: str) -> list[Path]:
    found = []
    for root in roots:
        found.extend(p for p in root.rglob(pattern) if p.is_file())
    # Shortest name first: libcublas.so.13 before libcublas.so.13.1.0.3 (the same file).
    return sorted(found, key=lambda p: (len(p.name), str(p)))


def preload() -> None:
    roots = _nvidia_roots()
    if sys.platform == "win32":
        dirs = {str(p.parent) for pattern in _WINDOWS for p in _find(roots, pattern)}
        dirs.update(d for d in os.environ.get("JUST_BA_DLL_PATH", "").split(";") if d)
        for d in sorted(dirs):
            if os.path.isdir(d):
                _handles.append(os.add_dll_directory(d))
        return
    for pattern in _LINUX:
        for path in _find(roots, pattern)[:1]:
            try:
                _handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
            except OSError:
                pass  # left to the loader's own search, and to the error at import


def import_error_hint(variant: str) -> str:
    cuda = variant[2:] if variant.startswith("cu") else "13"
    return (
        f"the CUDA backend of just-ba-{variant} needs NVIDIA's libraries: "
        f"pip install nvidia-cudss-cu{cuda} nvidia-cublas-cu{cuda} nvidia-cuda-nvrtc-cu{cuda} "
        "(installed with the package unless --no-deps was used), and an NVIDIA driver"
    )
