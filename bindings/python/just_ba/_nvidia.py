"""Finds the NVIDIA libraries the CUDA backend links, before ``_core`` is loaded.

The CUDA packages (just-ba-cu12, just-ba-cu13) link the CUDA runtime statically
and cuDSS as a shared library; cuDSS needs cuBLAS and NVRTC. None of them is in
the wheel: they come from NVIDIA's own pip packages (nvidia-cudss-cuXX, ...),
which install under ``nvidia/`` in a site-packages directory. They are loaded
here by full path, so the dynamic loader finds them by name when ``_core`` is
loaded (Linux), or their directories are added to the DLL search path (Windows).

They are looked for in the files of the installed ``nvidia-*`` distributions, then
under ``nvidia/`` of every site-packages directory (system and user) and every
entry of ``sys.path``, then in ``nvidia.__path__``. ``import nvidia`` alone is not
enough: another package (PyTorch's NVIDIA wheels in some images) may install a
regular ``nvidia/__init__.py``, which hides the copies in the other directories.

A CUDA toolkit and cuDSS installed elsewhere also work: on Linux through the
loader's usual search (LD_LIBRARY_PATH), on Windows through the directories in
JUST_BA_DLL_PATH (separated by ``;``).
"""

from __future__ import annotations

import ctypes
import fnmatch
import os
import sys
from pathlib import Path

# Load order: each library's dependencies first.
_LINUX = ("libcublasLt.so*", "libcublas.so*", "libnvrtc.so*", "libcudss.so*")
_WINDOWS = ("cublasLt64_*.dll", "cublas64_*.dll", "nvrtc64_*.dll", "cudss64_*.dll")

_handles: list = []  # keeps the loaded libraries and DLL directories alive


def _distribution_files() -> list[Path]:
    """Library files of the installed nvidia-* distributions, from their metadata."""
    from importlib import metadata

    found = []
    try:
        dists = list(metadata.distributions())
    except Exception:
        return found
    for dist in dists:
        try:
            name = (dist.metadata["Name"] or "").lower()
        except Exception:
            continue
        if not name.startswith("nvidia-"):
            continue
        for f in dist.files or []:
            if f.name.endswith((".dll", ".so")) or ".so." in f.name:
                try:
                    found.append(Path(dist.locate_file(f)))
                except Exception:
                    pass
    return found


def _nvidia_roots() -> list[Path]:
    """Every ``nvidia`` directory a pip wheel may have installed into."""
    import site

    candidates = []
    try:
        candidates += site.getsitepackages()
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    candidates += sys.path
    roots = [Path(p) / "nvidia" for p in candidates if p]
    try:
        import nvidia  # usually a namespace package; may be a regular one

        roots += [Path(p) for p in getattr(nvidia, "__path__", [])]
    except ImportError:
        pass
    out, seen = [], set()
    for r in roots:
        try:
            key = r.resolve()
        except OSError:
            continue
        if key not in seen and r.is_dir():
            seen.add(key)
            out.append(r)
    return out


def _candidates() -> list[Path]:
    files = [p for p in _distribution_files() if p.is_file()]
    for root in _nvidia_roots():
        files.extend(p for p in root.rglob("*") if p.is_file() and (".so" in p.name or p.name.endswith(".dll")))
    out, seen = [], set()
    for p in files:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _find(files: list[Path], pattern: str, cuda: str | None = None) -> list[Path]:
    found = [p for p in files if fnmatch.fnmatch(p.name, pattern)]
    # The CUDA major of this package first (libcublas.so.12 for cu12), then the
    # shortest name: libcublas.so.13 before libcublas.so.13.1.0.3 (the same file).
    def other_major(p: Path) -> bool:
        if not cuda or "cudss" in p.name:
            return False
        return not (f".so.{cuda}" in p.name or f"64_{cuda}" in p.name)

    return sorted(found, key=lambda p: (other_major(p), len(p.name), str(p)))


def preload(variant: str = "") -> None:
    cuda = variant[2:] if variant.startswith("cu") else None
    files = _candidates()
    if sys.platform == "win32":
        dirs = {str(p.parent) for pattern in _WINDOWS for p in _find(files, pattern, cuda)}
        dirs.update(d for d in os.environ.get("JUST_BA_DLL_PATH", "").split(";") if d)
        for d in sorted(dirs):
            if os.path.isdir(d):
                _handles.append(os.add_dll_directory(d))
        return
    for pattern in _LINUX:
        for path in _find(files, pattern, cuda)[:1]:
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
