"""Finds the NVIDIA libraries the CUDA backend links, before ``_core`` is loaded.

The CUDA packages (just-ba-cu12, just-ba-cu13) link the CUDA runtime statically
and cuDSS as a shared library; cuDSS needs cuBLAS and NVRTC. None of them is in
the wheel: they come from NVIDIA's own pip packages (nvidia-cudss-cuXX, ...),
which install under ``nvidia/`` in a site-packages directory. They are loaded
here by full path, so the dynamic loader finds them by name when ``_core`` is
loaded (Linux), or their directories are added to the DLL search path (Windows).

They are looked for in the files of the installed ``nvidia-*`` distributions, then
under ``nvidia/`` of every site-packages directory (system, user, and those on
``sys.path``), then in ``nvidia.__path__``; for each library the copy of this
package's CUDA major wins, then the first found. ``import nvidia`` alone is not
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


def _is_library(name: str) -> bool:
    return name.endswith((".dll", ".so")) or ".so." in name


def _distribution_files() -> list[Path]:
    """Library files of the installed nvidia-* distributions, from their RECORD."""
    from importlib import metadata

    found = []
    for dist in metadata.distributions():
        try:
            if not (dist.name or "").lower().startswith("nvidia-"):
                continue
            for f in dist.files or []:
                if _is_library(f.name):
                    found.append(Path(dist.locate_file(f)))
        except Exception:
            continue  # a broken distribution's metadata is skipped
    return found


def _nvidia_roots() -> list[Path]:
    """The ``nvidia`` directories of every site-packages directory, then nvidia.__path__."""
    import site

    dirs = []
    try:
        dirs += site.getsitepackages()
    except Exception:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass
    # Only sys.path entries that are package directories, not the script's folder or
    # the working directory, where a folder named nvidia may be anything.
    dirs += [p for p in sys.path if p and Path(p).name in ("site-packages", "dist-packages")]
    roots = [Path(p) / "nvidia" for p in dirs]
    try:
        import nvidia  # usually a namespace package; may be a regular one

        roots += [Path(p) for p in getattr(nvidia, "__path__", [])]
    except Exception:
        pass
    out, seen = [], set()
    for r in roots:
        try:
            if not r.is_dir():
                continue
            key = r.resolve()
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# Where NVIDIA's wheels put their libraries under nvidia/: <pkg>/lib (Linux),
# <pkg>/bin (Windows), one level deeper for some (bin/x86_64).
_LAYOUT = ("*/lib/*", "*/bin/*", "*/lib/*/*", "*/bin/*/*")


def _candidates() -> list[Path]:
    """Library files in search order: distribution files, then the nvidia directories."""
    files = _distribution_files()
    for root in _nvidia_roots():
        for pattern in _LAYOUT:
            try:
                files.extend(p for p in sorted(root.glob(pattern)) if _is_library(p.name))
            except OSError:
                pass
    out, seen = [], set()
    for p in files:
        try:
            if not p.is_file():
                continue
            key = p.resolve()
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _find(files: list[Path], pattern: str, cuda: str | None = None) -> list[Path]:
    """Files matching pattern: those of this package's CUDA major first (libcublas.so.12
    or a cu12 directory for cu12), then the shortest name (libcublas.so.13 before
    libcublas.so.13.1.0.3, the same file), then the search order."""
    found = [(k, p) for k, p in enumerate(files) if fnmatch.fnmatch(p.name, pattern)]

    def other_major(p: Path) -> bool:
        if not cuda:
            return False
        if f".so.{cuda}" in p.name or f"64_{cuda}" in p.name:
            return False
        # cuDSS's file names carry its own version: the CUDA major is in the path.
        return f"cu{cuda}" not in {part.lower() for part in p.parts}

    return [p for _, p in sorted(found, key=lambda kp: (other_major(kp[1]), len(kp[1].name), kp[0]))]


def preload(variant: str = "") -> None:
    """Never raises: what is not found here is left to the loader, and to the error at import."""
    try:
        _preload(variant[2:] if variant.startswith("cu") else None)
    except Exception:
        pass


def _preload(cuda: str | None) -> None:
    files = _candidates()
    if sys.platform == "win32":
        dirs = {str(p.parent) for pattern in _WINDOWS for p in _find(files, pattern, cuda)[:1]}
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
