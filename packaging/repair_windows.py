"""Repairs a Windows wheel of a CUDA package with delvewheel, leaving NVIDIA's DLLs
out (they come from NVIDIA's pip packages, D115):

    python packaging/repair_windows.py <wheel> <dest_dir> <root>

``<root>`` is fetch_cuda.py's destination; every DLL under it is excluded by name.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    wheel, dest, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    dlls = sorted({p.name for p in root.rglob("*.dll")})
    if not dlls:
        sys.exit(f"repair_windows.py: no DLLs under {root}")
    cmd = [sys.executable, "-m", "delvewheel", "repair", "-v", "-w", dest, "--no-dll", ";".join(dlls), wheel]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
