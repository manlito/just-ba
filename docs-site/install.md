# Install

| Package | Platforms | Backends |
|---|---|---|
| `pip install just-ba` | Linux x86_64, macOS arm64 (13 or later), Windows x86_64 | CPU; Metal on macOS |
| `pip install just-ba-cu12` | Linux x86_64, Windows x86_64 | CPU and CUDA 12 |
| `pip install just-ba-cu13` | Linux x86_64, Windows x86_64 | CPU and CUDA 13 |

Install one of them: all three provide the module `just_ba`. Python 3.12 or later;
one wheel per platform covers every later Python.

The CUDA packages need an NVIDIA driver for their CUDA version, and depend on
NVIDIA's own pip packages for cuDSS, cuBLAS and NVRTC, which pip installs with them.
`just-ba-cu12` also runs on the older GPU generations (Pascal, Volta) that CUDA 13
no longer supports.

Check what the installed package can do:

```python
import just_ba as jba
print(jba.__version__, jba.VARIANT, jba.compiled_backends())
print(jba.backend_available(jba.Backend.CUDA))  # this build has it and the machine can run it
```

## From source

A C++23 compiler, CMake 3.24 or later, and Python 3.12. From a checkout of the
repository:

```bash
pip install .
```

builds the CPU package (and Metal on macOS). The CUDA backend needs the CUDA toolkit
(12 or later) and, for its direct solver, cuDSS; the repository's README has the
CMake options.
