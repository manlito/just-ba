# just-ba

A minimal, high-performance bundle adjustment library in C++ with Python bindings,
for refining reconstructions that come with noisy camera poses from ARKit, ARCore
or another tracker. Early development: milestone 0 (build, generated residuals,
reference solver, file format). The plan is in [docs/](docs/README.md).

## Build

Needs CMake 3.24 or later and a C++23 compiler. On macOS the Metal backend is built
by default.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build --parallel
```

```bash
ctest --test-dir build --output-on-failure
```

## Reference solver (test-only)

Ceres 2.2.0 is built separately, with Eigen only, and never linked into the library.

```bash
cmake -DPREFIX=$PWD/build/ceres-2.2.0 -P third_party/ceres/build.cmake
```

```bash
python3 tests/data/fetch_bal.py
```

```bash
cmake -S . -B build -DJBA_WITH_CERES=ON -DCMAKE_PREFIX_PATH=$PWD/build/ceres-2.2.0
```

## Generated residuals

Residuals are written once in `residuals/symbolic/` and emitted for each target
into `residuals/generated/`, which is checked in. To regenerate:

```bash
uv venv --python 3.12 residuals/symbolic/.venv && uv pip install --python residuals/symbolic/.venv/bin/python -r residuals/symbolic/requirements.txt
```

```bash
residuals/symbolic/.venv/bin/python residuals/symbolic/generate.py
```

## Command line

```bash
build/cli/just-ba convert-bal problem.txt problem.jba
```

```bash
build/cli/just-ba info problem.jba
```

## Licence

Apache-2.0 (the licence file is added before the repository goes public).
