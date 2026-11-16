# References and acknowledgements

just-ba builds on published methods and on other people's software, and it is
checked against them. This page credits that work: what the code follows, the data
the tests run on, and the software the library and this site are made with.

## Methods the code follows

### Ceres Solver

Sameer Agarwal, Keir Mierle and the Ceres Solver Team. *Ceres Solver*.
<https://github.com/ceres-solver/ceres-solver>, <http://ceres-solver.org>. BSD
3-Clause licence.

- The trust-region loop follows Ceres's `TrustRegionMinimizer` and
  `LevenbergMarquardtStrategy`: the Jacobian column scaling, the damping, the radius
  updates and the tolerance tests. Iteration histories therefore compare directly.
- The robust losses have the same definitions and scale parameter as Ceres's losses
  of the same names. The correction of residuals and Jacobians under a robust loss
  follows Ceres's corrector.
- The iterative linear solver is the reduced camera system solved by conjugate
  gradients with camera-block preconditioning, as Ceres's `ITERATIVE_SCHUR` with
  `SCHUR_JACOBI`.
- Ceres is the correctness reference. Residuals, Jacobians, whole solves and
  covariances are checked against Ceres 2.2.0. It is linked only into the tests,
  never into the library or the Python packages.

### COLMAP

Johannes L. Schönberger and Jan-Michael Frahm. "Structure-from-Motion Revisited."
CVPR 2016. <https://colmap.github.io>. BSD 3-Clause licence.

- The camera models take COLMAP's names, ids and parameter order (see
  [Camera models](_generated/cameras.md)).
- Undistortion iterates as COLMAP's `IterativeUndistortion` does.
- When nothing else fixes the scale, the gauge holds one translation coordinate, as
  COLMAP does.
- `export_colmap` writes COLMAP's text model format.
- COLMAP 4.2.1, and its Caspar GPU bundle adjustment backend, are references in the
  [benchmarks](_generated/benchmarks.md).

### Bundle adjustment

Bill Triggs, Philip McLauchlan, Richard Hartley and Andrew Fitzgibbon. "Bundle
Adjustment — A Modern Synthesis." *Vision Algorithms: Theory and Practice*, LNCS
1883, Springer, 2000. The correction for robust losses above comes from this paper,
by way of Ceres.

### PyPose and `bae`

Chen Wang et al. "PyPose: A Library for Robot Learning with Physics-based
Optimization." CVPR 2023. <https://pypose.org>. PyPose's bundle adjuster `bae`
showed the direction this library takes: none of its code is used. `bae` 0.2 is
one of the comparisons in the benchmarks.

## Data

Sameer Agarwal, Noah Snavely, Steven M. Seitz and Richard Szeliski. "Bundle
Adjustment in the Large." ECCV 2010.
<https://grail.cs.washington.edu/projects/bal/>. The tests and benchmarks download
BAL problems from there. They are not redistributed.

## Software the library is built with

| Project | Used for | Licence |
|---|---|---|
| [wrenfold](https://wrenfold.org) | The residuals and their Jacobians are written symbolically and generated as C++, Metal, CUDA and WGSL code, which ships in the library. | MIT |
| [nanobind](https://github.com/wjakob/nanobind) | The Python bindings. | BSD 3-Clause |
| [NumPy](https://numpy.org) | Arrays in the Python API. | BSD 3-Clause |
| NVIDIA CUDA, cuDSS, cuBLAS and NVRTC | The CUDA backend. The CUDA packages install them from NVIDIA's own pip packages; they are not redistributed. | NVIDIA's licences |
| Apple Metal and Accelerate | The Metal backend, and the sparse factorization on Apple. | Part of macOS |
| [Dawn](https://github.com/google/dawn) and emdawnwebgpu | The WebGPU backend, native and in the browser. | BSD 3-Clause |
| [Emscripten](https://emscripten.org) | The WebAssembly build. | MIT and University of Illinois/NCSA |

## Tools for the tests, the packages and this site

- Ceres Solver 2.2.0 and [Eigen](https://eigen.tuxfamily.org) 3.4.0 (MPL 2.0): test
  references only.
- [scikit-build-core](https://github.com/scikit-build/scikit-build-core) and
  [cibuildwheel](https://github.com/pypa/cibuildwheel): the wheels.
- [Sphinx](https://www.sphinx-doc.org) (BSD 2-Clause),
  [MyST-Parser](https://github.com/executablebooks/MyST-Parser) (MIT),
  [SymPy](https://sympy.org) (BSD) for the formulas, and
  [MathJax](https://www.mathjax.org) (Apache-2.0) to display them: this site. Its
  theme is the project's own.
- The Inter and JetBrains Mono typefaces (SIL Open Font License), served by Google
  Fonts.

## Projects studied

These were read while deciding the library's language and design: MegBA, Graphite,
GTSAM's CUDA solver, Sprux and kornia-rs.
