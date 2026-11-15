# Backends

| Backend | Where | Precision | Linear solvers |
|---|---|---|---|
| `Backend.CPU` | everywhere | double | ITERATIVE_SCHUR |
| `Backend.METAL` | macOS on Apple Silicon (`just-ba`) | single on the GPU | ITERATIVE_SCHUR, SPARSE_SCHUR (Accelerate) |
| `Backend.CUDA` | NVIDIA GPUs (`just-ba-cu12`, `just-ba-cu13`) | single on the GPU | ITERATIVE_SCHUR, SPARSE_SCHUR (cuDSS) |

Choose one with `SolverOptions.backend` (or `RefineOptions.solver.backend`).
`backend_available(kind)` says whether the build has the backend and the machine
can run it; `compiled_backends()` lists what the build has. Asking for an
unavailable backend raises an error rather than falling back silently.

Both GPU backends eliminate the points by the Schur complement and solve the reduced
camera system either by preconditioned conjugate gradients (ITERATIVE_SCHUR) or by
forming it and factorizing it with a sparse direct solver (SPARSE_SCHUR).
`LinearSolver.AUTO`, the default, takes the backend's fastest for the problem:
SPARSE_SCHUR on the GPUs where it fits the device's memory, ITERATIVE_SCHUR
otherwise and on the CPU. `SolverSummary.linear_solver` reports which one ran.

The GPUs work in single precision, with the decisions of the trust region made in
double on the host; their results agree with the CPU's within single-precision
tolerance. The CPU backend is the reference and the fallback on machines without a
supported GPU. Results do not depend on the number of threads.
