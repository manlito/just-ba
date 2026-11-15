#pragma once

#include <chrono>

#include "jba/backend.h"
#include "jba/solver.h"

namespace jba {

// The Levenberg-Marquardt trust-region loop, on the host, identical for every
// backend (document 03). It follows Ceres's TrustRegionMinimizer and
// LevenbergMarquardtStrategy, so iteration histories compare directly:
// - Jacobian columns scaled by 1 / (1 + norm), computed at the first iteration;
// - damping D^2 = clamp(diag(J^T J), 1e-6, 1e32) / radius;
// - on success radius /= max(1/3, 1 - (2 rho - 1)^3); on failure radius /= v, v *= 2;
// - the parameter and function tolerances are tested on the candidate step before
//   it is accepted, and stop the solve without taking it.
// Fills everything in `summary` except the gauge, the costs by type and the linear
// solver. Times are measured from `start`, the moment the solver was called (document
// 05: set-up included); setup_seconds is the time to this function's entry.
void run_trust_region(Backend& backend, const SolverOptions& options, const IterationCallback& callback,
                      SolverSummary& summary, std::chrono::steady_clock::time_point start);

}  // namespace jba
