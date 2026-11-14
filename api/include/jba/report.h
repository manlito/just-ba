#pragma once

#include <cstdint>
#include <vector>

#include "jba/problem.h"
#include "jba/solver.h"

namespace jba {

// Per-residual diagnostics at the problem's current parameters (document 01,
// outputs): cost by residual type, reprojection errors, and how much each prior was
// down-weighted by its robust loss.
struct ResidualReport {
  // 0.5 * sum rho, with the losses of `options`, by type; their sum is the
  // solver's cost.
  double reprojection_cost = 0;
  double relative_prior_cost = 0;
  double global_prior_cost = 0;

  // Per observation: reprojection error in pixels (not divided by sigma), and
  // whether the point is in front of the camera.
  std::vector<double> reprojection_error;
  std::vector<char> in_front;

  // Per prior: whitened residual norm (standard deviations) and robust weight
  // rho'(s), 1 for a prior fully trusted, near 0 for one the solve overruled.
  std::vector<double> relative_prior_norm, relative_prior_weight;
  std::vector<double> global_prior_norm, global_prior_weight;

  // Priors whose weight fell below the threshold given to evaluate(): what the
  // caller's cleanup let through (document 02).
  std::vector<std::int32_t> downweighted_relative, downweighted_global;
};

ResidualReport evaluate(const Problem& problem, const SolverOptions& options, double downweight_threshold = 0.1);

}  // namespace jba
