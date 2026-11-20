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
  double gravity_cost = 0;
  double line_cost = 0;
  double depth_cost = 0;  // depth readings and the priors on free corrections
  // Priors on free intrinsics (D161). Kept apart from reprojection_cost, which stays
  // the pixels' own; the solver's reprojection entry (SolverSummary::*_cost_by_type)
  // is the sum of the two.
  double intrinsics_prior_cost = 0;
  // Per intrinsics block with a prior: the largest |k - target| / sigma over its
  // free parameters, and that parameter's index (COLMAP order); 0 and -1 for blocks
  // without one. Empty without intrinsics_prior_sigma.
  std::vector<double> intrinsics_prior_norm;
  std::vector<int> intrinsics_prior_param;

  // Per observation: reprojection error in pixels (not divided by sigma), and
  // whether the point is in front of the camera.
  std::vector<double> reprojection_error;
  std::vector<char> in_front;

  // Per prior: whitened residual norm (standard deviations) and robust weight
  // rho'(s), 1 for a prior fully trusted, near 0 for one the solve overruled.
  std::vector<double> relative_prior_norm, relative_prior_weight;
  std::vector<double> global_prior_norm, global_prior_weight;
  // Per gravity prior: angle between the measured and the solved gravity direction
  // (radians), and the robust weight.
  std::vector<double> gravity_angle, gravity_weight;
  // Per segment: the angle (radians) by which its assigned axis misses the segment's
  // plane, and the robust weight; NaN and 0 for segments not in the Manhattan term
  // (unassigned, or the term off).
  std::vector<double> line_angle, line_weight;
  // Per observation with a depth reading (NaN elsewhere, weight 0): the refined depth
  // of its point in the camera (as the readings measure it), the corrected reading
  // a d + b, and the robust weight. Applying the image's correction to the whole
  // depth map gives the refined map (document 02).
  std::vector<double> depth_refined, depth_corrected, depth_weight;

  // Priors whose weight fell below the threshold given to evaluate(): what the
  // caller's cleanup let through (document 02).
  std::vector<std::int32_t> downweighted_relative, downweighted_global, downweighted_gravity;
};

ResidualReport evaluate(const Problem& problem, const SolverOptions& options, double downweight_threshold = 0.1);

}  // namespace jba
