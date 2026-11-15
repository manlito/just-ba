#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "jba/problem.h"
#include "jba/solver.h"

namespace jba {

// Uncertainty of a solved problem (document 02, "Uncertainty of the result"): the
// inverse of the Gauss-Newton system at the current parameters, with the robust
// losses applied as in the solve, the camera side through the reduced system and
// points by back substitution. Meaningful as far as the input weights are: with
// default prior weights the values are relative, not calibrated. They are relative to
// the gauge the solve used (Covariance::gauge), and in the working frame.
struct CovarianceOptions {
  std::vector<std::int32_t> points;  // points to return, by index
  bool all_points = false;           // or every point
  // The reduced system is inverted densely; above this many camera-side unknowns the
  // call is refused (12,000 unknowns, 2,000 images, take about 1 GB).
  std::size_t max_unknowns = 20000;
};

struct Covariance {
  // Per image, 6 x 6 row-major in the pose tangent of D46: rotation (radians, camera
  // axes, R <- Exp(dtheta) R), then translation t. Zeros for images that are not
  // variables, and in held coordinates.
  std::vector<double> pose;
  // Per image, 3 x 3: the camera centre C = -R^T t, working frame, metres.
  std::vector<double> centre;
  // Per requested point, 3 x 3, working frame.
  std::vector<std::int32_t> points;
  std::vector<double> point;
  Gauge gauge;  // what was held
};

// Throws std::invalid_argument for an invalid problem, std::runtime_error when the
// system is singular (a freedom nothing holds) or too large.
Covariance covariance(const Problem& problem, const SolverOptions& options = {},
                      const CovarianceOptions& covariance_options = {});

}  // namespace jba
