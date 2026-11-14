#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "jba/loss.h"
#include "jba/priors.h"
#include "jba/problem.h"

namespace jba {

// How global freedoms (gauge) are removed when nothing else does it (document 02).
enum class GaugeMode {
  // When no image and no point is held fixed: in every group of connected images
  // whose frame nothing else defines, hold a reference pose, and the scale when
  // nothing metric sets it. See choose_gauge().
  Auto,
  // Leave the freedoms in; damping keeps the steps bounded.
  None,
};

// Images are connected when they share a point or a relative prior, or have global
// priors from the same set with a transform. Per group of connected images
// (document 02, Gauge):
// - a global prior from a set that is the working frame defines the frame: nothing
//   is held;
// - otherwise the reference image's pose is held: the one with the most
//   observations, then the most priors, then the lowest index;
// - the scale is then set by relative priors or a rigid set if there are any;
//   otherwise by holding the scale of a similarity set (the lowest index) if there
//   is one; otherwise, as COLMAP does, by holding one translation coordinate of the
//   image sharing the most points with the reference (its largest coordinate of
//   R2 (C2 - C1), D57).
struct Gauge {
  std::vector<int> fixed_images;                     // poses held fixed
  std::vector<std::array<int, 2>> fixed_translations;  // (image, coordinate 0..2)
  std::vector<int> fixed_set_scales;                 // prior sets whose scale is held
  bool empty() const { return fixed_images.empty() && fixed_translations.empty() && fixed_set_scales.empty(); }
};

struct SolverOptions {
  // Trust region (Levenberg-Marquardt). Defaults are Ceres's, so the two solvers
  // can be compared iteration by iteration.
  int max_iterations = 50;
  double function_tolerance = 1e-6;   // stop when |cost change| <= tol * cost
  double gradient_tolerance = 1e-10;  // stop when max |gradient| <= tol
  double parameter_tolerance = 1e-8;  // stop when |step| <= tol * (|x| + tol)
  double initial_trust_region_radius = 1e4;
  double max_trust_region_radius = 1e16;
  double min_trust_region_radius = 1e-32;
  double min_relative_decrease = 1e-3;
  int max_consecutive_invalid_steps = 5;  // the solve fails on reaching this count

  // Linear solver: implicit Schur complement, preconditioned conjugate gradients,
  // preconditioner = camera blocks of the Schur complement.
  int max_linear_iterations = 500;
  int min_linear_iterations = 0;
  double eta = 0.1;  // forcing parameter of the conjugate gradients stopping test

  Loss loss;  // applied to every reprojection residual
  // Pose priors are always robust (document 02). Scales are in standard deviations.
  // Provisional defaults, to be set from the synthetic jump tests and real captures.
  Loss relative_prior_loss{LossType::Cauchy, 3.0};
  Loss global_prior_loss{LossType::Cauchy, 3.0};
  PriorDefaults prior_defaults;  // weights for priors that come without any
  GaugeMode gauge = GaugeMode::Auto;

  // Worker threads for the CPU backend; 0 means all hardware threads. Results do
  // not depend on the thread count.
  int num_threads = 0;
};

struct IterationSummary {
  int iteration = 0;
  double cost = 0;              // after this iteration
  double cost_change = 0;       // of the candidate step, accepted or not
  double gradient_max_norm = 0;
  double step_norm = 0;         // of candidate - current, in stored coordinates
  double relative_decrease = 0;  // actual over predicted decrease
  double trust_region_radius = 0;
  int linear_iterations = 0;
  bool step_valid = false;
  bool step_accepted = false;
  double time_seconds = 0;      // wall time since the solve started
};

enum class Termination {
  Convergence,    // a tolerance was met
  NoConvergence,  // iteration limit reached
  Failure,        // numerical failure; parameters are the last accepted ones
  Cancelled,      // the callback returned false
};

std::string_view termination_name(Termination t);

struct SolverSummary {
  Termination termination = Termination::Failure;
  std::string message;
  double initial_cost = 0;
  double final_cost = 0;
  int iterations = 0;           // not counting iteration 0
  int accepted_steps = 0;
  int linear_iterations = 0;    // total
  double time_seconds = 0;
  std::vector<IterationSummary> history;  // iteration 0 first
  Gauge gauge;  // freedoms actually held fixed
};

// Called after every iteration, including iteration 0. Return false to stop; the
// solve then ends with Termination::Cancelled and the last accepted parameters.
// No printing happens inside the solver; this is the logging hook (document 01).
using IterationCallback = std::function<bool(const IterationSummary&)>;


// The gauge GaugeMode::Auto would apply to this problem, or an empty Gauge when
// none applies (any image or point already fixed). Exposed so reference solvers can
// apply the same one.
Gauge choose_gauge(const Problem& problem);

// Solves in place on the CPU. Throws std::invalid_argument for an invalid problem.
SolverSummary solve(Problem& problem, const SolverOptions& options = {},
                    const IterationCallback& callback = {});

}  // namespace jba
