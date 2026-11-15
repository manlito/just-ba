#include "jba/trust_region.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace jba {
namespace {

std::string format(const char* fmt, double a, double b) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), fmt, a, b);
  return buf;
}

}  // namespace

void run_trust_region(Backend& backend, const SolverOptions& o, const IterationCallback& callback,
                      SolverSummary& s) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };

  // A single-precision backend cannot resolve damping below a share of the diagonal,
  // so the radius is capped where it still changes the system.
  const double max_radius = std::min(o.max_trust_region_radius, backend.max_trust_region_radius());
  double radius = std::min(o.initial_trust_region_radius, max_radius);
  double decrease_factor = 2;
  int invalid_steps = 0;

  double x_cost = backend.cost();
  s.initial_cost = x_cost;
  s.final_cost = x_cost;
  backend.linearize(/*compute_scaling=*/true);
  double x_norm = backend.parameter_norm();

  IterationSummary it;
  it.iteration = 0;
  it.cost = x_cost;
  it.gradient_max_norm = backend.gradient_max_norm();
  it.trust_region_radius = radius;
  it.step_valid = true;
  it.step_accepted = true;
  it.time_seconds = elapsed();
  s.history.push_back(it);

  auto finish = [&](Termination t, std::string message) {
    s.termination = t;
    s.message = std::move(message);
    s.final_cost = x_cost;
    s.iterations = static_cast<int>(s.history.size()) - 1;
    s.time_seconds = elapsed();
  };

  if (callback && !callback(it)) return finish(Termination::Cancelled, "cancelled by the callback");
  if (!std::isfinite(x_cost)) return finish(Termination::Failure, "initial cost is not finite");
  if (it.gradient_max_norm <= o.gradient_tolerance)
    return finish(Termination::Convergence, format("gradient tolerance reached: %g <= %g", it.gradient_max_norm,
                                                   o.gradient_tolerance));

  for (;;) {
    if (static_cast<int>(s.history.size()) - 1 >= o.max_iterations)
      return finish(Termination::NoConvergence, "maximum number of iterations reached");

    IterationSummary cur;
    cur.iteration = static_cast<int>(s.history.size());
    const StepResult step = backend.compute_step(radius, o.eta, o.min_linear_iterations, o.max_linear_iterations);
    cur.linear_iterations = step.linear_iterations;
    s.linear_iterations += step.linear_iterations;
    cur.step_valid = step.valid;
    cur.step_norm = step.step_norm;

    bool accepted = false;
    // As with the tolerances below, the iteration that ends the solve is not recorded.
    if (!step.valid && step.precision_limit)
      return finish(Termination::Convergence, "predicted decrease below the backend's precision");
    if (!step.valid) {
      if (++invalid_steps >= o.max_consecutive_invalid_steps)  // as in Ceres
        return finish(Termination::Failure, "too many consecutive invalid steps");
      radius /= decrease_factor;
      decrease_factor *= 2;
    } else {
      invalid_steps = 0;
      double candidate = 0;
      if (!backend.candidate_cost(&candidate)) candidate = std::numeric_limits<double>::max();

      // Tolerances on the candidate, before it is accepted (as in Ceres).
      const double step_tolerance = o.parameter_tolerance * (x_norm + o.parameter_tolerance);
      // As in Ceres, the iteration that meets a tolerance is not recorded.
      if (step.step_norm <= step_tolerance) {
        return finish(Termination::Convergence,
                      format("parameter tolerance reached: step %g <= %g", step.step_norm, step_tolerance));
      }
      cur.cost_change = x_cost - candidate;
      if (std::abs(cur.cost_change) <= o.function_tolerance * x_cost) {
        return finish(Termination::Convergence, format("function tolerance reached: |cost change| %g <= %g",
                                                       std::abs(cur.cost_change), o.function_tolerance * x_cost));
      }

      cur.relative_decrease = cur.cost_change / step.model_cost_change;
      if (cur.relative_decrease > o.min_relative_decrease) {
        accepted = true;
        backend.accept_candidate();
        x_cost = candidate;
        x_norm = backend.parameter_norm();
        backend.linearize(false);
        const double q = 2 * cur.relative_decrease - 1;
        radius = std::min(max_radius, radius / std::max(1.0 / 3.0, 1.0 - q * q * q));
        decrease_factor = 2;
        ++s.accepted_steps;
      } else {
        radius /= decrease_factor;
        decrease_factor *= 2;
      }
    }

    cur.step_accepted = accepted;
    cur.cost = x_cost;
    cur.gradient_max_norm = backend.gradient_max_norm();
    cur.trust_region_radius = radius;
    cur.time_seconds = elapsed();
    s.history.push_back(cur);
    s.final_cost = x_cost;

    if (callback && !callback(cur)) return finish(Termination::Cancelled, "cancelled by the callback");
    if (accepted && cur.gradient_max_norm <= o.gradient_tolerance)
      return finish(Termination::Convergence, format("gradient tolerance reached: %g <= %g", cur.gradient_max_norm,
                                                     o.gradient_tolerance));
    if (radius < o.min_trust_region_radius)
      return finish(Termination::Convergence, format("trust region radius %g below %g", radius,
                                                     o.min_trust_region_radius));
  }
}

}  // namespace jba
