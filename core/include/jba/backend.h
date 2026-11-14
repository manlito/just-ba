#pragma once

// Interface between the trust-region loop (core, host) and a backend that owns the
// linear system (document 03). Internal; it will be frozen in milestone 4 from the
// CPU experience. Virtual calls happen a few times per outer iteration only.

namespace jba {

struct StepResult {
  bool valid = false;            // the linear solve succeeded and predicts a decrease
  double model_cost_change = 0;  // predicted decrease, > 0 when valid
  double step_norm = 0;          // norm of candidate - current, stored coordinates
  int linear_iterations = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;

  // Robustified cost 0.5 * sum rho(s) at the current parameters.
  virtual double cost() = 0;

  // Residuals, Jacobians and gradient at the current parameters. The Jacobian
  // column scaling is computed when `compute_scaling` is set (first call only, as
  // in Ceres) and reused afterwards.
  virtual void linearize(bool compute_scaling) = 0;

  // Max |gradient| in tangent coordinates, from the last linearize().
  virtual double gradient_max_norm() const = 0;

  // Norm of the free parameters (ambient coordinates).
  virtual double parameter_norm() const = 0;

  // Solves the damped subproblem for this trust-region radius and forms the
  // candidate parameters. Does not change the current parameters.
  virtual StepResult compute_step(double radius, double eta, int min_linear_iterations,
                                  int max_linear_iterations) = 0;

  // Cost at the candidate parameters. Returns false when it is not finite.
  virtual bool candidate_cost(double* cost) = 0;

  // Makes the candidate the current parameters.
  virtual void accept_candidate() = 0;
};

}  // namespace jba
