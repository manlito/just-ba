#pragma once

#include <memory>

#include "jba/backend.h"
#include "jba/problem.h"
#include "jba/solver.h"

namespace jba::metal {

// Metal backend (document 03): the per-observation work in single precision on the
// GPU (linearization of the reprojection residual, column norms and gradient, point
// and camera blocks, the implicit Schur product and the whole conjugate gradients
// loop with Ceres's stopping test), every decision on the host in double (the cost
// of the current and the candidate parameters, accepting a step, the retraction).
// Camera-side factors and depth rows, few by comparison, are linearized on the host
// in double and uploaded. Same linear solver as the CPU backend: implicit Schur
// complement with preconditioned conjugate gradients, preconditioner the camera
// blocks of the Schur complement. Results are the same from run to run.
//
// Throws std::runtime_error when no Metal device is present or a kernel fails.
class MetalBackend final : public Backend {
 public:
  MetalBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge);
  ~MetalBackend() override;

  double cost() override;
  void cost_by_type(double* out) override;
  void linearize(bool compute_scaling) override;
  double gradient_max_norm() const override;
  double parameter_norm() const override;
  StepResult compute_step(double radius, double eta, int min_linear_iterations, int max_linear_iterations) override;
  bool candidate_cost(double* cost) override;
  void accept_candidate() override;

  // Copies the current parameters into `problem`.
  void write_back(Problem& problem) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jba::metal
