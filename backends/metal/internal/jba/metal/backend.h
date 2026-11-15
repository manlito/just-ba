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
// in double and uploaded. Two linear solvers (D92): sparse_schur, the default here,
// forms the reduced camera system on the GPU and factorizes it on the host in double
// with Accelerate's sparse Cholesky, falling back to conjugate gradients when the
// single-precision system is not positive definite; iterative_schur runs the CPU
// backend's implicit Schur complement with preconditioned conjugate gradients
// entirely on the GPU. Results are the same from run to run.
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
  // 1e6: in single precision, damping below about 1e-6 of a diagonal entry is lost
  // in the sums, so a larger radius would not change the step.
  double max_trust_region_radius() const override { return 1e6; }

  // Copies the current parameters into `problem`.
  void write_back(Problem& problem) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jba::metal
