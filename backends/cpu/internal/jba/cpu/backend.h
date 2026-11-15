#pragma once

#include <memory>

#include "jba/backend.h"
#include "jba/problem.h"
#include "jba/solver.h"

namespace jba::cpu {

// CPU backend in double precision: the accuracy reference for the others
// (document 03). Linear solver: implicit Schur complement with preconditioned
// conjugate gradients; preconditioner = camera blocks of the Schur complement.
class CpuBackend final : public Backend {
 public:
  // `gauge` lists extra freedoms to hold fixed (see choose_gauge); it may be empty.
  CpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge);
  ~CpuBackend() override;

  double cost() override;
  void cost_by_type(double* out) override;
  void linearize(bool compute_scaling) override;
  double gradient_max_norm() const override;
  double parameter_norm() const override;
  StepResult compute_step(double radius, double eta, int min_linear_iterations, int max_linear_iterations) override;
  bool candidate_cost(double* cost) override;
  void accept_candidate() override;

  // Copies the current parameters into `problem` (the one passed at construction).
  void write_back(Problem& problem) const;

  // Covariance of the current parameters from the undamped, unscaled Gauss-Newton
  // system (document 02, "Uncertainty of the result"): 6 x 6 per image (pose
  // tangent: rotation, then translation), 3 x 3 for each point listed in `points`.
  // Held coordinates have zero rows and columns; images and points that are not
  // variables get zeros. Throws std::runtime_error when the system is singular (a
  // freedom no gauge removes) or larger than max_unknowns.
  void covariance(const std::vector<int>& points, std::size_t max_unknowns, std::vector<double>& pose_cov,
                  std::vector<double>& point_cov);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jba::cpu
