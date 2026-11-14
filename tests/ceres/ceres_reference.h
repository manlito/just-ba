#pragma once

// The same problem built in Ceres 2.2.0 (test-only reference, document 05).
//
// Independent of the library's math: the cost functor is hand-written with
// automatic differentiation, not the generated code. Shared with the library: the
// tangent convention (a matching pose manifold), the loss definitions, and the
// gauge choice, so both solvers see exactly the same problem.

#include <ceres/ceres.h>

#include <string>
#include <vector>

#include "jba/problem.h"
#include "jba/solver.h"

namespace jba_ceres {

// Unit quaternion (w, x, y, z) with the left rotation-vector tangent of D46.
class LeftRotationVectorQuaternion final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 3; }
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* J) const override;
  bool Minus(const double* y, const double* x, double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* J) const override;
};

// Geman-McClure in the library's definition; Ceres has no such loss.
class GemanMcClureLoss final : public ceres::LossFunction {
 public:
  explicit GemanMcClureLoss(double a) : a2_(a * a) {}
  void Evaluate(double s, double rho[3]) const override;

 private:
  double a2_;
};

// Reprojection cost function for one observation, parameters (pose 7, point 3,
// intrinsics), residual in pixels divided by sigma.
ceres::CostFunction* make_reprojection_cost(jba::CameraModel model, double ox, double oy, double inv_sigma);

// A ceres::LossFunction matching `loss`, or nullptr for none. Caller owns it.
ceres::LossFunction* make_loss(const jba::Loss& loss);

enum class LinearSolver { DenseSchur, SparseSchur, IterativeSchur };

struct Options {
  LinearSolver linear_solver = LinearSolver::IterativeSchur;
  int max_iterations = 50;
  double function_tolerance = 1e-6;
  double gradient_tolerance = 1e-10;
  double parameter_tolerance = 1e-8;
  int max_linear_iterations = 500;
  double eta = 0.1;
  jba::Loss loss;
  jba::Gauge gauge;  // the same freedoms the library would hold fixed
  int num_threads = 1;
};

struct Result {
  double initial_cost = 0;
  double final_cost = 0;
  int iterations = 0;
  int linear_iterations = 0;
  std::string termination;
  bool usable = false;
  double time_seconds = 0;
  std::vector<double> history_cost;  // iteration 0 first
  std::vector<int> history_linear_iterations;
  std::vector<double> history_radius;
  std::vector<double> history_relative_decrease;
  std::vector<bool> history_accepted;
};

// Solves `problem` in place with Ceres.
Result solve(jba::Problem& problem, const Options& options);

// From library options, so a parity run uses identical settings.
Options options_from(const jba::SolverOptions& o, const jba::Problem& problem, LinearSolver linear_solver);

}  // namespace jba_ceres
