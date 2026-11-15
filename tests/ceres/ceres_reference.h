#pragma once

// The same problem built in Ceres 2.2.0 (test-only reference, document 05).
//
// Independent of the library's math: the cost functor is hand-written with
// automatic differentiation, not the generated code. Shared with the library: the
// tangent convention (a matching pose manifold), the loss definitions, and the
// gauge choice, so both solvers see exactly the same problem.

#include <ceres/ceres.h>

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "jba/problem.h"
#include "jba/solver.h"

namespace jba_ceres {

// A string as a JSON string literal, quotes included.
inline std::string json_string(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

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

// Positive scale with a log tangent: s <- s * exp(ds), as the library updates a
// prior set's scale.
class LogScale final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 1; }
  int TangentSize() const override { return 1; }
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* J) const override;
  bool Minus(const double* y, const double* x, double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* J) const override;
};

// A pose (7 values) whose translation and heading about the vertical `up` are held,
// its tilt free (the gauge's heading_images, D83): tangent (delta0, delta1), the
// rotation step B (delta0, delta1, 0) of jba::heading_basis at the current pose.
class HeadingHeldPose final : public ceres::Manifold {
 public:
  explicit HeadingHeldPose(const double* up) : up_{up[0], up[1], up[2]} {}
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 2; }
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* J) const override;
  bool Minus(const double* y, const double* x, double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* J) const override;

 private:
  double up_[3];
};

// A unit quaternion rotated about a fixed axis `up` only: q <- Exp(theta up) q. The
// Manhattan frame with one degree of freedom (its heading about the vertical).
class HeadingQuaternion final : public ceres::Manifold {
 public:
  explicit HeadingQuaternion(const double* up) : up_{up[0], up[1], up[2]} {}
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 1; }
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* J) const override;
  bool Minus(const double* y, const double* x, double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* J) const override;

 private:
  double up_[3];
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

// Pose prior cost functions, written independently of the generated code with the
// residuals of residuals/symbolic/priors.py, whitened by W (6 x 6, row-major).
// Relative: parameters (pose i 7, pose j 7). Global: (pose 7, rigid 7, scale 1), or
// (pose 7) when the set is the working frame.
ceres::CostFunction* make_relative_prior_cost(const double* prior, const double* W);
ceres::CostFunction* make_global_prior_cost(const double* q_prior, const double* c_prior, const double* W,
                                            bool identity);

// Gravity prior: parameters (pose 7); residual R g_world - g_camera divided by sigma.
ceres::CostFunction* make_gravity_prior_cost(const double* g_world, const double* g_camera, double inv_sigma);

// Manhattan line: parameters (pose 7, frame 4); residual n . (R R_M e_axis) divided by
// sigma, with n the segment's plane normal in camera axes.
ceres::CostFunction* make_line_cost(const double* normal, int axis, double inv_sigma);

// Depth at a keypoint: parameters (pose 7, point 3, correction 2); residual
// (depth - b) / a - d, or a / (depth - b) - 1 / d, divided by sigma.
ceres::CostFunction* make_depth_cost(jba::DepthKind kind, jba::DepthForm form, double reading, double inv_sigma);
// Prior on a depth correction toward (1, 0): parameters (correction 2), weights per part.
ceres::CostFunction* make_depth_prior_cost(double w_scale, double w_offset);

// A ceres::LossFunction matching `loss`, or nullptr for none. Caller owns it.
ceres::LossFunction* make_loss(const jba::Loss& loss);

enum class LinearSolver { DenseSchur, SparseSchur, IterativeSchur };

struct Options {
  LinearSolver linear_solver = LinearSolver::IterativeSchur;
  // SPARSE_SCHUR's sparse Cholesky: Eigen's (the default, every platform), Apple's
  // Accelerate (a Ceres built with third_party/ceres/build.cmake -DACCELERATE=ON),
  // SuiteSparse's CHOLMOD, or NVIDIA's cuDSS (Ceres master built with CUDA): each
  // needs a Ceres built with it.
  enum class SparseLibrary { Eigen, Accelerate, SuiteSparse, Cuda };
  SparseLibrary sparse_library = SparseLibrary::Eigen;
  int max_iterations = 50;
  double function_tolerance = 1e-6;
  double gradient_tolerance = 1e-10;
  double parameter_tolerance = 1e-8;
  int max_linear_iterations = 500;
  double eta = 0.1;
  jba::Loss losses[jba::kNumLossTypes];  // by residual type (jba::kLossReprojection, ...)
  jba::PriorDefaults prior_defaults;
  jba::Gauge gauge;  // the same freedoms the library would hold fixed
  int num_threads = 1;
  bool solve = true;        // false: build the problem only (with covariance)
  bool covariance = false;  // Ceres's covariance estimation (dense SVD) at the end
  std::vector<int> covariance_points;
};

struct Result {
  double initial_cost = 0;
  double final_cost = 0;
  int iterations = 0;
  int linear_iterations = 0;
  std::string termination;
  bool usable = false;
  double time_seconds = 0;  // Ceres's total_time_in_seconds: from Solve() on
  double wall_seconds = 0;  // the whole call, building the Ceres problem included
  std::vector<double> history_cost;  // iteration 0 first
  std::vector<double> history_time;  // since the call started, per iteration
  std::vector<int> history_linear_iterations;
  std::vector<double> history_radius;
  std::vector<double> history_relative_decrease;
  std::vector<bool> history_accepted;
  // With Options::covariance: per image 6 x 6 in the pose tangent (held coordinates
  // zero; NaN for a heading-held pose, whose tangent is not the library's), and 3 x 3
  // per requested point.
  std::vector<double> pose_covariance, point_covariance;
};

// Solves `problem` in place with Ceres.
Result solve(jba::Problem& problem, const Options& options);

// From library options, so a parity run uses identical settings.
Options options_from(const jba::SolverOptions& o, const jba::Problem& problem, LinearSolver linear_solver);

}  // namespace jba_ceres
