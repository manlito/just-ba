#include "ceres_reference.h"

#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>

#include "jba/geometry.h"

namespace jba_ceres {

using jba::CameraModel;

// ---------------------------------------------------------------------------

bool LeftRotationVectorQuaternion::Plus(const double* x, const double* delta, double* x_plus_delta) const {
  double dq[4];
  jba::quat_exp(delta, dq);
  jba::quat_multiply(dq, x, x_plus_delta);
  jba::quat_normalize(x_plus_delta);
  return true;
}

// d/d(delta) of Exp(delta) * x at delta = 0 is 0.5 * (0, e_k) * x per column.
bool LeftRotationVectorQuaternion::PlusJacobian(const double* x, double* J) const {
  for (int k = 0; k < 3; ++k) {
    double e[4] = {0, 0, 0, 0};
    e[1 + k] = 0.5;
    double col[4];
    jba::quat_multiply(e, x, col);
    for (int r = 0; r < 4; ++r) J[3 * r + k] = col[r];
  }
  return true;
}

bool LeftRotationVectorQuaternion::Minus(const double* y, const double* x, double* y_minus_x) const {
  const double xc[4] = {x[0], -x[1], -x[2], -x[3]};
  double d[4];
  jba::quat_multiply(y, xc, d);
  if (d[0] < 0)
    for (double& v : d) v = -v;
  const double s = std::sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]);
  const double scale = s < 1e-12 ? 2.0 : 2.0 * std::atan2(s, d[0]) / s;
  for (int i = 0; i < 3; ++i) y_minus_x[i] = scale * d[1 + i];
  return true;
}

// At y = x, Minus is 2 * vec(y * conj(x)), linear in y.
bool LeftRotationVectorQuaternion::MinusJacobian(const double* x, double* J) const {
  const double xc[4] = {x[0], -x[1], -x[2], -x[3]};
  for (int c = 0; c < 4; ++c) {
    double e[4] = {0, 0, 0, 0};
    e[c] = 1;
    double col[4];
    jba::quat_multiply(e, xc, col);
    for (int r = 0; r < 3; ++r) J[4 * r + c] = 2 * col[1 + r];
  }
  return true;
}

void GemanMcClureLoss::Evaluate(double s, double rho[3]) const {
  const double d = a2_ + s;
  rho[0] = a2_ * s / d;
  rho[1] = a2_ * a2_ / (d * d);
  rho[2] = -2 * rho[1] / d;
}

// ---------------------------------------------------------------------------

namespace {

template <CameraModel M>
struct Reprojection {
  Reprojection(double x, double y, double inv_sigma) : ox(x), oy(y), w(inv_sigma) {}

  template <typename T>
  bool operator()(const T* pose, const T* point, const T* k, T* r) const {
    T p[3];
    ceres::UnitQuaternionRotatePoint(pose, point, p);
    p[0] += pose[4];
    p[1] += pose[5];
    p[2] += pose[6];
    const T u = p[0] / p[2], v = p[1] / p[2];
    T x, y;
    if constexpr (M == CameraModel::SimplePinhole) {
      x = k[0] * u + k[1];
      y = k[0] * v + k[2];
    } else if constexpr (M == CameraModel::Pinhole) {
      x = k[0] * u + k[2];
      y = k[1] * v + k[3];
    } else if constexpr (M == CameraModel::SimpleRadial) {
      const T d = T(1) + k[3] * (u * u + v * v);
      x = k[0] * u * d + k[1];
      y = k[0] * v * d + k[2];
    } else {
      const T r2 = u * u + v * v;
      const T d = T(1) + k[3] * r2 + k[4] * r2 * r2;
      x = k[0] * u * d + k[1];
      y = k[0] * v * d + k[2];
    }
    r[0] = (x - ox) * w;
    r[1] = (y - oy) * w;
    return true;
  }

  double ox, oy, w;
};

template <CameraModel M>
ceres::CostFunction* make_cost(double ox, double oy, double inv_sigma) {
  constexpr int np = M == CameraModel::SimplePinhole ? 3 : M == CameraModel::Radial ? 5 : 4;
  return new ceres::AutoDiffCostFunction<Reprojection<M>, 2, 7, 3, np>(new Reprojection<M>(ox, oy, inv_sigma));
}

}  // namespace

ceres::CostFunction* make_reprojection_cost(CameraModel m, double ox, double oy, double inv_sigma) {
  switch (m) {
    case CameraModel::SimplePinhole: return make_cost<CameraModel::SimplePinhole>(ox, oy, inv_sigma);
    case CameraModel::Pinhole: return make_cost<CameraModel::Pinhole>(ox, oy, inv_sigma);
    case CameraModel::SimpleRadial: return make_cost<CameraModel::SimpleRadial>(ox, oy, inv_sigma);
    case CameraModel::Radial: return make_cost<CameraModel::Radial>(ox, oy, inv_sigma);
    default: throw std::invalid_argument("camera model not supported by the Ceres harness yet");
  }
}

ceres::LossFunction* make_loss(const jba::Loss& loss) {
  switch (loss.type) {
    case jba::LossType::None: return nullptr;
    case jba::LossType::Huber: return new ceres::HuberLoss(loss.scale);
    case jba::LossType::Cauchy: return new ceres::CauchyLoss(loss.scale);
    case jba::LossType::GemanMcClure: return new GemanMcClureLoss(loss.scale);
  }
  return nullptr;
}

Options options_from(const jba::SolverOptions& o, const jba::Problem& problem, LinearSolver linear_solver) {
  Options c;
  c.linear_solver = linear_solver;
  c.max_iterations = o.max_iterations;
  c.function_tolerance = o.function_tolerance;
  c.gradient_tolerance = o.gradient_tolerance;
  c.parameter_tolerance = o.parameter_tolerance;
  c.max_linear_iterations = o.max_linear_iterations;
  c.eta = o.eta;
  c.loss = o.loss;
  c.gauge = o.gauge == jba::GaugeMode::Auto ? jba::choose_gauge(problem) : jba::Gauge{};
  c.num_threads = o.num_threads > 0 ? o.num_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  return c;
}

Result solve(jba::Problem& p, const Options& o) {
  const int np = jba::num_params(p.model);

  ceres::Problem::Options popts;
  popts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  popts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(popts);
  using PoseManifold = ceres::ProductManifold<LeftRotationVectorQuaternion, ceres::EuclideanManifold<3>>;
  using PartialPoseManifold = ceres::ProductManifold<LeftRotationVectorQuaternion, ceres::SubsetManifold>;
  auto pose_manifold = std::make_unique<PoseManifold>(LeftRotationVectorQuaternion{}, ceres::EuclideanManifold<3>{});
  std::vector<std::unique_ptr<ceres::Manifold>> owned;
  std::unique_ptr<ceres::LossFunction> loss(make_loss(o.loss));

  for (std::size_t k = 0; k < p.num_observations(); ++k) {
    const std::size_t i = static_cast<std::size_t>(p.obs_image[k]);
    const std::size_t c = static_cast<std::size_t>(p.intrinsics_of_image(i));
    const double sigma = p.obs_sigma.empty() ? 1.0 : p.obs_sigma[k];
    problem.AddResidualBlock(make_reprojection_cost(p.model, p.obs_xy[2 * k], p.obs_xy[2 * k + 1], 1.0 / sigma),
                             loss.get(), &p.poses[7 * i], &p.points[3 * static_cast<std::size_t>(p.obs_point[k])],
                             &p.intrinsics[static_cast<std::size_t>(np) * c]);
  }

  auto* ordering = new ceres::ParameterBlockOrdering;
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    double* pose = &p.poses[7 * i];
    if (!problem.HasParameterBlock(pose)) continue;
    if (static_cast<int>(i) == o.gauge.scale_image) {
      owned.push_back(std::make_unique<PartialPoseManifold>(
          LeftRotationVectorQuaternion{}, ceres::SubsetManifold(3, {o.gauge.scale_coordinate})));
      problem.SetManifold(pose, owned.back().get());
    } else {
      problem.SetManifold(pose, pose_manifold.get());
    }
    if ((!p.image_fixed.empty() && p.image_fixed[i]) || static_cast<int>(i) == o.gauge.reference_image)
      problem.SetParameterBlockConstant(pose);
    ordering->AddElementToGroup(pose, 1);
  }
  for (std::size_t j = 0; j < p.num_points(); ++j) {
    double* X = &p.points[3 * j];
    if (!problem.HasParameterBlock(X)) continue;
    if (!p.point_fixed.empty() && p.point_fixed[j]) problem.SetParameterBlockConstant(X);
    ordering->AddElementToGroup(X, 0);
  }
  for (std::size_t c = 0; c < p.num_intrinsics_blocks(); ++c) {
    double* K = &p.intrinsics[static_cast<std::size_t>(np) * c];
    if (!problem.HasParameterBlock(K)) continue;
    const std::uint8_t mask = p.intrinsics_free.empty() ? 0 : p.intrinsics_free[c];
    std::vector<int> constant;
    for (int a = 0; a < np; ++a) {
      if (!(mask & jba::intrinsics_group(p.model, a))) constant.push_back(a);
    }
    if (static_cast<int>(constant.size()) == np) {
      problem.SetParameterBlockConstant(K);
    } else if (!constant.empty()) {
      owned.push_back(std::make_unique<ceres::SubsetManifold>(np, constant));
      problem.SetManifold(K, owned.back().get());
    }
    ordering->AddElementToGroup(K, 1);
  }

  ceres::Solver::Options opts;
  opts.minimizer_type = ceres::TRUST_REGION;
  opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  opts.max_num_iterations = o.max_iterations;
  opts.function_tolerance = o.function_tolerance;
  opts.gradient_tolerance = o.gradient_tolerance;
  opts.parameter_tolerance = o.parameter_tolerance;
  opts.max_linear_solver_iterations = o.max_linear_iterations;
  opts.eta = o.eta;
  opts.num_threads = o.num_threads;
  opts.linear_solver_ordering.reset(ordering);
  switch (o.linear_solver) {
    case LinearSolver::DenseSchur: opts.linear_solver_type = ceres::DENSE_SCHUR; break;
    case LinearSolver::SparseSchur:
      opts.linear_solver_type = ceres::SPARSE_SCHUR;
      opts.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
      break;
    case LinearSolver::IterativeSchur:
      opts.linear_solver_type = ceres::ITERATIVE_SCHUR;
      opts.preconditioner_type = ceres::SCHUR_JACOBI;
      break;
  }
  opts.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  Result r;
  r.initial_cost = summary.initial_cost;
  r.final_cost = summary.final_cost;
  r.iterations = static_cast<int>(summary.iterations.size()) - 1;
  for (const auto& it : summary.iterations) {
    r.linear_iterations += it.linear_solver_iterations;
    r.history_cost.push_back(it.cost);
    r.history_linear_iterations.push_back(it.linear_solver_iterations);
    r.history_radius.push_back(it.trust_region_radius);
    r.history_relative_decrease.push_back(it.relative_decrease);
    r.history_accepted.push_back(it.step_is_successful);
  }
  r.termination = ceres::TerminationTypeToString(summary.termination_type);
  r.usable = summary.IsSolutionUsable();
  r.time_seconds = summary.total_time_in_seconds;
  return r;
}

}  // namespace jba_ceres
