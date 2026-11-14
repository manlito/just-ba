#include "ceres_reference.h"

#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>

#include "jba/geometry.h"
#include "jba/priors.h"

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

bool LogScale::Plus(const double* x, const double* delta, double* x_plus_delta) const {
  x_plus_delta[0] = x[0] * std::exp(delta[0]);
  return true;
}
bool LogScale::PlusJacobian(const double* x, double* J) const {
  J[0] = x[0];
  return true;
}
bool LogScale::Minus(const double* y, const double* x, double* y_minus_x) const {
  y_minus_x[0] = std::log(y[0] / x[0]);
  return true;
}
bool LogScale::MinusJacobian(const double* x, double* J) const {
  J[0] = 1 / x[0];
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
    } else if constexpr (M == CameraModel::Radial) {
      const T r2 = u * u + v * v;
      const T d = T(1) + k[3] * r2 + k[4] * r2 * r2;
      x = k[0] * u * d + k[1];
      y = k[0] * v * d + k[2];
    } else if constexpr (M == CameraModel::OpenCV) {
      // COLMAP's OpenCVCameraModel::Distortion.
      const T u2 = u * u, v2 = v * v, uv = u * v, r2 = u2 + v2;
      const T radial = k[4] * r2 + k[5] * r2 * r2;
      const T du = u * radial + T(2) * k[6] * uv + k[7] * (r2 + T(2) * u2);
      const T dv = v * radial + T(2) * k[7] * uv + k[6] * (r2 + T(2) * v2);
      x = k[0] * (u + du) + k[2];
      y = k[1] * (v + dv) + k[3];
    } else {
      // COLMAP's OpenCVFisheyeCameraModel::Distortion.
      const T r = ceres::sqrt(u * u + v * v);
      T uu = u, vv = v;
      if (r > T(1e-12)) {
        const T theta = ceres::atan(r);
        const T t2 = theta * theta, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
        const T thetad = theta * (T(1) + k[4] * t2 + k[5] * t4 + k[6] * t6 + k[7] * t8);
        uu = u * thetad / r;
        vv = v * thetad / r;
      }
      x = k[0] * uu + k[2];
      y = k[1] * vv + k[3];
    }
    r[0] = (x - ox) * w;
    r[1] = (y - oy) * w;
    return true;
  }

  double ox, oy, w;
};

template <CameraModel M>
ceres::CostFunction* make_cost(double ox, double oy, double inv_sigma) {
  constexpr int np = M == CameraModel::SimplePinhole ? 3
                     : M == CameraModel::Radial     ? 5
                     : (M == CameraModel::OpenCV || M == CameraModel::OpenCVFisheye) ? 8
                                                    : 4;
  return new ceres::AutoDiffCostFunction<Reprojection<M>, 2, 7, 3, np>(new Reprojection<M>(ox, oy, inv_sigma));
}

}  // namespace

ceres::CostFunction* make_reprojection_cost(CameraModel m, double ox, double oy, double inv_sigma) {
  switch (m) {
    case CameraModel::SimplePinhole: return make_cost<CameraModel::SimplePinhole>(ox, oy, inv_sigma);
    case CameraModel::Pinhole: return make_cost<CameraModel::Pinhole>(ox, oy, inv_sigma);
    case CameraModel::SimpleRadial: return make_cost<CameraModel::SimpleRadial>(ox, oy, inv_sigma);
    case CameraModel::Radial: return make_cost<CameraModel::Radial>(ox, oy, inv_sigma);
    case CameraModel::OpenCV: return make_cost<CameraModel::OpenCV>(ox, oy, inv_sigma);
    case CameraModel::OpenCVFisheye: return make_cost<CameraModel::OpenCVFisheye>(ox, oy, inv_sigma);
  }
  throw std::invalid_argument("unknown camera model");
}

namespace {

template <typename T>
void whiten(const double* W, const T* r, T* out) {
  for (int a = 0; a < 6; ++a) {
    out[a] = T(0);
    for (int c = 0; c < 6; ++c) out[a] += W[6 * a + c] * r[c];
  }
}

struct RelativePrior {
  double prior[7], W[36];

  template <typename T>
  bool operator()(const T* pi, const T* pj, T* out) const {
    const T qi_conj[4] = {pi[0], -pi[1], -pi[2], -pi[3]};
    T q_ji[4];
    ceres::QuaternionProduct(pj, qi_conj, q_ji);
    T Rti[3];
    ceres::UnitQuaternionRotatePoint(q_ji, pi + 4, Rti);
    const T qp_conj[4] = {T(prior[0]), T(-prior[1]), T(-prior[2]), T(-prior[3])};
    T qe[4];
    ceres::QuaternionProduct(qp_conj, q_ji, qe);
    T r[6];
    ceres::QuaternionToAngleAxis(qe, r);
    const T d[3] = {pj[4] - Rti[0] - prior[4], pj[5] - Rti[1] - prior[5], pj[6] - Rti[2] - prior[6]};
    ceres::UnitQuaternionRotatePoint(qp_conj, d, r + 3);
    whiten(W, r, out);
    return true;
  }
};

struct GlobalPrior {
  double q_prior[4], c_prior[3], W[36];

  template <typename T>
  bool eval(const T* pose, const T* rigid, const T* scale, T* out) const {
    const T qp_conj[4] = {T(q_prior[0]), T(-q_prior[1]), T(-q_prior[2]), T(-q_prior[3])};
    T q1[4], qe[4];
    ceres::QuaternionProduct(pose, rigid, q1);
    ceres::QuaternionProduct(q1, qp_conj, qe);
    T r[6];
    ceres::QuaternionToAngleAxis(qe, r);
    const T q_conj[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
    T C[3];
    ceres::UnitQuaternionRotatePoint(q_conj, pose + 4, C);
    const T d[3] = {-C[0] - rigid[4], -C[1] - rigid[5], -C[2] - rigid[6]};
    const T rs_conj[4] = {rigid[0], -rigid[1], -rigid[2], -rigid[3]};
    T local[3];
    ceres::UnitQuaternionRotatePoint(rs_conj, d, local);
    for (int a = 0; a < 3; ++a) r[3 + a] = local[a] / scale[0] - c_prior[a];
    whiten(W, r, out);
    return true;
  }
  template <typename T>
  bool operator()(const T* pose, const T* rigid, const T* scale, T* out) const {
    return eval(pose, rigid, scale, out);
  }
};

struct GlobalPriorIdentity {
  GlobalPrior g;
  template <typename T>
  bool operator()(const T* pose, T* out) const {
    const T rigid[7] = {T(1), T(0), T(0), T(0), T(0), T(0), T(0)};
    const T scale[1] = {T(1)};
    return g.eval(pose, rigid, scale, out);
  }
};

}  // namespace

ceres::CostFunction* make_relative_prior_cost(const double* prior, const double* W) {
  auto* f = new RelativePrior;
  std::copy_n(prior, 7, f->prior);
  std::copy_n(W, 36, f->W);
  return new ceres::AutoDiffCostFunction<RelativePrior, 6, 7, 7>(f);
}

ceres::CostFunction* make_global_prior_cost(const double* q_prior, const double* c_prior, const double* W,
                                            bool identity) {
  GlobalPrior g;
  std::copy_n(q_prior, 4, g.q_prior);
  std::copy_n(c_prior, 3, g.c_prior);
  std::copy_n(W, 36, g.W);
  if (identity) return new ceres::AutoDiffCostFunction<GlobalPriorIdentity, 6, 7>(new GlobalPriorIdentity{g});
  return new ceres::AutoDiffCostFunction<GlobalPrior, 6, 7, 7, 1>(new GlobalPrior(g));
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
  c.relative_prior_loss = o.relative_prior_loss;
  c.global_prior_loss = o.global_prior_loss;
  c.prior_defaults = o.prior_defaults;
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

  // Pose priors.
  std::unique_ptr<ceres::LossFunction> rel_loss(make_loss(o.relative_prior_loss));
  std::unique_ptr<ceres::LossFunction> glob_loss(make_loss(o.global_prior_loss));
  std::vector<double> W_rel, W_glob;
  jba::prior_whitening(p, o.prior_defaults, W_rel, W_glob);
  if (p.set_transform.empty() && p.num_prior_sets() > 0) {
    for (std::size_t s = 0; s < p.num_prior_sets(); ++s) p.set_transform.insert(p.set_transform.end(), {1, 0, 0, 0, 0, 0, 0, 1});
  }
  for (std::size_t k = 0; k < p.num_relative_priors(); ++k) {
    const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
    problem.AddResidualBlock(make_relative_prior_cost(&p.rel_pose[7 * k], &W_rel[36 * k]), rel_loss.get(),
                             &p.poses[7 * i], &p.poses[7 * j]);
  }
  for (std::size_t k = 0; k < p.num_global_priors(); ++k) {
    const auto i = static_cast<std::size_t>(p.glob_image[k]);
    const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
    const double* g = &p.glob_pose[7 * k];
    const double conj[4] = {g[0], -g[1], -g[2], -g[3]};
    double C[3];
    jba::quat_rotate(conj, g + 4, C);
    for (double& v : C) v = -v;
    const bool identity = p.relation_of_set(s) == jba::FrameRelation::Identity;
    auto* cost = make_global_prior_cost(g, C, &W_glob[36 * k], identity);
    if (identity) {
      problem.AddResidualBlock(cost, glob_loss.get(), &p.poses[7 * i]);
    } else {
      problem.AddResidualBlock(cost, glob_loss.get(), &p.poses[7 * i], &p.set_transform[8 * s],
                               &p.set_transform[8 * s + 7]);
    }
  }

  auto* ordering = new ceres::ParameterBlockOrdering;
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    double* pose = &p.poses[7 * i];
    if (!problem.HasParameterBlock(pose)) continue;
    std::vector<int> held;
    for (const auto& [img, c] : o.gauge.fixed_translations)
      if (img == static_cast<int>(i)) held.push_back(c);
    if (!held.empty()) {
      owned.push_back(std::make_unique<PartialPoseManifold>(LeftRotationVectorQuaternion{},
                                                            ceres::SubsetManifold(3, held)));
      problem.SetManifold(pose, owned.back().get());
    } else {
      problem.SetManifold(pose, pose_manifold.get());
    }
    const bool gauge_fixed = std::find(o.gauge.fixed_images.begin(), o.gauge.fixed_images.end(),
                                       static_cast<int>(i)) != o.gauge.fixed_images.end();
    if ((!p.image_fixed.empty() && p.image_fixed[i]) || gauge_fixed) problem.SetParameterBlockConstant(pose);
    ordering->AddElementToGroup(pose, 1);
  }
  auto scale_manifold = std::make_unique<LogScale>();
  for (std::size_t s = 0; s < p.num_prior_sets(); ++s) {
    double* rigid = &p.set_transform[8 * s];
    if (!problem.HasParameterBlock(rigid)) continue;
    problem.SetManifold(rigid, pose_manifold.get());
    problem.SetManifold(rigid + 7, scale_manifold.get());
    const bool scale_held = p.relation_of_set(s) != jba::FrameRelation::Similarity ||
                            std::find(o.gauge.fixed_set_scales.begin(), o.gauge.fixed_set_scales.end(),
                                      static_cast<int>(s)) != o.gauge.fixed_set_scales.end();
    if (scale_held) problem.SetParameterBlockConstant(rigid + 7);
    ordering->AddElementToGroup(rigid, 1);
    ordering->AddElementToGroup(rigid + 7, 1);
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
