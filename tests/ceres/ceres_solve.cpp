// Reference solve with Ceres 2.2.0 (test-only; document 05).
//
//   jba_ceres_solve <problem.jba | bal.txt> [--max-iterations N] [--linear-solver dense_schur|sparse_schur|iterative_schur]
//                   [--output solved.jba] [--threads N]
//
// Builds the same problem in Ceres with an independent cost functor (hand-written,
// automatic differentiation; not the generated code) and a pose manifold that
// matches just-ba's tangent: R <- Exp(dtheta) R, t <- t + dt. Prints a JSON summary.
// Exits non-zero if Ceres's initial cost disagrees with just-ba's CPU evaluation or
// the solve fails.

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "jba/cpu/cost.h"
#include "jba/geometry.h"
#include "jba/problem_io.h"

namespace {

using jba::CameraModel;

// ---------------------------------------------------------------------------
// Manifold: unit quaternion (w, x, y, z) with a left rotation-vector tangent.

class LeftRotationVectorQuaternion final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 3; }

  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
    double dq[4];
    jba::quat_exp(delta, dq);
    jba::quat_multiply(dq, x, x_plus_delta);
    jba::quat_normalize(x_plus_delta);
    return true;
  }

  // d/d(delta) of Exp(delta) * x at delta = 0 is 0.5 * (0, e_k) * x per column.
  bool PlusJacobian(const double* x, double* J) const override {
    for (int k = 0; k < 3; ++k) {
      double e[4] = {0, 0, 0, 0};
      e[1 + k] = 0.5;
      double col[4];
      jba::quat_multiply(e, x, col);
      for (int r = 0; r < 4; ++r) J[3 * r + k] = col[r];
    }
    return true;
  }

  bool Minus(const double* y, const double* x, double* y_minus_x) const override {
    const double xc[4] = {x[0], -x[1], -x[2], -x[3]};
    double d[4];
    jba::quat_multiply(y, xc, d);
    if (d[0] < 0) for (double& v : d) v = -v;
    const double s = std::sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]);
    const double scale = s < 1e-12 ? 2.0 : 2.0 * std::atan2(s, d[0]) / s;
    for (int i = 0; i < 3; ++i) y_minus_x[i] = scale * d[1 + i];
    return true;
  }

  // At y = x, Minus is 2 * vec(y * conj(x)), linear in y.
  bool MinusJacobian(const double* x, double* J) const override {
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
};

// ---------------------------------------------------------------------------
// Cost functor, written independently of the generated code.

template <CameraModel M>
struct Reprojection {
  Reprojection(double ox, double oy, double inv_sigma) : ox(ox), oy(oy), w(inv_sigma) {}

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

ceres::CostFunction* make_cost(CameraModel m, double ox, double oy, double inv_sigma) {
  switch (m) {
    case CameraModel::SimplePinhole: return make_cost<CameraModel::SimplePinhole>(ox, oy, inv_sigma);
    case CameraModel::Pinhole: return make_cost<CameraModel::Pinhole>(ox, oy, inv_sigma);
    case CameraModel::SimpleRadial: return make_cost<CameraModel::SimpleRadial>(ox, oy, inv_sigma);
    case CameraModel::Radial: return make_cost<CameraModel::Radial>(ox, oy, inv_sigma);
    default: throw std::invalid_argument("camera model not supported by the Ceres harness yet");
  }
}

// Indices of intrinsics held constant, from the freed-group mask.
std::vector<int> constant_intrinsics(CameraModel m, std::uint8_t free_mask) {
  std::vector<int> focal, pp, dist;
  switch (m) {
    case CameraModel::SimplePinhole: focal = {0}; pp = {1, 2}; break;
    case CameraModel::Pinhole: focal = {0, 1}; pp = {2, 3}; break;
    case CameraModel::SimpleRadial: focal = {0}; pp = {1, 2}; dist = {3}; break;
    case CameraModel::Radial: focal = {0}; pp = {1, 2}; dist = {3, 4}; break;
    default: break;
  }
  std::vector<int> out;
  if (!(free_mask & jba::kFocal)) out.insert(out.end(), focal.begin(), focal.end());
  if (!(free_mask & jba::kPrincipalPoint)) out.insert(out.end(), pp.begin(), pp.end());
  if (!(free_mask & jba::kDistortion)) out.insert(out.end(), dist.begin(), dist.end());
  return out;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <problem.jba | bal.txt> [--max-iterations N] [--linear-solver S] "
                           "[--output out.jba] [--threads N]\n", argv[0]);
      return 2;
    }
    const std::string input = argv[1];
    int max_iterations = 50;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::string linear = "sparse_schur", output;
    for (int a = 2; a + 1 < argc; a += 2) {
      const std::string key = argv[a], val = argv[a + 1];
      if (key == "--max-iterations") max_iterations = std::stoi(val);
      else if (key == "--linear-solver") linear = val;
      else if (key == "--output") output = val;
      else if (key == "--threads") threads = std::stoi(val);
      else throw std::invalid_argument("unknown option " + key);
    }

    jba::Problem p = ends_with(input, ".txt") ? jba::read_bal(input) : jba::read_problem(input);
    const double jba_initial = jba::cpu::reprojection_cost(p);
    const int np = jba::num_params(p.model);

    ceres::Problem::Options popts;
    popts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(popts);
    auto pose_manifold = std::make_unique<ceres::ProductManifold<LeftRotationVectorQuaternion, ceres::EuclideanManifold<3>>>(
        LeftRotationVectorQuaternion{}, ceres::EuclideanManifold<3>{});
    std::vector<std::unique_ptr<ceres::Manifold>> intrinsics_manifolds;

    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      const std::size_t i = static_cast<std::size_t>(p.obs_image[k]);
      const std::size_t c = static_cast<std::size_t>(p.intrinsics_of_image(i));
      const double sigma = p.obs_sigma.empty() ? 1.0 : p.obs_sigma[k];
      problem.AddResidualBlock(make_cost(p.model, p.obs_xy[2 * k], p.obs_xy[2 * k + 1], 1.0 / sigma), nullptr,
                               &p.poses[7 * i], &p.points[3 * static_cast<std::size_t>(p.obs_point[k])],
                               &p.intrinsics[static_cast<std::size_t>(np) * c]);
    }

    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;
    for (std::size_t i = 0; i < p.num_images(); ++i) {
      double* pose = &p.poses[7 * i];
      if (!problem.HasParameterBlock(pose)) continue;
      problem.SetManifold(pose, pose_manifold.get());
      if (!p.image_fixed.empty() && p.image_fixed[i]) problem.SetParameterBlockConstant(pose);
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
      const auto constant = constant_intrinsics(p.model, mask);
      if (static_cast<int>(constant.size()) == np) {
        problem.SetParameterBlockConstant(K);
      } else if (!constant.empty()) {
        intrinsics_manifolds.push_back(std::make_unique<ceres::SubsetManifold>(np, constant));
        problem.SetManifold(K, intrinsics_manifolds.back().get());
      }
      ordering->AddElementToGroup(K, 1);
    }

    double ceres_initial = 0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &ceres_initial, nullptr, nullptr, nullptr);

    ceres::Solver::Options opts;
    opts.minimizer_type = ceres::TRUST_REGION;
    opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    opts.max_num_iterations = max_iterations;
    opts.num_threads = threads;
    opts.linear_solver_ordering.reset(ordering);
    if (linear == "dense_schur") {
      opts.linear_solver_type = ceres::DENSE_SCHUR;
    } else if (linear == "sparse_schur") {
      opts.linear_solver_type = ceres::SPARSE_SCHUR;
      opts.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    } else if (linear == "iterative_schur") {
      opts.linear_solver_type = ceres::ITERATIVE_SCHUR;
      opts.preconditioner_type = ceres::SCHUR_JACOBI;
    } else {
      throw std::invalid_argument("unknown linear solver " + linear);
    }
    opts.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    if (!output.empty()) jba::write_problem(output, p);

    const double rel = std::abs(ceres_initial - jba_initial) / std::max(1.0, std::abs(ceres_initial));
    std::printf("{\n  \"input\": \"%s\",\n  \"ceres_version\": \"%s\",\n  \"model\": \"%s\",\n", input.c_str(),
                CERES_VERSION_STRING, std::string(jba::model_name(p.model)).c_str());
    std::printf("  \"images\": %zu, \"points\": %zu, \"observations\": %zu,\n", p.num_images(), p.num_points(),
                p.num_observations());
    std::printf("  \"linear_solver\": \"%s\", \"threads\": %d,\n", linear.c_str(), threads);
    std::printf("  \"initial_cost_ceres\": %.17g,\n  \"initial_cost_jba\": %.17g,\n  \"initial_cost_rel_diff\": %.3g,\n",
                ceres_initial, jba_initial, rel);
    std::printf("  \"final_cost\": %.17g,\n  \"iterations\": %zu,\n  \"termination\": \"%s\",\n",
                summary.final_cost, summary.iterations.size() - 1,
                ceres::TerminationTypeToString(summary.termination_type));
    std::printf("  \"time_seconds\": %.4f,\n  \"history\": [", summary.total_time_in_seconds);
    for (std::size_t it = 0; it < summary.iterations.size(); ++it) {
      const auto& s = summary.iterations[it];
      std::printf("%s\n    {\"iteration\": %d, \"cost\": %.17g, \"step_accepted\": %s, \"trust_region_radius\": %.6g, "
                  "\"linear_solver_iterations\": %d}",
                  it ? "," : "", s.iteration, s.cost, s.step_is_successful ? "true" : "false",
                  s.trust_region_radius, s.linear_solver_iterations);
    }
    std::printf("\n  ]\n}\n");

    if (rel > 1e-9) {
      std::fprintf(stderr, "initial cost mismatch between Ceres and just-ba: %.3g\n", rel);
      return 1;
    }
    if (!summary.IsSolutionUsable() || !(summary.final_cost <= summary.initial_cost)) {
      std::fprintf(stderr, "Ceres solve failed: %s\n", summary.message.c_str());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_ceres_solve: %s\n", e.what());
    return 1;
  }
}
