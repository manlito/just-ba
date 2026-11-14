#include "jba/report.h"

#include <cmath>
#include <stdexcept>

#include "jba/cpu/priors.h"
#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "jba/loss.h"
#include "jba/priors.h"

namespace jba {

ResidualReport evaluate(const Problem& p, const SolverOptions& o, double threshold) {
  if (auto err = validate(p); !err.empty()) throw std::invalid_argument("invalid problem: " + err);
  ResidualReport r;
  const int np = num_params(p.model);

  const std::size_t n_obs = p.num_observations();
  r.reprojection_error.resize(n_obs);
  r.in_front.resize(n_obs);
  for (std::size_t k = 0; k < n_obs; ++k) {
    const auto i = static_cast<std::size_t>(p.obs_image[k]);
    const double* pose = &p.poses[7 * i];
    const double* X = &p.points[3 * static_cast<std::size_t>(p.obs_point[k])];
    double res[2];
    cpu::reprojection<double>(p.model, pose, pose + 4, X,
                              &p.intrinsics[static_cast<std::size_t>(np) * static_cast<std::size_t>(p.intrinsics_of_image(i))],
                              &p.obs_xy[2 * k], res, nullptr, nullptr, nullptr);
    double pc[3];
    quat_rotate(pose, X, pc);
    r.in_front[k] = pc[2] + pose[6] > 0;
    const double e2 = res[0] * res[0] + res[1] * res[1];
    r.reprojection_error[k] = std::sqrt(e2);
    const double w = p.obs_sigma.empty() ? 1.0 : 1.0 / p.obs_sigma[k];
    double rho[3];
    evaluate_loss(o.loss, e2 * w * w, rho);
    r.reprojection_cost += 0.5 * rho[0];
  }

  std::vector<double> W_rel, W_glob;
  prior_whitening(p, o.prior_defaults, W_rel, W_glob);
  auto whitened_norm2 = [](const double* W, const double* res) {
    double s = 0;
    for (int a = 0; a < 6; ++a) {
      double v = 0;
      for (int c = 0; c < 6; ++c) v += W[6 * a + c] * res[c];
      s += v * v;
    }
    return s;
  };
  const std::size_t n_rel = p.num_relative_priors(), n_glob = p.num_global_priors();
  r.relative_prior_norm.resize(n_rel);
  r.relative_prior_weight.resize(n_rel);
  for (std::size_t k = 0; k < n_rel; ++k) {
    double res[6];
    cpu::relative_prior<double>(&p.poses[7 * static_cast<std::size_t>(p.rel_images[2 * k])],
                                &p.poses[7 * static_cast<std::size_t>(p.rel_images[2 * k + 1])], &p.rel_pose[7 * k], res,
                                nullptr, nullptr);
    const double s = whitened_norm2(&W_rel[36 * k], res);
    double rho[3];
    evaluate_loss(o.relative_prior_loss, s, rho);
    r.relative_prior_cost += 0.5 * rho[0];
    r.relative_prior_norm[k] = std::sqrt(s);
    r.relative_prior_weight[k] = rho[1];
    if (rho[1] < threshold) r.downweighted_relative.push_back(static_cast<std::int32_t>(k));
  }
  r.global_prior_norm.resize(n_glob);
  r.global_prior_weight.resize(n_glob);
  static constexpr double kId[8] = {1, 0, 0, 0, 0, 0, 0, 1};
  for (std::size_t k = 0; k < n_glob; ++k) {
    const auto s_idx = static_cast<std::size_t>(p.set_of_global_prior(k));
    const double* T = p.set_transform.empty() || p.relation_of_set(s_idx) == FrameRelation::Identity
                          ? kId
                          : &p.set_transform[8 * s_idx];
    const double* g = &p.glob_pose[7 * k];
    const double gc[4] = {g[0], -g[1], -g[2], -g[3]};
    double C[3];
    quat_rotate(gc, g + 4, C);
    for (double& v : C) v = -v;
    double res[6];
    cpu::global_prior<double>(&p.poses[7 * static_cast<std::size_t>(p.glob_image[k])], T, g, C, res, nullptr, nullptr,
                              nullptr);
    const double s = whitened_norm2(&W_glob[36 * k], res);
    double rho[3];
    evaluate_loss(o.global_prior_loss, s, rho);
    r.global_prior_cost += 0.5 * rho[0];
    r.global_prior_norm[k] = std::sqrt(s);
    r.global_prior_weight[k] = rho[1];
    if (rho[1] < threshold) r.downweighted_global.push_back(static_cast<std::int32_t>(k));
  }
  return r;
}

}  // namespace jba
