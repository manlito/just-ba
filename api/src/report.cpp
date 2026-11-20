#include "jba/report.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "jba/cpu/depth.h"
#include "jba/cpu/priors.h"
#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "jba/loss.h"
#include "jba/manhattan.h"
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

  // Priors on free intrinsics, with the solver's rules (D161): blocks used by an
  // observation, free parameters with a finite sigma.
  if (!p.intrinsics_prior_sigma.empty()) {
    const std::size_t n_blk = p.num_intrinsics_blocks();
    const std::vector<double>& target = p.intrinsics_prior.empty() ? p.intrinsics : p.intrinsics_prior;
    std::vector<char> used(n_blk, 0);
    for (std::size_t k = 0; k < n_obs; ++k) used[static_cast<std::size_t>(p.intrinsics_of_image(static_cast<std::size_t>(p.obs_image[k])))] = 1;
    r.intrinsics_prior_norm.assign(n_blk, 0.0);
    r.intrinsics_prior_param.assign(n_blk, -1);
    for (std::size_t c = 0; c < n_blk; ++c) {
      const std::uint8_t mask = p.intrinsics_free.empty() ? 0 : p.intrinsics_free[c];
      if (!used[c] || mask == 0) continue;
      for (int a = 0; a < np; ++a) {
        const std::size_t at = static_cast<std::size_t>(np) * c + static_cast<std::size_t>(a);
        const double sigma = p.intrinsics_prior_sigma[at];
        if (!(mask & intrinsics_group(p.model, a)) || !std::isfinite(sigma)) continue;
        const double v = std::isnan(target[at]) ? 0.0 : (p.intrinsics[at] - target[at]) / sigma;
        r.intrinsics_prior_cost += 0.5 * v * v;
        if (r.intrinsics_prior_param[c] < 0 || std::abs(v) > r.intrinsics_prior_norm[c]) {
          r.intrinsics_prior_norm[c] = std::abs(v);
          r.intrinsics_prior_param[c] = a;
        }
      }
    }
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

  double gw[3];
  gravity_direction(p, gw);
  const std::size_t n_grav = p.num_gravity_priors();
  r.gravity_angle.resize(n_grav);
  r.gravity_weight.resize(n_grav);
  for (std::size_t k = 0; k < n_grav; ++k) {
    double res[3];
    cpu::gravity_prior<double>(&p.poses[7 * static_cast<std::size_t>(p.grav_image[k])], gw, &p.grav_dir[3 * k], res,
                               static_cast<double*>(nullptr));
    const double chord = std::sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
    const double w = 1 / (p.grav_sigma.empty() ? o.prior_defaults.gravity_sigma : p.grav_sigma[k]);
    double rho[3];
    evaluate_loss(o.gravity_loss, chord * chord * w * w, rho);
    r.gravity_cost += 0.5 * rho[0];
    r.gravity_angle[k] = 2 * std::asin(std::min(1.0, chord / 2));
    r.gravity_weight[k] = rho[1];
    if (rho[1] < threshold) r.downweighted_gravity.push_back(static_cast<std::int32_t>(k));
  }

  if (!p.obs_depth.empty()) {
    r.depth_refined.assign(n_obs, std::nan(""));
    r.depth_corrected.assign(n_obs, std::nan(""));
    r.depth_weight.assign(n_obs, 0.0);
    for (std::size_t k = 0; k < n_obs; ++k) {
      const double d = p.obs_depth[k];
      if (std::isnan(d)) continue;
      const auto i = static_cast<std::size_t>(p.obs_image[k]);
      const double c[2] = {p.depth_correction.empty() ? 1.0 : p.depth_correction[2 * i],
                           p.depth_correction.empty() ? 0.0 : p.depth_correction[2 * i + 1]};
      const double* pose = &p.poses[7 * i];
      const double* X = &p.points[3 * static_cast<std::size_t>(p.obs_point[k])];
      double res;
      cpu::depth_residual<double>(p.depth_kind, p.depth_form, pose, X, d, c, &res, nullptr, nullptr, nullptr);
      const double w = 1 / depth_residual_sigma(p, k, o.prior_defaults);
      double rho[3];
      evaluate_loss(o.depth_loss, res * res * w * w, rho);
      r.depth_cost += 0.5 * rho[0];
      double pc[3];
      quat_rotate(pose, X, pc);
      for (int a = 0; a < 3; ++a) pc[a] += pose[4 + a];
      r.depth_refined[k] = p.depth_kind == DepthKind::Range ? std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]) : pc[2];
      r.depth_corrected[k] = c[0] * d + c[1];
      r.depth_weight[k] = rho[1];
    }
    // Priors on free corrections, as in the solver: images with a reading.
    std::vector<char> has_reading(p.num_images(), 0);
    for (std::size_t k = 0; k < n_obs; ++k)
      if (!std::isnan(p.obs_depth[k])) has_reading[static_cast<std::size_t>(p.obs_image[k])] = 1;
    for (std::size_t i = 0; i < p.depth_correction_free.size(); ++i) {
      const std::uint8_t m = p.depth_correction_free[i];
      if (!has_reading[i] || m == 0 || p.depth_correction.empty()) continue;
      if (m & kDepthScale) {
        const double v = (p.depth_correction[2 * i] - 1) / o.prior_defaults.depth_scale_sigma;
        r.depth_cost += 0.5 * v * v;
      }
      if (m & kDepthOffset) {
        const double v = p.depth_correction[2 * i + 1] / o.prior_defaults.depth_offset_sigma;
        r.depth_cost += 0.5 * v * v;
      }
    }
  }

  const std::size_t n_lines = p.num_lines();
  r.line_angle.assign(n_lines, std::nan(""));
  r.line_weight.assign(n_lines, 0.0);
  if (p.has_manhattan()) {
    for (std::size_t k = 0; k < n_lines; ++k) {
      double n[3];
      if (p.line_axis[k] < 0 || !line_normal(p, k, n)) continue;
      double axis[3] = {0, 0, 0}, res;
      axis[p.line_axis[k]] = 1;
      cpu::manhattan_line<double>(&p.poses[7 * static_cast<std::size_t>(p.line_image[k])], n, p.manhattan_frame.data(),
                                  axis, &res, static_cast<double*>(nullptr), static_cast<double*>(nullptr));
      const double w = 1 / line_residual_sigma(p, k, o.prior_defaults);
      double rho[3];
      evaluate_loss(o.line_loss, res * res * w * w, rho);
      r.line_cost += 0.5 * rho[0];
      r.line_angle[k] = std::asin(std::min(1.0, std::abs(res)));
      r.line_weight[k] = rho[1];
    }
  }
  return r;
}

}  // namespace jba
