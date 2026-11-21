#include "jba/priors.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "jba/geometry.h"

namespace jba {
namespace {

void diagonal(double rot_sigma, double trans_sigma, double* W) {
  std::fill(W, W + 36, 0.0);
  const double wr = 1 / rot_sigma, wt = 1 / trans_sigma;  // 1 / inf = 0: part left out
  for (int a = 0; a < 3; ++a) {
    W[7 * a] = wr;
    W[7 * (3 + a)] = wt;
  }
}

double rotation_angle(const double* q) {
  const double v = std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return 2 * std::atan2(v, std::abs(q[0]));
}

}  // namespace

void prior_whitening(const Problem& p, const PriorDefaults& d, std::vector<double>& relative,
                     std::vector<double>& global) {
  const std::size_t n_rel = p.num_relative_priors(), n_glob = p.num_global_priors();
  relative.assign(36 * n_rel, 0.0);
  global.assign(36 * n_glob, 0.0);
  for (std::size_t k = 0; k < n_rel; ++k) {
    double* W = &relative[36 * k];
    if (!p.rel_sqrt_info.empty()) {
      std::copy_n(&p.rel_sqrt_info[36 * k], 36, W);
    } else if (!p.rel_sigma.empty()) {
      diagonal(p.rel_sigma[2 * k], p.rel_sigma[2 * k + 1], W);
    } else {
      const double* T = &p.rel_pose[7 * k];
      // |t| of T_j T_i^-1 is the distance between the camera centres.
      const double dist = std::sqrt(T[4] * T[4] + T[5] * T[5] + T[6] * T[6]);
      diagonal(d.relative_rotation_sigma + d.relative_rotation_per_radian * rotation_angle(T),
               d.relative_translation_sigma + d.relative_translation_per_metre * dist, W);
    }
  }
  for (std::size_t k = 0; k < n_glob; ++k) {
    double* W = &global[36 * k];
    if (!p.glob_sqrt_info.empty()) {
      std::copy_n(&p.glob_sqrt_info[36 * k], 36, W);
    } else if (!p.glob_sigma.empty()) {
      diagonal(p.glob_sigma[2 * k], p.glob_sigma[2 * k + 1], W);
    } else {
      diagonal(d.global_rotation_sigma, d.global_position_sigma, W);
    }
  }
}

void add_relative_priors_from_poses(Problem& p, const std::vector<double>& poses,
                                    const std::vector<std::int32_t>& pairs) {
  if (pairs.size() % 2 != 0) throw std::invalid_argument("pairs must hold 2 values per pair");
  if (!p.rel_sigma.empty() || !p.rel_sqrt_info.empty())
    throw std::invalid_argument("existing relative priors carry weights; add these with weights too");
  const std::size_t n = poses.size() / 7;
  for (std::size_t k = 0; k < pairs.size(); k += 2) {
    const auto i = pairs[k], j = pairs[k + 1];
    if (i < 0 || j < 0 || static_cast<std::size_t>(i) >= n || static_cast<std::size_t>(j) >= n)
      throw std::invalid_argument("pair refers to a missing image");
    const double* Ti = &poses[7 * static_cast<std::size_t>(i)];
    const double* Tj = &poses[7 * static_cast<std::size_t>(j)];
    // T_j T_i^-1 = (R_j R_i^T, t_j - R_j R_i^T t_i).
    const double qi_conj[4] = {Ti[0], -Ti[1], -Ti[2], -Ti[3]};
    double q[4], Rt[3];
    quat_multiply(Tj, qi_conj, q);
    quat_normalize(q);
    quat_rotate(q, Ti + 4, Rt);
    p.rel_images.insert(p.rel_images.end(), {i, j});
    p.rel_pose.insert(p.rel_pose.end(), {q[0], q[1], q[2], q[3], Tj[4] - Rt[0], Tj[5] - Rt[1], Tj[6] - Rt[2]});
  }
}

void set_intrinsics_prior(Problem& p, double focal_fraction, double principal_point_sigma, double distortion_sigma) {
  for (double v : {focal_fraction, principal_point_sigma, distortion_sigma})
    if (!(v > 0)) throw std::invalid_argument("set_intrinsics_prior: sigmas must be positive (+inf for none)");
  const int np = num_params(p.model);
  p.intrinsics_prior = p.intrinsics;
  p.intrinsics_prior_sigma.resize(p.intrinsics.size());
  for (std::size_t at = 0; at < p.intrinsics.size(); ++at) {
    double& s = p.intrinsics_prior_sigma[at];
    switch (intrinsics_group(p.model, static_cast<int>(at % static_cast<std::size_t>(np)))) {
      case kFocal: s = focal_fraction * std::abs(p.intrinsics[at]); break;
      case kPrincipalPoint: s = principal_point_sigma; break;
      case kDistortion: s = distortion_sigma; break;
    }
    if (!(s > 0) || !std::isfinite(1 / s))
      throw std::invalid_argument("set_intrinsics_prior: a sigma of 0 or too small (a focal length of 0?)");
  }
}

bool IntrinsicsGroupTerms::any() const {
  return std::any_of(member.begin(), member.end(), [](std::uint8_t m) { return m != 0; });
}

IntrinsicsGroupTerms intrinsics_group_terms(const Problem& p) {
  IntrinsicsGroupTerms t;
  if (p.intrinsics_group.empty()) return t;
  const auto np = static_cast<std::size_t>(num_params(p.model));
  const std::size_t n_blk = p.num_intrinsics_blocks(), G = p.num_intrinsics_groups();
  t.groups = G;
  t.member.assign(n_blk, 0);
  t.tie_w.assign(G * np, 0.0);
  t.anchor_w.assign(G * np, 0.0);
  t.mean.assign(G * np, std::nan(""));
  t.target.assign(G * np, std::nan(""));
  // Members: grouped blocks used by an observation, free or not.
  std::vector<char> used(n_blk, 0);
  for (auto i : p.obs_image) used[static_cast<std::size_t>(p.intrinsics_of_image(static_cast<std::size_t>(i)))] = 1;
  std::vector<std::size_t> count(G, 0);
  std::vector<double> sum(G * np, 0.0);
  std::vector<char> free_any(G * np, 0);
  for (std::size_t c = 0; c < n_blk; ++c) {
    const std::int32_t g = p.intrinsics_group[c];
    if (g < 0 || !used[c]) continue;
    const auto gs = static_cast<std::size_t>(g);
    ++count[gs];
    const std::uint8_t mask = p.intrinsics_free.empty() ? 0 : p.intrinsics_free[c];
    for (std::size_t a = 0; a < np; ++a) {
      sum[gs * np + a] += p.intrinsics[c * np + a];
      if (mask & intrinsics_group(p.model, static_cast<int>(a))) free_any[gs * np + a] = 1;
    }
  }
  std::vector<char> tied(G, 0);
  for (std::size_t g = 0; g < G; ++g) {
    if (count[g] == 0) continue;
    for (std::size_t a = 0; a < np; ++a) {
      const std::size_t at = g * np + a;
      const double members_mean = sum[at] / static_cast<double>(count[g]);
      const double sigma = p.intrinsics_group_sigma[at];
      const double anchor = p.intrinsics_group_prior_sigma.empty() ? std::numeric_limits<double>::infinity()
                                                                    : p.intrinsics_group_prior_sigma[at];
      // A single member without a target has nothing to share: the mean would follow it.
      if (!free_any[at] || !std::isfinite(sigma) || (count[g] < 2 && !std::isfinite(anchor))) continue;
      tied[g] = 1;
      t.tie_w[at] = 1 / sigma;
      const double given_mean = p.intrinsics_group_mean.empty() ? std::nan("") : p.intrinsics_group_mean[at];
      t.mean[at] = std::isnan(given_mean) ? members_mean : given_mean;
      if (std::isfinite(anchor)) {
        t.anchor_w[at] = 1 / anchor;
        const double given = p.intrinsics_group_prior.empty() ? std::nan("") : p.intrinsics_group_prior[at];
        t.target[at] = std::isnan(given) ? members_mean : given;
      }
    }
  }
  for (std::size_t c = 0; c < n_blk; ++c) {
    const std::int32_t g = p.intrinsics_group[c];
    if (g >= 0 && used[c] && tied[static_cast<std::size_t>(g)]) t.member[c] = 1;
  }
  return t;
}

void set_intrinsics_groups(Problem& p, const std::vector<std::int32_t>& group_of_block, double focal_fraction,
                           double principal_point_sigma, double distortion_sigma, double group_focal_fraction,
                           double group_principal_point_sigma, double group_distortion_sigma) {
  for (double v : {focal_fraction, principal_point_sigma, distortion_sigma, group_focal_fraction,
                   group_principal_point_sigma, group_distortion_sigma})
    if (!(v > 0)) throw std::invalid_argument("set_intrinsics_groups: sigmas must be positive (+inf for none)");
  const auto np = static_cast<std::size_t>(num_params(p.model));
  const std::size_t n_blk = p.num_intrinsics_blocks();
  if (group_of_block.size() != n_blk) throw std::invalid_argument("set_intrinsics_groups: one group per block");
  std::int32_t last = -1;
  for (auto g : group_of_block) {
    if (g < -1) throw std::invalid_argument("set_intrinsics_groups: a group must be -1 or more");
    last = std::max(last, g);
  }
  const auto G = static_cast<std::size_t>(last + 1);
  // Each group's mean over all its blocks: the target and the scale of focal sigmas.
  std::vector<double> mean(G * np, 0.0);
  std::vector<std::size_t> count(G, 0);
  for (std::size_t c = 0; c < n_blk; ++c) {
    if (group_of_block[c] < 0) continue;
    const auto g = static_cast<std::size_t>(group_of_block[c]);
    ++count[g];
    for (std::size_t a = 0; a < np; ++a) mean[g * np + a] += p.intrinsics[c * np + a];
  }
  p.intrinsics_group = group_of_block;
  p.intrinsics_group_sigma.assign(G * np, std::numeric_limits<double>::infinity());
  p.intrinsics_group_prior_sigma.assign(G * np, std::numeric_limits<double>::infinity());
  p.intrinsics_group_prior.assign(G * np, std::nan(""));
  p.intrinsics_group_mean.clear();
  for (std::size_t g = 0; g < G; ++g) {
    if (count[g] == 0) continue;  // no block: no tie, nothing to fill
    for (std::size_t a = 0; a < np; ++a) {
      const std::size_t at = g * np + a;
      mean[at] /= static_cast<double>(count[g]);
      p.intrinsics_group_prior[at] = mean[at];
      double& s = p.intrinsics_group_sigma[at];
      double& sg = p.intrinsics_group_prior_sigma[at];
      switch (intrinsics_group(p.model, static_cast<int>(a))) {
        case kFocal: s = focal_fraction * std::abs(mean[at]); sg = group_focal_fraction * std::abs(mean[at]); break;
        case kPrincipalPoint: s = principal_point_sigma; sg = group_principal_point_sigma; break;
        case kDistortion: s = distortion_sigma; sg = group_distortion_sigma; break;
      }
      if (!(s > 0) || !std::isfinite(1 / s) || !(sg > 0) || !std::isfinite(1 / sg))
        throw std::invalid_argument("set_intrinsics_groups: a sigma of 0 or too small (a focal length of 0?)");
    }
  }
}

double depth_residual_sigma(const Problem& p, std::size_t k, const PriorDefaults& d) {
  const double reading = p.obs_depth[k];
  const double sigma = p.obs_depth_sigma.empty() ? d.depth_sigma + d.depth_sigma_per_metre * reading : p.obs_depth_sigma[k];
  return p.depth_form == DepthForm::Inverse ? sigma / (reading * reading) : sigma;
}

}  // namespace jba
