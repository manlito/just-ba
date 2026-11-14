#include "jba/priors.h"

#include <algorithm>
#include <cmath>
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

}  // namespace jba
