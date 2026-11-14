#include <algorithm>
#include <cmath>
#include <vector>

#include "jba/geometry.h"
#include "jba/loss.h"
#include "jba/solver.h"

namespace jba {

std::string_view loss_name(LossType type) {
  switch (type) {
    case LossType::None: return "none";
    case LossType::Huber: return "huber";
    case LossType::Cauchy: return "cauchy";
    case LossType::GemanMcClure: return "geman_mcclure";
  }
  return "none";
}

std::optional<LossType> loss_from_name(std::string_view name) {
  for (LossType t : {LossType::None, LossType::Huber, LossType::Cauchy, LossType::GemanMcClure}) {
    if (loss_name(t) == name) return t;
  }
  return std::nullopt;
}

std::string_view termination_name(Termination t) {
  switch (t) {
    case Termination::Convergence: return "convergence";
    case Termination::NoConvergence: return "no_convergence";
    case Termination::Failure: return "failure";
    case Termination::Cancelled: return "cancelled";
  }
  return "failure";
}

Gauge choose_gauge(const Problem& p) {
  Gauge g;
  const bool any_image_fixed = std::any_of(p.image_fixed.begin(), p.image_fixed.end(), [](auto v) { return v != 0; });
  const bool any_point_fixed = std::any_of(p.point_fixed.begin(), p.point_fixed.end(), [](auto v) { return v != 0; });
  if (any_image_fixed || any_point_fixed) return g;

  const std::size_t n_img = p.num_images();
  std::vector<std::size_t> count(n_img, 0);
  for (auto i : p.obs_image) ++count[static_cast<std::size_t>(i)];
  std::size_t ref = 0;
  for (std::size_t i = 1; i < n_img; ++i) {
    if (count[i] > count[ref]) ref = i;
  }
  if (n_img < 2 || count[ref] == 0) return g;

  // Points seen by the reference image, then shared counts per other image.
  std::vector<char> seen(p.num_points(), 0);
  for (std::size_t k = 0; k < p.num_observations(); ++k) {
    if (static_cast<std::size_t>(p.obs_image[k]) == ref) seen[static_cast<std::size_t>(p.obs_point[k])] = 1;
  }
  std::vector<std::size_t> shared(n_img, 0);
  for (std::size_t k = 0; k < p.num_observations(); ++k) {
    const auto i = static_cast<std::size_t>(p.obs_image[k]);
    if (i != ref && seen[static_cast<std::size_t>(p.obs_point[k])]) ++shared[i];
  }
  std::size_t second = ref;
  for (std::size_t i = 0; i < n_img; ++i) {
    if (i == ref) continue;
    if (second == ref || shared[i] > shared[second]) second = i;
  }
  if (second == ref || shared[second] == 0) return g;

  // With the reference pose fixed, the remaining freedom is a scaling about its
  // centre C1, which moves t2 along R2 (C2 - C1) = -t2 - R2 C1. Hold the largest
  // coordinate of that direction.
  const double* p1 = &p.poses[7 * ref];
  const double* p2 = &p.poses[7 * second];
  const double conj1[4] = {p1[0], -p1[1], -p1[2], -p1[3]};
  double C1[3];
  jba::quat_rotate(conj1, p1 + 4, C1);
  for (double& v : C1) v = -v;
  double R2C1[3];
  jba::quat_rotate(p2, C1, R2C1);
  int coord = 0;
  double best = -1;
  for (int c = 0; c < 3; ++c) {
    const double v = std::abs(-p2[4 + c] - R2C1[c]);
    if (v > best) {
      best = v;
      coord = c;
    }
  }
  g.reference_image = static_cast<int>(ref);
  g.scale_image = static_cast<int>(second);
  g.scale_coordinate = coord;
  return g;
}

}  // namespace jba
