#include "jba/cpu/cost.h"

#include <cstddef>

#include "jba/cpu/reprojection.h"

namespace jba::cpu {

double reprojection_cost(const Problem& p) {
  const std::size_t np = static_cast<std::size_t>(num_params(p.model));
  // Summed in fixed blocks, so the order will not depend on thread count later.
  constexpr std::size_t kBlock = 1024;
  double total = 0.0;
  const std::size_t n = p.num_observations();
  for (std::size_t begin = 0; begin < n; begin += kBlock) {
    const std::size_t end = begin + kBlock < n ? begin + kBlock : n;
    double block = 0.0;
    for (std::size_t k = begin; k < end; ++k) {
      const std::size_t i = static_cast<std::size_t>(p.obs_image[k]);
      const double* pose = &p.poses[7 * i];
      const double* intr = &p.intrinsics[np * static_cast<std::size_t>(p.intrinsics_of_image(i))];
      const double* X = &p.points[3 * static_cast<std::size_t>(p.obs_point[k])];
      double r[2] = {0, 0};
      reprojection<double>(p.model, pose, pose + 4, X, intr, &p.obs_xy[2 * k], r, nullptr, nullptr, nullptr);
      const double s = p.obs_sigma.empty() ? 1.0 : p.obs_sigma[k];
      block += (r[0] * r[0] + r[1] * r[1]) / (s * s);
    }
    total += block;
  }
  return 0.5 * total;
}

}  // namespace jba::cpu
