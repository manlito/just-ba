// Covariances against the actual spread of errors over repeated noisy runs
// (document 05): the same scene, fresh keypoint noise each run, solved from the
// truth; the predicted variance of each pose coordinate and camera centre must match
// the observed one. (Agreement with Ceres's covariance is in tests/ceres.)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "jba/covariance.h"
#include "jba/geometry.h"
#include "jba/priors.h"
#include "jba/solver.h"
#include "synthetic.h"

int main() {
  const auto truth = jba_test::random_problem(jba::CameraModel::Pinhole, 8, 120, 21, 0.0);
  const int runs = 300;
  const double noise = 0.5;
  const std::size_t n = truth.num_images();
  std::vector<double> observed(6 * n, 0), predicted(6 * n, 0), obs_c(3 * n, 0), pred_c(3 * n, 0);
  std::mt19937_64 rng(22);
  jba_test::Normal gauss(0, noise);
  jba::Gauge gauge;
  for (int run = 0; run < runs; ++run) {
    auto p = truth;
    for (double& v : p.obs_xy) v += gauss(rng);
    p.obs_sigma.assign(p.num_observations(), noise);
    jba::SolverOptions o;
    o.function_tolerance = 1e-12;
    jba::solve(p, o);
    const auto cov = jba::covariance(p, o);
    gauge = cov.gauge;
    for (std::size_t i = 0; i < n; ++i) {
      const double* e = &p.poses[7 * i];
      const double* t = &truth.poses[7 * i];
      const double tc[4] = {t[0], -t[1], -t[2], -t[3]};
      double dq[4];
      jba::quat_multiply(e, tc, dq);  // Exp(dtheta) = R_e R_t^T
      const double s = std::sqrt(dq[1] * dq[1] + dq[2] * dq[2] + dq[3] * dq[3]);
      const double f = s < 1e-15 ? 2.0 : 2 * std::atan2(s, dq[0]) / s;
      const double d[6] = {f * dq[1], f * dq[2], f * dq[3], e[4] - t[4], e[5] - t[5], e[6] - t[6]};
      for (int a = 0; a < 6; ++a) {
        observed[6 * i + static_cast<std::size_t>(a)] += d[a] * d[a] / runs;
        predicted[6 * i + static_cast<std::size_t>(a)] += cov.pose[36 * i + static_cast<std::size_t>(7 * a)] / runs;
      }
      double Ce[3], Ct[3];
      const double ec[4] = {e[0], -e[1], -e[2], -e[3]};
      jba::quat_rotate(ec, e + 4, Ce);
      jba::quat_rotate(tc, t + 4, Ct);
      for (int a = 0; a < 3; ++a) {
        const double dc = Ce[a] - Ct[a];
        obs_c[3 * i + static_cast<std::size_t>(a)] += dc * dc / runs;
        pred_c[3 * i + static_cast<std::size_t>(a)] += cov.centre[9 * i + static_cast<std::size_t>(4 * a)] / runs;
      }
    }
  }
  std::vector<double> ratios;
  for (std::size_t c = 0; c < 6 * n; ++c)
    if (predicted[c] > 0) ratios.push_back(observed[c] / predicted[c]);
  for (std::size_t c = 0; c < 3 * n; ++c)
    if (pred_c[c] > 0) ratios.push_back(obs_c[c] / pred_c[c]);
  std::sort(ratios.begin(), ratios.end());
  const double med = ratios[ratios.size() / 2];
  std::printf("observed / predicted variance over %d runs, %zu coordinates: min %.3f, median %.3f, max %.3f\n", runs,
              ratios.size(), ratios.front(), med, ratios.back());
  // 300 runs estimate a variance to about 8% (one sigma).
  EXPECT(med > 0.9 && med < 1.1);
  EXPECT(ratios.front() > 0.7 && ratios.back() < 1.35);
  // Held coordinates have zero covariance and do not move (up to this test's rounding).
  EXPECT(gauge.fixed_images.size() == 1);
  const auto ref = static_cast<std::size_t>(gauge.fixed_images[0]);
  for (int a = 0; a < 6; ++a)
    EXPECT(predicted[6 * ref + static_cast<std::size_t>(a)] == 0 && observed[6 * ref + static_cast<std::size_t>(a)] < 1e-30);

  // A system singular at a group prior's mean (D165: ties so loose their weight
  // underflows) is refused with the block named, as any other block is.
  {
    auto p = truth;
    const std::vector<double> block = p.intrinsics;
    p.intrinsics.clear();
    for (std::size_t i = 0; i < n; ++i) p.intrinsics.insert(p.intrinsics.end(), block.begin(), block.end());
    p.image_intrinsics.resize(n);
    for (std::size_t i = 0; i < n; ++i) p.image_intrinsics[i] = static_cast<std::int32_t>(i);
    p.intrinsics_free.assign(n, jba::kFocal);
    const double inf = std::numeric_limits<double>::infinity();
    jba::set_intrinsics_groups(p, std::vector<std::int32_t>(n, 0), inf, inf, inf, inf, inf, inf);
    p.intrinsics_group_sigma[0] = p.intrinsics_group_sigma[1] = 1e200;
    jba::SolverOptions o;
    jba::solve(p, o);
    std::string what;
    try {
      jba::covariance(p, o);
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    std::printf("singular at a group's mean: %s\n", what.c_str());
    EXPECT(what.find("mean of intrinsics group 0") != std::string::npos);
  }
  return TEST_MAIN_RESULT();
}
