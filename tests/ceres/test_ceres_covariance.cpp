// Covariances against Ceres's covariance estimation (dense SVD) on the same solved
// problem, with the same gauge and losses (document 05).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "ceres_reference.h"
#include "check.h"
#include "jba/covariance.h"
#include "jba/geometry.h"
#include "jba/priors.h"
#include "synthetic.h"

namespace {

double worst = 0;

void compare(const char* name, jba::Problem p, jba::SolverOptions o) {
  o.function_tolerance = 1e-14;
  o.parameter_tolerance = 1e-14;
  o.max_iterations = 200;
  jba::solve(p, o);
  jba::CovarianceOptions co;
  co.points = {0, 7, 19};
  const auto ours = jba::covariance(p, o, co);
  auto copts = jba_ceres::options_from(o, p, jba_ceres::LinearSolver::DenseSchur);
  copts.solve = false;
  copts.covariance = true;
  copts.covariance_points = {0, 7, 19};
  copts.num_threads = 1;
  auto q = p;
  const auto ref = jba_ceres::solve(q, copts);
  double here = 0;
  int compared = 0;
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    const double* a = &ours.pose[36 * i];
    const double* b = &ref.pose_covariance[36 * i];
    if (std::isnan(b[0])) continue;
    double scale = 0, diff = 0;
    for (int k = 0; k < 36; ++k) {
      scale = std::max(scale, std::abs(b[k]));
      diff = std::max(diff, std::abs(a[k] - b[k]));
    }
    if (scale == 0) {
      EXPECT(diff == 0);
      continue;
    }
    here = std::max(here, diff / scale);
    ++compared;
  }
  for (std::size_t k = 0; k < co.points.size(); ++k) {
    double scale = 0, diff = 0;
    for (int a = 0; a < 9; ++a) {
      scale = std::max(scale, std::abs(ref.point_covariance[9 * k + static_cast<std::size_t>(a)]));
      diff = std::max(diff, std::abs(ours.point[9 * k + static_cast<std::size_t>(a)] - ref.point_covariance[9 * k + static_cast<std::size_t>(a)]));
    }
    here = std::max(here, diff / scale);
  }
  std::printf("%-34s %d poses and 3 points, worst relative difference %.3g\n", name, compared, here);
  EXPECT(compared > 0);
  EXPECT(here < 1e-7);
  worst = std::max(worst, here);
}

void perturb(jba::Problem& p, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  jba_test::Normal n(0, 1);
  for (double& v : p.points) v += 0.01 * n(rng);
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    const double step[6] = {0.005 * n(rng), 0.005 * n(rng), 0.005 * n(rng), 0.01 * n(rng), 0.01 * n(rng), 0.01 * n(rng)};
    double moved[7];
    jba::pose_retract(&p.poses[7 * i], step, moved);
    std::copy_n(moved, 7, &p.poses[7 * i]);
  }
}

}  // namespace

int main() {
  using jba::CameraModel;
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 10, 200, 3);
    perturb(p, 4);
    compare("reprojection, automatic gauge", p, {});
    jba::SolverOptions o;
    o.loss = {jba::LossType::Cauchy, 1.0};
    compare("reprojection, Cauchy", p, o);
  }
  {
    auto p = jba_test::random_problem(CameraModel::Radial, 10, 200, 5);
    p.intrinsics_free = {jba::kFocal | jba::kDistortion};
    perturb(p, 6);
    compare("free shared intrinsics", p, {});
  }
  {
    // A prior on the free intrinsics (D161): toward values away from the start, and
    // with one parameter free but without a prior.
    auto p = jba_test::random_problem(CameraModel::Radial, 10, 200, 14);
    p.intrinsics_free = {jba::kFocal | jba::kDistortion};
    jba::set_intrinsics_prior(p, 0.002, std::numeric_limits<double>::infinity(), 0.001);
    p.intrinsics_prior_sigma[4] = std::numeric_limits<double>::infinity();  // k2 fully free
    p.intrinsics[0] *= 1.01;
    perturb(p, 15);
    compare("free intrinsics with a prior", p, {});
  }
  {
    // The group prior (D165): five blocks of two images each in one group, focal and
    // k1 tied to the mean, the mean's focal tied to a target away from the start.
    auto p = jba_test::random_problem(CameraModel::SimpleRadial, 10, 200, 16);
    const std::vector<double> block = p.intrinsics;
    p.intrinsics.clear();
    for (int c = 0; c < 5; ++c) p.intrinsics.insert(p.intrinsics.end(), block.begin(), block.end());
    p.image_intrinsics.resize(p.num_images());
    for (std::size_t i = 0; i < p.num_images(); ++i) p.image_intrinsics[i] = static_cast<std::int32_t>(i / 2);
    p.intrinsics_free.assign(5, jba::kFocal | jba::kDistortion);
    const double inf = std::numeric_limits<double>::infinity();
    jba::set_intrinsics_groups(p, {0, 0, 0, 0, 0}, 0.003, inf, 0.001, 0.02, inf, inf);
    p.intrinsics_group_prior = {1.01 * block[0], std::nan(""), std::nan(""), std::nan("")};  // away from the start
    perturb(p, 17);
    compare("free intrinsics in a group", p, {});
  }
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 10, 200, 7);
    std::vector<std::int32_t> pairs;
    for (int i = 0; i < 10; ++i) pairs.insert(pairs.end(), {i, (i + 1) % 10});
    jba::add_relative_priors_from_poses(p, p.poses, pairs);
    p.glob_image = {3, 6};
    p.glob_pose.assign(p.poses.begin() + 21, p.poses.begin() + 28);
    p.glob_pose.insert(p.glob_pose.end(), p.poses.begin() + 42, p.poses.begin() + 49);
    p.set_relation = {jba::FrameRelation::Similarity};
    perturb(p, 8);
    compare("relative priors, similarity set", p, {});
  }
  {
    auto p = jba_test::random_problem(CameraModel::SimpleRadial, 10, 200, 9);
    p.obs_depth.assign(p.num_observations(), std::nan(""));
    std::mt19937_64 rng(10);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      if (rng() % 2) continue;
      const double* T = &p.poses[7 * static_cast<std::size_t>(p.obs_image[k])];
      double pc[3];
      jba::quat_rotate(T, &p.points[3 * static_cast<std::size_t>(p.obs_point[k])], pc);
      p.obs_depth[k] = pc[2] + T[6];
    }
    p.depth_correction_free.assign(p.num_images(), jba::kDepthScale | jba::kDepthOffset);
    perturb(p, 11);
    compare("depth, free corrections", p, {});
  }
  {
    // Gravity priors: the automatic gauge holds the reference image's heading and
    // translation (D83), so its rotation covariance is rank 2, in the pose tangent.
    auto p = jba_test::random_problem(CameraModel::Pinhole, 10, 200, 12);
    p.gravity_world = {0.0, 1.0, 0.0};
    for (std::size_t i = 0; i < p.num_images(); ++i) {
      double g[3];
      jba::quat_rotate(&p.poses[7 * i], p.gravity_world.data(), g);
      p.grav_image.push_back(static_cast<std::int32_t>(i));
      p.grav_dir.insert(p.grav_dir.end(), g, g + 3);
    }
    perturb(p, 13);
    compare("gravity priors, heading held", p, {});
  }
  std::printf("worst relative difference %.3g\n", worst);
  return TEST_MAIN_RESULT();
}
