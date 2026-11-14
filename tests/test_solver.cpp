// Solver behaviour that needs no reference solver: convergence on exact data,
// cancellation, determinism across thread counts, gauge and fixed blocks honoured,
// fixed intrinsics groups untouched, robust loss down-weights outliers.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "jba/geometry.h"
#include "jba/solver.h"
#include "synthetic.h"

using jba::CameraModel;

namespace {

// Perturbs points and poses of a problem in place (all of them; callers re-fix).
void perturb(jba::Problem& p, std::uint64_t seed, double point_sigma = 0.02, double rot_sigma = 0.01,
             double trans_sigma = 0.02) {
  std::mt19937_64 rng(seed);
  jba_test::Normal n(0, 1);
  for (double& v : p.points) v += point_sigma * n(rng);
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    double* pose = &p.poses[7 * i];
    const double step[6] = {rot_sigma * n(rng), rot_sigma * n(rng), rot_sigma * n(rng),
                            trans_sigma * n(rng), trans_sigma * n(rng), trans_sigma * n(rng)};
    double moved[7];
    jba::pose_retract(pose, step, moved);
    for (int a = 0; a < 7; ++a) pose[a] = moved[a];
  }
}

}  // namespace

int main() {
  // Exact observations, perturbed start: the cost goes to zero.
  for (CameraModel model : {CameraModel::SimplePinhole, CameraModel::Pinhole, CameraModel::SimpleRadial,
                            CameraModel::Radial}) {
    auto p = jba_test::random_problem(model, 10, 200, 21, /*pixel_noise=*/0.0);
    perturb(p, 22);
    jba::SolverOptions o;
    o.function_tolerance = 1e-12;
    o.max_iterations = 100;
    const auto s = jba::solve(p, o);
    std::printf("%-15s initial %.6g final %.3g iterations %d (%s)\n", std::string(jba::model_name(model)).c_str(),
                s.initial_cost, s.final_cost, s.iterations, s.message.c_str());
    EXPECT(s.termination == jba::Termination::Convergence);
    EXPECT(s.final_cost < 1e-12 * s.initial_cost);
    EXPECT(s.history.size() == static_cast<std::size_t>(s.iterations) + 1);
    for (std::size_t i = 1; i < s.history.size(); ++i) EXPECT(s.history[i].cost <= s.history[i - 1].cost);
  }

  // Cancellation: stop after iteration 2, keep the last accepted parameters.
  {
    auto p = jba_test::random_problem(CameraModel::SimpleRadial, 8, 150, 23);
    perturb(p, 24);
    int calls = 0;
    const auto s = jba::solve(p, {}, [&](const jba::IterationSummary& it) {
      ++calls;
      return it.iteration < 2;
    });
    EXPECT(s.termination == jba::Termination::Cancelled);
    EXPECT(s.iterations == 2);
    EXPECT(calls == 3);
    EXPECT(s.final_cost == s.history.back().cost);
  }

  // Same result, bit for bit, for any thread count.
  {
    const auto base = [] {
      auto p = jba_test::random_problem(CameraModel::Radial, 12, 400, 25);
      perturb(p, 26);
      return p;
    }();
    std::vector<double> first;
    double first_cost = 0;
    for (int threads : {1, 2, 3, 8}) {
      auto p = base;
      jba::SolverOptions o;
      o.num_threads = threads;
      const auto s = jba::solve(p, o);
      if (first.empty()) {
        first = p.points;
        first_cost = s.final_cost;
      } else {
        EXPECT(p.points == first);
        EXPECT(s.final_cost == first_cost);
      }
    }
  }

  // Gauge: the reference pose and the held translation coordinate do not move.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 9, 200, 27);
    perturb(p, 28);
    const auto g = jba::choose_gauge(p);
    EXPECT(g.fixed_images.size() == 1 && g.fixed_translations.size() == 1 && g.fixed_set_scales.empty());
    const int ref = g.fixed_images.at(0), second = g.fixed_translations.at(0)[0];
    EXPECT(second != ref);
    const auto before = p.poses;
    const auto s = jba::solve(p);
    EXPECT(s.gauge.fixed_images == g.fixed_images);
    for (int a = 0; a < 7; ++a)
      EXPECT(p.poses[7 * static_cast<std::size_t>(ref) + static_cast<std::size_t>(a)] ==
             before[7 * static_cast<std::size_t>(ref) + static_cast<std::size_t>(a)]);
    const std::size_t held = 7 * static_cast<std::size_t>(second) + 4 + static_cast<std::size_t>(g.fixed_translations[0][1]);
    EXPECT(p.poses[held] == before[held]);
    EXPECT(s.final_cost < s.initial_cost);
  }

  // No gauge when something is fixed already; fixed blocks do not move.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 9, 200, 29);
    perturb(p, 30);
    p.image_fixed.assign(p.num_images(), 0);
    p.image_fixed[3] = 1;
    p.point_fixed.assign(p.num_points(), 0);
    p.point_fixed[7] = 1;
    EXPECT(jba::choose_gauge(p).empty());
    const auto before = p;
    const auto s = jba::solve(p);
    EXPECT(s.gauge.empty());
    for (int a = 0; a < 7; ++a) EXPECT(p.poses[21 + static_cast<std::size_t>(a)] == before.poses[21 + static_cast<std::size_t>(a)]);
    for (int a = 0; a < 3; ++a) EXPECT(p.points[21 + static_cast<std::size_t>(a)] == before.points[21 + static_cast<std::size_t>(a)]);
    EXPECT(p.intrinsics == before.intrinsics);  // fixed by default
  }

  // Freed focal length is refined; the principal point and distortion stay.
  {
    auto truth = jba_test::random_problem(CameraModel::Radial, 10, 300, 31, 0.0);
    auto p = truth;
    perturb(p, 32);
    p.intrinsics_free = {jba::kFocal};
    p.intrinsics[0] *= 1.02;
    jba::SolverOptions o;
    o.function_tolerance = 1e-12;
    o.max_iterations = 100;
    const auto s = jba::solve(p, o);
    EXPECT(s.final_cost < 1e-10 * s.initial_cost);
    for (int a = 1; a < 5; ++a) EXPECT(p.intrinsics[static_cast<std::size_t>(a)] == truth.intrinsics[static_cast<std::size_t>(a)]);
    std::printf("focal: truth %.6f, start %.6f, solved %.6f\n", truth.intrinsics[0], truth.intrinsics[0] * 1.02,
                p.intrinsics[0]);
    // Gauge fixes scale, so the focal length is recovered exactly up to noise (none).
    EXPECT_NEAR(p.intrinsics[0], truth.intrinsics[0], 1e-6 * truth.intrinsics[0]);
  }

  // Robust loss: with gross outliers, Cauchy lands nearer the truth than no loss.
  {
    auto truth = jba_test::random_problem(CameraModel::SimpleRadial, 10, 300, 33, 0.3);
    auto noisy = truth;
    std::mt19937_64 rng(34);
    jba_test::Uniform u(0, 1);
    for (std::size_t k = 0; k < noisy.num_observations(); ++k) {
      if (u(rng) < 0.05) noisy.obs_xy[2 * k] += 40;
    }
    perturb(noisy, 35);
    auto error = [&](const jba::Problem& p) {
      double e = 0;
      for (std::size_t j = 0; j < p.num_points(); ++j)
        for (int a = 0; a < 3; ++a) {
          const double d = p.points[3 * j + static_cast<std::size_t>(a)] - truth.points[3 * j + static_cast<std::size_t>(a)];
          e += d * d;
        }
      return std::sqrt(e / static_cast<double>(p.num_points()));
    };
    jba::SolverOptions plain, robust;
    plain.gauge = robust.gauge = jba::GaugeMode::None;
    robust.loss = {jba::LossType::Cauchy, 2.0};
    auto a = noisy, b = noisy;
    jba::solve(a, plain);
    jba::solve(b, robust);
    std::printf("point RMS error: no loss %.4g, Cauchy %.4g\n", error(a), error(b));
    EXPECT(error(b) < 0.5 * error(a));
  }

  // Gauge with priors, per group of connected images.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 6, 50, 40);
    // Relative priors: only the reference pose is held; the priors set the scale.
    auto rel = p;
    jba::add_relative_priors_from_poses(rel, rel.poses, {0, 1, 1, 2});
    auto g = jba::choose_gauge(rel);
    EXPECT(g.fixed_images.size() == 1 && g.fixed_translations.empty() && g.fixed_set_scales.empty());
    // A global set that is the working frame: nothing is held.
    auto glob = p;
    glob.glob_image = {3};
    glob.glob_pose.assign(p.poses.begin() + 21, p.poses.begin() + 28);
    EXPECT(jba::choose_gauge(glob).empty());
    // A similarity set alone: the reference pose and the set's scale.
    glob.set_relation = {jba::FrameRelation::Similarity};
    g = jba::choose_gauge(glob);
    EXPECT(g.fixed_images.size() == 1 && g.fixed_translations.empty() && g.fixed_set_scales == std::vector<int>{0});
    // A rigid set fixes the scale itself.
    glob.set_relation = {jba::FrameRelation::Rigid};
    g = jba::choose_gauge(glob);
    EXPECT(g.fixed_images.size() == 1 && g.fixed_translations.empty() && g.fixed_set_scales.empty());
    // Two groups joined by nothing: a pose graph with two components gets two
    // reference poses, and no scale is held (relative priors are metric).
    jba::Problem two;
    two.model = CameraModel::Pinhole;
    two.intrinsics = p.intrinsics;
    two.poses = p.poses;
    jba::add_relative_priors_from_poses(two, two.poses, {0, 1, 1, 2, 3, 4, 4, 5});
    g = jba::choose_gauge(two);
    EXPECT((g.fixed_images == std::vector<int>{1, 4}) && g.fixed_translations.empty());  // most priors
  }

  // With priors, results are still bit-identical for any thread count, and the
  // reference pose does not move.
  {
    auto base = jba_test::random_problem(CameraModel::Radial, 12, 400, 41);
    std::vector<std::int32_t> pairs;
    for (int i = 0; i < 12; ++i) pairs.insert(pairs.end(), {i, (i + 1) % 12});
    jba::add_relative_priors_from_poses(base, base.poses, pairs);
    perturb(base, 42);
    std::vector<double> first;
    for (int threads : {1, 3, 8}) {
      auto p = base;
      jba::SolverOptions o;
      o.num_threads = threads;
      const auto s = jba::solve(p, o);
      EXPECT(s.final_cost < s.initial_cost);
      if (first.empty()) first = p.poses;
      else EXPECT(p.poses == first);
      const auto ref = static_cast<std::size_t>(s.gauge.fixed_images.at(0));
      for (int a = 0; a < 7; ++a) EXPECT(p.poses[7 * ref + static_cast<std::size_t>(a)] == base.poses[7 * ref + static_cast<std::size_t>(a)]);
    }
  }

  // Invalid problems are refused.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 3, 10, 36);
    p.obs_point[0] = 999;
    EXPECT_THROWS(jba::solve(p));
  }

  return TEST_MAIN_RESULT();
}
