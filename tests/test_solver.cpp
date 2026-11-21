// Solver behaviour that needs no reference solver: convergence on exact data,
// cancellation, determinism across thread counts (split gathers too), gauge and
// fixed blocks honoured, fixed intrinsics groups untouched, robust loss down-weights
// outliers.

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "check.h"
#include "jba/geometry.h"
#include "jba/priors.h"
#include "jba/report.h"
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

// Equal, NaN matching NaN.
bool same_nan(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])))) return false;
  return true;
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

  // The same with blocks over a gather unit (16,384 observations), whose sums are
  // split and added in order: the shared intrinsics block with focal and distortion
  // free (16 x 2500 observations), and pose blocks (3 images x 20,000 points).
  for (const auto& [images, points, free] : {std::tuple{16, 2500, jba::kFocal | jba::kDistortion},
                                             std::tuple{3, 20000, 0}}) {
    const auto base = [&] {
      auto p = jba_test::random_problem(CameraModel::Radial, images, points, 41);
      perturb(p, 42);
      p.intrinsics_free = {static_cast<std::uint8_t>(free)};
      p.intrinsics[0] *= 1.01;
      return p;
    }();
    jba::Problem first;
    double first_cost = 0;
    for (int threads : {1, 3, 8}) {
      auto p = base;
      jba::SolverOptions o;
      o.num_threads = threads;
      o.max_iterations = 10;
      const auto s = jba::solve(p, o);
      if (threads == 1) {
        first = p;
        first_cost = s.final_cost;
      } else {
        EXPECT(p.points == first.points);
        EXPECT(p.poses == first.poses);
        EXPECT(p.intrinsics == first.intrinsics);
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

  // Prior on free intrinsics (D161): no sigma (or all +inf) is exactly the free
  // solve; a huge sigma is free to rounding, a tiny one fixed; a moderate one lands
  // between. solve() writes the target back, so a second solve anchors to the same
  // values. Cost by type: the reprojection entry is the report's reprojection plus
  // intrinsics prior cost.
  {
    const auto truth = jba_test::random_problem(CameraModel::Radial, 10, 300, 51, 0.5);
    auto start = truth;
    perturb(start, 52);
    start.intrinsics_free = {jba::kFocal | jba::kDistortion};
    start.intrinsics[0] *= 1.05;
    start.intrinsics[3] += 0.02;
    jba::SolverOptions o;
    o.function_tolerance = 1e-12;
    o.max_iterations = 100;
    const double inf = std::numeric_limits<double>::infinity();
    auto solved = [&](const std::vector<double>& sigma, jba::SolverSummary* out = nullptr) {
      auto p = start;
      p.intrinsics_prior_sigma = sigma;
      const auto s = jba::solve(p, o);
      if (out) *out = s;
      return p;
    };
    jba::SolverSummary s_free, s_inf;
    const auto free = solved({}, &s_free);
    const auto all_inf = solved(std::vector<double>(5, inf), &s_inf);
    EXPECT(all_inf.intrinsics == free.intrinsics && all_inf.poses == free.poses && all_inf.points == free.points);
    EXPECT(s_inf.final_cost == s_free.final_cost && s_inf.iterations == s_free.iterations);
    EXPECT(free.intrinsics_prior.empty());
    // The focal length is weakly determined here (free lands at 511 for a truth of
    // 520), so two solves that differ only in rounding stop a little apart along that
    // valley: compare the cost tightly and the intrinsics loosely.
    jba::SolverSummary s_loose;
    const auto loose = solved({1e9, 1e9, 1e9, 1e9, 1e9}, &s_loose);
    std::printf("intrinsics prior: sigma 1e9 cost %.17g iterations %d, free %.17g iterations %d, focal %.9f vs %.9f\n",
                s_loose.final_cost, s_loose.iterations, s_free.final_cost, s_free.iterations, loose.intrinsics[0],
                free.intrinsics[0]);
    EXPECT_NEAR(s_loose.final_cost, s_free.final_cost, 1e-9 * s_free.final_cost);
    for (int a = 0; a < 5; ++a)
      EXPECT_NEAR(loose.intrinsics[static_cast<std::size_t>(a)], free.intrinsics[static_cast<std::size_t>(a)],
                  1e-5 * (1 + std::abs(free.intrinsics[static_cast<std::size_t>(a)])));
    const auto tight = solved({1e-7, 1e-7, 1e-7, 1e-9, 1e-9});
    for (int a = 0; a < 5; ++a)
      EXPECT_NEAR(tight.intrinsics[static_cast<std::size_t>(a)], start.intrinsics[static_cast<std::size_t>(a)],
                  1e-6 * (1 + std::abs(start.intrinsics[static_cast<std::size_t>(a)])));
    // Moderate: focal within about 1 % of its start, k1 and k2 near theirs.
    auto p = start;
    jba::set_intrinsics_prior(p, 0.01, inf, 0.005);
    EXPECT(p.intrinsics_prior == start.intrinsics);
    jba::SolverSummary s;
    {
      auto q = start;
      q.intrinsics_prior_sigma = p.intrinsics_prior_sigma;  // empty target: the start
      s = jba::solve(q, o);
      EXPECT(q.intrinsics_prior == start.intrinsics);  // written back
      p = q;
    }
    const double f0 = start.intrinsics[0], f_free = free.intrinsics[0], f = p.intrinsics[0];
    std::printf("intrinsics prior: focal start %.3f, free %.3f, regularized %.3f, truth %.3f\n", f0, f_free, f,
                truth.intrinsics[0]);
    EXPECT(std::abs(f - f0) < std::abs(f_free - f0) && (f - f0) * (f_free - f0) > 0);
    EXPECT(p.intrinsics[1] == start.intrinsics[1] && p.intrinsics[2] == start.intrinsics[2]);  // principal point fixed
    for (int when = 0; when < 2; ++when) {
      const auto& by = when == 0 ? s.initial_cost_by_type : s.final_cost_by_type;
      const double total = when == 0 ? s.initial_cost : s.final_cost;
      auto at = when == 0 ? start : p;
      at.intrinsics_prior = start.intrinsics;
      at.intrinsics_prior_sigma = p.intrinsics_prior_sigma;
      const auto rep = jba::evaluate(at, o);
      double sum = 0;
      for (double v : by) sum += v;
      EXPECT_NEAR(sum, total, 1e-12 * total);
      EXPECT_NEAR(by[jba::kLossReprojection], rep.reprojection_cost + rep.intrinsics_prior_cost,
                  1e-9 * (1 + by[jba::kLossReprojection]));
      if (when == 0) EXPECT(rep.intrinsics_prior_cost == 0);
      else EXPECT(rep.intrinsics_prior_cost > 0 && rep.intrinsics_prior_param.size() == 1 &&
                  rep.intrinsics_prior_param[0] >= 0 && rep.intrinsics_prior_norm[0] > 0);
    }
    // Write-back covers the blocks that got a prior only: sigma with nothing freed
    // leaves the target empty; with two blocks, the fixed one keeps NaN (its value at
    // a later solve) and the regularized one gets its resolved target. A NaN entry
    // in a given target means that parameter's starting value.
    {
      auto q = start;
      q.intrinsics_free.clear();
      q.intrinsics_prior_sigma = p.intrinsics_prior_sigma;
      jba::solve(q, o);
      EXPECT(q.intrinsics_prior.empty());
      auto two = start;
      two.intrinsics.insert(two.intrinsics.end(), start.intrinsics.begin(), start.intrinsics.end());
      two.image_intrinsics.assign(two.num_images(), 0);
      for (std::size_t i = 5; i < two.num_images(); ++i) two.image_intrinsics[i] = 1;
      two.intrinsics_free = {jba::kFocal, 0};
      two.intrinsics_prior_sigma.assign(10, 1.0);
      two.intrinsics_prior.assign(10, std::nan(""));
      two.intrinsics_prior[0] = start.intrinsics[0] + 3;  // focal of block 0 given
      const auto two_start = two.intrinsics;
      jba::solve(two, o);
      EXPECT(two.intrinsics_prior[0] == start.intrinsics[0] + 3);
      for (std::size_t a = 1; a < 5; ++a) EXPECT(two.intrinsics_prior[a] == two_start[a]);
      for (std::size_t a = 5; a < 10; ++a) EXPECT(std::isnan(two.intrinsics_prior[a]));
    }
    // A second solve from the result anchors to the first target, not to where it is.
    auto again = p;
    jba::solve(again, o);
    EXPECT(again.intrinsics_prior == start.intrinsics);
    EXPECT_NEAR(again.intrinsics[0], p.intrinsics[0], 1e-6 * p.intrinsics[0]);
  }

  // The hierarchical prior by group (D165). Per-image blocks of one camera whose focal
  // lengths differ a little (autofocus), a poor calibration to start from, and a few
  // images that see little: grouped, the focal lengths land nearer the truth than
  // freed one by one or each pulled toward the poor calibration (D161), and the mean
  // corrects the calibration.
  {
    const int n_img = 30;
    auto truth = jba_test::random_problem(CameraModel::Radial, n_img, 300, 61, 0.0);
    const std::size_t np = 5;
    const double f0 = truth.intrinsics[0];
    std::mt19937_64 rng(62);
    jba_test::Normal jitter(0, 0.004), noise(0, 0.5);
    const std::vector<double> block = truth.intrinsics;
    truth.intrinsics.clear();
    truth.image_intrinsics.resize(static_cast<std::size_t>(n_img));
    for (int i = 0; i < n_img; ++i) {
      truth.intrinsics.insert(truth.intrinsics.end(), block.begin(), block.end());
      truth.intrinsics[np * static_cast<std::size_t>(i)] = f0 * (1 + jitter(rng));
      truth.image_intrinsics[static_cast<std::size_t>(i)] = i;
    }
    // Observations from each image's own focal length; images 0 to 4 keep 20 each.
    jba::Problem obs = truth;
    obs.obs_image.clear();
    obs.obs_point.clear();
    obs.obs_xy.clear();
    std::vector<int> seen(static_cast<std::size_t>(n_img), 0);
    for (std::size_t k = 0; k < truth.num_observations(); ++k) {
      const auto i = static_cast<std::size_t>(truth.obs_image[k]);
      if (i < 5 && seen[i] >= 20) continue;
      ++seen[i];
      const double* pose = &truth.poses[7 * i];
      double pc[3];
      jba::quat_rotate(pose, &truth.points[3 * static_cast<std::size_t>(truth.obs_point[k])], pc);
      for (int c = 0; c < 3; ++c) pc[c] += pose[4 + c];
      double px, py;
      jba_test::reference_project(CameraModel::Radial, &truth.intrinsics[np * i], pc, &px, &py);
      obs.obs_image.push_back(truth.obs_image[k]);
      obs.obs_point.push_back(truth.obs_point[k]);
      obs.obs_xy.insert(obs.obs_xy.end(), {px + noise(rng), py + noise(rng)});
    }
    auto start = obs;
    perturb(start, 63);
    for (int i = 0; i < n_img; ++i) {
      std::copy(block.begin(), block.end(), start.intrinsics.begin() + static_cast<std::ptrdiff_t>(np * static_cast<std::size_t>(i)));
      start.intrinsics[np * static_cast<std::size_t>(i)] = 1.05 * f0;  // the calibration, 5 % off
    }
    start.intrinsics_free.assign(static_cast<std::size_t>(n_img), jba::kFocal);
    jba::SolverOptions o;
    o.function_tolerance = 1e-12;
    o.max_iterations = 200;
    const double inf = std::numeric_limits<double>::infinity();
    auto focal_rms = [&](const jba::Problem& p, int from, int to) {
      double s = 0;
      for (int i = from; i < to; ++i) {
        const double d = p.intrinsics[np * static_cast<std::size_t>(i)] / truth.intrinsics[np * static_cast<std::size_t>(i)] - 1;
        s += d * d;
      }
      return std::sqrt(s / (to - from));
    };
    double true_mean = 0;
    for (int i = 0; i < n_img; ++i) true_mean += truth.intrinsics[np * static_cast<std::size_t>(i)] / n_img;
    const std::vector<std::int32_t> one_group(static_cast<std::size_t>(n_img), 0);

    auto free = start;
    const auto s_free = jba::solve(free, o);
    auto per_block = start;
    jba::set_intrinsics_prior(per_block, 0.01, inf, inf);
    jba::solve(per_block, o);
    auto grouped = start;
    jba::set_intrinsics_groups(grouped, one_group, 0.005, inf, inf, 0.05, inf, inf);
    EXPECT(grouped.intrinsics_group_mean.empty() && grouped.intrinsics_group_prior.empty());
    const auto s_grouped = jba::solve(grouped, o);
    const double mean = grouped.intrinsics_group_mean[0];
    std::printf("intrinsics groups: focal rms free %.5f (weak %.5f), per block %.5f (weak %.5f), grouped %.5f (weak "
                "%.5f); mean %.5f of the true mean; %d and %d iterations\n",
                focal_rms(free, 0, n_img), focal_rms(free, 0, 5), focal_rms(per_block, 0, n_img), focal_rms(per_block, 0, 5),
                focal_rms(grouped, 0, n_img), focal_rms(grouped, 0, 5), mean / true_mean, s_free.iterations,
                s_grouped.iterations);
    EXPECT(s_grouped.termination == jba::Termination::Convergence);
    EXPECT(focal_rms(grouped, 0, n_img) < focal_rms(free, 0, n_img));
    EXPECT(focal_rms(grouped, 0, n_img) < focal_rms(per_block, 0, n_img));
    EXPECT(focal_rms(grouped, 0, 5) < focal_rms(free, 0, 5));
    EXPECT(std::abs(mean / true_mean - 1) < 0.003);
    // Written back: the mean of the tied parameter only (the rest NaN), the target.
    EXPECT(grouped.intrinsics_group_mean.size() == np && std::isnan(grouped.intrinsics_group_mean[1]));
    EXPECT(grouped.intrinsics_group_prior[0] == 1.05 * f0);
    // Cost by type: the reprojection entry is the report's reprojection plus the group
    // prior; at the start the ties cost nothing (every block at the mean).
    for (int when = 0; when < 2; ++when) {
      auto at = when == 0 ? start : grouped;
      if (when == 0) {
        jba::set_intrinsics_groups(at, one_group, 0.005, inf, inf, 0.05, inf, inf);
      }
      const auto rep = jba::evaluate(at, o);
      const auto& by = when == 0 ? s_grouped.initial_cost_by_type : s_grouped.final_cost_by_type;
      EXPECT_NEAR(by[jba::kLossReprojection], rep.reprojection_cost + rep.intrinsics_group_cost,
                  1e-9 * (1 + by[jba::kLossReprojection]));
      EXPECT(rep.intrinsics_group_norm.size() == static_cast<std::size_t>(n_img) &&
             rep.intrinsics_group_target_norm.size() == 1);
      if (when == 0) EXPECT(rep.intrinsics_group_cost == 0);
      else EXPECT(rep.intrinsics_group_cost > 0 && rep.intrinsics_group_param[0] == 0 &&
                  rep.intrinsics_group_target_param[0] == 0 && rep.intrinsics_group_target_norm[0] > 0);
    }
    // A second solve starts from the mean it wrote and anchors to the same target.
    auto again = grouped;
    jba::solve(again, o);
    EXPECT(same_nan(again.intrinsics_group_prior, grouped.intrinsics_group_prior));
    EXPECT_NEAR(again.intrinsics_group_mean[0], mean, 1e-6 * mean);

    // Limits: every sigma +inf builds nothing (the free solve, bit for bit); a group of
    // one without a target builds nothing either; a tiny tie with a free mean gives
    // every block one focal length.
    {
      auto all_inf = start;
      jba::set_intrinsics_groups(all_inf, one_group, inf, inf, inf, inf, inf, inf);
      const auto s = jba::solve(all_inf, o);
      EXPECT(all_inf.intrinsics == free.intrinsics && all_inf.poses == free.poses && s.final_cost == s_free.final_cost);
      EXPECT(all_inf.intrinsics_group_mean.empty());
      auto alone = start;
      std::vector<std::int32_t> each(static_cast<std::size_t>(n_img));
      for (int i = 0; i < n_img; ++i) each[static_cast<std::size_t>(i)] = i;
      jba::set_intrinsics_groups(alone, each, 0.001, inf, inf, inf, inf, inf);
      jba::solve(alone, o);
      EXPECT(alone.intrinsics == free.intrinsics && alone.intrinsics_group_mean.empty());
      auto shared = start;
      jba::set_intrinsics_groups(shared, one_group, 1e-9, inf, inf, inf, inf, inf);
      jba::solve(shared, o);
      for (int i = 1; i < n_img; ++i)
        EXPECT_NEAR(shared.intrinsics[np * static_cast<std::size_t>(i)], shared.intrinsics[0], 1e-6 * f0);
      EXPECT_NEAR(shared.intrinsics_group_mean[0], shared.intrinsics[0], 1e-6 * f0);
    }
    // A group of one with a target is the block prior with the two variances summed.
    {
      auto one = start;
      one.intrinsics = std::vector<double>(start.intrinsics.begin(), start.intrinsics.begin() + static_cast<std::ptrdiff_t>(np));
      one.image_intrinsics.clear();
      one.intrinsics_free = {jba::kFocal};
      auto as_group = one, as_block = one;
      jba::set_intrinsics_groups(as_group, {0}, 0.003, inf, inf, 0.004, inf, inf);
      jba::set_intrinsics_prior(as_block, 0.005, inf, inf);
      jba::solve(as_group, o);
      jba::solve(as_block, o);
      std::printf("intrinsics groups: group of one %.9f, block prior %.9f\n", as_group.intrinsics[0], as_block.intrinsics[0]);
      EXPECT_NEAR(as_group.intrinsics[0], as_block.intrinsics[0], 1e-6 * f0);
    }
    // A block in the group that no observation uses (a placeholder) does not vote: not
    // in the helper's focal sigma, the target or the mean, so the solve is the same as
    // without it, and the block keeps its value.
    {
      auto extra = start;
      extra.intrinsics.insert(extra.intrinsics.end(), block.begin(), block.end());
      extra.intrinsics[np * static_cast<std::size_t>(n_img)] = 5;
      extra.intrinsics_free.push_back(jba::kFocal);
      auto groups_extra = one_group;
      groups_extra.push_back(0);
      jba::set_intrinsics_groups(extra, groups_extra, 0.005, inf, inf, 0.05, inf, inf);
      auto plain = start;
      jba::set_intrinsics_groups(plain, one_group, 0.005, inf, inf, 0.05, inf, inf);
      EXPECT(extra.intrinsics_group_sigma == plain.intrinsics_group_sigma &&
             extra.intrinsics_group_prior_sigma == plain.intrinsics_group_prior_sigma);
      jba::solve(extra, o);
      jba::solve(plain, o);
      EXPECT(std::equal(plain.intrinsics.begin(), plain.intrinsics.end(), extra.intrinsics.begin()));
      EXPECT(extra.intrinsics[np * static_cast<std::size_t>(n_img)] == 5);
      EXPECT(same_nan(extra.intrinsics_group_mean, plain.intrinsics_group_mean) &&
             same_nan(extra.intrinsics_group_prior, plain.intrinsics_group_prior));
      // No block in a group: no groups, a valid problem.
      auto none = start;
      jba::set_intrinsics_groups(none, std::vector<std::int32_t>(static_cast<std::size_t>(n_img), -1), 0.005, inf, inf, 0.05,
                                 inf, inf);
      EXPECT(none.intrinsics_group.empty() && none.intrinsics_group_sigma.empty() && jba::validate(none).empty());
    }
    // Members freed differently, two groups, and the block prior on top: the solver's
    // cost by type agrees with evaluate(), and a second solve has nothing left to do.
    {
      auto mixed = start;
      for (int i = 0; i < n_img; ++i)
        mixed.intrinsics_free[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(i % 3 == 0 ? jba::kFocal | jba::kDistortion : (i % 3 == 1 ? jba::kDistortion : jba::kFocal));
      std::vector<std::int32_t> two(static_cast<std::size_t>(n_img));
      for (int i = 0; i < n_img; ++i) two[static_cast<std::size_t>(i)] = i % 2;
      jba::set_intrinsics_groups(mixed, two, 0.005, inf, 0.002, 0.05, inf, inf);
      jba::set_intrinsics_prior(mixed, 0.03, inf, 0.01);
      const auto s = jba::solve(mixed, o);
      const auto rep = jba::evaluate(mixed, o);
      EXPECT_NEAR(s.final_cost_by_type[jba::kLossReprojection],
                  rep.reprojection_cost + rep.intrinsics_prior_cost + rep.intrinsics_group_cost,
                  1e-9 * s.final_cost_by_type[jba::kLossReprojection]);
      EXPECT(rep.intrinsics_group_cost > 0 && rep.intrinsics_prior_cost > 0);
      auto again2 = mixed;
      const auto s2 = jba::solve(again2, o);
      std::printf("intrinsics groups, mixed masks: %d iterations, then %d\n", s.iterations, s2.iterations);
      EXPECT(s2.iterations <= 1 && std::abs(s2.final_cost - s.final_cost) <= 1e-9 * s.final_cost);
    }
    // A fixed member pulls the mean and its siblings: blocks 0 to 4 held at the truth's
    // mean, the others free and tied tightly, the mean free.
    {
      auto held = start;
      for (int i = 0; i < 5; ++i) {
        held.intrinsics[np * static_cast<std::size_t>(i)] = true_mean;
        held.intrinsics_free[static_cast<std::size_t>(i)] = 0;
      }
      jba::set_intrinsics_groups(held, one_group, 1e-6, inf, inf, inf, inf, inf);
      jba::solve(held, o);
      for (int i = 5; i < n_img; ++i) EXPECT_NEAR(held.intrinsics[np * static_cast<std::size_t>(i)], true_mean, 1e-3 * f0);
      EXPECT_NEAR(held.intrinsics_group_mean[0], true_mean, 1e-3 * f0);
    }
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

  // Gravity sets the vertical (D83): a scene tilted as a whole, consistent with its
  // observations and relative priors but not with gravity, levels itself. Only the
  // reference's translation and heading are held, so its tilt follows gravity.
  {
    auto truth = jba_test::random_problem(CameraModel::Pinhole, 8, 200, 43, 0.0);
    std::vector<std::int32_t> pairs;
    for (int i = 0; i < 8; ++i) pairs.insert(pairs.end(), {i, (i + 1) % 8});
    jba::add_relative_priors_from_poses(truth, truth.poses, pairs);
    truth.gravity_world = {0, 0.6, 0.8};
    for (int i = 0; i < 8; ++i) {
      double g[3];
      jba::quat_rotate(&truth.poses[7 * static_cast<std::size_t>(i)], truth.gravity_world.data(), g);
      truth.grav_image.push_back(i);
      truth.grav_dir.insert(truth.grav_dir.end(), g, g + 3);
    }
    // Tilt everything by Q about a horizontal axis: X' = Q X, R' = R Q^T, t' = t.
    const double tilt[3] = {0.06, 0, 0};
    double Q[4];
    jba::quat_exp(tilt, Q);
    const double Qc[4] = {Q[0], -Q[1], -Q[2], -Q[3]};
    auto p = truth;
    for (std::size_t j = 0; j < p.num_points(); ++j) jba::quat_rotate(Q, &truth.points[3 * j], &p.points[3 * j]);
    for (std::size_t i = 0; i < p.num_images(); ++i) {
      jba::quat_multiply(&truth.poses[7 * i], Qc, &p.poses[7 * i]);
      jba::quat_normalize(&p.poses[7 * i]);
    }
    const auto g = jba::choose_gauge(p);
    EXPECT(g.fixed_images.empty() && g.heading_images.size() == 1 && g.fixed_translations.empty());
    jba::SolverOptions o;
    o.function_tolerance = 1e-14;
    o.parameter_tolerance = 1e-14;
    o.max_iterations = 100;
    const auto start = p;
    const auto s = jba::solve(p, o);
    std::printf("tilted scene with gravity priors: cost %.3g -> %.3g\n", s.initial_cost, s.final_cost);
    EXPECT(s.final_cost < 1e-16 * s.initial_cost);
    double worst = 0;
    for (std::size_t k = 0; k < p.num_gravity_priors(); ++k) {
      double gc[3];
      jba::quat_rotate(&p.poses[7 * static_cast<std::size_t>(p.grav_image[k])], p.gravity_world.data(), gc);
      for (int a = 0; a < 3; ++a) worst = std::max(worst, std::abs(gc[a] - p.grav_dir[3 * k + static_cast<std::size_t>(a)]));
    }
    EXPECT(worst < 1e-8);
    const auto ref = static_cast<std::size_t>(g.heading_images[0]);
    for (std::size_t a = 4; a < 7; ++a) EXPECT(p.poses[7 * ref + a] == start.poses[7 * ref + a]);
    // Without gravity priors the same start is already a minimum: the tilt is a gauge freedom.
    auto plain = start;
    plain.grav_image.clear();
    plain.grav_dir.clear();
    EXPECT(jba::choose_gauge(plain).heading_images.empty());
  }

  // Cost by residual type (document 01): sums to the total and agrees with the
  // report, type by type, at the start and at the end.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 8, 150, 44);
    std::vector<std::int32_t> pairs;
    for (int i = 0; i < 8; ++i) pairs.insert(pairs.end(), {i, (i + 1) % 8});
    jba::add_relative_priors_from_poses(p, p.poses, pairs);
    p.glob_image = {2};
    p.glob_pose.assign(p.poses.begin() + 14, p.poses.begin() + 21);
    double g[3];
    jba::quat_rotate(&p.poses[0], std::vector<double>{0, -1, 0}.data(), g);
    p.grav_image = {0};
    p.grav_dir.assign(g, g + 3);
    p.obs_depth.assign(p.num_observations(), std::nan(""));
    for (std::size_t k = 0; k < p.num_observations(); k += 3) p.obs_depth[k] = 5.0;
    p.depth_correction_free.assign(p.num_images(), jba::kDepthScale);
    perturb(p, 45);
    jba::SolverOptions o;
    o.loss = {jba::LossType::Cauchy, 2.0};
    const auto start = p;
    const auto s = jba::solve(p, o);
    for (int when = 0; when < 2; ++when) {
      const auto& by = when == 0 ? s.initial_cost_by_type : s.final_cost_by_type;
      const double total = when == 0 ? s.initial_cost : s.final_cost;
      const auto rep = jba::evaluate(when == 0 ? start : p, o);
      double sum = 0;
      for (double v : by) sum += v;
      EXPECT_NEAR(sum, total, 1e-12 * total);
      const double expect[jba::kNumLossTypes] = {rep.reprojection_cost, rep.relative_prior_cost, rep.global_prior_cost,
                                                 rep.gravity_cost, rep.line_cost, rep.depth_cost};
      for (std::size_t t = 0; t < jba::kNumLossTypes; ++t) EXPECT_NEAR(by[t], expect[t], 1e-9 * (1 + expect[t]));
    }
    EXPECT(s.final_cost_by_type[jba::kLossDepth] > 0 && s.final_cost_by_type[jba::kLossGravity] >= 0);
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
    auto q = jba_test::random_problem(CameraModel::Pinhole, 3, 10, 36);
    q.points[4] = std::nan("");
    EXPECT_THROWS(jba::solve(q));
  }

  // A start with no finite cost ends in Failure, with the parameters untouched.
  // An image with zero translation sees a point at the origin at depth exactly
  // zero, whatever its rotation and however the platform rounds.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 4, 40, 37);
    const std::size_t i = static_cast<std::size_t>(p.obs_image[0]), j = static_cast<std::size_t>(p.obs_point[0]);
    for (std::size_t a = 0; a < 3; ++a) p.poses[7 * i + 4 + a] = p.points[3 * j + a] = 0.0;
    const auto before = p;
    const auto s = jba::solve(p);
    EXPECT(s.termination == jba::Termination::Failure);
    EXPECT(s.iterations == 0);
    EXPECT(p.poses == before.poses && p.points == before.points);
  }

  // An image and a point without observations are left alone; the same point
  // observed twice in one image is accepted; the rest still converges.
  {
    auto p = jba_test::random_problem(CameraModel::SimpleRadial, 8, 150, 38, 0.0);
    perturb(p, 39);
    for (int a = 0; a < 7; ++a) p.poses.push_back(p.poses[static_cast<std::size_t>(a)]);
    p.points.insert(p.points.end(), {1.0, 2.0, 3.0});
    p.obs_image.push_back(p.obs_image[0]);
    p.obs_point.push_back(p.obs_point[0]);
    p.obs_xy.push_back(p.obs_xy[0]);
    p.obs_xy.push_back(p.obs_xy[1]);
    const std::size_t image = p.num_images() - 1, point = p.num_points() - 1;
    const auto before = p;
    jba::SolverOptions o;
    o.function_tolerance = 1e-12;
    o.max_iterations = 100;
    const auto s = jba::solve(p, o);
    EXPECT(s.termination == jba::Termination::Convergence);
    EXPECT(s.final_cost < 1e-12 * s.initial_cost);
    for (std::size_t a = 0; a < 7; ++a) EXPECT(p.poses[7 * image + a] == before.poses[7 * image + a]);
    for (std::size_t a = 0; a < 3; ++a) EXPECT(p.points[3 * point + a] == before.points[3 * point + a]);
  }

  // No observations at all: nothing to do, and that is convergence.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 3, 10, 40);
    p.obs_image.clear();
    p.obs_point.clear();
    p.obs_xy.clear();
    const auto before = p;
    const auto s = jba::solve(p);
    EXPECT(s.termination == jba::Termination::Convergence);
    EXPECT(s.iterations == 0 && s.final_cost == 0);
    EXPECT(p.poses == before.poses && p.points == before.points);
  }

  // The step norm is measured in stored coordinates, as in Ceres: for a pose step
  // that is nearly all rotation it is about half the rotation angle.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 6, 200, 41, 0.0);
    const auto truth = p;
    p.point_fixed.assign(p.num_points(), 1);
    p.image_fixed.assign(p.num_images(), 1);
    p.image_fixed[2] = 0;
    const double step[6] = {0.02, 0, 0, 0, 0, 0};
    double moved[7];
    jba::pose_retract(&truth.poses[14], step, moved);
    for (std::size_t a = 0; a < 7; ++a) p.poses[14 + a] = moved[a];
    const auto s = jba::solve(p);
    double stored = 0;
    for (std::size_t a = 0; a < 7; ++a) stored += (p.poses[14 + a] - moved[a]) * (p.poses[14 + a] - moved[a]);
    double sum = 0;
    for (const auto& h : s.history) sum += h.step_accepted ? h.step_norm : 0.0;
    std::printf("step norms: summed %.6g, start to end in stored coordinates %.6g\n", sum, std::sqrt(stored));
    EXPECT(s.history.size() >= 2 && s.history[1].step_accepted);
    EXPECT(s.history[1].step_norm < 0.75 * 0.02);  // a tangent norm would be about 0.02
    EXPECT(sum >= std::sqrt(stored) * (1 - 1e-9));
  }

  return TEST_MAIN_RESULT();
}
