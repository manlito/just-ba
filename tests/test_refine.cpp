// End to end on synthetic scenes (document 05): keypoints, matches and noisy pose
// priors through refine(), checked against ground truth. Prints the numbers each
// check is based on.

#include <cmath>
#include <cstdio>
#include <string>

#include "check.h"
#include "jba/initialize.h"
#include "jba/refine.h"
#include "scene.h"

using jba_test::SceneKind;
using jba_test::SceneOptions;

namespace {

constexpr double kDeg = 3.141592653589793 / 180;

struct Outcome {
  // before: the starting poses built from the priors alone (what refine is given,
  // in effect); tracker: the tracker's own poses.
  jba_test::PoseError before, tracker, after;
  jba::RefineResult result;
};

Outcome run(const char* name, const SceneOptions& so, jba::RefineOptions ro = {}, bool scale_free = false) {
  const auto scene = jba_test::make_scene(so);
  Outcome out;
  const auto init = jba::initialize_poses(scene.input.scene, ro.solver.prior_defaults);
  std::vector<char> placed(init.group.size());
  for (std::size_t i = 0; i < placed.size(); ++i) placed[i] = init.group[i] >= 0;
  out.before = jba_test::pose_error(init.poses, scene.truth.poses, scale_free, placed);
  out.tracker = jba_test::pose_error(scene.tracker_poses, scene.truth.poses, scale_free);
  out.result = jba::refine(scene.input, ro);
  out.after = jba_test::pose_error(out.result.problem.poses, scene.truth.poses, scale_free, out.result.refined);
  const auto& r = out.result;
  std::printf("%-28s images %3zu points %5zu obs %6zu priors %3zu/%zu | before: centre %.4f m (max %.4f), rot %.3f deg"
              " | after: centre %.4f m (max %.4f), rot %.4f deg (max %.3f) | groups %d, down-weighted %zu/%zu, scale %.5f\n",
              name, r.problem.num_images(), r.problem.num_points(), r.problem.num_observations(),
              r.problem.num_relative_priors(), r.problem.num_global_priors(), out.before.median_centre,
              out.before.max_centre, out.before.median_rotation / kDeg, out.after.median_centre, out.after.max_centre,
              out.after.median_rotation / kDeg, out.after.max_rotation / kDeg, r.num_groups,
              r.report.downweighted_relative.size(), r.report.downweighted_global.size(), out.after.scale);
  return out;
}

}  // namespace

int main() {
  // A room walked with a drifting tracker: refinement removes the drift.
  {
    SceneOptions so;
    so.kind = SceneKind::Room;
    so.drift_heading_per_m = 0.02;
    so.drift_trans_per_m = 0.02;
    const auto o = run("room, drift", so);
    EXPECT(o.after.median_centre < 0.01);
    EXPECT(o.after.max_centre < 0.03);
    // This tracker drifts in heading about twice as much as the ARCore captures the
    // default prior weights come from (D77), so those weights trust it a little too
    // much: 0.115 deg here, 0.090 with the looser weights before D77.
    EXPECT(o.after.median_rotation < 0.12 * kDeg);
    EXPECT(o.after.median_centre < 0.25 * o.before.median_centre);
    EXPECT(o.result.num_groups == 1);
  }

  // Priors already accurate: refinement must not make them worse (document 05).
  {
    SceneOptions so;
    so.drift_heading_per_m = 0;
    so.drift_tilt_per_m = 0;
    so.drift_trans_per_m = 0;
    so.prior_noise_rot = 0.0003;
    so.prior_noise_trans = 0.001;
    jba::RefineOptions ro;
    ro.solver.prior_defaults.relative_rotation_sigma = 0.0005;
    ro.solver.prior_defaults.relative_translation_sigma = 0.002;
    const auto o = run("room, accurate priors", so, ro);
    EXPECT(o.after.median_centre <= 1.5 * o.before.median_centre + 0.001);
    EXPECT(o.after.median_rotation <= 1.5 * o.before.median_rotation + 0.01 * kDeg);
  }

  // Wrong matches: tracks are repaired and outliers removed.
  {
    SceneOptions so;
    so.drift_heading_per_m = 0.02;
    so.wrong_matches = 0.1;
    const auto o = run("room, 10% wrong matches", so);
    EXPECT(o.after.median_centre < 0.015);
    EXPECT(o.after.median_rotation < 0.15 * kDeg);
  }

  // Uneven priors: after the caller's cleanup, 5% of the relative priors are still
  // noticeably wrong. The result stays near the clean one and every bad prior is
  // reported as down-weighted.
  {
    SceneOptions so;
    so.bad_prior_fraction = 0.05;
    so.seed = 3;
    const auto scene = jba_test::make_scene(so);
    const auto o = run("room, 5% bad priors", so);
    SceneOptions clean = so;
    clean.bad_prior_fraction = 0;
    const auto c = run("room, same without them", clean);
    EXPECT(!scene.bad_relative.empty());
    EXPECT(o.after.median_centre < 1.5 * c.after.median_centre + 0.001);
    EXPECT(o.after.median_rotation < 1.5 * c.after.median_rotation + 0.01 * kDeg);
    std::size_t found = 0;
    for (auto k : scene.bad_relative)
      for (auto d : o.result.report.downweighted_relative) found += d == k;
    std::printf("  bad priors %zu, reported %zu of them, %zu reported in all\n", scene.bad_relative.size(), found,
                o.result.report.downweighted_relative.size());
    EXPECT(found == scene.bad_relative.size());
    EXPECT(o.result.report.downweighted_relative.size() == found);  // nothing good reported
  }

  // Tracking break with the priors across it removed: two groups, each refined in
  // its own frame (checked separately against the truth).
  {
    SceneOptions so;
    so.drift_heading_per_m = 0.01;
    so.jump_probability = 0.0;
    so.prior_neighbours = 3;
    auto scene = jba_test::make_scene(so);
    // Remove every prior that crosses frame 30 in capture order.
    jba::Problem& sc = scene.input.scene;
    jba::Problem kept = sc;
    kept.rel_images.clear();
    kept.rel_pose.clear();
    for (std::size_t k = 0; k < sc.num_relative_priors(); ++k) {
      const bool a = sc.rel_images[2 * k] < 30, b = sc.rel_images[2 * k + 1] < 30;
      if (a != b) continue;
      kept.rel_images.insert(kept.rel_images.end(), {sc.rel_images[2 * k], sc.rel_images[2 * k + 1]});
      kept.rel_pose.insert(kept.rel_pose.end(), &sc.rel_pose[7 * k], &sc.rel_pose[7 * k] + 7);
    }
    sc = kept;
    const auto r = jba::refine(scene.input);
    EXPECT(r.num_groups == 2);
    for (int g = 0; g < 2; ++g) {
      std::vector<char> use(r.group.size(), 0);
      for (std::size_t i = 0; i < use.size(); ++i) use[i] = r.group[i] == g;
      const auto e = jba_test::pose_error(r.problem.poses, scene.truth.poses, false, use);
      std::printf("%-28s group %d: centre %.4f m, rot %.4f deg\n", "room, broken session", g, e.median_centre,
                  e.median_rotation / kDeg);
      EXPECT(e.median_centre < 0.01);
    }
  }

  // Images that no match reaches, or too few: still solved (through their priors)
  // but not reported as refined, with their observation counts.
  {
    SceneOptions so;
    so.drift_heading_per_m = 0.02;
    auto scene = jba_test::make_scene(so);
    constexpr int kBlind = 10, kSparse = 20;
    std::vector<std::int32_t> kept;
    int sparse_kept = 0;
    const auto& m = scene.input.matches;
    for (std::size_t k = 0; k + 3 < m.size(); k += 4) {
      const bool blind = m[k] == kBlind || m[k + 2] == kBlind;
      const bool sparse = m[k] == kSparse || m[k + 2] == kSparse;
      if (blind || (sparse && sparse_kept++ >= 10)) continue;
      kept.insert(kept.end(), m.begin() + static_cast<std::ptrdiff_t>(k), m.begin() + static_cast<std::ptrdiff_t>(k) + 4);
    }
    scene.input.matches = kept;
    const auto r = jba::refine(scene.input);
    std::printf("%-28s observations: image %d %d, image %d %d\n", "room, blind images", kBlind,
                r.num_observations[kBlind], kSparse, r.num_observations[kSparse]);
    EXPECT(r.num_observations.size() == r.problem.num_images());
    EXPECT(r.num_observations[kBlind] == 0 && !r.refined[kBlind] && r.group[kBlind] == 0);
    EXPECT(r.num_observations[kSparse] < 30 && !r.refined[kSparse]);
    std::size_t refined = 0;
    for (char x : r.refined) refined += x != 0;
    EXPECT(refined == r.problem.num_images() - 2);
    // With no minimum, every solved image counts again, the blind one included.
    jba::RefineOptions ro;
    ro.min_observations = 0;
    const auto r0 = jba::refine(scene.input, ro);
    EXPECT(r0.refined[kBlind] && r0.refined[kSparse]);
  }

  // Global priors from a drift-free source with arbitrary frame and scale,
  // combined with metric relative priors: the similarity recovers the true scale.
  {
    SceneOptions so;
    so.drift_heading_per_m = 0.02;
    so.global_priors = true;
    so.global_relation = jba::FrameRelation::Similarity;
    so.global_noise_rot = 0.02;
    so.global_noise_pos = 0.05;
    so.global_outliers = 0.05;
    const auto scene = jba_test::make_scene(so);
    const auto o = run("room, relative + network", so);
    EXPECT(o.after.median_centre < 0.01);
    const double s_est = o.result.problem.set_transform[7], s_true = scene.truth_global_transform[7];
    std::printf("  set scale: estimated %.5f, true %.5f\n", s_est, s_true);
    // 60 noisy centres at 6% of the scene size each: about 1% standard deviation.
    EXPECT(std::abs(s_est / s_true - 1) < 0.03);
    std::size_t found = 0;
    for (auto k : scene.bad_global)
      for (auto d : o.result.report.downweighted_global) found += d == k;
    std::printf("  bad global priors %zu, reported %zu of them\n", scene.bad_global.size(), found);
    EXPECT(found == scene.bad_global.size());
  }

  // Global priors alone, as the working frame (network poses only): the result is
  // in the network's frame and scale.
  {
    SceneOptions so;
    so.relative_priors = false;
    so.global_priors = true;
    so.global_relation = jba::FrameRelation::Identity;
    so.global_noise_rot = 0.01;
    so.global_noise_pos = 0.03;
    const auto o = run("room, network only", so, {}, false);
    EXPECT(o.after.median_centre < 0.01);
    EXPECT(std::abs(o.after.scale - 1) < 0.005);
  }

  // Frames in shuffled order: same accuracy.
  {
    SceneOptions so;
    so.drift_heading_per_m = 0.02;
    so.shuffle_frames = true;
    const auto o = run("room, shuffled frames", so);
    EXPECT(o.after.median_centre < 0.01);
  }

  // Two rooms joined by a doorway.
  {
    SceneOptions so;
    so.kind = SceneKind::House;
    so.images = 80;
    so.points = 20000;
    const auto o = run("house, two rooms", so);
    EXPECT(o.after.median_centre < 0.015);
    EXPECT(o.after.median_centre < 0.5 * o.before.median_centre);
  }

  // A building walked around from outside: the loop the tracker did not close.
  {
    SceneOptions so;
    so.kind = SceneKind::Exterior;
    so.images = 72;
    so.points = 16000;
    const auto o = run("exterior, open loop", so);
    EXPECT(o.after.median_centre < 0.03);
    EXPECT(o.after.median_centre < 0.5 * o.before.median_centre);
  }

  // Wide-angle and distorted cameras.
  for (jba::CameraModel model : {jba::CameraModel::OpenCV, jba::CameraModel::OpenCVFisheye}) {
    SceneOptions so;
    so.model = model;
    so.drift_heading_per_m = 0.02;
    const auto o = run(model == jba::CameraModel::OpenCV ? "room, OPENCV" : "room, OPENCV_FISHEYE", so);
    EXPECT(o.after.median_centre < 0.01);
  }

  return TEST_MAIN_RESULT();
}
