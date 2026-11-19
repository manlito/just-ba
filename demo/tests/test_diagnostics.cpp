// The mapper's diagnostics (D151): every model it grows, why an image was left out of
// the one returned (the points of it the image sees, a rejected PnP), each registered
// image's reprojection error, and the share of the images a pair's inliers cover, with
// the rule that drops pairs whose inliers cover too little. On the generated orbit,
// whole and cut in two halves that share no pair.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mapper.h"
#include "scenes.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++failures;
}

std::size_t count(const std::vector<char>& v) {
  std::size_t n = 0;
  for (char c : v) n += c != 0;
  return n;
}

}  // namespace

int main() {
  const demo::Scene scene = demo::make_scene("orbit");
  demo::MapperOptions mo;
  demo::thresholds_for_noise(scene.noise_px, mo);
  const std::size_t n = scene.capture.num_images();

  // The whole orbit: one model, every image in it, errors near the keypoints' noise.
  {
    const demo::Reconstruction r = demo::Mapper(mo).run(scene.capture);
    check(r.ok(), "orbit: the run succeeds (" + r.error + ")");
    check(r.all_models.size() == static_cast<std::size_t>(r.models), "orbit: every model grown is listed");
    check(r.kept_model >= 0 && r.kept_model < static_cast<int>(r.all_models.size()), "orbit: the kept model is one of them");
    const demo::Reconstruction::Model& kept = r.all_models[static_cast<std::size_t>(r.kept_model)];
    check(kept.registered == r.registered, "orbit: the kept model's images are the result's");
    check(kept.points.size() == r.problem.points.size(), "orbit: the kept model's points are the result's");
    check(r.image_points_seen.size() == n && r.image_pnp_inliers.size() == n && r.image_error_px.size() == n,
          "orbit: one value per image");
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!r.registered[i]) continue;
      check(std::isfinite(r.image_error_px[i]), "orbit: a registered image has an error");
      worst = std::max(worst, r.image_error_px[i]);
      check(r.image_points_seen[i] >= static_cast<std::int32_t>(mo.min_pnp_inliers), "orbit: a registered image sees points");
    }
    check(worst > 0.0 && worst < 4.0 * scene.noise_px, "orbit: per-image errors near the noise (" + std::to_string(worst) + ")");
    // Without image sizes there is no coverage.
    bool nan = true;
    for (float c : r.pair_coverage) nan = nan && std::isnan(c);
    check(r.pair_coverage.size() == scene.capture.pairs.size() && nan, "orbit: no coverage without image sizes");
    std::printf("orbit: %zu of %zu registered, worst image %.3f px\n", count(r.registered), n, worst);
  }

  // With the image sizes: coverage in (0, 1] for verified pairs; a rule no pair meets
  // leaves nothing to start from.
  {
    demo::Capture c = scene.capture;
    c.image_size = {static_cast<std::int32_t>(scene.width), static_cast<std::int32_t>(scene.height)};
    const demo::Reconstruction r = demo::Mapper(mo).run(c);
    check(r.ok(), "orbit with sizes: the run succeeds");
    int finite = 0;
    bool in_range = true;
    for (std::size_t i = 0; i < c.pairs.size(); ++i) {
      if (!std::isfinite(r.pair_coverage[i])) continue;
      ++finite;
      in_range = in_range && r.pair_coverage[i] > 0.0f && r.pair_coverage[i] <= 1.0f;
    }
    check(finite >= r.pairs_kept && in_range, "orbit with sizes: verified pairs have a coverage in (0, 1]");
    check(r.pairs_narrow == 0, "orbit with sizes: no rule, no pair left out");

    demo::MapperOptions strict = mo;
    strict.min_pair_coverage = 1.01;
    const demo::Reconstruction s = demo::Mapper(strict).run(c);
    check(!s.ok() && s.pairs_kept == 0, "coverage above 1: no pair kept, nothing to start from");
    check(s.pairs_narrow == r.pairs_kept, "coverage above 1: every pair kept before is too narrow");
    std::printf("orbit with sizes: %d pairs kept, %d left out at coverage 1.01\n", r.pairs_kept, s.pairs_narrow);
  }

  // Cut in two: pairs between the halves removed. Two models, each holding one half;
  // the half left out of the kept model sees none of its points.
  {
    demo::Capture c = scene.capture;
    const std::int32_t half = static_cast<std::int32_t>(n / 2);
    std::vector<demo::Pair> kept_pairs;
    for (const demo::Pair& p : c.pairs)
      if ((p.a < half) == (p.b < half)) kept_pairs.push_back(p);
    c.pairs = kept_pairs;
    const demo::Reconstruction r = demo::Mapper(mo).run(c);
    check(r.ok(), "two halves: the run succeeds (" + r.error + ")");
    check(r.all_models.size() == 2, "two halves: two models (" + std::to_string(r.all_models.size()) + ")");
    std::size_t both = 0, either = 0;
    for (std::size_t i = 0; i < n; ++i) {
      int in = 0;
      for (const auto& m : r.all_models) in += m.registered[i] != 0;
      both += in == 2;
      either += in >= 1;
    }
    check(both == 0, "two halves: no image in both models");
    check(either >= n - 2, "two halves: (nearly) every image in a model");
    bool unseen = true;
    for (std::size_t i = 0; i < n; ++i)
      if (!r.registered[i]) unseen = unseen && r.image_points_seen[i] < static_cast<std::int32_t>(mo.min_pnp_inliers);
    check(unseen, "two halves: the other half sees too few of the kept model's points");
    for (std::size_t k = 0; k < r.all_models.size(); ++k) {
      const auto& m = r.all_models[k];
      check(m.first_pair >= 0 && m.first_pair < static_cast<int>(c.pairs.size()), "two halves: a model names its first pair");
      check(std::isfinite(m.mean_error_px) && m.mean_error_px > 0.0 && !m.points.empty(), "two halves: a model has points");
      std::printf("two halves: model %zu%s, %zu images, %zu points, %.3f px\n", k + 1,
                  static_cast<int>(k) == r.kept_model ? " (kept)" : "", count(m.registered), m.points.size() / 3,
                  m.mean_error_px);
    }
  }

  if (failures) return 1;
  std::printf("all checks passed\n");
  return 0;
}
