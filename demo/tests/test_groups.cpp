// The mapper's group prior on intrinsics (D166): a block per image whose focal
// lengths differ a little (autofocus), a calibration 5 % off. Grouped, the per-image
// focal lengths land nearer the truth than refined one by one; a grouping with every
// block at -1 is the ungrouped run bit for bit; bad groups are refused; two groups, a
// guessed focal length and blocks below refine_min_images work; camera.group
// survives a capture file.
//
// Also the live demo's cameras (D167: a block per camera, image size and focal length)
// and the focal search per block (D168: two cameras of one guess get a factor each).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "capture_io.h"
#include "mapper.h"
#include "scenes.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++failures;
}

// The capture the mapper is given: every block at the mean focal length, 5 % off.
demo::Capture given(const demo::Scene& scene) {
  demo::Capture c = scene.capture;
  const std::size_t np = static_cast<std::size_t>(jba::num_params(c.model));
  const std::size_t blocks = c.intrinsics.size() / np;
  const std::size_t n_focal = c.model == jba::CameraModel::Pinhole ? 2 : 1;
  for (std::size_t k = 0; k < n_focal; ++k) {
    double mean = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) mean += c.intrinsics[np * b + k] / static_cast<double>(blocks);
    for (std::size_t b = 0; b < blocks; ++b) c.intrinsics[np * b + k] = 1.05 * mean;
  }
  return c;
}

// Per registered image: its block's focal lengths (fx, and fy for PINHOLE) against
// the truth's, RMS.
double focal_rms(const demo::Scene& scene, const demo::Reconstruction& r) {
  const std::size_t np = static_cast<std::size_t>(jba::num_params(scene.capture.model));
  const std::size_t n_focal = scene.capture.model == jba::CameraModel::Pinhole ? 2 : 1;
  double sq = 0.0;
  for (std::size_t k = 0; k < r.problem.num_images(); ++k) {
    const std::size_t b = static_cast<std::size_t>(r.problem.intrinsics_of_image(k));
    for (std::size_t f = 0; f < n_focal; ++f)
      sq += std::pow(r.problem.intrinsics[np * b + f] / scene.capture.intrinsics[np * b + f] - 1.0, 2);
  }
  return std::sqrt(sq / static_cast<double>(n_focal * r.problem.num_images()));
}

demo::MapperOptions options(std::uint64_t seed, double noise) {
  demo::MapperOptions mo;
  mo.seed = 7 + static_cast<std::uint32_t>(seed);
  demo::thresholds_for_noise(noise, mo);
  mo.refine_intrinsics = jba::kFocal | jba::kDistortion;
  return mo;
}

bool same_result(const demo::Reconstruction& a, const demo::Reconstruction& b) {
  return a.ok() && b.ok() && a.problem.poses == b.problem.poses && a.problem.points == b.problem.points &&
         a.problem.intrinsics == b.problem.intrinsics;
}

}  // namespace

int main() {
  double free_sum = 0.0, grouped_sum = 0.0;
  for (std::uint64_t seed = 1; seed <= 4; ++seed) {
    demo::SceneOptions so;
    so.seed = seed;
    so.focal_jitter = 0.004;
    const demo::Scene scene = demo::make_scene("orbit", so);  // the scene whose focal lengths are weakest
    demo::Mapper mapper(options(seed, so.noise_px));
    demo::Capture c = given(scene);
    const demo::Reconstruction free = mapper.run(c);
    c.intrinsics_group.assign(c.intrinsics.size() / static_cast<std::size_t>(jba::num_params(c.model)), 0);
    const demo::Reconstruction grouped = mapper.run(c);
    check(free.ok() && grouped.ok(), "both reconstruct, seed " + std::to_string(seed));
    if (!free.ok() || !grouped.ok()) continue;
    check(grouped.problem.num_images() == free.problem.num_images(), "the same images registered");
    check(free.problem.intrinsics_group.empty(), "no groups without camera.group");
    check(grouped.problem.intrinsics_group_mean.size() == static_cast<std::size_t>(jba::num_params(c.model)) &&
              std::isfinite(grouped.problem.intrinsics_group_mean[0]),
          "the mean is estimated");
    const double f = focal_rms(scene, free), g = focal_rms(scene, grouped);
    std::printf("seed %d: focal rms free %.4f %%, grouped %.4f %%\n", static_cast<int>(seed), 100.0 * f, 100.0 * g);
    free_sum += f;
    grouped_sum += g;
  }
  check(grouped_sum < 0.9 * free_sum, "grouped focal lengths nearer the truth on average");

  {
    demo::SceneOptions so;
    so.seed = 5;
    so.focal_jitter = 0.004;
    const demo::Scene scene = demo::make_scene("wall", so);
    const std::size_t np = static_cast<std::size_t>(jba::num_params(scene.capture.model));
    demo::Capture c = given(scene);
    const std::size_t blocks = c.intrinsics.size() / np;
    demo::Mapper mapper(options(5, so.noise_px));
    const demo::Reconstruction free = mapper.run(c);
    // Every block at -1: no groups, the same run bit for bit.
    demo::Capture none = c;
    none.intrinsics_group.assign(blocks, -1);
    check(same_result(free, mapper.run(none)), "every block at -1 is the ungrouped run");
    // Bad groupings and options are refused, not thrown.
    demo::Capture bad = c;
    bad.intrinsics_group.assign(blocks + 1, 0);
    check(!mapper.run(bad).ok(), "a group per block or the capture is refused");
    bad.intrinsics_group.assign(blocks, 0);
    bad.intrinsics_group[1] = -2;
    check(!mapper.run(bad).ok(), "a group below -1 is refused");
    bad.intrinsics_group[1] = static_cast<std::int32_t>(blocks);
    check(!mapper.run(bad).ok(), "a group beyond the blocks is refused");
    demo::MapperOptions bad_options = options(5, so.noise_px);
    bad_options.group_focal_fraction = 0.0;
    demo::Capture grouped = c;
    grouped.intrinsics_group.assign(blocks, 0);
    check(!demo::Mapper(bad_options).run(grouped).ok(), "a sigma of 0 is refused");
    // Two groups, one block left out: two means.
    demo::Capture two = c;
    two.intrinsics_group.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) two.intrinsics_group[b] = static_cast<std::int32_t>(b % 2);
    two.intrinsics_group[0] = -1;
    const demo::Reconstruction r2 = mapper.run(two);
    check(r2.ok() && r2.problem.intrinsics_group_mean.size() == 2 * np && std::isfinite(r2.problem.intrinsics_group_mean[np]),
          "two groups, two means");
    // A guessed focal length 30 % off: searched for, then grouped.
    demo::Capture guessed = grouped;
    for (std::size_t b = 0; b < blocks; ++b) guessed.intrinsics[np * b] *= 1.3 / 1.05;
    guessed.focal_unknown.assign(blocks, 1);
    demo::MapperOptions search = options(5, so.noise_px);
    search.search_focal = true;
    const demo::Reconstruction rg = demo::Mapper(search).run(guessed);
    check(rg.ok() && focal_rms(scene, rg) < 0.01, "a guessed focal length is searched for and grouped");
    std::printf("wall, guessed focal: rms %.4f %%\n", 100.0 * (rg.ok() ? focal_rms(scene, rg) : 1.0));
    // refine_min_images 2 with a block per image: grouped, each block is refined (its
    // group has enough images), so none stays at the given 5 % off.
    demo::MapperOptions two_min = options(5, so.noise_px);
    two_min.refine_min_images = 2;
    const demo::Reconstruction rm = demo::Mapper(two_min).run(grouped);
    check(rm.ok() && focal_rms(scene, rm) < 0.01, "blocks of one image are refined when their group has enough");
    std::printf("wall, refine_min_images 2: rms %.4f %% (given 5 %% off)\n", 100.0 * (rm.ok() ? focal_rms(scene, rm) : 1.0));
  }

  // camera.group in a capture file.
  {
    demo::SceneOptions so;
    so.focal_jitter = 0.004;
    const demo::Scene scene = demo::make_scene("wall", so);
    demo::CaptureFile file = demo::to_capture_file(scene);
    const std::size_t blocks = file.capture.intrinsics.size() / static_cast<std::size_t>(jba::num_params(file.capture.model));
    check(blocks == scene.capture.num_images(), "a block per image");
    file.capture.intrinsics_group.assign(blocks, 0);
    file.capture.intrinsics_group.back() = -1;
    const auto path = (std::filesystem::temp_directory_path() / "jba_demo_groups.jbc").string();
    demo::write_capture(path, file);
    const demo::CaptureFile back = demo::read_capture(path);
    std::filesystem::remove(path);
    check(back.capture.intrinsics_group == file.capture.intrinsics_group, "camera.group round-trips");
    check(back.capture.block_camera.empty(), "no camera.id without cameras");
    file.capture.block_camera.assign(blocks, 3);
    file.capture.block_camera[0] = 1;
    demo::write_capture(path, file);
    check(demo::read_capture(path).capture.block_camera == file.capture.block_camera, "camera.id round-trips");
    std::filesystem::remove(path);
    demo::Capture bad = file.capture;
    bad.block_camera.pop_back();
    check(!demo::Mapper(options(0, so.noise_px)).run(bad).ok(), "a camera per block or the capture is refused");
  }
  // The live demo's blocks: two videos of one size and unknown focal length are two
  // cameras; photos of one size and focal length share a block.
  {
    const std::int32_t size[] = {1280, 720, 1280, 720, 1280, 720, 1280, 720, 1600, 1200, 1280, 720};
    const double nan = std::nan(""), focal[] = {nan, nan, nan, nan, 1500.0, 0.0};
    const std::int32_t camera[] = {1, 1, 2, 2, 0, 1};
    demo::Capture c;
    demo::simple_radial_blocks(c, 6, size, focal, camera);
    check(c.image_intrinsics == std::vector<std::int32_t>({0, 0, 1, 1, 2, 0}), "a block per camera");
    check(c.focal_unknown == std::vector<std::uint8_t>({1, 1, 0}), "both videos' focal lengths are guesses");
    check(c.intrinsics.size() == 12 && c.intrinsics[0] == 1.2 * 1280 && c.intrinsics[4] == 1.2 * 1280 &&
              c.intrinsics[8] == 1500.0 && c.intrinsics[1] == 640.0 && c.intrinsics[2] == 360.0,
          "the guess is 1.2 times the larger side, the principal point the centre");
    check(c.image_size == std::vector<std::int32_t>({1280, 720, 1280, 720, 1600, 1200}), "sizes per block");
    check(c.block_camera == std::vector<std::int32_t>({1, 2, 0}), "each block's camera");
    demo::simple_radial_blocks(c, 6, size, focal, nullptr);
    check(c.image_intrinsics == std::vector<std::int32_t>({0, 0, 0, 0, 1, 0}) && c.focal_unknown.size() == 2,
          "without cameras, a block per size and focal length");
  }

  // Two cameras, the second's focal length half the first's, both given 1.5 times the
  // first's and marked a guess: searched per block, each start lands within 5 % of its
  // truth; with one factor for both, one of them can't.
  for (const char* name : {"orbit", "wall", "grid"}) {
    demo::SceneOptions so;
    so.seed = 3;
    so.second_focal = 0.5;
    const demo::Scene scene = demo::make_scene(name, so);
    const std::size_t np = static_cast<std::size_t>(jba::num_params(scene.capture.model));
    demo::Capture c = scene.capture;
    check(c.intrinsics.size() == 2 * np, "two blocks");
    const std::size_t n_focal = c.model == jba::CameraModel::Pinhole ? 2 : 1;
    const double truth[2] = {c.intrinsics[0], c.intrinsics[np]};
    for (std::size_t k = 0; k < n_focal; ++k) {
      c.intrinsics[k] *= 1.5;
      c.intrinsics[np + k] = c.intrinsics[k];
    }
    c.focal_unknown.assign(2, 1);
    demo::MapperOptions search = options(3, so.noise_px);
    search.search_focal = true;
    const demo::Reconstruction r = demo::Mapper(search).run(c);
    check(r.ok() && r.focal_factors.size() == 2, std::string(name) + ": a factor per block");
    if (r.ok() && r.focal_factors.size() == 2) {
      for (std::size_t b = 0; b < 2; ++b)
        check(std::abs(c.intrinsics[np * b] * r.focal_factors[b] / truth[b] - 1.0) < 0.05,
              std::string(name) + ": camera " + std::to_string(b + 1) + "'s search lands on its focal length");
      check(r.focal_factor == std::min(r.focal_factors[0], r.focal_factors[1]), "the scalar is the lower median");
      check(focal_rms(scene, r) < 0.01, std::string(name) + ": both refined to their truth");
      std::printf("%s, two cameras: searched x%.3f and x%.3f (truth x%.3f and x%.3f), refined rms %.4f %%\n", name,
                  r.focal_factors[0], r.focal_factors[1], truth[0] / c.intrinsics[0], truth[1] / c.intrinsics[np],
                  100.0 * focal_rms(scene, r));
    }
    search.search_focal_per_block = false;
    const demo::Reconstruction joint = demo::Mapper(search).run(c);
    check(joint.focal_factors.size() == 2 && joint.focal_factors[0] == joint.focal_factors[1] &&
              joint.focal_factor == joint.focal_factors[0],
          std::string(name) + ": one factor without the per-block search");
    // The two blocks as one camera's (its photos of two sizes, say), or with no cameras
    // named: one factor, the search before D168 bit for bit. A 3-pair block gets no
    // factor of its own.
    for (const std::vector<std::int32_t>& cameras : {std::vector<std::int32_t>{7, 7}, std::vector<std::int32_t>{}}) {
      demo::Capture same = c;
      same.block_camera = cameras;
      search.search_focal_per_block = true;
      const demo::Reconstruction r1 = demo::Mapper(search).run(same);
      check(same_result(r1, joint) && r1.focal_factors == joint.focal_factors && r1.focal_factor == joint.focal_factor,
            std::string(name) + ": blocks of one camera share a factor, as before");
    }
    // One guessed block beside a known one: from its own pairs too.
    demo::Capture half = c;
    for (std::size_t k = 0; k < n_focal; ++k) half.intrinsics[k] = scene.capture.intrinsics[k];
    half.focal_unknown = {0, 1};
    search.search_focal_per_block = true;
    const demo::Reconstruction one = demo::Mapper(search).run(half);
    check(one.ok() && one.focal_factors.size() == 2 && one.focal_factors[0] == 1.0 &&
              std::abs(half.intrinsics[np] * one.focal_factors[1] / truth[1] - 1.0) < 0.05 && focal_rms(scene, one) < 0.01,
          std::string(name) + ": one guessed block beside a known one is found");
    // A capture of one block: the search before D168, bit for bit.
    demo::SceneOptions plain;
    plain.seed = 3;
    const demo::Scene single = demo::make_scene(name, plain);
    demo::Capture guess = single.capture;
    for (std::size_t k = 0; k < n_focal; ++k) guess.intrinsics[k] *= 1.5;
    guess.focal_unknown.assign(1, 1);
    const demo::Reconstruction with = demo::Mapper(search).run(guess);
    search.search_focal_per_block = false;
    check(same_result(with, demo::Mapper(search).run(guess)) && with.focal_factors == std::vector<double>({with.focal_factor}),
          std::string(name) + ": one block is searched as before");
    std::printf("%s, one known: searched x%.3f, refined rms %.4f %%\n", name, one.focal_factors[1], 100.0 * focal_rms(scene, one));
  }
  if (failures == 0) std::printf("all passed\n");
  return failures == 0 ? 0 : 1;
}
