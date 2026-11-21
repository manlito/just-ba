// The mapper's group prior on intrinsics (D166): a block per image whose focal
// lengths differ a little (autofocus), a calibration 5 % off. Grouped, the per-image
// focal lengths land nearer the truth than refined one by one; without groups the
// mapper is as before; camera.group survives a capture file.

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

// Per registered image: its block's focal length against the truth's, RMS.
double focal_rms(const demo::Scene& scene, const demo::Reconstruction& r) {
  const std::size_t np = static_cast<std::size_t>(jba::num_params(scene.capture.model));
  double sq = 0.0;
  for (std::size_t k = 0; k < r.problem.num_images(); ++k) {
    const std::size_t b = static_cast<std::size_t>(r.problem.intrinsics_of_image(k));
    sq += std::pow(r.problem.intrinsics[np * b] / scene.capture.intrinsics[np * b] - 1.0, 2);
  }
  return std::sqrt(sq / static_cast<double>(r.problem.num_images()));
}

}  // namespace

int main() {
  double free_sum = 0.0, grouped_sum = 0.0;
  for (std::uint64_t seed = 1; seed <= 4; ++seed) {
    demo::SceneOptions so;
    so.seed = seed;
    so.focal_jitter = 0.004;
    const demo::Scene scene = demo::make_scene("orbit", so);  // the scene whose focal lengths are weakest
    demo::MapperOptions mo;
    mo.seed = 7 + static_cast<std::uint32_t>(seed);
    demo::thresholds_for_noise(so.noise_px, mo);
    mo.refine_intrinsics = jba::kFocal | jba::kDistortion;
    demo::Mapper mapper(mo);
    demo::Capture c = given(scene);
    const demo::Reconstruction free = mapper.run(c);
    c.intrinsics_group.assign(c.intrinsics.size() / static_cast<std::size_t>(jba::num_params(c.model)), 0);
    const demo::Reconstruction grouped = mapper.run(c);
    check(free.ok() && grouped.ok(), "both reconstruct, seed " + std::to_string(seed));
    if (!free.ok() || !grouped.ok()) continue;
    check(grouped.problem.num_images() == free.problem.num_images(), "the same images registered");
    check(free.problem.intrinsics_group.empty(), "no groups without camera.group");
    check(grouped.problem.intrinsics_group_mean.size() == 4 && std::isfinite(grouped.problem.intrinsics_group_mean[0]),
          "the mean is estimated");
    const double f = focal_rms(scene, free), g = focal_rms(scene, grouped);
    std::printf("seed %d: focal rms free %.4f %%, grouped %.4f %%\n", static_cast<int>(seed), 100.0 * f, 100.0 * g);
    free_sum += f;
    grouped_sum += g;
  }
  check(grouped_sum < 0.9 * free_sum, "grouped focal lengths nearer the truth on average");

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
  }
  if (failures == 0) std::printf("all passed\n");
  return failures == 0 ? 0 : 1;
}
