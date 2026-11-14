// Writes a synthetic problem for the parity tests, with perturbed starting values.
//
//   jba_make_problem <out.jba> [--model NAME] [--images N] [--points N] [--seed N]
//                    [--noise PX] [--outliers FRACTION] [--free-intrinsics MASK]
//                    [--per-image-intrinsics] [--fix-images N] [--fix-points N] [--sigma PX]
//
// MASK is the IntrinsicsGroup bit mask: 1 focal, 2 principal point, 4 distortion.
// Not the synthetic generator of document 05 (milestone 2).

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "jba/geometry.h"
#include "jba/problem_io.h"
#include "synthetic.h"

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <out.jba> [options]\n", argv[0]);
      return 2;
    }
    const std::string out = argv[1];
    jba::CameraModel model = jba::CameraModel::SimpleRadial;
    int images = 12, points = 300, fix_images = 0, fix_points = 0, free_mask = 0;
    std::uint64_t seed = 5;
    double noise = 0.5, outliers = 0, sigma = 0;
    bool per_image = false;
    for (int a = 2; a < argc; ++a) {
      const std::string key = argv[a];
      if (key == "--per-image-intrinsics") {
        per_image = true;
        continue;
      }
      if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      const std::string val = argv[++a];
      if (key == "--model") {
        auto m = jba::model_from_name(val);
        if (!m) throw std::invalid_argument("unknown model " + val);
        model = *m;
      } else if (key == "--images") images = std::stoi(val);
      else if (key == "--points") points = std::stoi(val);
      else if (key == "--seed") seed = std::stoull(val);
      else if (key == "--noise") noise = std::stod(val);
      else if (key == "--outliers") outliers = std::stod(val);
      else if (key == "--free-intrinsics") free_mask = std::stoi(val);
      else if (key == "--fix-images") fix_images = std::stoi(val);
      else if (key == "--fix-points") fix_points = std::stoi(val);
      else if (key == "--sigma") sigma = std::stod(val);
      else throw std::invalid_argument("unknown option " + key);
    }

    auto p = jba_test::random_problem(model, images, points, seed, noise);
    std::mt19937_64 rng(seed + 1000);
    std::normal_distribution<double> n(0, 1);
    std::uniform_real_distribution<double> u(0, 1);

    // Gross outliers: move a fraction of observations by 20 to 60 pixels.
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      if (u(rng) < outliers) {
        const double angle = 6.283185307179586 * u(rng), len = 20 + 40 * u(rng);
        p.obs_xy[2 * k] += len * std::cos(angle);
        p.obs_xy[2 * k + 1] += len * std::sin(angle);
      }
    }
    if (sigma > 0) p.obs_sigma.assign(p.num_observations(), sigma);

    if (fix_images > 0) {
      p.image_fixed.assign(p.num_images(), 0);
      for (int i = 0; i < fix_images && i < images; ++i) p.image_fixed[static_cast<std::size_t>(i)] = 1;
    }
    if (fix_points > 0) {
      p.point_fixed.assign(p.num_points(), 0);
      for (int j = 0; j < fix_points && j < points; ++j) p.point_fixed[static_cast<std::size_t>(3 * j % points)] = 1;
    }

    // Perturb what is free.
    for (std::size_t j = 0; j < p.num_points(); ++j) {
      if (!p.point_fixed.empty() && p.point_fixed[j]) continue;
      for (int a = 0; a < 3; ++a) p.points[3 * j + static_cast<std::size_t>(a)] += 0.02 * n(rng);
    }
    for (std::size_t i = 0; i < p.num_images(); ++i) {
      if (!p.image_fixed.empty() && p.image_fixed[i]) continue;
      double* pose = &p.poses[7 * i];
      const double step[6] = {0.01 * n(rng), 0.01 * n(rng), 0.01 * n(rng), 0.02 * n(rng), 0.02 * n(rng), 0.02 * n(rng)};
      double moved[7];
      jba::pose_retract(pose, step, moved);
      for (int a = 0; a < 7; ++a) pose[a] = moved[a];
    }
    const int np = jba::num_params(model);
    if (per_image) {
      const std::vector<double> block = p.intrinsics;
      p.intrinsics.clear();
      for (int i = 0; i < images; ++i) p.intrinsics.insert(p.intrinsics.end(), block.begin(), block.end());
      p.image_intrinsics.resize(static_cast<std::size_t>(images));
      for (int i = 0; i < images; ++i) p.image_intrinsics[static_cast<std::size_t>(i)] = i;
    }
    if (free_mask != 0) {
      p.intrinsics_free.assign(p.num_intrinsics_blocks(), static_cast<std::uint8_t>(free_mask));
      for (std::size_t c = 0; c < p.num_intrinsics_blocks(); ++c) {
        for (int a = 0; a < np; ++a) {
          double& v = p.intrinsics[c * static_cast<std::size_t>(np) + static_cast<std::size_t>(a)];
          const auto g = jba::intrinsics_group(model, a);
          if (!(free_mask & g)) continue;
          if (g == jba::kFocal) v *= 1 + 0.01 * n(rng);
          else if (g == jba::kPrincipalPoint) v += 2 * n(rng);
          else v += 0.005 * n(rng);
        }
      }
    }
    jba::write_problem(out, p);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_make_problem: %s\n", e.what());
    return 1;
  }
}
