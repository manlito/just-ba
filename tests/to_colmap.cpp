// One problem in two forms, for the GPU speed comparison with COLMAP's Caspar backend
// (milestone 5, document 05): the library's problem file and a COLMAP text model.
//
//   jba_to_colmap <problem.jba | bal.txt> <out.jba> <colmap_dir> [--simple-radial]
//
// Caspar takes the reprojection residual on SIMPLE_RADIAL or PINHOLE cameras only, so
// --simple-radial turns RADIAL into SIMPLE_RADIAL by dropping k2 (BAL's second radial
// coefficient). A principal point at the origin (BAL) is moved to the middle of an
// image that holds the observations, and the observations with it, which leaves every
// residual as it was. Observations of a point behind its camera are dropped, as
// COLMAP drops them, and so are points left without observations. Priors, depth and
// segments are refused: Caspar has none of them.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/export.h"
#include "jba/geometry.h"
#include "jba/problem.h"
#include "jba/problem_io.h"

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::fprintf(stderr, "usage: %s <problem.jba | bal.txt> <out.jba> <colmap_dir> [--simple-radial]\n", argv[0]);
      return 2;
    }
    const std::string in = argv[1];
    bool simple_radial = false;
    for (int a = 4; a < argc; ++a) {
      if (std::string(argv[a]) == "--simple-radial") simple_radial = true;
      else throw std::invalid_argument(std::string("unknown option ") + argv[a]);
    }
    jba::Problem p = ends_with(in, ".txt") ? jba::read_bal(in) : jba::read_problem(in);
    if (p.num_relative_priors() || p.num_global_priors() || p.num_gravity_priors() || p.num_lines() ||
        !p.obs_depth.empty())
      throw std::invalid_argument("priors, segments and depth have no COLMAP or Caspar form");

    const std::size_t n_blk = p.num_intrinsics_blocks();
    if (simple_radial) {
      if (p.model != jba::CameraModel::Radial) throw std::invalid_argument("--simple-radial needs a RADIAL problem");
      std::vector<double> k;
      for (std::size_t b = 0; b < n_blk; ++b) k.insert(k.end(), &p.intrinsics[5 * b], &p.intrinsics[5 * b + 4]);
      p.intrinsics = k;
      p.model = jba::CameraModel::SimpleRadial;
    }
    const int np = jba::num_params(p.model);
    const bool one_focal = p.model == jba::CameraModel::SimplePinhole || p.model == jba::CameraModel::SimpleRadial ||
                           p.model == jba::CameraModel::Radial;
    const int cx = one_focal ? 1 : 2;

    // Points in front of their cameras only.
    std::vector<std::size_t> keep;
    std::vector<int> seen(p.num_points(), 0);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      const double* T = &p.poses[7 * static_cast<std::size_t>(p.obs_image[k])];
      double pc[3];
      jba::quat_rotate(T, &p.points[3 * static_cast<std::size_t>(p.obs_point[k])], pc);
      if (pc[2] + T[6] <= 0) continue;
      keep.push_back(k);
      ++seen[static_cast<std::size_t>(p.obs_point[k])];
    }
    std::vector<std::int32_t> new_index(p.num_points(), -1);
    std::vector<double> points;
    std::vector<std::uint8_t> point_fixed;
    for (std::size_t j = 0; j < p.num_points(); ++j) {
      if (seen[j] == 0) continue;
      new_index[j] = static_cast<std::int32_t>(points.size() / 3);
      points.insert(points.end(), &p.points[3 * j], &p.points[3 * j + 3]);
      if (!p.point_fixed.empty()) point_fixed.push_back(p.point_fixed[j]);
    }
    std::vector<std::int32_t> obs_image, obs_point;
    std::vector<double> obs_xy, obs_sigma;
    for (std::size_t k : keep) {
      obs_image.push_back(p.obs_image[k]);
      obs_point.push_back(new_index[static_cast<std::size_t>(p.obs_point[k])]);
      obs_xy.insert(obs_xy.end(), &p.obs_xy[2 * k], &p.obs_xy[2 * k + 2]);
      if (!p.obs_sigma.empty()) obs_sigma.push_back(p.obs_sigma[k]);
    }
    const std::size_t dropped_obs = p.num_observations() - keep.size(), dropped_pts = p.num_points() - points.size() / 3;
    p.points = points;
    p.point_fixed = point_fixed;
    p.obs_image = obs_image;
    p.obs_point = obs_point;
    p.obs_xy = obs_xy;
    p.obs_sigma = obs_sigma;

    // Each block's image: the principal point in its middle; a principal point at the
    // origin moves to the middle of a frame that holds its observations.
    std::vector<double> half(2 * n_blk, 0);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      const auto b = static_cast<std::size_t>(p.intrinsics_of_image(static_cast<std::size_t>(p.obs_image[k])));
      const double* K = &p.intrinsics[static_cast<std::size_t>(np) * b];
      for (int c = 0; c < 2; ++c) half[2 * b + c] = std::max(half[2 * b + c], std::abs(p.obs_xy[2 * k + c] - K[cx + c]));
    }
    std::vector<double> shift(2 * n_blk, 0);
    std::vector<std::int32_t> size(2 * n_blk);
    for (std::size_t b = 0; b < n_blk; ++b) {
      double* K = &p.intrinsics[static_cast<std::size_t>(np) * b];
      for (int c = 0; c < 2; ++c) {
        const double h = std::ceil(half[2 * b + c]) + 1;
        if (K[cx + c] == 0) {
          shift[2 * b + c] = h;
          K[cx + c] = h;
        }
        size[2 * b + c] = static_cast<std::int32_t>(std::ceil(2 * std::max(h, K[cx + c])));
      }
    }
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      const auto b = static_cast<std::size_t>(p.intrinsics_of_image(static_cast<std::size_t>(p.obs_image[k])));
      for (int c = 0; c < 2; ++c) p.obs_xy[2 * k + c] += shift[2 * b + c];
    }

    jba::write_problem(argv[2], p);
    std::filesystem::create_directories(argv[3]);
    jba::ColmapExportOptions eo;
    eo.image_size = size;
    jba::export_colmap(p, argv[3], eo);
    std::printf("{\"images\": %zu, \"points\": %zu, \"observations\": %zu, \"model\": \"%s\", "
                "\"dropped_observations\": %zu, \"dropped_points\": %zu}\n",
                p.num_images(), p.num_points(), p.num_observations(), std::string(jba::model_name(p.model)).c_str(),
                dropped_obs, dropped_pts);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
