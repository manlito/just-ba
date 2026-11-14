#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "jba/camera_model.h"

namespace jba {

// A section of a problem file that this version does not interpret. Kept verbatim
// so reading and writing a file never loses data (document 11).
struct RawSection {
  std::string name;
  std::uint8_t dtype = 0;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> bytes;
};

// Bundle adjustment problem in structure-of-arrays form (document 03).
//
// Poses map world to camera, camera axes x right, y down, z forward. A pose is
// stored as qw, qx, qy, qz, tx, ty, tz.
struct Problem {
  CameraModel model = CameraModel::Pinhole;

  // One block of num_params(model) values per intrinsics block. Normally one block
  // per session (decision D11); the BAL harness uses one per image.
  std::vector<double> intrinsics;
  // Per block: IntrinsicsGroup mask of freed groups. Empty means all fixed.
  std::vector<std::uint8_t> intrinsics_free;

  // Per image.
  std::vector<double> poses;                    // 7 per image
  std::vector<std::int32_t> image_intrinsics;   // block index; empty means all 0
  std::vector<std::uint8_t> image_fixed;        // empty means all free

  // Per point.
  std::vector<double> points;                   // 3 per point
  std::vector<std::uint8_t> point_fixed;        // empty means all free

  // Per observation.
  std::vector<std::int32_t> obs_image;
  std::vector<std::int32_t> obs_point;
  std::vector<double> obs_xy;                   // 2 per observation, pixels
  std::vector<double> obs_sigma;                // empty means 1 pixel

  std::vector<RawSection> extra;

  std::size_t num_images() const { return poses.size() / 7; }
  std::size_t num_points() const { return points.size() / 3; }
  std::size_t num_observations() const { return obs_image.size(); }
  std::size_t num_intrinsics_blocks() const {
    return intrinsics.size() / static_cast<std::size_t>(num_params(model));
  }
  std::int32_t intrinsics_of_image(std::size_t image) const {
    return image_intrinsics.empty() ? 0 : image_intrinsics[image];
  }
};

// Checks sizes and index ranges. Returns an empty string when valid, otherwise a
// description of the first problem found.
std::string validate(const Problem& problem);

}  // namespace jba
