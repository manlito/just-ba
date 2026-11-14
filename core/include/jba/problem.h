#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "jba/camera_model.h"

namespace jba {

// How a set of global pose priors relates to the working frame (D19, document 02).
// The transform maps the set's frame into the working frame:
// X_working = s * R * X_set + t.
enum class FrameRelation : std::int32_t {
  Identity = 0,    // the set is the working frame; nothing is estimated
  Rigid = 1,       // R and t are estimated, s stays 1
  Similarity = 2,  // R, t and s are estimated
};

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

  // One block of num_params(model) values per intrinsics block, all of the same
  // model. Normally one block per session (D11); several are accepted, for example
  // one per image for BAL (D53).
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

  // Relative pose priors (document 02). Prior k ties images i = rel_images[2k] and
  // j = rel_images[2k + 1] to the motion T_j * T_i^-1 (camera i coordinates to
  // camera j coordinates), stored like a pose: qw qx qy qz tx ty tz.
  // Residual: rotation (radians) then translation (metres, camera i axes).
  std::vector<std::int32_t> rel_images;         // 2 per prior
  std::vector<double> rel_pose;                 // 7 per prior
  // Weights, at most one of the two. sigma: rotation (rad) and translation (m)
  // standard deviations; +inf leaves that part out. sqrt_info: 6 x 6 row-major
  // whitening W, residual W r. Both empty: defaults from the solver options.
  std::vector<double> rel_sigma;                // 2 per prior
  std::vector<double> rel_sqrt_info;            // 36 per prior

  // Global pose priors. Prior k ties image glob_image[k] to a pose in the frame of
  // set glob_set[k], stored world to camera like image poses. Residual: rotation
  // (radians) then camera centre (in the set's own units).
  std::vector<std::int32_t> glob_image;         // 1 per prior
  std::vector<std::int32_t> glob_set;           // empty means all in set 0
  std::vector<double> glob_pose;                // 7 per prior
  std::vector<double> glob_sigma;               // 2 per prior: rotation (rad), position (set units)
  std::vector<double> glob_sqrt_info;           // 36 per prior

  // Per global prior set. Empty means one set, the working frame.
  std::vector<FrameRelation> set_relation;
  // 8 per set: qw qx qy qz tx ty tz s. Starting value, and the estimate after a
  // solve. Empty means the identity for every set.
  std::vector<double> set_transform;

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
  std::size_t num_relative_priors() const { return rel_images.size() / 2; }
  std::size_t num_global_priors() const { return glob_image.size(); }
  std::size_t num_prior_sets() const {
    return !set_relation.empty() ? set_relation.size() : (glob_image.empty() ? 0 : 1);
  }
  std::int32_t set_of_global_prior(std::size_t k) const { return glob_set.empty() ? 0 : glob_set[k]; }
  FrameRelation relation_of_set(std::size_t s) const {
    return set_relation.empty() ? FrameRelation::Identity : set_relation[s];
  }
};

// Checks sizes and index ranges. Returns an empty string when valid, otherwise a
// description of the first problem found.
std::string validate(const Problem& problem);

}  // namespace jba
