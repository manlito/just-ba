#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "jba/covariance.h"
#include "jba/problem.h"
#include "jba/report.h"
#include "jba/tracks.h"

namespace jba {

// Export of a solved problem (document 01, outputs; D89): COLMAP's text model for
// interchange, and the library's own problem file with result sections for what
// COLMAP's format cannot hold.

struct ColmapExportOptions {
  // Per image: the NAME column. Empty: "image_000000" and so on.
  std::vector<std::string> image_names;
  // Per intrinsics block: width, height. Empty: 2 cx and 2 cy, rounded up. COLMAP's
  // pixel convention puts the centre of the first pixel at (0.5, 0.5); keypoints
  // and principal points are written as they are, so they should follow it.
  std::vector<std::int32_t> image_size;
  // Per image, 1 to export. Empty: every image with an observation.
  std::vector<char> images;
  // Optional: every keypoint of each image and the keypoint of each observation, so
  // POINTS2D lists all keypoints as COLMAP does (unobserved ones with POINT3D_ID -1).
  // Without them, POINTS2D holds the observations.
  const Keypoints* keypoints = nullptr;
  std::vector<std::int32_t> obs_keypoint;
  std::vector<std::uint8_t> point_color;  // 3 per point; empty: grey
};

// Writes cameras.txt, images.txt and points3D.txt into `directory` (which must
// exist). IDs are indices plus one: camera = intrinsics block, image, point. A point
// is written when it has an observation in an exported image; its ERROR is the mean
// reprojection error there. Throws std::runtime_error.
void export_colmap(const Problem& problem, const std::string& directory, const ColmapExportOptions& options = {});

// The problem file of document 11 with `result.*` sections added (any already there
// are replaced): covariances, the residual report, and the keypoint of each
// observation, each when given. Sections are listed in document 11. Throws
// std::runtime_error.
void write_result(const std::string& path, const Problem& problem, const Covariance* covariance,
                  const ResidualReport* report, const std::vector<std::int32_t>* obs_keypoint = nullptr);

}  // namespace jba
