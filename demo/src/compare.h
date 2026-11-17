// A reconstruction against another of the same capture (document 05, level 3): COLMAP's
// on the same matches, or a generated scene's truth. Poses are compared on the images
// both registered, after aligning them.
#pragma once

#include <cstddef>

#include "capture_io.h"
#include "mapper.h"

namespace demo {

struct Side {
  std::size_t registered = 0, points = 0, observations = 0;
  double mean_track_length = 0.0;
  double mean_error_px = 0.0;  // mean reprojection error of its observations
};

struct Comparison {
  Side ours, reference;
  std::size_t common = 0;                       // images both registered
  // Over the common images: rotation error after the best common rotation; camera
  // centre error after the similarity that best aligns the centres, as a percentage of
  // the diagonal of the box around the reference's centres, and in its units.
  double rot_median_deg = 0.0, rot_max_deg = 0.0;
  double centre_median_pct = 0.0, centre_max_pct = 0.0;
  double centre_median = 0.0;
};

// `capture` gives the intrinsics for the reference's reprojection errors when it has
// none of its own.
Comparison compare(const Capture& capture, const Reference& ref, const Reconstruction& r);

}  // namespace demo
