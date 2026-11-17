// A reconstruction of a generated scene against its ground truth, after the
// similarity that best aligns the camera centres.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "mapper.h"
#include "scenes.h"

namespace demo {

struct Metrics {
  std::size_t images = 0, registered = 0, points = 0, observations = 0;
  double mean_error_px = 0.0;
  double rot_median_deg = 0.0, rot_max_deg = 0.0;
  double centre_median_pct = 0.0, centre_max_pct = 0.0;  // of the scene extent
  double point_median_pct = 0.0;
  // Pair verification: pairs with no true overlap kept, and the inlier masks of the
  // kept pairs against the true matches.
  int false_pairs_kept = 0;
  double mask_precision = 0.0, mask_recall = 0.0;
  // The similarity from the reconstruction to the truth: X_true = s R X + t.
  double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, t[3] = {0, 0, 0}, s = 1.0;
  bool pass = false;
  std::string why;  // the first threshold missed
};

// Pass: every image registered, median rotation error below 0.5 degrees, median
// camera-centre error below 1 % of the scene extent, mean reprojection below 1.5 px or
// twice the keypoint noise, whichever is larger.
Metrics evaluate(const Scene& scene, const Reconstruction& r);

}  // namespace demo
