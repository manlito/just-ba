// Generated captures with ground truth: cameras, points, noisy keypoints, and the
// matches of every pair corrupted the way a matcher's output is (wrong pairings inside
// overlapping pairs, and pairs that do not overlap at all but still carry matches).
// The layouts are poselib-gpu's incremental test's; the draws are the demo's own.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "capture_io.h"
#include "mapper.h"

namespace demo {

struct SceneOptions {
  std::uint64_t seed = 0;
  double noise_px = 0.5;          // standard deviation per coordinate
  double wrong_fraction = 0.25;   // of each overlapping pair's matches
  int false_pairs = -1;           // pairs with no overlap that get random matches; -1: the scene's own
};

struct Scene {
  std::string name;
  Capture capture;
  double width = 1600.0, height = 1200.0;
  double noise_px = 0.5;                            // of the keypoints, per coordinate
  std::vector<double> poses;                        // 7 per image, world to camera (truth)
  std::vector<double> points;                       // 3 per point (truth)
  std::vector<std::vector<std::int32_t>> kp_point;  // per image: the point of each keypoint
  std::vector<std::vector<std::uint8_t>> correct;   // per pair, per match: both keypoints see one point
  std::vector<int> shared;                          // per pair: points really seen by both
};

// "orbit", "wall" or "grid". Throws std::invalid_argument for another name.
Scene make_scene(const std::string& name, const SceneOptions& options = {});
const std::vector<std::string>& scene_names();

// The scene as a capture file, its truth as the reference: every image registered, the
// points seen by two images or more, every keypoint of such a point an observation.
CaptureFile to_capture_file(const Scene& scene);

}  // namespace demo
