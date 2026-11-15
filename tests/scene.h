#pragma once

// Synthetic generator (document 05): scenes shaped like the main use case, with
// ground truth, keypoints and matches, and tracker-like pose priors that drift,
// jump, and carry noise. Deterministic for a seed on every platform.

#include <cstdint>
#include <vector>

#include "jba/camera_model.h"
#include "jba/problem.h"
#include "jba/refine.h"

namespace jba_test {

enum class SceneKind {
  Room,      // walls, floor and ceiling of one room; the camera walks a loop inside, looking out
  House,     // two rooms joined by a doorway; thin connection between them
  Exterior,  // a box building walked around from outside, looking in; a loop the tracker has not closed
};

// Line segments for the Manhattan prior (document 05): on the surfaces along the
// building's axes, plus distractors.
enum class LineMode {
  None,
  Manhattan,  // along the axes only
  Mixed,      // 60% along the axes; the rest sloped in their surface, in random directions, or on the floor and
              // ceiling at a second heading (a wing 30 degrees off)
  Random,     // random directions only: a scene that is not Manhattan
};

struct SceneOptions {
  SceneKind kind = SceneKind::Room;
  jba::CameraModel model = jba::CameraModel::Pinhole;
  int images = 60;
  int points = 12000;  // spread over the surfaces; a few hundred keypoints per image
  std::uint64_t seed = 1;
  double pixel_noise = 0.5;
  double wrong_matches = 0.0;   // fraction of matches pointing at a random keypoint
  int match_min_shared = 15;    // image pairs sharing at least this many points are matched

  // Tracker: camera-to-world poses that drift (a random walk whose standard
  // deviation grows with the square root of the distance travelled, mostly in
  // heading) and may jump (a relocalisation offset).
  double drift_heading_per_m = 0.005;  // rad per sqrt(m)
  double drift_tilt_per_m = 0.0005;
  double drift_trans_per_m = 0.005;    // m per sqrt(m)
  double jump_probability = 0.0;      // per frame
  double prior_noise_rot = 0.001;     // white noise on each relative prior
  double prior_noise_trans = 0.003;
  int prior_neighbours = 3;           // relative priors between frames up to this far apart in capture order
  // Relative priors spanning a jump are removed (the caller's cleanup, D17) except
  // this fraction. Priors across one jump all carry the same offset, so this is a
  // tracking break, the caller's to detect.
  double bad_priors_kept = 0.0;
  // Fraction of relative priors that are individually wrong after the cleanup
  // (document 05, "uneven priors"): 0.05 to 0.2 rad and 0.1 to 0.4 m off.
  double bad_prior_fraction = 0.0;

  // Global priors from a drift-free source (a network): every frame, in a frame
  // related to the truth by `global_relation` (random transform), with noise.
  bool global_priors = false;
  jba::FrameRelation global_relation = jba::FrameRelation::Similarity;
  double global_noise_rot = 0.01;
  double global_noise_pos = 0.05;  // in the source's units
  double global_outliers = 0.0;    // fraction of global priors that are badly wrong

  bool relative_priors = true;
  bool shuffle_frames = false;  // frames in random order (D12)

  LineMode lines = LineMode::None;
  int line_count = 400;        // 3-D segments; each image sees some of them
  double line_noise = 0.5;     // endpoint noise, pixels
  double min_line_length = 30;  // pixels
};

struct Scene {
  jba::Problem truth;                   // true poses, points, intrinsics (no observations needed)
  std::vector<double> tracker_poses;    // world to camera, as the tracker reports them
  std::vector<double> truth_global_transform;  // 8: the global source's frame to the truth frame
  std::vector<std::int32_t> bad_relative;      // relative priors that are wrong (jumps left in, or corrupted)
  std::vector<std::int32_t> bad_global;        // global priors that are outliers
  std::vector<std::int32_t> capture_order;     // frame index of each capture step
  // Per 2-D segment in input.scene: the true building axis (0 world x, 1 world z,
  // 2 vertical) or -1 for a distractor. The true Manhattan frame (building axes to
  // world, world y down) is truth_frame.
  std::vector<std::int32_t> line_truth_axis;
  double truth_frame[4] = {1, 0, 0, 0};
  jba::RefineInput input;
};

Scene make_scene(const SceneOptions& options);

// Accuracy of estimated poses against the truth after the best alignment of the
// estimate onto the truth (rigid, or similarity when `with_scale`), over the images
// marked in `use` (all when empty): median and max camera centre error (truth
// units), median and max rotation error (radians).
struct PoseError {
  double median_centre = 0, max_centre = 0, median_rotation = 0, max_rotation = 0;
  double scale = 1;  // of the alignment
};
PoseError pose_error(const std::vector<double>& estimate, const std::vector<double>& truth, bool with_scale,
                     const std::vector<char>& use = {});

}  // namespace jba_test
