// A minimal incremental reconstruction (document 04, part C; D132). poselib-gpu
// verifies the matches of every pair, gives the first two poses and registers each
// further image by PnP; this library builds the tracks, triangulates and adjusts.
// Part of the demo app, not of the library: only the demo links poselib-gpu.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "jba/problem.h"
#include "jba/solver.h"
#include "jba/tracks.h"
#include "jpo/jpo.h"

namespace demo {

// The caller's matches of one image pair: keypoint ka[k] of image a with kb[k] of b.
struct Pair {
  std::int32_t a = 0, b = 0;
  std::vector<std::int32_t> ka, kb;
};

// Calibrated cameras of one model (normally one for the whole capture), the keypoints
// of every image and the matches of the pairs the caller's matcher tried.
struct Capture {
  jba::CameraModel model = jba::CameraModel::Pinhole;
  std::vector<double> intrinsics;               // num_params(model) per block
  std::vector<std::int32_t> image_intrinsics;   // per image: its block; empty: block 0 for all
  std::vector<std::vector<double>> keypoints;  // per image: 2 per keypoint, pixels
  std::vector<Pair> pairs;
  std::size_t num_images() const { return keypoints.size(); }
  std::int32_t block(std::size_t image) const { return image_intrinsics.empty() ? 0 : image_intrinsics[image]; }
};

struct MapperOptions {
  std::int32_t pose_backend = JPO_BACKEND_AUTO;    // poselib-gpu's backend
  jba::BackendKind ba_backend = jba::BackendKind::Cpu;
  std::uint32_t seed = 7;
  int threads = 0;  // both libraries; 0: all

  // RANSAC inlier thresholds, pixels (poselib-gpu's defaults). Set from the keypoints'
  // noise: about twice its standard deviation for pairs.
  double pair_max_error_px = 1.0;
  double pnp_max_error_px = 4.0;

  // COLMAP's defaults where it has one.
  std::uint32_t min_pair_inliers = 30;     // a pair's inliers go to the tracks from this many
  double min_pair_inlier_ratio = 0.25;     // and this share of its matches
  std::uint32_t min_first_pair_inliers = 100;
  std::uint32_t min_pnp_inliers = 15;      // COLMAP: 30; the generated scenes are small
  double min_pnp_inlier_ratio = 0.25;
  double max_reproj_px = 4.0;              // the filter after each adjustment
  double min_tri_angle_deg = 1.5;
  // Largest angle between a keypoint's ray and its point for the keypoint to join the
  // point, at triangulation and when continuing a track, before the adjustment and its
  // reprojection filter. 0: max_reproj_px seen through the camera.
  double tri_max_angle_error_deg = 0.0;
  // Tracks with only two observations are not triangulated (COLMAP's mapper does the
  // same by default, Mapper.tri_ignore_two_view_tracks): they constrain little and are
  // the likeliest to be a wrong match.
  bool ignore_two_view_tracks = false;
  std::size_t pnp_batch = 4;               // images tried in one PnP call
  int local_iterations = 25;               // each adjustment after a registration
  int final_iterations = 50;
  // Cauchy scale of the adjustments, pixels; 0: max_reproj_px, so every observation the
  // filter accepts counts fully (D139). A smaller scale fits each point to the majority
  // of its keypoints and leaves valid ones, far from the image centre, to the filter.
  double loss_scale = 0.0;
};

// Thresholds for keypoints whose noise is sigma pixels per coordinate. The defaults hold
// up to half a pixel; above that the thresholds grow with it.
inline void thresholds_for_noise(double sigma, MapperOptions& o) {
  o.pair_max_error_px = sigma > 0.5 ? 2.0 * sigma : 1.0;
  o.pnp_max_error_px = sigma > 1.0 ? 4.0 * sigma : 4.0;
  o.max_reproj_px = sigma > 1.0 ? 4.0 * sigma : 4.0;
}

// The defaults of COLMAP's incremental mapper and two-view geometry, for real captures:
// RANSAC at 4 px for pairs and 12 px for PnP, at least 30 PnP inliers, filter at 4 px.
// Long-standing COLMAP defaults; not checked against 4.2.1 on this machine.
inline void colmap_thresholds(MapperOptions& o) {
  o.pair_max_error_px = 4.0;
  o.pnp_max_error_px = 12.0;
  o.min_pnp_inliers = 30;
  o.min_pnp_inlier_ratio = 0.25;
  o.min_first_pair_inliers = 100;
  o.max_reproj_px = 4.0;
  o.min_tri_angle_deg = 1.5;
  o.ignore_two_view_tracks = true;
  o.tri_max_angle_error_deg = 2.0;  // COLMAP's create_max_angle_error and continue_max_angle_error
}

// The map after a step, for a caller that draws the reconstruction as it grows.
struct Snapshot {
  std::string stage;                       // "first pair", "register", "final"
  std::vector<std::int32_t> added;         // images registered in this step
  const std::vector<double>* poses;        // 7 per image; meaningful where registered
  const std::vector<char>* registered;
  std::vector<double> points;              // 3 per triangulated point
  std::vector<std::int32_t> point_track;   // per point: its track, the same across steps
  double mean_error_px = 0.0;
};
using StepCallback = std::function<void(const Snapshot&)>;

struct Timings {
  double verify = 0, tracks = 0, pnp = 0, triangulate = 0, adjust = 0, total = 0;  // seconds
};

struct Reconstruction {
  std::string error;  // empty on success
  bool ok() const { return error.empty(); }

  std::vector<double> poses;        // 7 per image of the capture
  std::vector<char> registered;
  // The last adjustment: registered images and their points, observations kept by
  // the filter. image_of maps its images to the capture's; obs_keypoint gives each
  // observation's keypoint in its image.
  jba::Problem problem;
  std::vector<std::int32_t> image_of;
  std::vector<std::int32_t> obs_keypoint;
  std::vector<std::int32_t> point_track;  // per point of `problem`: its track in `all_tracks`
  jba::Tracks all_tracks;                  // every track built from the verified matches
  jba::TrackStats track_stats;             // of building them
  double mean_error_px = 0.0;

  // Pair verification.
  std::vector<jpo_pose_result> pair_results;
  std::vector<std::uint8_t> pair_inliers;  // one byte per match, every pair in order
  std::vector<std::uint32_t> pair_offsets;
  std::vector<char> pair_kept;
  int pairs_ok = 0, pairs_degenerate = 0, pairs_failed = 0, pairs_kept = 0;

  std::size_t tracks = 0;  // built from the matches, before any split
  int first_a = -1, first_b = -1;
  std::uint32_t first_inliers = 0;
  double first_angle_deg = 0.0;
  int pnp_calls = 0, pnp_problems = 0, pnp_rejected = 0, adjustments = 0;
  std::string pose_backend;           // what poselib-gpu ran on
  Timings time;
};

// One poselib-gpu context, kept across reconstructions (on a GPU its start-up is
// paid once).
class Mapper {
 public:
  explicit Mapper(const MapperOptions& options);  // throws std::runtime_error
  ~Mapper();
  Mapper(const Mapper&) = delete;
  Mapper& operator=(const Mapper&) = delete;

  Reconstruction run(const Capture& capture, const StepCallback& on_step = {});
  const MapperOptions& options() const { return options_; }
  // Options for the next runs. The context keeps the pose backend, seed and threads it
  // was created with; everything else, the solver's backend included, can change.
  void set_options(const MapperOptions& options) { options_ = options; }

 private:
  MapperOptions options_;
  jpo_ctx* ctx_ = nullptr;
};

}  // namespace demo
