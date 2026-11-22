#pragma once

#include <cstdint>
#include <vector>

#include "jba/camera_model.h"
#include "jba/triangulation.h"

namespace jba {

// Keypoints of every image, concatenated: image i owns keypoints
// begin[i] .. begin[i + 1] - 1 of xy (2 values each, pixels). Keypoint indices in
// matches and tracks are local to their image.
struct Keypoints {
  std::vector<std::int64_t> begin;  // num_images + 1
  std::vector<double> xy;
  std::size_t num_images() const { return begin.empty() ? 0 : begin.size() - 1; }
  const double* at(std::int32_t image, std::int32_t keypoint) const {
    return &xy[2 * static_cast<std::size_t>(begin[static_cast<std::size_t>(image)] + keypoint)];
  }
};

// Tracks: track t has observations begin[t] .. begin[t + 1] - 1, each an
// (image, keypoint) pair, at most one per image, in increasing image order.
struct Tracks {
  std::vector<std::int64_t> begin = {0};
  std::vector<std::int32_t> image;
  std::vector<std::int32_t> keypoint;
  std::size_t size() const { return begin.size() - 1; }
};

// Cameras and poses for the geometric stage: one camera model, intrinsics blocks
// as in a Problem, world-to-camera poses (7 per image).
struct Cameras {
  CameraModel model = CameraModel::Pinhole;
  std::vector<double> intrinsics;
  std::vector<std::int32_t> image_intrinsics;  // empty means block 0 for every image
  std::vector<double> poses;
};

// Ray of a keypoint in world coordinates. Returns false if it cannot be unprojected.
bool keypoint_ray(const Cameras& cameras, std::int32_t image, const double* xy, Ray& ray);

enum class ConflictPolicy {
  Drop,    // a component with two keypoints in one image is removed (OpenMVG)
  Repair,  // it is split into tracks by triangulation from the poses (stage 2)
  // Conflict free by construction: matches are applied best first (by score, see
  // build_tracks), and two tracks are joined only if they share no image. Needs no
  // poses; O(M log M).
  Greedy,
};

struct TrackOptions {
  int min_views = 2;   // document 04: 2 with poses, 3 without
  int max_views = 100;  // longer tracks keep their first max_views observations
  ConflictPolicy conflict_policy = ConflictPolicy::Repair;  // needs cameras; Drop without
  // Also run the geometric check on clean components (document 04, stage 2,
  // "optionally for every track"): removes wrong matches inside clean tracks.
  bool check_clean = true;
  TriangulationOptions triangulation;  // stage 2 thresholds
  // Repair: a conflicted component with more observations than this is first split
  // by the greedy merge over its own matches, and the pieces go through stage 2.
  // Stage 2 grows faster than linearly with a component's size, and dense matches
  // over many pairs chain wrong matches into components of many thousands. 0: no cap.
  int max_repair_observations = 500;
  // Stage 2 runs on components in parallel; 0 means all hardware threads. Results do
  // not depend on the thread count.
  int num_threads = 0;
};

struct TrackStats {
  std::size_t rejected_matches = 0;  // failed the two-view check against the poses
  std::size_t components = 0;  // connected components of the match graph
  std::size_t clean = 0;       // with at most one keypoint per image
  std::size_t conflicted = 0;
  std::size_t tracks = 0;      // tracks produced
  std::size_t dropped_observations = 0;  // keypoints in components that produced no track
  std::size_t largest_component = 0;         // observations in the largest component
  std::size_t largest_component_images = 0;  // distinct images in it
  // Repair: conflicted components above max_repair_observations, split greedily
  // before stage 2.
  std::size_t split_components = 0;
  // Greedy (and Repair's split): matches not applied because their two tracks
  // already share an image.
  std::size_t refused_matches = 0;
};

// Matches: 4 values each (image_a, keypoint_a, image_b, keypoint_b). Matches are
// assumed verified by the caller but may still be wrong (document 04).
// With cameras and ConflictPolicy::Repair, each match is first checked against the
// poses: the two keypoint rays must meet in front of both cameras within the stage 2
// inlier angle of each other's epipolar plane.
// The same check runs with ConflictPolicy::Greedy when cameras are given.
// Stage 1: union-find over the matched keypoints. Clean components become tracks.
// Conflicted ones are dropped, or with cameras and ConflictPolicy::Repair split by
// repeated robust triangulation (stage 2).
// ConflictPolicy::Greedy builds the tracks by joining best matches first and never
// makes a conflict; with cameras and check_clean, stage 2 then checks every track.
// `scores` (optional, one per match, higher is better, for example a matcher's
// confidence) give the order of the greedy merge. Without them the order is the
// epipolar error against the poses when cameras are given (smallest first), else
// the order of the matches. Deterministic.
Tracks build_tracks(const Keypoints& keypoints, const std::vector<std::int32_t>& matches,
                    const TrackOptions& options, const Cameras* cameras = nullptr, TrackStats* stats = nullptr,
                    const std::vector<double>* scores = nullptr);

}  // namespace jba
