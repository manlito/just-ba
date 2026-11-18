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

// Two sets of poses of the same images, on the images both registered: rotation error
// after the best common rotation; camera centre error after the similarity that best
// aligns the centres, as a percentage of the diagonal of the box around b's centres,
// and in b's units. b is the reference (COLMAP's, or the truth).
struct PoseAgreement {
  std::size_t common = 0;
  double rot_median_deg = 0.0, rot_max_deg = 0.0;
  double centre_median_pct = 0.0, centre_max_pct = 0.0;
  double centre_median = 0.0;
};
PoseAgreement compare_poses(const std::vector<char>& registered_a, const std::vector<double>& poses_a,
                            const std::vector<char>& registered_b, const std::vector<double>& poses_b);

struct Comparison {
  Side ours, reference;
  std::size_t common = 0;                       // images both registered
  // Over the common images: rotation error after the best common rotation; camera
  // centre error after the similarity that best aligns the centres, as a percentage of
  // the diagonal of the box around the reference's centres, and in its units.
  double rot_median_deg = 0.0, rot_max_deg = 0.0;
  double centre_median_pct = 0.0, centre_max_pct = 0.0;
  double centre_median = 0.0;

  // How the two sides group the same keypoints into points. An observation is shared
  // when both sides have that keypoint of that image in a point.
  std::size_t shared_obs = 0, ours_only_obs = 0, reference_only_obs = 0;
  // Reference points with shared observations, and the number of our points those go to
  // (1: the same track; more: we split it).
  std::size_t reference_points_seen = 0, reference_points_split = 0;
  double ours_per_reference_point = 0.0;
  // Our points with shared observations, and those whose observations belong to two or
  // more reference points (we joined what the reference keeps apart).
  std::size_t our_points_seen = 0, our_points_joined = 0;
  // The reference's observations we do not have: in one of our points' tracks but
  // filtered out; in a track of ours with no point; in no track of ours at all.
  std::size_t missing_filtered = 0, missing_untriangulated = 0, missing_untracked = 0;
  // Our points with no observation in common with the reference, and their length.
  std::size_t ours_only_points = 0, ours_only_two_view = 0;
};

// `capture` gives the intrinsics for the reference's reprojection errors when it has
// none of its own.
Comparison compare(const Capture& capture, const Reference& ref, const Reconstruction& r);

}  // namespace demo
