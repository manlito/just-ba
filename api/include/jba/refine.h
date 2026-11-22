#pragma once

#include <cstdint>
#include <vector>

#include "jba/initialize.h"
#include "jba/manhattan.h"
#include "jba/problem.h"
#include "jba/report.h"
#include "jba/solver.h"
#include "jba/tracks.h"

namespace jba {

// The main use case end to end (document 01): keypoints, matches (or tracks) and
// noisy pose priors in, a refined reconstruction out.
//
//   1. Starting poses from the priors (initialize_poses); images without priors are
//      left out and reported.
//   2. Optionally a pose-only solve over the priors. With the Manhattan prior on, the
//      segments are then assigned (and the prior refused where the support test
//      fails), and the pose-only solve is repeated with them, which removes heading
//      drift before anything is triangulated.
//   3. Rounds of: tracks from the matches with the current poses (stages 1 and 2 of
//      document 04), robust triangulation, solve, removal of outlier observations
//      and weak points, solve again. Each round starts from the previous poses with
//      a tighter triangulation threshold, and segments are assigned again to the
//      current frame with a tighter angle (halved each round, down to
//      manhattan_min_angle_deg).
struct RefineInput {
  // Camera, pose priors and prior sets, as in a Problem. `poses` may be empty (all
  // identity) or hold starting poses: a relative-prior group's root keeps its pose.
  // Points and observations are ignored.
  Problem scene;
  Keypoints keypoints;
  std::vector<double> keypoint_sigma;  // per keypoint, pixels; empty means 1
  // Depth at keypoints (document 02, section 4a): per keypoint, NaN for none; empty
  // means no depth. scene.depth_kind, depth_form, depth_correction and
  // depth_correction_free say what the readings are and what is estimated.
  std::vector<double> keypoint_depth;
  std::vector<double> keypoint_depth_sigma;  // per keypoint, metres; empty: the default
  // Either matches (4 values each: image_a, keypoint_a, image_b, keypoint_b) or
  // tracks; tracks take precedence when not empty.
  std::vector<std::int32_t> matches;
  // Optional, one per match, higher is better (a matcher's confidence): the order of
  // ConflictPolicy::Greedy and of Repair's split of oversized components.
  std::vector<double> match_score;
  Tracks tracks;
  // Line segments for the Manhattan prior are scene.line_image, line_xy and
  // line_sigma; their axes are assigned by refine (scene.line_axis is ignored).
  // scene.manhattan_frame is the frame when estimate_manhattan_frame is off, and is
  // ignored otherwise.
};

struct RefineOptions {
  SolverOptions solver = [] {
    SolverOptions o;
    o.loss = {LossType::Cauchy, 2.0};  // document 02: reprojection is robust by default
    return o;
  }();
  TrackOptions tracks;
  int rounds = 2;
  bool prior_solve = true;             // step 2
  double max_reprojection_error = 4;   // pixels; COLMAP's default (document 04)
  double min_tri_angle_deg = 1.5;
  double downweight_threshold = 0.1;   // for the report's down-weighted priors
  bool largest_group_only = false;     // otherwise every group is solved, each in its own frame
  // An image counts as refined when at least this many inlier observations constrain
  // its pose (COLMAP registers an image from 30 inliers). Images below it are still
  // solved, but their poses come from the priors alone or nearly so (D76).
  int min_observations = 30;
  // Manhattan prior from scene's line segments (document 02, section 5). Opt-in: it
  // is strong, and wrong where the scene is not Manhattan.
  bool use_manhattan = false;
  // With use_manhattan: estimate the building's axes from the segments (the vote of
  // assign_lines over the starting poses). Off when the working frame is already
  // Manhattan aligned (D164): the frame is scene.manhattan_frame, or aligned_frame()
  // when that is empty, and the segments are only assigned to its axes. The support
  // test, the solves and the later rounds are the same either way.
  bool estimate_manhattan_frame = true;
  ManhattanOptions manhattan;
  double manhattan_min_angle_deg = 1;
};

struct RefineResult {
  // Solved problem: poses, points, inlier observations, priors, set transforms, and
  // the losses refine solved with (loss.* sections when written).
  Problem problem;
  std::vector<std::int32_t> obs_keypoint;  // keypoint (in its image) of each observation
  std::vector<std::int32_t> group;         // per image, from initialize_poses; -1 left out (no prior places it)
  int num_groups = 0;
  // Per image: 1 if it was solved and has at least min_observations inlier
  // observations in `problem`. Images in a group but below that are solved with the
  // rest; their poses follow the priors.
  std::vector<char> refined;
  std::vector<std::int32_t> num_observations;  // per image, inlier observations in `problem`
  ResidualReport report;
  std::vector<SolverSummary> summaries;    // every solve, in order
  TrackStats track_stats;                  // of the last round
  std::size_t rejected_observations = 0;   // in the last round's filtering
  // Priors of `problem` by their index in the input (priors of images left out are
  // dropped). The report's down-weighted lists are already in input numbering.
  std::vector<std::int32_t> relative_prior_index, global_prior_index, gravity_prior_index;
  // The last assignment of the segments (one entry per input segment; segments of
  // images left out are -1), the frame as solved, and whether the support test
  // accepted the prior. Empty when use_manhattan is off.
  ManhattanAssignment manhattan;
};

RefineResult refine(const RefineInput& input, const RefineOptions& options = {});

}  // namespace jba
