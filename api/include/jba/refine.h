#pragma once

#include <cstdint>
#include <vector>

#include "jba/initialize.h"
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
//   2. Optionally a pose-only solve over the priors.
//   3. Rounds of: tracks from the matches with the current poses (stages 1 and 2 of
//      document 04), robust triangulation, solve, removal of outlier observations
//      and weak points, solve again. Each round starts from the previous poses with
//      a tighter triangulation threshold.
struct RefineInput {
  // Camera, pose priors and prior sets, as in a Problem. `poses` may be empty (all
  // identity) or hold starting poses: a relative-prior group's root keeps its pose.
  // Points and observations are ignored.
  Problem scene;
  Keypoints keypoints;
  std::vector<double> keypoint_sigma;  // per keypoint, pixels; empty means 1
  // Either matches (4 values each: image_a, keypoint_a, image_b, keypoint_b) or
  // tracks; tracks take precedence when not empty.
  std::vector<std::int32_t> matches;
  Tracks tracks;
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
};

struct RefineResult {
  // Solved problem: poses, points, inlier observations, priors, set transforms.
  Problem problem;
  std::vector<std::int32_t> obs_keypoint;  // keypoint (in its image) of each observation
  std::vector<std::int32_t> group;         // per image, from initialize_poses; -1 not refined
  int num_groups = 0;
  std::vector<char> refined;               // per image: 1 if its pose was solved
  ResidualReport report;
  std::vector<SolverSummary> summaries;    // every solve, in order
  TrackStats track_stats;                  // of the last round
  std::size_t rejected_observations = 0;   // in the last round's filtering
  // Priors of `problem` by their index in the input (priors of images left out are
  // dropped). The report's down-weighted lists are already in input numbering.
  std::vector<std::int32_t> relative_prior_index, global_prior_index;
};

RefineResult refine(const RefineInput& input, const RefineOptions& options = {});

}  // namespace jba
