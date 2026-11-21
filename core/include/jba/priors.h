#pragma once

#include <vector>

#include "jba/problem.h"

namespace jba {

// Weights used for pose priors that arrive without any (D21). Trackers report no
// uncertainty, so relative priors get a standard deviation that grows with the
// motion between the two frames (the LaMAR heuristic, document 02): frames far
// apart are tied more loosely than neighbours.
//
// Relative values measured on ARCore captures (D77); global values are still
// provisional (D21).
struct PriorDefaults {
  // Relative priors: sigma = base + per_unit * motion.
  double relative_rotation_sigma = 0.001;             // rad
  double relative_rotation_per_radian = 0.012;        // rad per rad of rotation
  double relative_translation_sigma = 0.005;          // m
  double relative_translation_per_metre = 0.02;       // m per m travelled
  // Global priors (provisional).
  double global_rotation_sigma = 0.02;                // rad
  double global_position_sigma = 0.05;                // the set's own units
  // Gravity priors (provisional, D82).
  double gravity_sigma = 0.01;                        // rad
  // Manhattan line segments (provisional, D84, D163): the residual is in pixels, with
  // sigma = max(endpoint sigma, line_min_sigma * length in pixels / sqrt(2));
  // line_min_sigma is a floor on the sine of the image angle.
  double line_endpoint_sigma = 1.0;                   // pixels
  double line_min_sigma = 0.002;
  // Depth at keypoints (provisional, D87): sigma = base + per_metre * reading, metres;
  // for the inverse form divided by the reading squared. Prior on a free depth
  // correction toward (1, 0); +inf leaves that part without one.
  double depth_sigma = 0.02;
  double depth_sigma_per_metre = 0.02;
  double depth_scale_sigma = 0.1;
  double depth_offset_sigma = 0.1;                    // metres
};

// Whitening matrices W (6 x 6, row-major, residual W r) for every relative and
// global prior: from sqrt_info when given, else from sigma (a +inf sigma gives zero
// rows: that part is left out), else from `defaults`.
void prior_whitening(const Problem& problem, const PriorDefaults& defaults, std::vector<double>& relative,
                     std::vector<double>& global);

// Standard deviation of observation k's depth residual, in the residual's units:
// metres for the linear form, 1 / metres for the inverse (sigma / reading^2). From
// obs_depth_sigma when given, else base + per_metre * reading (D87).
double depth_residual_sigma(const Problem& problem, std::size_t k, const PriorDefaults& defaults);

// Builds relative priors from per-image poses and a list of image pairs, for callers
// whose poses are already clean (document 02): each pair (i, j) gets the prior
// T_j * T_i^-1 computed from `poses` (7 per image, world to camera). Appended to
// `problem`; weights are left to the defaults.
void add_relative_priors_from_poses(Problem& problem, const std::vector<double>& poses,
                                    const std::vector<std::int32_t>& pairs);

// Prior on free intrinsics by group (D161): sets intrinsics_prior to the current
// intrinsics and intrinsics_prior_sigma per parameter, for every block: focal
// lengths focal_fraction times their value (0.01 keeps them within about 1 %), the
// principal point principal_point_sigma pixels, distortion coefficients
// distortion_sigma. +inf leaves that group fully free. Only groups freed in
// intrinsics_free are refined at all; the mask stays the caller's.
void set_intrinsics_prior(Problem& problem, double focal_fraction, double principal_point_sigma,
                          double distortion_sigma);

// The hierarchical intrinsics prior (D165) as the solver builds it, resolved from a
// problem's fields, for the solver, evaluate() and the Ceres reference alike. [G, P]
// arrays; a weight of 0 means no residual.
struct IntrinsicsGroupTerms {
  std::size_t groups = 0;
  std::vector<std::uint8_t> member;  // per block: 1 when it is tied to its group's mean
  std::vector<double> tie_w;         // 1 / sigma of member - mean, 0 where not tied
  std::vector<double> anchor_w;      // 1 / sigma of mean - target, 0 for none
  std::vector<double> mean;          // the mean the solve starts from
  std::vector<double> target;        // the target of the mean
  bool any() const;
};
IntrinsicsGroupTerms intrinsics_group_terms(const Problem& problem);

// Hierarchical prior by group of blocks (D165): intrinsics_group = group_of_block (one
// entry per block, -1 for none) and, per group, the sigmas of the ties to its mean
// (focal lengths focal_fraction times the group's mean focal length, the principal
// point principal_point_sigma pixels, distortion distortion_sigma) and of the mean to
// its target, the group's current mean (group_*: the same units; +inf leaves the mean
// free). The mean is cleared, so a solve starts it at the members' mean. Only groups
// freed in intrinsics_free are refined; the mask stays the caller's.
void set_intrinsics_groups(Problem& problem, const std::vector<std::int32_t>& group_of_block,
                           double focal_fraction, double principal_point_sigma, double distortion_sigma,
                           double group_focal_fraction, double group_principal_point_sigma,
                           double group_distortion_sigma);

}  // namespace jba
