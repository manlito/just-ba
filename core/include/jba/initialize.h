#pragma once

#include <cstdint>
#include <vector>

#include "jba/priors.h"
#include "jba/problem.h"

namespace jba {

// Starting poses built from the priors alone (document 02, "Starting poses and
// connectivity").
//
// - Relative priors link images into groups. In each group the priors are chained
//   outward from a root along a maximum-trust spanning tree. Trust is the squared
//   norm of the prior's whitening matrix, divided by 1 + (e / 3)^2 where e is the
//   median disagreement, in standard deviations, with the triangles of priors it
//   closes, so a wrong prior is not chained through when its neighbours disagree.
//   The root is the image with the most relative priors; it keeps its current
//   pose, so a caller's frame carries over.
// - Global priors from a set that is the working frame place their group: the
//   chained poses are moved as a whole by the rigid transform that best aligns
//   them to those priors (refit without outlying pairs). An image with no relative
//   prior starts at its global prior.
// - A set with a transform gets a starting transform from the images that are
//   already placed and have a prior in it (similarity or rigid fit); images placed
//   only through such a set follow from it. Sets are resolved in index order until
//   nothing changes.
// - Images with no prior keep their current pose and get group -1.
struct PoseInitialization {
  std::vector<double> poses;          // 7 per image
  std::vector<double> set_transform;  // 8 per prior set
  // Group of each image: images in one group are placed in one consistent frame.
  // Group 0 is the working frame when global priors define it (or the largest
  // group otherwise); further groups by decreasing size. -1: no prior.
  std::vector<std::int32_t> group;
  int num_groups = 0;
};

PoseInitialization initialize_poses(const Problem& problem, const PriorDefaults& defaults = {});

// Rotation and translation (and scale, if `with_scale`) that best map points `from`
// onto points `to` (3 per point): to ~ s R from + t. The rotation is the average of
// the per-pair rotations `R_pairs` (4 per pair, quaternions) when given, which
// stays defined for one or two points; otherwise from the points (Horn's method).
// Returns qw qx qy qz tx ty tz s.
void fit_similarity(const std::vector<double>& from, const std::vector<double>& to,
                    const std::vector<double>* rotations, bool with_scale, double* out);

// Chordal L2 average of unit quaternions (eigenvector of sum q q^T), sign-invariant.
void average_quaternions(const std::vector<double>& quats, double* out);

}  // namespace jba
