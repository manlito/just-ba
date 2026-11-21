#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "jba/priors.h"
#include "jba/problem.h"

namespace jba {

// Manhattan prior from line segments (document 02, section 5): the caller detects
// segments; the library estimates the building's axes, assigns each segment to an
// axis or to none, and refuses the prior where the scene is not Manhattan.

// Unit normal, in camera axes, of the plane through the camera centre and segment k
// (the normalised cross product of its endpoint bearings). False when an endpoint
// cannot be unprojected or the bearings coincide.
bool line_normal(const Problem& problem, std::size_t k, double* n);

// Normalized coordinates (u0, v0, u1, v1) of a segment's endpoints xy (x0, y0, x1, y1,
// pixels) under intrinsics K: the inputs of the line residual (D163). False when an
// endpoint cannot be undistorted.
bool line_uv(CameraModel model, const double* K, const double* xy, double* uv);

// Standard deviation of segment k's residual, in pixels (D163): its endpoint sigma,
// at least defaults.line_min_sigma (the sine of an image angle) times its length in
// pixels over sqrt(2).
double line_residual_sigma(const Problem& problem, std::size_t k, const PriorDefaults& defaults);

struct ManhattanOptions {
  // 1: the working frame is gravity-aligned (tracker poses or gravity priors) and
  // only the heading of the walls about the vertical is estimated; 3: the whole
  // rotation, the vertical taken as the strongest direction among the segments.
  int dof = 1;
  // A segment fits an axis when the angle, in the image, between the segment and the
  // direction towards the axis's vanishing point is below this (D163).
  double max_angle_deg = 3;
  // Support test: the share of informative horizontal segment length (see
  // ManhattanAssignment::support) the frame must explain, and the fewest segments it
  // must assign. Provisional (D85).
  double min_support = 0.35;
  int min_segments = 20;
  // Segments whose plane is within this of horizontal carry no heading: any
  // horizontal direction lies nearly in it. They are left out of the vote and of
  // the support test.
  double horizontal_plane_deg = 10;
};

struct ManhattanAssignment {
  bool accepted = false;  // the support test passed
  std::string message;    // why it was refused, when it was
  std::vector<double> frame;         // 4: building axes to working frame
  std::vector<std::int32_t> axis;    // per segment: 0, 1, 2 or -1
  int assigned[3] = {0, 0, 0};       // segments per axis
  int unusable = 0;                  // segments whose endpoints could not be unprojected
  // Length-weighted share of the informative segments (not along the vertical, plane
  // not horizontal) that fit one of the horizontal axes. About 0.07 for segments in
  // random directions at a 3 degree threshold; above 0.5 in a room.
  double support = 0;
};

// Estimates the frame from the segments seen from the problem's current poses (a vote
// over headings modulo 90 degrees about the vertical), assigns every segment that
// fits exactly one axis, and runs the support test. With `frame` given (4 values),
// nothing is estimated: segments are assigned to its axes and the test is run.
ManhattanAssignment assign_lines(const Problem& problem, const ManhattanOptions& options,
                                 const double* frame = nullptr);

// The frame of a scene whose working axes are already the building's (D164): axis 2
// is up (-gravity_world), axis 0 the working axis most nearly horizontal (the lowest
// index on a tie), made exactly horizontal, and axis 1 completes a right-handed
// basis. With world y down: e0 = x, e1 = z, e2 = -y; with z up, the identity.
std::vector<double> aligned_frame(const Problem& problem);

// Stores an assignment in the problem (line_axis, manhattan_frame, manhattan_dof),
// which turns the Manhattan term on in solve(). A refused assignment clears them.
void apply_assignment(Problem& problem, const ManhattanAssignment& assignment, int dof);

}  // namespace jba
