#pragma once

#include <cstdint>
#include <vector>

namespace jba {

// Triangulation from known poses (document 04, part B). Works on rays: a camera
// centre C and a unit direction d in world coordinates per observation, so every
// camera model is handled the same way once its keypoints are unprojected, and
// errors are angles, fair to wide-angle cameras.

struct Ray {
  double C[3];
  double d[3];
};

struct TriangulationOptions {
  // An observation is an inlier when the angle between its ray and the direction
  // from its camera to the point is below this, and the point is in front.
  double max_angle_error_deg = 2.0;
  // The largest angle between two inlier rays at the point must reach this.
  double min_tri_angle_deg = 1.5;
  int min_views = 2;
  // Two-view samples tried: every pair when there are at most this many, else this
  // many pairs drawn with a fixed seed (results do not depend on the run).
  int max_samples = 200;
};

// Least-squares point closest to all rays: sum (I - d d^T) X = sum (I - d d^T) C.
// Returns false when the rays are (nearly) parallel.
bool triangulate_rays(const Ray* rays, std::size_t n, double* X);

// Angle (radians) between ray r and the direction from its centre to X; pi if X is
// behind it.
double ray_angle_error(const Ray& r, const double* X);

// Largest angle (radians) between the directions from the centres to X.
double triangulation_angle(const Ray* rays, std::size_t n, const double* X);

struct TriangulationResult {
  bool ok = false;
  double X[3] = {0, 0, 0};
  std::vector<char> inlier;  // per ray
  int num_inliers = 0;
};

// RANSAC over two-view samples, then a least-squares refit on the inliers and a
// final inlier test. `group` optionally gives each ray a key (for example its
// image); at most one inlier per key is kept, the one with the smallest error.
TriangulationResult triangulate_robust(const std::vector<Ray>& rays, const TriangulationOptions& options,
                                       const std::vector<std::int32_t>* group = nullptr);

}  // namespace jba
