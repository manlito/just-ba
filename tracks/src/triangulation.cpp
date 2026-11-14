#include "jba/triangulation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace jba {
namespace {

constexpr double kPi = 3.141592653589793;

double deg(double rad) { return rad * 180 / kPi; }

double angle_between(const double* a, const double* b) {
  const double cx = a[1] * b[2] - a[2] * b[1], cy = a[2] * b[0] - a[0] * b[2], cz = a[0] * b[1] - a[1] * b[0];
  const double s = std::sqrt(cx * cx + cy * cy + cz * cz);
  const double c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return std::atan2(s, c);
}

// Inliers of X, with at most one per group key (smallest error wins).
int classify(const std::vector<Ray>& rays, const double* X, double max_err, const std::vector<std::int32_t>* group,
             std::vector<char>& inlier, double& total_error) {
  const std::size_t n = rays.size();
  inlier.assign(n, 0);
  std::vector<double> err(n);
  total_error = 0;
  for (std::size_t i = 0; i < n; ++i) {
    err[i] = ray_angle_error(rays[i], X);
    inlier[i] = err[i] < max_err;
  }
  if (group) {
    // Keep the best ray per key.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      if ((*group)[a] != (*group)[b]) return (*group)[a] < (*group)[b];
      return err[a] < err[b];
    });
    std::int32_t last_key = 0;
    bool have_key = false, taken = false;
    for (std::size_t q = 0; q < n; ++q) {
      const std::size_t i = order[q];
      if (!have_key || (*group)[i] != last_key) {
        last_key = (*group)[i];
        have_key = true;
        taken = false;
      }
      if (!inlier[i]) continue;
      if (taken) inlier[i] = 0;
      else taken = true;
    }
  }
  int count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (inlier[i]) {
      ++count;
      total_error += err[i];
    }
  }
  return count;
}

}  // namespace

bool triangulate_rays(const Ray* rays, std::size_t n, double* X) {
  double A[9] = {}, b[3] = {};
  for (std::size_t i = 0; i < n; ++i) {
    const double* d = rays[i].d;
    const double* C = rays[i].C;
    double M[9];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) M[3 * r + c] = (r == c ? 1.0 : 0.0) - d[r] * d[c];
    for (int a = 0; a < 9; ++a) A[a] += M[a];
    for (int r = 0; r < 3; ++r) b[r] += M[3 * r] * C[0] + M[3 * r + 1] * C[1] + M[3 * r + 2] * C[2];
  }
  // Cramer's rule with a conditioning test: A is symmetric positive semidefinite
  // with trace 2n; nearly parallel rays make its smallest eigenvalue tiny.
  const double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
                     A[2] * (A[3] * A[7] - A[4] * A[6]);
  const double trace = A[0] + A[4] + A[8];
  if (!(det > 1e-12 * trace * trace * trace)) return false;
  double inv[9];
  inv[0] = A[4] * A[8] - A[5] * A[7];
  inv[1] = A[2] * A[7] - A[1] * A[8];
  inv[2] = A[1] * A[5] - A[2] * A[4];
  inv[3] = A[5] * A[6] - A[3] * A[8];
  inv[4] = A[0] * A[8] - A[2] * A[6];
  inv[5] = A[2] * A[3] - A[0] * A[5];
  inv[6] = A[3] * A[7] - A[4] * A[6];
  inv[7] = A[1] * A[6] - A[0] * A[7];
  inv[8] = A[0] * A[4] - A[1] * A[3];
  for (int r = 0; r < 3; ++r) X[r] = (inv[3 * r] * b[0] + inv[3 * r + 1] * b[1] + inv[3 * r + 2] * b[2]) / det;
  return std::isfinite(X[0]) && std::isfinite(X[1]) && std::isfinite(X[2]);
}

double ray_angle_error(const Ray& r, const double* X) {
  const double v[3] = {X[0] - r.C[0], X[1] - r.C[1], X[2] - r.C[2]};
  if (!(v[0] * r.d[0] + v[1] * r.d[1] + v[2] * r.d[2] > 0)) return kPi;
  return angle_between(v, r.d);
}

double triangulation_angle(const Ray* rays, std::size_t n, const double* X) {
  double best = 0;
  for (std::size_t a = 0; a < n; ++a) {
    const double va[3] = {X[0] - rays[a].C[0], X[1] - rays[a].C[1], X[2] - rays[a].C[2]};
    for (std::size_t b = a + 1; b < n; ++b) {
      const double vb[3] = {X[0] - rays[b].C[0], X[1] - rays[b].C[1], X[2] - rays[b].C[2]};
      best = std::max(best, angle_between(va, vb));
    }
  }
  return best;
}

TriangulationResult triangulate_robust(const std::vector<Ray>& rays, const TriangulationOptions& o,
                                       const std::vector<std::int32_t>* group) {
  TriangulationResult out;
  const std::size_t n = rays.size();
  if (n < 2 || static_cast<int>(n) < o.min_views) return out;
  const double max_err = o.max_angle_error_deg * kPi / 180;

  // Candidate pairs: all of them, or a fixed-seed sample.
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  const std::size_t all = n * (n - 1) / 2;
  if (all <= static_cast<std::size_t>(o.max_samples)) {
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = a + 1; b < n; ++b) pairs.emplace_back(a, b);
  } else {
    std::mt19937_64 rng(0x6a6261);
    while (pairs.size() < static_cast<std::size_t>(o.max_samples)) {
      const std::size_t a = static_cast<std::size_t>(rng() % n), b = static_cast<std::size_t>(rng() % n);
      if (a != b) pairs.emplace_back(std::min(a, b), std::max(a, b));
    }
  }

  int best_count = 0;
  double best_error = 0;
  double best_X[3] = {0, 0, 0};
  std::vector<char> inl;
  for (const auto& [a, b] : pairs) {
    if (group && (*group)[a] == (*group)[b]) continue;
    const Ray two[2] = {rays[a], rays[b]};
    double X[3];
    if (!triangulate_rays(two, 2, X)) continue;
    double err;
    const int count = classify(rays, X, max_err, group, inl, err);
    if (count > best_count || (count == best_count && count > 0 && err < best_error)) {
      best_count = count;
      best_error = err;
      std::copy_n(X, 3, best_X);
    }
  }
  if (best_count < std::max(2, o.min_views)) return out;

  // Refit on the inliers, twice, re-testing each time.
  double X[3] = {best_X[0], best_X[1], best_X[2]};
  double err;
  classify(rays, X, max_err, group, inl, err);
  for (int round = 0; round < 2; ++round) {
    std::vector<Ray> chosen;
    for (std::size_t i = 0; i < n; ++i)
      if (inl[i]) chosen.push_back(rays[i]);
    double Y[3];
    if (!triangulate_rays(chosen.data(), chosen.size(), Y)) break;
    std::vector<char> inl2;
    double err2;
    const int count = classify(rays, Y, max_err, group, inl2, err2);
    if (count < std::max(2, o.min_views)) break;
    std::copy_n(Y, 3, X);
    inl.swap(inl2);
  }
  // Every inlier in front, and a sufficient angle.
  std::vector<Ray> chosen;
  for (std::size_t i = 0; i < n; ++i)
    if (inl[i]) chosen.push_back(rays[i]);
  if (static_cast<int>(chosen.size()) < std::max(2, o.min_views)) return out;
  if (deg(triangulation_angle(chosen.data(), chosen.size(), X)) < o.min_tri_angle_deg) return out;
  out.ok = true;
  std::copy_n(X, 3, out.X);
  out.inlier = inl;
  out.num_inliers = static_cast<int>(chosen.size());
  return out;
}

}  // namespace jba
