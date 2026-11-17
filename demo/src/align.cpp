#include "align.h"

#include <algorithm>
#include <cmath>

#include "geometry.h"

namespace demo {
namespace {

// Largest eigenvector of a symmetric 4 x 4 matrix (cyclic Jacobi).
void largest_eigenvector4(double A[4][4], double v[4]) {
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) {
        if (std::abs(A[p][q]) < 1e-300) continue;
        const double theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 4; ++k) {
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          const double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
  }
  int best = 0;
  for (int k = 1; k < 4; ++k)
    if (A[k][k] > A[best][best]) best = k;
  for (int k = 0; k < 4; ++k) v[k] = V[k][best];
}

}  // namespace

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// The similarity to ~ s R from + t best in least squares (Umeyama; the rotation as
// Horn's quaternion, the same optimum without an SVD).
void fit_similarity(const std::vector<double>& from, const std::vector<double>& to, double R[9], double t[3], double* s) {
  const std::size_t n = from.size() / 3;
  double ma[3] = {0, 0, 0}, mb[3] = {0, 0, 0};
  for (std::size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) {
      ma[d] += from[3 * i + d] / static_cast<double>(n);
      mb[d] += to[3 * i + d] / static_cast<double>(n);
    }
  double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, var = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 0; r < 3; ++r) {
      const double a = from[3 * i + r] - ma[r];
      var += a * a;
      for (int c = 0; c < 3; ++c) S[r][c] += a * (to[3 * i + c] - mb[c]);
    }
  double N[4][4] = {
      {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
      {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
      {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
      {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  double q[4];
  largest_eigenvector4(N, q);
  quat_to_rotmat(q, R);
  double num = 0.0;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) num += R[3 * c + r] * S[r][c];  // trace(R S)
  *s = var > 0.0 ? num / var : 1.0;
  for (int r = 0; r < 3; ++r) t[r] = mb[r] - *s * (R[3 * r] * ma[0] + R[3 * r + 1] * ma[1] + R[3 * r + 2] * ma[2]);
}

// The rotation Q that best maps the estimated rotations onto the true ones (Re Q' ~ Rg),
// as the normalized mean of the quaternions of Re' Rg (signs aligned): the chordal mean,
// close enough at these errors.
void fit_rotation(const std::vector<double>& est_q, const std::vector<double>& true_q, double Q[9]) {
  double sum[4] = {0, 0, 0, 0};
  for (std::size_t i = 0; i < est_q.size() / 4; ++i) {
    double Re[9], Rg[9], Ret[9], D[9], q[4];
    quat_to_rotmat(&est_q[4 * i], Re);
    quat_to_rotmat(&true_q[4 * i], Rg);
    transpose3(Re, Ret);
    mat3_mul(Ret, Rg, D);  // Q' for this image
    rotmat_to_quat(D, q);
    const double sign = (q[0] * sum[0] + q[1] * sum[1] + q[2] * sum[2] + q[3] * sum[3]) < 0 ? -1.0 : 1.0;
    for (int k = 0; k < 4; ++k) sum[k] += sign * q[k];
  }
  double Qt[9];
  quat_to_rotmat(sum, Qt);  // normalizes
  transpose3(Qt, Q);
}

double bbox_diagonal(const std::vector<double>& pts) {
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (std::size_t i = 0; i < pts.size(); ++i) {
    lo[i % 3] = std::min(lo[i % 3], pts[i]);
    hi[i % 3] = std::max(hi[i % 3], pts[i]);
  }
  return std::hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}


}  // namespace demo
