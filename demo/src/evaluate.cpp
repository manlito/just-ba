#include "evaluate.h"

#include <algorithm>
#include <cmath>

#include "geometry.h"

namespace demo {
namespace {

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

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

}  // namespace

Metrics evaluate(const Scene& scene, const Reconstruction& r) {
  Metrics mt;
  const std::size_t n_img = scene.poses.size() / 7;
  mt.images = n_img;
  mt.points = r.problem.num_points();
  mt.observations = r.problem.num_observations();
  mt.mean_error_px = r.mean_error_px;

  // Pair verification against the truth.
  std::size_t kept_matches = 0, kept_correct = 0, correct_total = 0;
  for (std::size_t i = 0; i < r.pair_kept.size(); ++i) {
    if (!r.pair_kept[i]) continue;
    if (scene.shared[i] == 0) ++mt.false_pairs_kept;
    for (std::size_t k = 0; k < scene.correct[i].size(); ++k) {
      correct_total += scene.correct[i][k];
      if (!r.pair_inliers[r.pair_offsets[i] + k]) continue;
      ++kept_matches;
      kept_correct += scene.correct[i][k];
    }
  }
  mt.mask_precision = kept_matches ? static_cast<double>(kept_correct) / static_cast<double>(kept_matches) : 0.0;
  mt.mask_recall = correct_total ? static_cast<double>(kept_correct) / static_cast<double>(correct_total) : 0.0;

  // The scene extent: the box around the cameras and every point some image saw.
  std::vector<double> all, true_centres;
  std::vector<char> seen(scene.points.size() / 3, 0);
  for (std::size_t i = 0; i < n_img; ++i) {
    double C[3];
    camera_centre(&scene.poses[7 * i], C);
    true_centres.insert(true_centres.end(), C, C + 3);
    for (int j : scene.kp_point[i]) seen[j] = 1;
  }
  all = true_centres;
  for (std::size_t j = 0; j < seen.size(); ++j)
    if (seen[j]) all.insert(all.end(), &scene.points[3 * j], &scene.points[3 * j] + 3);
  const double extent = bbox_diagonal(all);

  std::vector<double> est, gt;
  std::vector<int> images;
  for (std::size_t i = 0; i < r.registered.size(); ++i) {
    if (!r.registered[i]) continue;
    double C[3];
    camera_centre(&r.poses[7 * i], C);
    est.insert(est.end(), C, C + 3);
    gt.insert(gt.end(), &true_centres[3 * i], &true_centres[3 * i] + 3);
    images.push_back(static_cast<int>(i));
  }
  mt.registered = images.size();
  if (!r.ok()) {
    mt.why = r.error;
    return mt;
  }
  if (images.size() < 3) {
    mt.why = "fewer than three images registered";
    return mt;
  }
  fit_similarity(est, gt, mt.R, mt.t, &mt.s);
  // Rotations are compared after their own best common rotation, not the centres'
  // alignment: with short baselines (the grid's 20 cm) the rotation of a fit to the
  // centres is less certain than the rotations themselves.
  std::vector<double> est_q, true_q;
  for (int i : images) {
    est_q.insert(est_q.end(), &r.poses[7 * static_cast<std::size_t>(i)], &r.poses[7 * static_cast<std::size_t>(i)] + 4);
    true_q.insert(true_q.end(), &scene.poses[7 * static_cast<std::size_t>(i)], &scene.poses[7 * static_cast<std::size_t>(i)] + 4);
  }
  double Q[9], Qt[9];
  fit_rotation(est_q, true_q, Q);
  transpose3(Q, Qt);
  std::vector<double> rot_err, centre_err;
  for (std::size_t k = 0; k < images.size(); ++k) {
    double c[3];
    for (int d = 0; d < 3; ++d)
      c[d] = mt.s * (mt.R[3 * d] * est[3 * k] + mt.R[3 * d + 1] * est[3 * k + 1] + mt.R[3 * d + 2] * est[3 * k + 2]) + mt.t[d];
    centre_err.push_back(std::hypot(c[0] - gt[3 * k], c[1] - gt[3 * k + 1], c[2] - gt[3 * k + 2]));
    double Re[9], Rg[9], Ra[9];
    quat_to_rotmat(&r.poses[7 * static_cast<std::size_t>(images[k])], Re);
    quat_to_rotmat(&scene.poses[7 * static_cast<std::size_t>(images[k])], Rg);
    mat3_mul(Re, Qt, Ra);
    rot_err.push_back(rotation_angle_deg(Ra, Rg));
  }
  mt.rot_median_deg = median(rot_err);
  mt.rot_max_deg = *std::max_element(rot_err.begin(), rot_err.end());
  mt.centre_median_pct = 100.0 * median(centre_err) / extent;
  mt.centre_max_pct = 100.0 * *std::max_element(centre_err.begin(), centre_err.end()) / extent;

  // Points: each is the scene point most of its keypoints come from.
  std::vector<std::vector<int>> ids(r.problem.num_points());
  for (std::size_t k = 0; k < r.problem.num_observations(); ++k) {
    const int image = r.image_of[static_cast<std::size_t>(r.problem.obs_image[k])];
    ids[static_cast<std::size_t>(r.problem.obs_point[k])].push_back(scene.kp_point[image][r.obs_keypoint[k]]);
  }
  std::vector<double> point_err;
  for (std::size_t j = 0; j < ids.size(); ++j) {
    if (ids[j].empty()) continue;
    std::sort(ids[j].begin(), ids[j].end());
    const double* X = &r.problem.points[3 * j];
    const double* G = &scene.points[3 * static_cast<std::size_t>(ids[j][ids[j].size() / 2])];
    double c[3];
    for (int d = 0; d < 3; ++d) c[d] = mt.s * (mt.R[3 * d] * X[0] + mt.R[3 * d + 1] * X[1] + mt.R[3 * d + 2] * X[2]) + mt.t[d];
    point_err.push_back(std::hypot(c[0] - G[0], c[1] - G[1], c[2] - G[2]));
  }
  mt.point_median_pct = 100.0 * median(point_err) / extent;

  if (mt.registered != mt.images) mt.why = "not every image registered";
  else if (!(mt.rot_median_deg < 0.5)) mt.why = "median rotation error";
  else if (!(mt.centre_median_pct < 1.0)) mt.why = "median centre error";
  else if (!(mt.mean_error_px < std::max(1.5, 2.0 * scene.noise_px))) mt.why = "mean reprojection error";
  else mt.pass = true;
  return mt;
}

}  // namespace demo
