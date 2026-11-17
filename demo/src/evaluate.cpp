#include "evaluate.h"

#include <algorithm>
#include <cmath>

#include "align.h"
#include "geometry.h"

namespace demo {

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
