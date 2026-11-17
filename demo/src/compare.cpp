#include "compare.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "align.h"
#include "geometry.h"
#include "jba/camera_model.h"

namespace demo {

Comparison compare(const Capture& capture, const Reference& ref, const Reconstruction& r) {
  Comparison c;
  const std::size_t n_img = capture.num_images();
  const std::size_t np = static_cast<std::size_t>(jba::num_params(capture.model));

  // Ours.
  for (char g : r.registered) c.ours.registered += g != 0;
  c.ours.points = r.problem.num_points();
  c.ours.observations = r.problem.num_observations();
  c.ours.mean_track_length = c.ours.points ? static_cast<double>(c.ours.observations) / static_cast<double>(c.ours.points) : 0.0;
  c.ours.mean_error_px = r.mean_error_px;

  // The reference, its reprojection errors through its own intrinsics.
  for (char g : ref.registered) c.reference.registered += g != 0;
  c.reference.points = ref.points.size() / 3;
  c.reference.observations = ref.obs_image.size();
  c.reference.mean_track_length =
      c.reference.points ? static_cast<double>(c.reference.observations) / static_cast<double>(c.reference.points) : 0.0;
  const std::vector<double>& intr = ref.intrinsics.empty() ? capture.intrinsics : ref.intrinsics;
  double sum = 0.0;
  std::size_t counted = 0;
  for (std::size_t k = 0; k < ref.obs_image.size(); ++k) {
    const std::size_t i = static_cast<std::size_t>(ref.obs_image[k]);
    const double* pose = &ref.poses[7 * i];
    double R[9], Y[3], u, v;
    quat_to_rotmat(pose, R);
    apply(R, pose + 4, &ref.points[3 * static_cast<std::size_t>(ref.obs_point[k])], Y);
    if (!(Y[2] > 0.0) ||
        !jba::project(capture.model, &intr[np * static_cast<std::size_t>(capture.block(i))], Y, &u, &v))
      continue;
    const double* xy = &capture.keypoints[i][2 * static_cast<std::size_t>(ref.obs_keypoint[k])];
    sum += std::hypot(u - xy[0], v - xy[1]);
    ++counted;
  }
  c.reference.mean_error_px = counted ? sum / static_cast<double>(counted) : 0.0;

  // Poses of the images both registered.
  std::vector<double> ours_c, ref_c, ours_q, ref_q, all_ref_c;
  for (std::size_t i = 0; i < n_img; ++i) {
    if (!ref.registered[i]) continue;
    double C[3];
    camera_centre(&ref.poses[7 * i], C);
    all_ref_c.insert(all_ref_c.end(), C, C + 3);
    if (i >= r.registered.size() || !r.registered[i]) continue;
    ref_c.insert(ref_c.end(), C, C + 3);
    camera_centre(&r.poses[7 * i], C);
    ours_c.insert(ours_c.end(), C, C + 3);
    ours_q.insert(ours_q.end(), &r.poses[7 * i], &r.poses[7 * i] + 4);
    ref_q.insert(ref_q.end(), &ref.poses[7 * i], &ref.poses[7 * i] + 4);
  }
  c.common = ours_c.size() / 3;
  if (c.common < 3) return c;

  double Rs[9], ts[3], s, Q[9], Qt[9];
  fit_similarity(ours_c, ref_c, Rs, ts, &s);
  fit_rotation(ours_q, ref_q, Q);
  transpose3(Q, Qt);
  const double extent = bbox_diagonal(all_ref_c);
  std::vector<double> rot, centre;
  for (std::size_t k = 0; k < c.common; ++k) {
    double X[3];
    for (int d = 0; d < 3; ++d)
      X[d] = s * (Rs[3 * d] * ours_c[3 * k] + Rs[3 * d + 1] * ours_c[3 * k + 1] + Rs[3 * d + 2] * ours_c[3 * k + 2]) + ts[d];
    centre.push_back(std::hypot(X[0] - ref_c[3 * k], X[1] - ref_c[3 * k + 1], X[2] - ref_c[3 * k + 2]));
    double Ro[9], Rr[9], Ra[9];
    quat_to_rotmat(&ours_q[4 * k], Ro);
    quat_to_rotmat(&ref_q[4 * k], Rr);
    mat3_mul(Ro, Qt, Ra);
    rot.push_back(rotation_angle_deg(Ra, Rr));
  }
  c.rot_median_deg = median(rot);
  c.rot_max_deg = *std::max_element(rot.begin(), rot.end());
  c.centre_median = median(centre);
  c.centre_median_pct = extent > 0 ? 100.0 * c.centre_median / extent : 0.0;
  c.centre_max_pct = extent > 0 ? 100.0 * *std::max_element(centre.begin(), centre.end()) / extent : 0.0;
  return c;
}

}  // namespace demo
