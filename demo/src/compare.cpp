#include "compare.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
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

  // Track agreement: the reference's point of each (image, keypoint), then ours.
  auto key = [](std::int64_t image, std::int64_t keypoint) { return (image << 32) | keypoint; };
  std::unordered_map<std::int64_t, std::int32_t> ref_point;
  ref_point.reserve(ref.obs_image.size());
  for (std::size_t k = 0; k < ref.obs_image.size(); ++k) ref_point[key(ref.obs_image[k], ref.obs_keypoint[k])] = ref.obs_point[k];
  std::unordered_map<std::int32_t, std::unordered_set<std::int32_t>> ours_of_ref;  // reference point -> our points
  std::unordered_map<std::int32_t, std::unordered_set<std::int32_t>> refs_of_ours;  // our point -> reference points
  std::unordered_set<std::int64_t> ours_keys;
  for (std::size_t k = 0; k < r.problem.num_observations(); ++k) {
    const std::int64_t image = r.image_of[static_cast<std::size_t>(r.problem.obs_image[k])];
    const std::int64_t kk = key(image, r.obs_keypoint[k]);
    ours_keys.insert(kk);
    const auto it = ref_point.find(kk);
    if (it == ref_point.end()) {
      ++c.ours_only_obs;
      continue;
    }
    ++c.shared_obs;
    ours_of_ref[it->second].insert(r.problem.obs_point[k]);
    refs_of_ours[r.problem.obs_point[k]].insert(it->second);
  }
  std::unordered_map<std::int64_t, std::int32_t> track_of_key;  // our tracks' keypoints
  for (std::size_t t = 0; t < r.all_tracks.size(); ++t)
    for (std::int64_t k = r.all_tracks.begin[t]; k < r.all_tracks.begin[t + 1]; ++k)
      track_of_key[key(r.all_tracks.image[k], r.all_tracks.keypoint[k])] = static_cast<std::int32_t>(t);
  std::unordered_set<std::int32_t> tracks_with_point(r.point_track.begin(), r.point_track.end());
  for (const auto& [kk, p] : ref_point) {
    if (ours_keys.count(kk)) continue;
    ++c.reference_only_obs;
    const auto t = track_of_key.find(kk);
    if (t == track_of_key.end()) ++c.missing_untracked;
    else if (tracks_with_point.count(t->second)) ++c.missing_filtered;
    else ++c.missing_untriangulated;
  }
  std::vector<int> obs_per_point(r.problem.num_points(), 0);
  for (std::size_t k = 0; k < r.problem.num_observations(); ++k) ++obs_per_point[static_cast<std::size_t>(r.problem.obs_point[k])];
  for (std::size_t j = 0; j < obs_per_point.size(); ++j) {
    if (refs_of_ours.count(static_cast<std::int32_t>(j))) continue;
    ++c.ours_only_points;
    c.ours_only_two_view += obs_per_point[j] == 2;
  }
  std::size_t total = 0;
  for (const auto& [p, ours] : ours_of_ref) {
    total += ours.size();
    c.reference_points_split += ours.size() > 1;
  }
  c.reference_points_seen = ours_of_ref.size();
  c.ours_per_reference_point = c.reference_points_seen ? static_cast<double>(total) / static_cast<double>(c.reference_points_seen) : 0.0;
  c.our_points_seen = refs_of_ours.size();
  for (const auto& [p, refs] : refs_of_ours) c.our_points_joined += refs.size() > 1;

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
