// Geometry in front of the solver: unprojection, triangulation, track building and
// starting poses from priors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "check.h"
#include "jba/camera_model.h"
#include "jba/geometry.h"
#include "jba/initialize.h"
#include "jba/priors.h"
#include "jba/tracks.h"
#include "jba/triangulation.h"
#include "synthetic.h"

using jba::CameraModel;

namespace {

void centre(const double* pose, double* C) {
  const double c[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  jba::quat_rotate(c, pose + 4, C);
  for (int a = 0; a < 3; ++a) C[a] = -C[a];
}

double pose_distance(const double* a, const double* b) {
  double Ca[3], Cb[3];
  centre(a, Ca);
  centre(b, Cb);
  double d = 0;
  for (int c = 0; c < 3; ++c) d += (Ca[c] - Cb[c]) * (Ca[c] - Cb[c]);
  // Angle of a^-1 b by atan2: acos near 1 cannot resolve angles below about 1e-8.
  const double ac[4] = {a[0], -a[1], -a[2], -a[3]};
  double e[4];
  jba::quat_multiply(ac, b, e);
  const double v = std::sqrt(e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);
  return std::sqrt(d) + 2 * std::atan2(v, std::abs(e[0]));
}

}  // namespace

int main() {
  std::mt19937_64 rng(3);
  jba_test::Uniform u(-1, 1);

  // Unprojection inverts projection for every model, up to wide angles for the
  // fisheye.
  for (CameraModel m : {CameraModel::SimplePinhole, CameraModel::Pinhole, CameraModel::SimpleRadial,
                        CameraModel::Radial, CameraModel::OpenCV, CameraModel::OpenCVFisheye}) {
    double k[8];
    jba_test::typical_intrinsics(m, k);
    const double reach = m == CameraModel::OpenCVFisheye ? 2.5 : 0.7;
    double worst = 0;
    for (int n = 0; n < 2000; ++n) {
      const double pc[3] = {reach * u(rng), reach * 0.75 * u(rng), 1};
      double x, y, b[3];
      EXPECT(jba::project(m, k, pc, &x, &y));
      EXPECT(jba::unproject(m, k, x, y, b));
      const double nrm = std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + 1);
      for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(b[c] - pc[c] / nrm));
    }
    // The principal point itself.
    double b[3];
    const double cx = m == CameraModel::SimplePinhole || m == CameraModel::SimpleRadial || m == CameraModel::Radial ? k[1] : k[2];
    const double cy = m == CameraModel::SimplePinhole || m == CameraModel::SimpleRadial || m == CameraModel::Radial ? k[2] : k[3];
    EXPECT(jba::unproject(m, k, cx, cy, b) && std::abs(b[2] - 1) < 1e-15);
    std::printf("%-15s unproject round trip: worst bearing error %.3g\n", std::string(jba::model_name(m)).c_str(), worst);
    EXPECT(worst < 1e-10);
  }
  // Behind the camera, no projection.
  {
    double k[4] = {500, 500, 320, 240}, x, y;
    const double pc[3] = {0, 0, -1};
    EXPECT(!jba::project(CameraModel::Pinhole, k, pc, &x, &y));
  }

  // Triangulation: exact rays meet at the point; outliers are rejected; a point
  // seen along nearly parallel rays is refused.
  {
    const double X[3] = {0.3, -0.2, 5};
    std::vector<jba::Ray> rays;
    for (int i = 0; i < 6; ++i) {
      jba::Ray r;
      r.C[0] = -1 + 0.4 * i;
      r.C[1] = 0.1 * u(rng);
      r.C[2] = 0;
      double d[3] = {X[0] - r.C[0], X[1] - r.C[1], X[2] - r.C[2]};
      const double nrm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      for (int c = 0; c < 3; ++c) r.d[c] = d[c] / nrm;
      rays.push_back(r);
    }
    double Y[3];
    EXPECT(jba::triangulate_rays(rays.data(), rays.size(), Y));
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(Y[c], X[c], 1e-12);
    // Two rays pointing elsewhere.
    rays[1].d[0] += 0.3;
    rays[4].d[1] -= 0.3;
    for (int i : {1, 4}) {
      const double nrm = std::sqrt(rays[i].d[0] * rays[i].d[0] + rays[i].d[1] * rays[i].d[1] + rays[i].d[2] * rays[i].d[2]);
      for (double& v : rays[i].d) v /= nrm;
    }
    const auto res = jba::triangulate_robust(rays, {});
    EXPECT(res.ok && res.num_inliers == 4 && !res.inlier[1] && !res.inlier[4]);
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(res.X[c], X[c], 1e-9);
    // One key twice: only the better ray is kept.
    std::vector<std::int32_t> keys = {0, 1, 2, 3, 4, 0};
    const auto res2 = jba::triangulate_robust(rays, {}, &keys);
    EXPECT(res2.ok && !(res2.inlier[0] && res2.inlier[5]));
    // Nearly parallel rays: angle below 1.5 degrees.
    std::vector<jba::Ray> close = {rays[0], rays[2]};
    for (auto& r : close) r.C[0] *= 0.01;
    EXPECT(!jba::triangulate_robust(close, {}).ok);
  }

  // Track building: clean components become tracks; a fused component is split by
  // the geometric repair, or dropped without cameras.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 5, 40, 11, 0.0);
    // Keypoints: observation k of image i, in problem order.
    jba::Keypoints kp;
    kp.begin.assign(1, 0);
    std::vector<std::vector<int>> kp_of_point(5, std::vector<int>(40, -1));
    for (int i = 0; i < 5; ++i) {
      int count = 0;
      for (std::size_t k = 0; k < p.num_observations(); ++k) {
        if (p.obs_image[k] != i) continue;
        kp.xy.insert(kp.xy.end(), {p.obs_xy[2 * k], p.obs_xy[2 * k + 1]});
        kp_of_point[static_cast<std::size_t>(i)][static_cast<std::size_t>(p.obs_point[k])] = count++;
      }
      kp.begin.push_back(static_cast<std::int64_t>(kp.xy.size() / 2));
    }
    std::vector<std::int32_t> matches;
    for (int j = 0; j < 40; ++j)
      for (int a = 0; a < 4; ++a) {
        const int ka = kp_of_point[static_cast<std::size_t>(a)][static_cast<std::size_t>(j)];
        const int kb = kp_of_point[static_cast<std::size_t>(a + 1)][static_cast<std::size_t>(j)];
        if (ka >= 0 && kb >= 0) matches.insert(matches.end(), {a, ka, a + 1, kb});
      }
    // One wrong match fuses points 0 and 1.
    matches.insert(matches.end(), {0, kp_of_point[0][0], 1, kp_of_point[1][1]});
    jba::TrackOptions to;
    to.conflict_policy = jba::ConflictPolicy::Drop;
    jba::TrackStats st;
    auto t = jba::build_tracks(kp, matches, to, nullptr, &st);
    std::printf("tracks without cameras: %zu tracks, %zu conflicted, %zu observations dropped\n", t.size(), st.conflicted,
                st.dropped_observations);
    EXPECT(st.conflicted == 1 && t.size() == 38);
    jba::Cameras cams{p.model, p.intrinsics, {}, p.poses};
    to.conflict_policy = jba::ConflictPolicy::Repair;
    t = jba::build_tracks(kp, matches, to, &cams, &st);
    std::printf("tracks with repair: %zu tracks, %zu observations dropped\n", t.size(), st.dropped_observations);
    EXPECT(t.size() == 40);
    EXPECT(st.dropped_observations <= 1);
    // Every track holds keypoints of one point only.
    bool pure = true;
    for (std::size_t q = 0; q < t.size(); ++q) {
      int point = -1;
      for (auto o = t.begin[q]; o < t.begin[q + 1]; ++o) {
        const auto i = static_cast<std::size_t>(t.image[static_cast<std::size_t>(o)]);
        int j = -1;
        for (int c = 0; c < 40; ++c)
          if (kp_of_point[i][static_cast<std::size_t>(c)] == t.keypoint[static_cast<std::size_t>(o)]) j = c;
        if (point < 0) point = j;
        pure = pure && j == point;
      }
    }
    EXPECT(pure);
  }

  // Starting poses: exact relative priors reproduce the poses up to the root's
  // pose; groups are found; a similarity set gets its transform.
  {
    auto p = jba_test::random_problem(CameraModel::Pinhole, 8, 0, 13);
    const auto truth = p.poses;
    jba::add_relative_priors_from_poses(p, truth, {0, 1, 1, 2, 2, 3, 0, 3, 4, 5, 5, 6, 6, 7});
    // Start from wrong poses except the roots, which keep theirs.
    for (std::size_t i = 0; i < 8; ++i) {
      if (i == 0 || i == 5) continue;  // roots: the most priors (ties: lowest index)
      for (int a = 4; a < 7; ++a) p.poses[7 * i + static_cast<std::size_t>(a)] += 3;
    }
    auto init = jba::initialize_poses(p);
    EXPECT(init.num_groups == 2);
    EXPECT(init.group[0] == init.group[3] && init.group[4] == init.group[7] && init.group[0] != init.group[4]);
    double worst = 0;
    for (std::size_t i = 0; i < 8; ++i) worst = std::max(worst, pose_distance(&init.poses[7 * i], &truth[7 * i]));
    std::printf("chained relative priors: worst pose error %.3g\n", worst);
    EXPECT(worst < 1e-12);

    // Global priors in a similarity frame on group 0, identity priors on group 1:
    // group 1 is the working frame; the set's transform comes from group... none
    // placed there yet, so it is taken as is and group 0 keeps its own frame.
    jba::Problem q = p;
    q.glob_image = {4, 6};
    q.glob_pose.assign(truth.begin() + 28, truth.begin() + 35);
    q.glob_pose.insert(q.glob_pose.end(), truth.begin() + 42, truth.begin() + 49);
    init = jba::initialize_poses(q);
    EXPECT(init.group[4] == 0);  // the working frame comes first
    worst = 0;
    for (std::size_t i = 4; i < 8; ++i) worst = std::max(worst, pose_distance(&init.poses[7 * i], &truth[7 * i]));
    EXPECT(worst < 1e-12);

    // Similarity set: images 0..3 have priors in a frame scaled by 2 and moved;
    // with group 0 placed by identity priors on image 0, the transform is found.
    jba::Problem r = p;
    const double S[8] = {0.8, 0.6, 0, 0, 1, 2, 3, 2};  // set to working: s R X + t
    r.set_relation = {jba::FrameRelation::Identity, jba::FrameRelation::Similarity};
    r.glob_image = {0, 0, 1, 2, 3};
    r.glob_set = {0, 1, 1, 1, 1};
    r.glob_pose.assign(truth.begin(), truth.begin() + 7);
    for (int i : {0, 1, 2, 3}) {
      const double* T = &truth[7 * static_cast<std::size_t>(i)];
      double g[7], Rt[3];
      jba::quat_multiply(T, S, g);
      jba::quat_normalize(g);
      jba::quat_rotate(T, S + 4, Rt);
      for (int c = 0; c < 3; ++c) g[4 + c] = (Rt[c] + T[4 + c]) / S[7];
      r.glob_pose.insert(r.glob_pose.end(), g, g + 7);
    }
    init = jba::initialize_poses(r);
    double terr = 0;
    for (int a = 0; a < 8; ++a) terr = std::max(terr, std::abs(init.set_transform[8 + static_cast<std::size_t>(a)] - S[a]));
    std::printf("similarity set transform: worst error %.3g\n", terr);
    EXPECT(terr < 1e-9);
  }

  return TEST_MAIN_RESULT();
}
