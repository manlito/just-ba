// The mapper. Ported from poselib-gpu's incremental reconstruction test, with its
// rules unchanged; this version uses only poselib-gpu's public header and this
// library's camera models.
#include "mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "geometry.h"
#include "jba/camera_model.h"
#include "jba/report.h"
#include "jba/triangulation.h"

namespace demo {
namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// The same camera for poselib-gpu: model ids and parameter order are COLMAP's in both.
jpo_camera to_jpo(const Capture& c) {
  jpo_camera cam;
  std::memset(&cam, 0, sizeof(cam));
  cam.model = static_cast<std::int32_t>(c.model);
  std::copy(c.intrinsics.begin(), c.intrinsics.end(), cam.params);
  return cam;
}

double focal(const Capture& c) {
  switch (c.model) {
    case jba::CameraModel::SimplePinhole:
    case jba::CameraModel::SimpleRadial:
    case jba::CameraModel::Radial:
      return c.intrinsics[0];
    default:
      return 0.5 * (c.intrinsics[0] + c.intrinsics[1]);
  }
}

struct Map {
  const Capture* capture = nullptr;
  jba::Keypoints keypoints;
  jba::Tracks tracks;
  jba::Cameras cams;                            // poses of every image; meaningful where registered
  std::vector<char> registered;
  std::vector<std::vector<std::int64_t>> image_obs;  // per image: its observations (indices into tracks)
  std::vector<std::int32_t> obs_track;
  std::vector<char> has_point;                  // per track
  std::vector<double> X;                        // 3 per track
  std::vector<char> obs_in;                     // per observation: an inlier of its track's point
  std::size_t n_images() const { return registered.size(); }
};

// Reprojection error in pixels, negative when the point cannot be projected.
double reprojection_error(const Map& m, int image, const double* xy, const double* X) {
  const double* pose = &m.cams.poses[7 * static_cast<std::size_t>(image)];
  double R[9], Y[3], u, v;
  quat_to_rotmat(pose, R);
  apply(R, pose + 4, X, Y);
  if (!(Y[2] > 0.0) || !jba::project(m.cams.model, m.cams.intrinsics.data(), Y, &u, &v)) return -1.0;
  return std::hypot(u - xy[0], v - xy[1]);
}

// New points for tracks that have none, and new observations for points that exist,
// from the images registered so far.
void update_points(Map& m, const MapperOptions& o) {
  jba::TriangulationOptions tri;
  // The library's default (2 degrees) is for noisy tracker poses; here poses come from
  // PnP and adjustment, so the angle is the reprojection threshold seen through the camera.
  tri.max_angle_error_deg = std::atan(o.max_reproj_px / focal(*m.capture)) * 180.0 / kPi;
  tri.min_tri_angle_deg = o.min_tri_angle_deg;
  tri.min_views = 2;
  std::vector<jba::Ray> rays;
  std::vector<std::int64_t> idx;
  for (std::size_t t = 0; t < m.tracks.size(); ++t) {
    const std::int64_t b = m.tracks.begin[t], e = m.tracks.begin[t + 1];
    if (m.has_point[t]) {
      for (std::int64_t k = b; k < e; ++k) {
        const int image = m.tracks.image[k];
        if (m.obs_in[k] || !m.registered[image]) continue;
        const double err = reprojection_error(m, image, m.keypoints.at(image, m.tracks.keypoint[k]), &m.X[3 * t]);
        if (err >= 0.0 && err <= o.max_reproj_px) m.obs_in[k] = 1;
      }
      continue;
    }
    rays.clear();
    idx.clear();
    for (std::int64_t k = b; k < e; ++k) {
      const int image = m.tracks.image[k];
      jba::Ray r;
      if (!m.registered[image] || !jba::keypoint_ray(m.cams, image, m.keypoints.at(image, m.tracks.keypoint[k]), r))
        continue;
      rays.push_back(r);
      idx.push_back(k);
    }
    if (rays.size() < 2) continue;
    const jba::TriangulationResult res = jba::triangulate_robust(rays, tri);
    if (!res.ok) continue;
    m.has_point[t] = 1;
    for (int d = 0; d < 3; ++d) m.X[3 * t + d] = res.X[d];
    for (std::size_t k = 0; k < idx.size(); ++k) m.obs_in[idx[k]] = res.inlier[k];
  }
}

struct Adjusted {
  jba::Problem problem;
  std::vector<std::int32_t> image_of;     // problem image -> capture image
  std::vector<std::int32_t> track_of;     // problem point -> track
  std::vector<std::int32_t> obs_keypoint;  // problem observation -> keypoint in its image
  double mean_error_px = 0.0;
};

// Bundle adjustment of everything registered, then the usual filter: observations
// with a large reprojection error or behind the camera are dropped, and points left
// with fewer than two views or a small triangulation angle are removed (their tracks
// can be triangulated again later).
Adjusted adjust(Map& m, const MapperOptions& o, int max_iterations) {
  Adjusted a;
  jba::Problem& p = a.problem;
  p.model = m.cams.model;
  p.intrinsics = m.cams.intrinsics;  // calibrated: every intrinsic stays fixed
  std::vector<int> dense(m.n_images(), -1);
  for (std::size_t i = 0; i < m.n_images(); ++i)
    if (m.registered[i]) {
      dense[i] = static_cast<int>(a.image_of.size());
      a.image_of.push_back(static_cast<std::int32_t>(i));
      p.poses.insert(p.poses.end(), &m.cams.poses[7 * i], &m.cams.poses[7 * i] + 7);
    }
  std::vector<std::int64_t> obs_of;  // problem observation -> observation of the map
  for (std::size_t t = 0; t < m.tracks.size(); ++t) {
    if (!m.has_point[t]) continue;
    int n = 0;
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k)
      n += m.obs_in[k] && m.registered[m.tracks.image[k]];
    if (n < 2) continue;
    const int j = static_cast<int>(a.track_of.size());
    a.track_of.push_back(static_cast<std::int32_t>(t));
    p.points.insert(p.points.end(), &m.X[3 * t], &m.X[3 * t] + 3);
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k) {
      const int image = m.tracks.image[k];
      if (!m.obs_in[k] || !m.registered[image]) continue;
      const double* xy = m.keypoints.at(image, m.tracks.keypoint[k]);
      p.obs_image.push_back(dense[image]);
      p.obs_point.push_back(j);
      p.obs_xy.push_back(xy[0]);
      p.obs_xy.push_back(xy[1]);
      obs_of.push_back(k);
    }
  }

  jba::SolverOptions so;
  so.loss = {jba::LossType::Cauchy, 2.0};  // the library's own choice in refine()
  so.backend = o.ba_backend;
  so.num_threads = o.threads;
  so.max_iterations = max_iterations;
  // The gauge is free (no priors): GaugeMode::Auto holds one pose and one translation
  // coordinate, so the frame and scale set by the first pair carry through.
  jba::solve(p, so);

  for (std::size_t k = 0; k < a.image_of.size(); ++k)
    std::copy(&p.poses[7 * k], &p.poses[7 * k] + 7, &m.cams.poses[7 * static_cast<std::size_t>(a.image_of[k])]);
  for (std::size_t j = 0; j < a.track_of.size(); ++j)
    std::copy(&p.points[3 * j], &p.points[3 * j] + 3, &m.X[3 * static_cast<std::size_t>(a.track_of[j])]);

  so.backend = jba::BackendKind::Cpu;  // the report is the host's
  const jba::ResidualReport rep = jba::evaluate(p, so);
  double sum = 0.0;
  std::size_t kept = 0;
  for (std::size_t k = 0; k < obs_of.size(); ++k) {
    if (rep.in_front[k] && rep.reprojection_error[k] <= o.max_reproj_px) {
      sum += rep.reprojection_error[k];
      ++kept;
    } else {
      m.obs_in[obs_of[k]] = 0;
    }
  }
  a.mean_error_px = kept ? sum / static_cast<double>(kept) : 0.0;
  std::vector<jba::Ray> rays;
  for (int t : a.track_of) {
    rays.clear();
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k) {
      const int image = m.tracks.image[k];
      jba::Ray r;
      if (m.obs_in[k] && m.registered[image] &&
          jba::keypoint_ray(m.cams, image, m.keypoints.at(image, m.tracks.keypoint[k]), r))
        rays.push_back(r);
    }
    if (rays.size() >= 2 && jba::triangulation_angle(rays.data(), rays.size(), &m.X[3 * t]) * 180.0 / kPi >= o.min_tri_angle_deg)
      continue;
    m.has_point[t] = 0;
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k) m.obs_in[k] = 0;
  }

  // The problem as it stands after the filter: kept observations only.
  jba::Problem kept_problem;
  kept_problem.model = p.model;
  kept_problem.intrinsics = p.intrinsics;
  kept_problem.poses = p.poses;
  std::vector<int> point_index(a.track_of.size(), -1);
  std::vector<std::int32_t> track_of;
  for (std::size_t k = 0; k < obs_of.size(); ++k) {
    if (!m.obs_in[obs_of[k]]) continue;
    const int j = p.obs_point[k];
    if (point_index[j] < 0) {
      point_index[j] = static_cast<int>(track_of.size());
      track_of.push_back(a.track_of[j]);
      kept_problem.points.insert(kept_problem.points.end(), &p.points[3 * j], &p.points[3 * j] + 3);
    }
    kept_problem.obs_image.push_back(p.obs_image[k]);
    kept_problem.obs_point.push_back(point_index[j]);
    kept_problem.obs_xy.push_back(p.obs_xy[2 * k]);
    kept_problem.obs_xy.push_back(p.obs_xy[2 * k + 1]);
    a.obs_keypoint.push_back(m.tracks.keypoint[obs_of[k]]);
  }
  a.problem = std::move(kept_problem);
  a.track_of = std::move(track_of);
  return a;
}

void report(const StepCallback& cb, const char* stage, const std::vector<std::int32_t>& added, const Map& m,
            const Adjusted& a) {
  if (!cb) return;
  Snapshot s;
  s.stage = stage;
  s.added = added;
  s.poses = &m.cams.poses;
  s.registered = &m.registered;
  s.points = a.problem.points;
  s.point_track = a.track_of;
  s.mean_error_px = a.mean_error_px;
  cb(s);
}

// Choosing the first pair: among pairs with status OK and enough inliers, the one with
// the most inliers whose median triangulation angle is at least 16 degrees (COLMAP's
// init_min_tri_angle); if there is none, the same with 8, 4, then 2 degrees.
int choose_first_pair(const Capture& c, const Reconstruction& r, const MapperOptions& o) {
  for (const double floor_deg : {16.0, 8.0, 4.0, 2.0}) {
    int best = -1;
    for (std::size_t i = 0; i < c.pairs.size(); ++i) {
      const jpo_pose_result& p = r.pair_results[i];
      if (!r.pair_kept[i] || p.status != JPO_OK || p.n_inliers < o.min_first_pair_inliers ||
          p.median_tri_angle_deg < floor_deg)
        continue;
      if (best < 0 || p.n_inliers > r.pair_results[best].n_inliers) best = static_cast<int>(i);
    }
    if (best >= 0) return best;
  }
  return -1;
}

}  // namespace

Mapper::Mapper(const MapperOptions& options) : options_(options) {
  jpo_config config;
  jpo_config_init(&config);
  config.backend = options.pose_backend;
  config.seed = options.seed;
  config.threads = options.threads;
  ctx_ = jpo_create(&config);
  if (!ctx_) throw std::runtime_error(std::string("poselib-gpu: ") + jpo_last_error());
}

Mapper::~Mapper() { jpo_destroy(ctx_); }

Reconstruction Mapper::run(const Capture& c, const StepCallback& on_step) {
  const MapperOptions& o = options_;
  const auto t_start = Clock::now();
  Reconstruction run;
  run.pose_backend = jpo_backend_name(ctx_);
  Map m;
  m.capture = &c;
  const std::size_t n_img = c.num_images();
  m.registered.assign(n_img, 0);
  if (static_cast<int>(c.intrinsics.size()) != jba::num_params(c.model)) {
    run.error = "the intrinsics do not match the camera model";
    return run;
  }
  const jpo_camera camera = to_jpo(c);

  // -- 1. Verify every pair: one call for the whole capture.
  auto t0 = Clock::now();
  std::vector<std::uint32_t>& offsets = run.pair_offsets;
  std::vector<double> xy_a, xy_b;
  std::vector<std::uint32_t> cam_index(c.pairs.size(), 0);
  offsets.push_back(0);
  for (const Pair& p : c.pairs) {
    if (p.ka.size() != p.kb.size()) {
      run.error = "a pair has more keypoints on one side than the other";
      return run;
    }
    for (std::size_t k = 0; k < p.ka.size(); ++k) {
      xy_a.insert(xy_a.end(), &c.keypoints[p.a][2 * p.ka[k]], &c.keypoints[p.a][2 * p.ka[k]] + 2);
      xy_b.insert(xy_b.end(), &c.keypoints[p.b][2 * p.kb[k]], &c.keypoints[p.b][2 * p.kb[k]] + 2);
    }
    offsets.push_back(static_cast<std::uint32_t>(xy_a.size() / 2));
  }
  jpo_pairs pairs;
  std::memset(&pairs, 0, sizeof(pairs));
  pairs.n_pairs = static_cast<std::uint32_t>(c.pairs.size());
  pairs.n_cameras = 1;
  pairs.offsets = offsets.data();
  pairs.xy_a = xy_a.data();
  pairs.xy_b = xy_b.data();
  pairs.cameras = &camera;
  pairs.camera_a = cam_index.data();
  pairs.camera_b = cam_index.data();
  run.pair_results.resize(c.pairs.size());
  run.pair_inliers.assign(offsets.back(), 0);
  jpo_ransac_options pair_ransac;
  jpo_ransac_options_init(&pair_ransac);
  pair_ransac.max_error_px = o.pair_max_error_px;
  if (jpo_relative_pose(ctx_, &pairs, &pair_ransac, run.pair_results.data(), run.pair_inliers.data()) != 0) {
    run.error = std::string("relative pose: ") + jpo_last_error();
    return run;
  }
  run.time.verify = seconds_since(t0);

  // The matches that go on to track building: the inliers of every pair with enough of
  // them. A DEGENERATE pair (no observable baseline) is kept too: its inliers are good
  // matches and connect images that the pairs with a baseline may not.
  std::vector<std::int32_t> verified;
  run.pair_kept.assign(c.pairs.size(), 0);
  for (std::size_t i = 0; i < c.pairs.size(); ++i) {
    const Pair& p = c.pairs[i];
    const jpo_pose_result& r = run.pair_results[i];
    if (r.status == JPO_OK) ++run.pairs_ok;
    else if (r.status == JPO_DEGENERATE) ++run.pairs_degenerate;
    else ++run.pairs_failed;
    if (r.status != JPO_OK && r.status != JPO_DEGENERATE) continue;
    if (r.n_inliers < o.min_pair_inliers || r.n_inliers < o.min_pair_inlier_ratio * static_cast<double>(p.ka.size()))
      continue;
    run.pair_kept[i] = 1;
    ++run.pairs_kept;
    for (std::size_t k = 0; k < p.ka.size(); ++k)
      if (run.pair_inliers[offsets[i] + k]) verified.insert(verified.end(), {p.a, p.ka[k], p.b, p.kb[k]});
  }

  // -- 2. Tracks. No poses yet, so only the combinatorial stage runs: a component with
  // two keypoints in one image is dropped.
  t0 = Clock::now();
  m.keypoints.begin.push_back(0);
  for (std::size_t i = 0; i < n_img; ++i) {
    m.keypoints.xy.insert(m.keypoints.xy.end(), c.keypoints[i].begin(), c.keypoints[i].end());
    m.keypoints.begin.push_back(static_cast<std::int64_t>(m.keypoints.xy.size() / 2));
  }
  jba::TrackOptions track_options;
  track_options.min_views = 2;
  track_options.conflict_policy = jba::ConflictPolicy::Drop;
  m.tracks = jba::build_tracks(m.keypoints, verified, track_options);
  run.tracks = m.tracks.size();
  run.time.tracks = seconds_since(t0);
  const std::size_t n_obs = m.tracks.image.size();
  m.image_obs.assign(n_img, {});
  m.obs_track.assign(n_obs, 0);
  for (std::size_t t = 0; t < m.tracks.size(); ++t)
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k) {
      m.obs_track[k] = static_cast<std::int32_t>(t);
      m.image_obs[m.tracks.image[k]].push_back(k);
    }
  m.has_point.assign(m.tracks.size(), 0);
  m.X.assign(3 * m.tracks.size(), 0.0);
  m.obs_in.assign(n_obs, 0);
  m.cams.model = c.model;
  m.cams.intrinsics = c.intrinsics;
  m.cams.poses.assign(7 * n_img, 0.0);
  for (std::size_t i = 0; i < n_img; ++i) m.cams.poses[7 * i] = 1.0;

  auto finish = [&](Adjusted&& a) {
    run.poses = m.cams.poses;
    run.registered = m.registered;
    run.problem = std::move(a.problem);
    run.image_of = std::move(a.image_of);
    run.obs_keypoint = std::move(a.obs_keypoint);
    run.mean_error_px = a.mean_error_px;
    run.time.total = seconds_since(t_start);
  };
  auto timed_points = [&] {
    const auto t = Clock::now();
    update_points(m, o);
    run.time.triangulate += seconds_since(t);
  };
  auto timed_adjust = [&](int iterations) {
    const auto t = Clock::now();
    Adjusted a = adjust(m, o, iterations);
    ++run.adjustments;
    run.time.adjust += seconds_since(t);
    return a;
  };

  // -- 3. First pair, first points, first adjustment. The first image is the origin;
  // the second is the relative pose as returned (camera a to camera b is then world to
  // camera b), whose unit translation fixes the scale of the map.
  const int first = choose_first_pair(c, run, o);
  if (first < 0) {
    run.error = "no pair is fit to start from";
    run.time.total = seconds_since(t_start);
    return run;
  }
  run.first_a = c.pairs[first].a;
  run.first_b = c.pairs[first].b;
  run.first_inliers = run.pair_results[first].n_inliers;
  run.first_angle_deg = run.pair_results[first].median_tri_angle_deg;
  const double identity[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::copy(identity, identity + 7, &m.cams.poses[7 * static_cast<std::size_t>(run.first_a)]);
  std::copy(run.pair_results[first].pose, run.pair_results[first].pose + 7,
            &m.cams.poses[7 * static_cast<std::size_t>(run.first_b)]);
  m.registered[run.first_a] = m.registered[run.first_b] = 1;
  timed_points();
  Adjusted last = timed_adjust(o.local_iterations);
  report(on_step, "first pair", {run.first_a, run.first_b}, m, last);

  // -- 4. Register the remaining images. Those that see the most triangulated points go
  // first; up to pnp_batch are resected in one call, all against the same map.
  std::vector<int> failed_with(n_img, -1);  // points an image saw when its resection last failed
  for (;;) {
    timed_points();
    std::vector<std::pair<int, int>> seen;  // (points seen, image)
    for (std::size_t i = 0; i < n_img; ++i) {
      if (m.registered[i]) continue;
      int n = 0;
      for (std::int64_t k : m.image_obs[i]) n += m.has_point[m.obs_track[k]];
      // An image that failed is tried again only once it sees more points than it did then.
      if (n >= static_cast<int>(o.min_pnp_inliers) && n > failed_with[i]) seen.emplace_back(n, static_cast<int>(i));
    }
    if (seen.empty()) break;
    std::sort(seen.begin(), seen.end(),
              [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
    // Only images that see at least half as many points as the best one: a weakly
    // connected image is better registered later, against a larger map.
    std::size_t n_cand = 1;
    while (n_cand < seen.size() && n_cand < o.pnp_batch && 2 * seen[n_cand].first >= seen[0].first) ++n_cand;

    t0 = Clock::now();
    std::vector<std::uint32_t> off = {0}, cam(n_cand, 0);
    std::vector<double> xy, xyz;
    for (std::size_t k = 0; k < n_cand; ++k) {
      const int image = seen[k].second;
      for (std::int64_t ob : m.image_obs[image]) {
        const std::int32_t t = m.obs_track[ob];
        if (!m.has_point[t]) continue;
        const double* kp = m.keypoints.at(image, m.tracks.keypoint[ob]);
        xy.insert(xy.end(), kp, kp + 2);
        xyz.insert(xyz.end(), &m.X[3 * t], &m.X[3 * t] + 3);
      }
      off.push_back(static_cast<std::uint32_t>(xy.size() / 2));
    }
    jpo_resections res;
    std::memset(&res, 0, sizeof(res));
    res.n_images = static_cast<std::uint32_t>(n_cand);
    res.n_cameras = 1;
    res.offsets = off.data();
    res.xy = xy.data();
    res.xyz = xyz.data();
    res.cameras = &camera;
    res.camera = cam.data();
    std::vector<jpo_pose_result> out(n_cand);
    jpo_ransac_options pnp_ransac;
    jpo_ransac_options_init(&pnp_ransac);
    pnp_ransac.max_error_px = o.pnp_max_error_px;
    if (jpo_absolute_pose(ctx_, &res, &pnp_ransac, out.data(), nullptr) != 0) {
      run.error = std::string("absolute pose: ") + jpo_last_error();
      finish(std::move(last));
      return run;
    }
    ++run.pnp_calls;
    run.pnp_problems += static_cast<int>(n_cand);
    run.time.pnp += seconds_since(t0);

    std::vector<std::int32_t> added;
    for (std::size_t k = 0; k < n_cand; ++k) {
      const int image = seen[k].second;
      const std::uint32_t n = off[k + 1] - off[k];
      const bool ok = out[k].status == JPO_OK && out[k].n_inliers >= o.min_pnp_inliers &&
                      out[k].n_inliers >= o.min_pnp_inlier_ratio * static_cast<double>(n);
      if (!ok) {
        failed_with[image] = seen[k].first;
        ++run.pnp_rejected;
        continue;
      }
      std::copy(out[k].pose, out[k].pose + 7, &m.cams.poses[7 * static_cast<std::size_t>(image)]);
      m.registered[image] = 1;
      added.push_back(image);
    }
    if (added.empty()) continue;  // the rejected images wait until the map grows; others get their turn
    timed_points();
    last = timed_adjust(o.local_iterations);
    report(on_step, "register", added, m, last);
  }

  // -- 5. Final adjustment: complete the tracks, adjust, filter, and adjust what is left.
  timed_points();
  last = timed_adjust(o.final_iterations);
  last = timed_adjust(o.final_iterations);
  report(on_step, "final", {}, m, last);
  finish(std::move(last));
  return run;
}

}  // namespace demo
