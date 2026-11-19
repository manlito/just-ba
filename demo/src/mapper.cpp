// The mapper. Ported from poselib-gpu's incremental reconstruction test, with its
// rules unchanged; this version uses only poselib-gpu's public header and this
// library's camera models.
#include "mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "align.h"
#include "geometry.h"
#include "jba/camera_model.h"
#include "jba/report.h"
#include "jba/triangulation.h"

namespace demo {

double median_focal(jba::CameraModel model, const std::vector<double>& intrinsics) {
  const std::size_t np = static_cast<std::size_t>(jba::num_params(model));
  const bool one_focal = model == jba::CameraModel::SimplePinhole || model == jba::CameraModel::SimpleRadial ||
                         model == jba::CameraModel::Radial;
  std::vector<double> f;
  for (std::size_t b = 0; b < intrinsics.size() / np; ++b)
    f.push_back(one_focal ? intrinsics[np * b] : 0.5 * (intrinsics[np * b] + intrinsics[np * b + 1]));
  if (f.empty()) return 0.0;
  std::nth_element(f.begin(), f.begin() + f.size() / 2, f.end());
  return f[f.size() / 2];
}

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// The same cameras for poselib-gpu: model ids and parameter order are COLMAP's in both.
std::vector<jpo_camera> to_jpo(jba::CameraModel model, const std::vector<double>& intrinsics) {
  const std::size_t np = static_cast<std::size_t>(jba::num_params(model));
  std::vector<jpo_camera> cams(intrinsics.size() / np);
  for (std::size_t b = 0; b < cams.size(); ++b) {
    std::memset(&cams[b], 0, sizeof(jpo_camera));
    cams[b].model = static_cast<std::int32_t>(model);
    std::copy(&intrinsics[np * b], &intrinsics[np * b] + np, cams[b].params);
  }
  return cams;
}


// Empty when the capture is well formed, else what is wrong.
std::string check(const Capture& c) {
  const int np = jba::num_params(c.model);
  if (np <= 0 || c.intrinsics.empty() || c.intrinsics.size() % static_cast<std::size_t>(np) != 0)
    return "the intrinsics do not match the camera model";
  const std::int32_t blocks = static_cast<std::int32_t>(c.intrinsics.size() / static_cast<std::size_t>(np));
  if (!c.image_intrinsics.empty() && c.image_intrinsics.size() != c.num_images())
    return "image_intrinsics needs one block per image";
  for (std::int32_t b : c.image_intrinsics)
    if (b < 0 || b >= blocks) return "an image names an intrinsics block that does not exist";
  if (!c.prior_position.empty() && c.prior_position.size() != 3 * c.num_images())
    return "prior_position needs 3 values per image";
  if (!c.prior_sigma.empty() && c.prior_sigma.size() != 2 * c.num_images())
    return "prior_sigma needs 2 values per image";
  if (!c.image_size.empty() && c.image_size.size() != 2 * static_cast<std::size_t>(blocks))
    return "image_size needs a width and a height per intrinsics block";
  for (const Pair& p : c.pairs) {
    if (p.ka.size() != p.kb.size()) return "a pair has more keypoints on one side than the other";
    if (p.a < 0 || p.b < 0 || p.a == p.b || static_cast<std::size_t>(std::max(p.a, p.b)) >= c.num_images())
      return "a pair names an image that does not exist";
    const std::size_t na = c.keypoints[p.a].size() / 2, nb = c.keypoints[p.b].size() / 2;
    for (std::size_t k = 0; k < p.ka.size(); ++k)
      if (p.ka[k] < 0 || p.kb[k] < 0 || static_cast<std::size_t>(p.ka[k]) >= na || static_cast<std::size_t>(p.kb[k]) >= nb)
        return "a match names a keypoint that does not exist";
  }
  return {};
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
  std::vector<int> track_images;                // per track: the images it spans
  std::vector<char> obs_dead;                   // per observation: moved to a track split off
  std::vector<char> obs_filtered;               // per observation: dropped from its track's point by the filter
  std::vector<std::int64_t> split_tried;        // per track: leftovers at its last failed split
  bool aligned = false;                         // in the frame of the position priors (D150)
  int aligned_with = 0;                         // registered images with a prior at the last (re)alignment
  std::size_t n_images() const { return registered.size(); }
};

// Area of the convex hull of `xy` (2 values per point), Andrew's monotone chain; 0 for
// fewer than three points or points on a line.
double hull_area(std::vector<std::pair<double, double>>& p) {
  if (p.size() < 3) return 0.0;
  std::sort(p.begin(), p.end());
  auto cross = [](const std::pair<double, double>& o, const std::pair<double, double>& a,
                  const std::pair<double, double>& b) {
    return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
  };
  std::vector<std::pair<double, double>> h(2 * p.size());
  std::size_t k = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0.0) --k;
    h[k++] = p[i];
  }
  for (std::size_t i = p.size() - 1, t = k + 1; i > 0; --i) {
    while (k >= t && cross(h[k - 2], h[k - 1], p[i - 1]) <= 0.0) --k;
    h[k++] = p[i - 1];
  }
  double area = 0.0;
  for (std::size_t i = 0; i + 1 < k; ++i) area += h[i].first * h[i + 1].second - h[i + 1].first * h[i].second;
  return 0.5 * std::abs(area);
}

// The share of each image of pair `i` the convex hull of its inliers covers, the smaller
// of the two (D151); NaN without the images' sizes.
float pair_coverage(const Capture& c, const Pair& p, const std::uint8_t* inlier) {
  if (c.image_size.empty()) return std::numeric_limits<float>::quiet_NaN();
  double share = 1.0;
  for (const bool side_a : {true, false}) {
    const std::int32_t image = side_a ? p.a : p.b;
    const std::size_t b = static_cast<std::size_t>(c.block(static_cast<std::size_t>(image)));
    const double area = static_cast<double>(c.image_size[2 * b]) * static_cast<double>(c.image_size[2 * b + 1]);
    if (!(area > 0.0)) return std::numeric_limits<float>::quiet_NaN();
    std::vector<std::pair<double, double>> pts;
    const std::vector<std::int32_t>& kp = side_a ? p.ka : p.kb;
    for (std::size_t k = 0; k < kp.size(); ++k)
      if (inlier[k]) pts.emplace_back(c.keypoints[image][2 * kp[k]], c.keypoints[image][2 * kp[k] + 1]);
    share = std::min(share, hull_area(pts) / area);
  }
  return static_cast<float>(share);
}

// Whether the run uses the capture's position priors.
bool uses_priors(const Capture& c, const MapperOptions& o) { return o.use_position_priors && !c.prior_position.empty(); }

// An image's prior standard deviations, horizontal and vertical: the capture's, else
// the options' defaults.
void prior_sigmas(const Capture& c, const MapperOptions& o, std::size_t image, double* h, double* v) {
  *h = o.prior_sigma_horizontal;
  *v = o.prior_sigma_vertical;
  if (c.prior_sigma.empty()) return;
  const double sh = c.prior_sigma[2 * image], sv = c.prior_sigma[2 * image + 1];
  if (std::isfinite(sh) && sh > 0.0) *h = sh;
  if (std::isfinite(sv) && sv > 0.0) *v = sv;
}

// Distance from a camera centre to its prior in standard deviations (Mahalanobis, the
// horizontal and vertical sigmas on their axes).
double prior_distance(const Capture& c, const MapperOptions& o, std::size_t image, const double C[3]) {
  double h, v;
  prior_sigmas(c, o, image, &h, &v);
  const double* P = &c.prior_position[3 * image];
  const double dx = (C[0] - P[0]) / h, dy = (C[1] - P[1]) / h, dz = (C[2] - P[2]) / v;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Root mean square distance of points (3 values each) from their best line: how far
// they spread across it.
double spread_across_line(const std::vector<double>& pts) {
  const std::size_t n = pts.size() / 3;
  if (n < 2) return 0.0;
  double mean[3] = {0, 0, 0};
  for (std::size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) mean[d] += pts[3 * i + d] / static_cast<double>(n);
  double S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 0; r < 3; ++r)
      for (int q = 0; q < 3; ++q) S[3 * r + q] += (pts[3 * i + r] - mean[r]) * (pts[3 * i + q] - mean[q]) / static_cast<double>(n);
  // The largest eigenvalue by power iteration; the rest of the trace is the spread.
  double v[3] = {1.0, 0.7, 0.3}, lambda = 0.0;
  for (int it = 0; it < 100; ++it) {
    double w[3] = {0, 0, 0};
    for (int r = 0; r < 3; ++r)
      for (int q = 0; q < 3; ++q) w[r] += S[3 * r + q] * v[q];
    const double norm = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (!(norm > 0.0)) break;
    for (int d = 0; d < 3; ++d) v[d] = w[d] / norm;
    lambda = norm;
  }
  return std::sqrt(std::max(0.0, S[0] + S[4] + S[8] - lambda));
}

// The median horizontal standard deviation of some images' priors.
double median_sigma_h(const Capture& c, const MapperOptions& o, const std::vector<int>& images) {
  std::vector<double> h;
  for (int i : images) {
    double sh, sv;
    prior_sigmas(c, o, static_cast<std::size_t>(i), &sh, &sv);
    h.push_back(sh);
  }
  return median(h);
}

// The tracks: connected components of the verified matches, two keypoints or more,
// in a fixed order. A component with two keypoints in one image (a wrong match joined
// two points, or a repeated pattern) is kept, unlike the library's build_tracks with
// ConflictPolicy::Drop, which has no poses to split it with here: triangulation takes
// at most one keypoint per image (update_points), so the wrong ones stay out (D140).
jba::Tracks components(const jba::Keypoints& kp, const std::vector<std::int32_t>& matches, jba::TrackStats& st) {
  const std::size_t n = static_cast<std::size_t>(kp.begin.back());
  std::vector<std::int64_t> parent(n);
  for (std::size_t i = 0; i < n; ++i) parent[i] = static_cast<std::int64_t>(i);
  auto find = [&](std::int64_t x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
      x = parent[static_cast<std::size_t>(x)];
    }
    return x;
  };
  std::vector<char> matched(n, 0);
  for (std::size_t k = 0; k + 3 < matches.size(); k += 4) {
    const std::int64_t a = kp.begin[static_cast<std::size_t>(matches[k])] + matches[k + 1];
    const std::int64_t b = kp.begin[static_cast<std::size_t>(matches[k + 2])] + matches[k + 3];
    matched[static_cast<std::size_t>(a)] = matched[static_cast<std::size_t>(b)] = 1;
    const std::int64_t ra = find(a), rb = find(b);
    if (ra != rb) parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
  }
  // Members of each root, in keypoint order; components ordered by their first keypoint.
  std::vector<std::int64_t> count(n, 0), start(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i)
    if (matched[i]) ++count[static_cast<std::size_t>(find(static_cast<std::int64_t>(i)))];
  std::vector<std::int32_t> image_of(n);
  for (std::size_t i = 0; i + 1 < kp.begin.size(); ++i)
    for (std::int64_t g = kp.begin[i]; g < kp.begin[i + 1]; ++g) image_of[static_cast<std::size_t>(g)] = static_cast<std::int32_t>(i);
  jba::Tracks t;
  std::vector<std::int64_t> slot(n, -1);  // root -> its track
  for (std::size_t i = 0; i < n; ++i) {
    if (!matched[i] || find(static_cast<std::int64_t>(i)) != static_cast<std::int64_t>(i) || count[i] < 2) continue;
    slot[i] = static_cast<std::int64_t>(t.size());
    t.begin.push_back(t.begin.back() + count[i]);
  }
  t.image.resize(static_cast<std::size_t>(t.begin.back()));
  t.keypoint.resize(t.image.size());
  std::vector<std::int64_t> fill(t.begin.begin(), t.begin.end() - 1);
  for (std::size_t i = 0; i < n; ++i) {
    if (!matched[i]) continue;
    const std::int64_t s = slot[static_cast<std::size_t>(find(static_cast<std::int64_t>(i)))];
    if (s < 0) continue;
    const std::size_t at = static_cast<std::size_t>(fill[static_cast<std::size_t>(s)]++);
    t.image[at] = image_of[i];
    t.keypoint[at] = static_cast<std::int32_t>(static_cast<std::int64_t>(i) - kp.begin[static_cast<std::size_t>(image_of[i])]);
  }
  st = {};
  st.components = t.size();
  st.tracks = t.size();
  for (std::size_t c = 0; c < t.size(); ++c)
    for (std::int64_t k = t.begin[c] + 1; k < t.begin[c + 1]; ++k)
      if (t.image[static_cast<std::size_t>(k)] == t.image[static_cast<std::size_t>(k - 1)]) {
        ++st.conflicted;
        break;
      }
  return t;
}

// Reprojection error in pixels, negative when the point cannot be projected.
double reprojection_error(const Map& m, int image, const double* xy, const double* X) {
  const double* pose = &m.cams.poses[7 * static_cast<std::size_t>(image)];
  double R[9], Y[3], u, v;
  quat_to_rotmat(pose, R);
  apply(R, pose + 4, X, Y);
  const double* intr = &m.cams.intrinsics[static_cast<std::size_t>(jba::num_params(m.cams.model)) *
                                          static_cast<std::size_t>(m.capture->block(static_cast<std::size_t>(image)))];
  if (!(Y[2] > 0.0) || !jba::project(m.cams.model, intr, Y, &u, &v)) return -1.0;
  return std::hypot(u - xy[0], v - xy[1]);
}

// A track whose point leaves keypoints of three registered images or more unused may
// hold a second point (two points joined by a wrong match, or a repeated pattern):
// those keypoints are triangulated on their own and, if they make a point, move to a
// new track, as the library's ConflictPolicy::Repair splits tracks once poses exist.
// Three images, so stray wrong matches do not become points (D140).
void split_leftovers(Map& m, std::size_t t, const jba::TriangulationOptions& tri) {
  const std::int64_t b = m.tracks.begin[t], e = m.tracks.begin[t + 1];
  std::vector<jba::Ray> rays;
  std::vector<std::int64_t> idx;
  std::vector<std::int32_t> group;
  int images = 0;
  for (std::int64_t k = b; k < e; ++k) {
    const int image = m.tracks.image[k];
    jba::Ray r;
    if (m.obs_in[k] || m.obs_dead[k] || !m.registered[image] ||
        !jba::keypoint_ray(m.cams, image, m.keypoints.at(image, m.tracks.keypoint[k]), r))
      continue;
    images += group.empty() || group.back() != image;  // track entries are in image order
    rays.push_back(r);
    idx.push_back(k);
    group.push_back(image);
  }
  if (images < 3 || static_cast<std::int64_t>(idx.size()) <= m.split_tried[t]) return;
  const jba::TriangulationResult res = jba::triangulate_robust(rays, tri, &group);
  if (!res.ok || res.num_inliers < 3) {
    m.split_tried[t] = static_cast<std::int64_t>(idx.size());
    return;
  }
  const std::int32_t nt = static_cast<std::int32_t>(m.tracks.size());
  for (std::size_t q = 0; q < idx.size(); ++q) {
    if (!res.inlier[q]) continue;
    const std::int64_t k = idx[q];
    const std::int64_t nk = static_cast<std::int64_t>(m.tracks.image.size());
    m.tracks.image.push_back(m.tracks.image[k]);
    m.tracks.keypoint.push_back(m.tracks.keypoint[k]);
    m.obs_track.push_back(nt);
    m.obs_in.push_back(1);
    m.obs_dead.push_back(0);
    m.obs_filtered.push_back(0);
    m.image_obs[m.tracks.image[k]].push_back(nk);
    m.obs_dead[k] = 1;
  }
  m.tracks.begin.push_back(static_cast<std::int64_t>(m.tracks.image.size()));
  // The images the original track still spans (the two-view rule reads it).
  int left = 0, last = -1;
  for (std::int64_t k = b; k < e; ++k)
    if (!m.obs_dead[k] && m.tracks.image[k] != last) {
      last = m.tracks.image[k];
      ++left;
    }
  m.track_images[t] = left;
  m.has_point.push_back(1);
  m.X.insert(m.X.end(), res.X, res.X + 3);
  m.track_images.push_back(res.num_inliers);
  m.split_tried.push_back(0);
}

// New points for tracks that have none, and new observations for points that exist,
// from the images registered so far. `split`: also split tracks (split_leftovers);
// only right after an adjustment, as a split is never undone and poses fresh from PnP
// could agree on a duplicate of a point.
void update_points(Map& m, const MapperOptions& o, bool split) {
  jba::TriangulationOptions tri;
  // The library's default (2 degrees) is for noisy tracker poses. Without an angle of
  // its own, the reprojection threshold seen through the camera: poses here come from
  // PnP and adjustment.
  tri.max_angle_error_deg = o.tri_max_angle_error_deg > 0.0
                                ? o.tri_max_angle_error_deg
                                : std::atan(o.max_reproj_px / median_focal(m.cams.model, m.cams.intrinsics)) * 180.0 / kPi;
  const double max_angle_rad = tri.max_angle_error_deg * kPi / 180.0;
  tri.min_tri_angle_deg = o.min_tri_angle_deg;
  tri.min_views = 2;
  std::vector<jba::Ray> rays;
  std::vector<std::int64_t> idx;
  std::vector<std::int32_t> group;
  std::vector<std::pair<double, std::int64_t>> cand;  // (error, observation)
  std::vector<int> used;                               // images with an inlier in the track
  for (std::size_t t = 0; t < m.tracks.size(); ++t) {
    const std::int64_t b = m.tracks.begin[t], e = m.tracks.begin[t + 1];
    if (o.ignore_two_view_tracks && m.track_images[t] <= 2) continue;
    if (m.has_point[t]) {
      // Continue the track: each registered image without an inlier gets its closest
      // keypoint, if within the threshold (an angle when one is set, else pixels). A
      // keypoint the filter dropped from this point comes back only within the filter's
      // own threshold (COLMAP's complete_max_reproj_error): let in again by the wider
      // angle it would be adjusted and dropped every time, and would never be left over
      // for split_leftovers to make the point it belongs to.
      used.clear();
      cand.clear();
      for (std::int64_t k = b; k < e; ++k)
        if (m.obs_in[k]) used.push_back(m.tracks.image[k]);
      for (std::int64_t k = b; k < e; ++k) {
        const int image = m.tracks.image[k];
        if (m.obs_in[k] || m.obs_dead[k] || !m.registered[image] || std::find(used.begin(), used.end(), image) != used.end())
          continue;
        const double* xy = m.keypoints.at(image, m.tracks.keypoint[k]);
        const double err = reprojection_error(m, image, xy, &m.X[3 * t]);
        if (err < 0.0) continue;
        if (m.obs_filtered[k]) {
          if (err > o.max_reproj_px) continue;
        } else if (o.tri_max_angle_error_deg > 0.0) {
          jba::Ray r;
          if (!jba::keypoint_ray(m.cams, image, xy, r) || jba::ray_angle_error(r, &m.X[3 * t]) > max_angle_rad) continue;
        } else if (err > o.max_reproj_px) {
          continue;
        }
        cand.emplace_back(err, k);
      }
      std::sort(cand.begin(), cand.end());
      for (const auto& [err, k] : cand) {
        const int image = m.tracks.image[k];
        if (std::find(used.begin(), used.end(), image) != used.end()) continue;
        used.push_back(image);
        m.obs_in[k] = 1;
      }
      if (split) split_leftovers(m, t, tri);
      continue;
    }
    rays.clear();
    idx.clear();
    group.clear();
    for (std::int64_t k = b; k < e; ++k) {
      const int image = m.tracks.image[k];
      jba::Ray r;
      if (m.obs_dead[k] || !m.registered[image] ||
          !jba::keypoint_ray(m.cams, image, m.keypoints.at(image, m.tracks.keypoint[k]), r))
        continue;
      rays.push_back(r);
      idx.push_back(k);
      group.push_back(image);
    }
    if (rays.size() < 2) continue;
    // At most one inlier per image: a track may hold two keypoints of one image.
    const jba::TriangulationResult res = jba::triangulate_robust(rays, tri, &group);
    if (!res.ok) continue;
    m.has_point[t] = 1;
    for (int d = 0; d < 3; ++d) m.X[3 * t + d] = res.X[d];
    for (std::size_t k = 0; k < idx.size(); ++k) m.obs_in[idx[k]] = res.inlier[k];
    if (split) split_leftovers(m, t, tri);
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
// can be triangulated again later). `on_iteration` goes to the solver; when it stops the
// solve, the map is left as it was.
Adjusted adjust(Map& m, const MapperOptions& o, int max_iterations, const jba::IterationCallback& on_iteration) {
  Adjusted a;
  jba::Problem& p = a.problem;
  p.model = m.cams.model;
  p.intrinsics = m.cams.intrinsics;
  std::vector<int> dense(m.n_images(), -1);
  for (std::size_t i = 0; i < m.n_images(); ++i)
    if (m.registered[i]) {
      dense[i] = static_cast<int>(a.image_of.size());
      a.image_of.push_back(static_cast<std::int32_t>(i));
      p.poses.insert(p.poses.end(), &m.cams.poses[7 * i], &m.cams.poses[7 * i] + 7);
      if (!m.capture->image_intrinsics.empty()) p.image_intrinsics.push_back(m.capture->image_intrinsics[i]);
    }
  if (o.refine_intrinsics) {
    // The groups asked for, in each block enough registered images use.
    const std::size_t blocks = p.intrinsics.size() / static_cast<std::size_t>(jba::num_params(p.model));
    std::vector<int> users(blocks, 0);
    for (std::int32_t i : a.image_of) ++users[static_cast<std::size_t>(m.capture->block(static_cast<std::size_t>(i)))];
    p.intrinsics_free.assign(blocks, 0);
    for (std::size_t b = 0; b < blocks; ++b)
      if (users[b] >= o.refine_min_images) p.intrinsics_free[b] = o.refine_intrinsics;
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

  // Position priors once the map is in their frame (D150): the camera centre only (the
  // rotation rows of the whitening are zero), in the working frame, so nothing else
  // holds the gauge.
  if (m.aligned && uses_priors(*m.capture, o))
    for (std::size_t k = 0; k < a.image_of.size(); ++k) {
      const std::size_t i = static_cast<std::size_t>(a.image_of[k]);
      if (!m.capture->has_prior(i)) continue;
      double h, v;
      prior_sigmas(*m.capture, o, i, &h, &v);
      const double* P = &m.capture->prior_position[3 * i];
      p.glob_image.push_back(static_cast<std::int32_t>(k));
      p.glob_pose.insert(p.glob_pose.end(), {1.0, 0.0, 0.0, 0.0, -P[0], -P[1], -P[2]});
      double W[36] = {};
      W[3 * 6 + 3] = W[4 * 6 + 4] = 1.0 / h;
      W[5 * 6 + 5] = 1.0 / v;
      p.glob_sqrt_info.insert(p.glob_sqrt_info.end(), W, W + 36);
    }

  jba::SolverOptions so;
  so.loss = {jba::LossType::Cauchy, o.loss_scale > 0.0 ? o.loss_scale : o.max_reproj_px};
  so.backend = o.ba_backend;
  so.num_threads = o.threads;
  so.max_iterations = max_iterations;
  // Without priors the gauge is free: GaugeMode::Auto holds one pose and one translation
  // coordinate, so the frame and scale set by the first pair carry through. With them
  // (a set that is the working frame) nothing is held.
  // Stopped by the callback: the run ends, so nothing more is needed.
  if (jba::solve(p, so, on_iteration).termination == jba::Termination::Cancelled) return a;

  for (std::size_t k = 0; k < a.image_of.size(); ++k)
    std::copy(&p.poses[7 * k], &p.poses[7 * k] + 7, &m.cams.poses[7 * static_cast<std::size_t>(a.image_of[k])]);
  for (std::size_t j = 0; j < a.track_of.size(); ++j)
    std::copy(&p.points[3 * j], &p.points[3 * j] + 3, &m.X[3 * static_cast<std::size_t>(a.track_of[j])]);
  m.cams.intrinsics = p.intrinsics;  // refined or as they were

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
      m.obs_filtered[obs_of[k]] = 1;
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
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k) m.obs_in[k] = m.obs_filtered[k] = 0;
  }

  // The problem as it stands after the filter: kept observations only.
  jba::Problem kept_problem;
  kept_problem.model = p.model;
  kept_problem.intrinsics = p.intrinsics;
  kept_problem.intrinsics_free = p.intrinsics_free;
  kept_problem.poses = p.poses;
  kept_problem.image_intrinsics = p.image_intrinsics;
  kept_problem.glob_image = p.glob_image;
  kept_problem.glob_pose = p.glob_pose;
  kept_problem.glob_sqrt_info = p.glob_sqrt_info;
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

// A step of one model, kept until the model is chosen: only the chosen model's steps
// reach the caller.
struct Step {
  std::string stage;
  std::vector<std::int32_t> added;
  std::vector<double> poses;
  std::vector<char> registered;
  std::vector<double> points;
  std::vector<std::int32_t> point_track;
  double mean_error_px = 0.0;
  bool aligned = false;
};

void report(std::vector<Step>* steps, const char* stage, const std::vector<std::int32_t>& added, const Map& m,
            const Adjusted& a) {
  if (!steps) return;
  steps->push_back({stage, added, m.cams.poses, m.registered, a.problem.points, a.track_of, a.mean_error_px, m.aligned});
}

// Choosing the first pair: among pairs with status OK and enough inliers, the one with
// the most inliers whose median triangulation angle is at least 16 degrees (COLMAP's
// init_min_tri_angle); if there is none, the same with 8, 4, then 2 degrees.
// Only pairs whose two images are in no earlier model, as COLMAP starts a new model.
// With position priors (D150), pairs of two images with a prior come first, under every
// angle, so the model can move into the priors' frame early; then the others.
int choose_first_pair(const Capture& c, const Reconstruction& r, const MapperOptions& o,
                      const std::vector<char>& in_a_model) {
  const bool priors = uses_priors(c, o);
  for (const bool need_priors : {true, false}) {
    if (need_priors && !priors) continue;
    for (const double floor_deg : {16.0, 8.0, 4.0, 2.0}) {
      int best = -1;
      for (std::size_t i = 0; i < c.pairs.size(); ++i) {
        const jpo_pose_result& p = r.pair_results[i];
        const std::size_t a = static_cast<std::size_t>(c.pairs[i].a), b = static_cast<std::size_t>(c.pairs[i].b);
        if (in_a_model[a] || in_a_model[b]) continue;
        if (need_priors && !(c.has_prior(a) && c.has_prior(b))) continue;
        if (!r.pair_kept[i] || p.status != JPO_OK || p.n_inliers < o.min_first_pair_inliers ||
            p.median_tri_angle_deg < floor_deg)
          continue;
        if (best < 0 || p.n_inliers > r.pair_results[best].n_inliers) best = static_cast<int>(i);
      }
      if (best >= 0) return best;
    }
  }
  return -1;
}

// Moving a model into the frame of its position priors (D150): the similarity of the
// registered images' camera centres onto their priors. The fit is the triple's whose
// distances to the priors have the smallest median (least median: no threshold, and up
// to half the priors may be wrong), refitted on the centres within prior_inlier_sigmas
// of their prior. Distances are in the priors' sigmas, scaled up by the model's own
// error where that is the larger (precise priors next to a model of a few images): the
// median distance over that of a 3-D normal's length. False, and the map unchanged,
// while fewer than prior_min_images agree, or their priors spread across their line by
// less than prior_min_spread of those scaled sigmas, too little to set the rotation
// about it. Called again as the model grows, it refits an aligned model (near the
// identity when the two agree).
bool align_to_priors(Map& m, const MapperOptions& o) {
  const Capture& c = *m.capture;
  std::vector<int> images;
  std::vector<double> centres, priors;
  for (std::size_t i = 0; i < m.n_images(); ++i) {
    if (!m.registered[i] || !c.has_prior(i)) continue;
    double C[3];
    camera_centre(&m.cams.poses[7 * i], C);
    images.push_back(static_cast<int>(i));
    centres.insert(centres.end(), C, C + 3);
    priors.insert(priors.end(), &c.prior_position[3 * i], &c.prior_position[3 * i] + 3);
  }
  const std::size_t n = images.size();
  const std::size_t need = static_cast<std::size_t>(std::max(3, o.prior_min_images));
  if (n < need || spread_across_line(priors) < o.prior_min_spread * median_sigma_h(c, o, images)) return false;

  struct Sim {
    double R[9], t[3], s;
  };
  auto fit = [&](const std::vector<std::size_t>& use, Sim& sim) {
    std::vector<double> from, to;
    for (std::size_t q : use) {
      from.insert(from.end(), &centres[3 * q], &centres[3 * q] + 3);
      to.insert(to.end(), &priors[3 * q], &priors[3 * q] + 3);
    }
    fit_similarity(from, to, sim.R, sim.t, &sim.s);
    return std::isfinite(sim.s) && sim.s > 0.0;
  };
  auto distances = [&](const Sim& sim) {
    std::vector<double> d(n);
    for (std::size_t q = 0; q < n; ++q) {
      double Y[3];
      const double zero[3] = {0, 0, 0};
      apply(sim.R, zero, &centres[3 * q], Y);
      const double X[3] = {sim.s * Y[0] + sim.t[0], sim.s * Y[1] + sim.t[1], sim.s * Y[2] + sim.t[2]};
      d[q] = prior_distance(c, o, static_cast<std::size_t>(images[q]), X);
    }
    return d;
  };
  // Triples: all of them when few, else 200 drawn with the run's seed.
  Sim sim;
  double best_median = std::numeric_limits<double>::infinity();
  auto try_triple = [&](std::size_t x, std::size_t y, std::size_t z) {
    Sim f;
    if (!fit({x, y, z}, f)) return;
    const double med = median(distances(f));
    if (med < best_median) {
      best_median = med;
      sim = f;
    }
  };
  if (n <= 12) {
    for (std::size_t x = 0; x < n; ++x)
      for (std::size_t y = x + 1; y < n; ++y)
        for (std::size_t z = y + 1; z < n; ++z) try_triple(x, y, z);
  } else {
    Rng rng(o.seed + 0x5eedull);
    for (int k = 0; k < 200; ++k) {
      const std::size_t x = static_cast<std::size_t>(rng.next() % n);
      const std::size_t y = static_cast<std::size_t>(rng.next() % n);
      const std::size_t z = static_cast<std::size_t>(rng.next() % n);
      if (x != y && y != z && x != z) try_triple(x, y, z);
    }
  }
  if (!std::isfinite(best_median)) return false;
  // The inliers of a fit, and the scale of its distances: 1 (the priors' sigmas) or
  // more, from the median of the distances given (a 3-D normal's length has median
  // 1.538 sigma).
  constexpr double kNormal3Median = 1.5382;
  double scale = 1.0;
  auto inliers_of = [&](const Sim& f) {
    const std::vector<double> d = distances(f);
    scale = std::max(1.0, median(d) / kNormal3Median);
    std::vector<std::size_t> in;
    for (std::size_t q = 0; q < n; ++q)
      if (d[q] < o.prior_inlier_sigmas * scale) in.push_back(q);
    return in;
  };
  std::vector<std::size_t> in = inliers_of(sim);
  for (int round = 0; round < 3 && in.size() >= need; ++round) {
    if (!fit(in, sim)) return false;
    std::vector<std::size_t> next = inliers_of(sim);
    if (next == in) break;
    in = std::move(next);
  }
  if (in.size() < need || !fit(in, sim)) return false;
  // The scale over the inliers of the final fit.
  {
    const std::vector<double> d = distances(sim);
    std::vector<double> din;
    for (std::size_t q : in) din.push_back(d[q]);
    scale = std::max(1.0, median(din) / kNormal3Median);
  }
  std::vector<double> in_priors;
  std::vector<int> in_images;
  for (std::size_t q : in) {
    in_priors.insert(in_priors.end(), &priors[3 * q], &priors[3 * q] + 3);
    in_images.push_back(images[q]);
  }
  if (spread_across_line(in_priors) < o.prior_min_spread * scale * median_sigma_h(c, o, in_images)) return false;

  // X' = s R X + t. A pose (Rc, tc) becomes (Rc R^T, s tc - Rc R^T t): the same camera,
  // its coordinates scaled by s.
  double Rt[9];
  transpose3(sim.R, Rt);
  for (std::size_t i = 0; i < m.n_images(); ++i) {
    if (!m.registered[i]) continue;
    double* pose = &m.cams.poses[7 * i];
    double Rc[9], Rn[9], u[3];
    quat_to_rotmat(pose, Rc);
    mat3_mul(Rc, Rt, Rn);
    const double zero[3] = {0, 0, 0};
    apply(Rn, zero, sim.t, u);
    for (int k = 0; k < 3; ++k) pose[4 + k] = sim.s * pose[4 + k] - u[k];
    rotmat_to_quat(Rn, pose);
  }
  for (std::size_t t = 0; t < m.tracks.size(); ++t) {
    if (!m.has_point[t]) continue;
    double Y[3];
    const double zero[3] = {0, 0, 0};
    apply(sim.R, zero, &m.X[3 * t], Y);
    for (int k = 0; k < 3; ++k) m.X[3 * t + k] = sim.s * Y[k] + sim.t[k];
  }
  m.aligned = true;
  m.aligned_with = static_cast<int>(n);
  return true;
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

namespace {

// search_focal: the factor on the guessed focal lengths that best explains the pairs
// with the most matches, by poselib-gpu's MSAC score summed over them. 1 when there is
// nothing to search; -1 when poselib-gpu fails.
double search_focal(jpo_ctx* ctx, const Capture& c, const MapperOptions& o) {
  const std::size_t np = static_cast<std::size_t>(jba::num_params(c.model));
  const std::size_t blocks = c.intrinsics.size() / np;
  if (c.focal_unknown.size() != blocks ||
      std::none_of(c.focal_unknown.begin(), c.focal_unknown.end(), [](std::uint8_t u) { return u != 0; }))
    return 1.0;
  constexpr std::size_t kPairs = 50, kSteps = 61;
  constexpr double kLow = 0.25, kHigh = 4.0;
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < c.pairs.size(); ++i)
    if (c.pairs[i].ka.size() >= std::max<std::size_t>(30, o.min_pair_inliers) &&
        (c.focal_unknown[c.block(c.pairs[i].a)] || c.focal_unknown[c.block(c.pairs[i].b)]))
      order.push_back(i);
  if (order.empty()) return 1.0;
  std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
    return c.pairs[x].ka.size() != c.pairs[y].ka.size() ? c.pairs[x].ka.size() > c.pairs[y].ka.size() : x < y;
  });
  if (order.size() > kPairs) order.resize(kPairs);

  std::vector<double> factor(kSteps);
  for (std::size_t k = 0; k < kSteps; ++k)
    factor[k] = kLow * std::pow(kHigh / kLow, static_cast<double>(k) / static_cast<double>(kSteps - 1));
  std::vector<double> intr;
  for (std::size_t k = 0; k < kSteps; ++k)
    for (std::size_t b = 0; b < blocks; ++b)
      for (std::size_t j = 0; j < np; ++j) {
        const double v = c.intrinsics[np * b + j];
        const bool scale = c.focal_unknown[b] && jba::intrinsics_group(c.model, static_cast<int>(j)) == jba::kFocal;
        intr.push_back(scale ? v * factor[k] : v);
      }
  const std::vector<jpo_camera> cameras = to_jpo(c.model, intr);
  std::vector<std::uint32_t> offsets = {0}, cam_a, cam_b;
  std::vector<double> xy_a, xy_b;
  for (std::size_t k = 0; k < kSteps; ++k)
    for (std::size_t i : order) {
      const Pair& p = c.pairs[i];
      cam_a.push_back(static_cast<std::uint32_t>(k * blocks + static_cast<std::size_t>(c.block(p.a))));
      cam_b.push_back(static_cast<std::uint32_t>(k * blocks + static_cast<std::size_t>(c.block(p.b))));
      for (std::size_t m = 0; m < p.ka.size(); ++m) {
        xy_a.insert(xy_a.end(), &c.keypoints[p.a][2 * p.ka[m]], &c.keypoints[p.a][2 * p.ka[m]] + 2);
        xy_b.insert(xy_b.end(), &c.keypoints[p.b][2 * p.kb[m]], &c.keypoints[p.b][2 * p.kb[m]] + 2);
      }
      offsets.push_back(static_cast<std::uint32_t>(xy_a.size() / 2));
    }
  jpo_pairs pairs;
  std::memset(&pairs, 0, sizeof(pairs));
  pairs.n_pairs = static_cast<std::uint32_t>(cam_a.size());
  pairs.n_cameras = static_cast<std::uint32_t>(cameras.size());
  pairs.offsets = offsets.data();
  pairs.xy_a = xy_a.data();
  pairs.xy_b = xy_b.data();
  pairs.cameras = cameras.data();
  pairs.camera_a = cam_a.data();
  pairs.camera_b = cam_b.data();
  std::vector<jpo_pose_result> results(cam_a.size());
  jpo_ransac_options ransac;
  jpo_ransac_options_init(&ransac);
  ransac.max_error_px = o.pair_max_error_px;
  if (jpo_relative_pose(ctx, &pairs, &ransac, results.data(), nullptr) != 0) return -1.0;
  const double worst = o.pair_max_error_px * o.pair_max_error_px;  // MSAC's cap per match
  std::size_t best = 0;
  double best_score = 0.0;
  for (std::size_t k = 0; k < kSteps; ++k) {
    double score = 0.0;
    for (std::size_t q = 0; q < order.size(); ++q) {
      const jpo_pose_result& r = results[k * order.size() + q];
      const double n = static_cast<double>(c.pairs[order[q]].ka.size());
      score += r.status == JPO_OK || r.status == JPO_DEGENERATE ? r.score : n * worst;
    }
    if (k == 0 || score < best_score) {
      best = k;
      best_score = score;
    }
  }
  return factor[best];
}

}  // namespace

Reconstruction Mapper::run(const Capture& c, const StepCallback& on_step, const ProgressCallback& on_progress) {
  const MapperOptions& o = options_;
  const auto t_start = Clock::now();
  Reconstruction run;
  run.pose_backend = jpo_backend_name(ctx_);
  Map m;
  m.capture = &c;
  const std::size_t n_img = c.num_images();
  m.registered.assign(n_img, 0);
  run.error = check(c);
  if (!run.ok()) return run;

  // Live progress (D146). Once the callback returns false it is not called again, and
  // every stage checks `stopped` on its way out.
  bool stopped = false;
  int model_index = -1;
  auto progress = [&](const char* stage, const Snapshot* live, int iteration = -1, double cost = 0.0) {
    if (!on_progress || stopped) return !stopped;
    Progress p;
    p.stage = stage;
    p.model = model_index;
    p.registered = static_cast<int>(std::count(m.registered.begin(), m.registered.end(), 1));
    p.images = static_cast<int>(n_img);
    p.adjustments = run.adjustments;
    p.iteration = iteration;
    p.cost = cost;
    p.live = live;
    stopped = !on_progress(p);
    return !stopped;
  };
  jba::IterationCallback on_iteration;  // empty without a progress callback, as before
  if (on_progress)
    on_iteration = [&](const jba::IterationSummary& s) { return progress("adjust", nullptr, s.iteration, s.cost); };
  auto cancel = [&] {
    run.cancelled = true;
    run.error = "cancelled";
    run.time.total = seconds_since(t_start);
  };
  // The starting intrinsics: the capture's, with the searched focal lengths. Pair
  // verification uses them; PnP the map's, refined as it grows.
  std::vector<double> start = c.intrinsics;
  if (o.search_focal) {
    const auto ts = Clock::now();
    run.focal_factor = search_focal(ctx_, c, o);
    if (run.focal_factor < 0.0) {
      run.error = std::string("focal search: ") + jpo_last_error();
      return run;
    }
    const std::size_t np = static_cast<std::size_t>(jba::num_params(c.model));
    for (std::size_t b = 0; b < start.size() / np; ++b)
      for (std::size_t j = 0; j < np; ++j)
        if (!c.focal_unknown.empty() && c.focal_unknown[b] && jba::intrinsics_group(c.model, static_cast<int>(j)) == jba::kFocal)
          start[np * b + j] *= run.focal_factor;
    run.time.focal_search = seconds_since(ts);
    if (!progress("focal search", nullptr)) {
      cancel();
      return run;
    }
  }
  const std::vector<jpo_camera> cameras = to_jpo(c.model, start);

  // -- 1. Verify every pair: one call for the whole capture.
  auto t0 = Clock::now();
  std::vector<std::uint32_t>& offsets = run.pair_offsets;
  std::vector<double> xy_a, xy_b;
  std::vector<std::uint32_t> cam_a, cam_b;
  offsets.push_back(0);
  for (const Pair& p : c.pairs) {
    cam_a.push_back(static_cast<std::uint32_t>(c.block(static_cast<std::size_t>(p.a))));
    cam_b.push_back(static_cast<std::uint32_t>(c.block(static_cast<std::size_t>(p.b))));
    for (std::size_t k = 0; k < p.ka.size(); ++k) {
      xy_a.insert(xy_a.end(), &c.keypoints[p.a][2 * p.ka[k]], &c.keypoints[p.a][2 * p.ka[k]] + 2);
      xy_b.insert(xy_b.end(), &c.keypoints[p.b][2 * p.kb[k]], &c.keypoints[p.b][2 * p.kb[k]] + 2);
    }
    offsets.push_back(static_cast<std::uint32_t>(xy_a.size() / 2));
  }
  jpo_pairs pairs;
  std::memset(&pairs, 0, sizeof(pairs));
  pairs.n_pairs = static_cast<std::uint32_t>(c.pairs.size());
  pairs.n_cameras = static_cast<std::uint32_t>(cameras.size());
  pairs.offsets = offsets.data();
  pairs.xy_a = xy_a.data();
  pairs.xy_b = xy_b.data();
  pairs.cameras = cameras.data();
  pairs.camera_a = cam_a.data();
  pairs.camera_b = cam_b.data();
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
  if (!progress("verify", nullptr)) {
    cancel();
    return run;
  }

  // The matches that go on to track building: the inliers of every pair with enough of
  // them. A DEGENERATE pair (no observable baseline) is kept too: its inliers are good
  // matches and connect images that the pairs with a baseline may not.
  // With the images' sizes, a pair also needs its inliers spread over min_pair_coverage
  // of both images (D151).
  std::vector<std::int32_t> verified;
  run.pair_kept.assign(c.pairs.size(), 0);
  run.pair_coverage.assign(c.pairs.size(), std::numeric_limits<float>::quiet_NaN());
  for (std::size_t i = 0; i < c.pairs.size(); ++i) {
    const Pair& p = c.pairs[i];
    const jpo_pose_result& r = run.pair_results[i];
    if (r.status == JPO_OK) ++run.pairs_ok;
    else if (r.status == JPO_DEGENERATE) ++run.pairs_degenerate;
    else ++run.pairs_failed;
    if (r.status != JPO_OK && r.status != JPO_DEGENERATE) continue;
    run.pair_coverage[i] = pair_coverage(c, p, &run.pair_inliers[offsets[i]]);
    if (r.n_inliers < o.min_pair_inliers || r.n_inliers < o.min_pair_inlier_ratio * static_cast<double>(p.ka.size()))
      continue;
    if (o.min_pair_coverage > 0.0 && std::isfinite(run.pair_coverage[i]) && run.pair_coverage[i] < o.min_pair_coverage) {
      ++run.pairs_narrow;
      continue;
    }
    run.pair_kept[i] = 1;
    ++run.pairs_kept;
    for (std::size_t k = 0; k < p.ka.size(); ++k)
      if (run.pair_inliers[offsets[i] + k]) verified.insert(verified.end(), {p.a, p.ka[k], p.b, p.kb[k]});
  }

  // -- 2. Tracks: the connected components of the verified matches. No poses yet; a
  // component with two keypoints in one image is kept and sorted out at triangulation.
  t0 = Clock::now();
  m.keypoints.begin.push_back(0);
  for (std::size_t i = 0; i < n_img; ++i) {
    m.keypoints.xy.insert(m.keypoints.xy.end(), c.keypoints[i].begin(), c.keypoints[i].end());
    m.keypoints.begin.push_back(static_cast<std::int64_t>(m.keypoints.xy.size() / 2));
  }
  m.tracks = components(m.keypoints, verified, run.track_stats);
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
  m.track_images.assign(m.tracks.size(), 0);
  for (std::size_t t = 0; t < m.tracks.size(); ++t) {
    int n = 0;
    for (std::int64_t k = m.tracks.begin[t]; k < m.tracks.begin[t + 1]; ++k)
      n += k == m.tracks.begin[t] || m.tracks.image[k] != m.tracks.image[k - 1];
    m.track_images[t] = n;
  }
  m.has_point.assign(m.tracks.size(), 0);
  m.X.assign(3 * m.tracks.size(), 0.0);
  m.obs_in.assign(n_obs, 0);
  m.obs_dead.assign(n_obs, 0);
  m.obs_filtered.assign(n_obs, 0);
  m.split_tried.assign(m.tracks.size(), 0);
  m.cams.model = c.model;
  m.cams.intrinsics = start;
  m.cams.image_intrinsics = c.image_intrinsics;
  m.cams.poses.assign(7 * n_img, 0.0);
  for (std::size_t i = 0; i < n_img; ++i) m.cams.poses[7 * i] = 1.0;
  if (!progress("tracks", nullptr)) {
    cancel();
    return run;
  }

  const Map fresh = m;  // the map before any model: tracks built, nothing registered

  auto finish = [&](Map& mm, Adjusted&& a) {
    run.poses = mm.cams.poses;
    run.registered = mm.registered;
    run.problem = std::move(a.problem);
    run.image_of = std::move(a.image_of);
    run.obs_keypoint = std::move(a.obs_keypoint);
    run.point_track = std::move(a.track_of);
    run.all_tracks = mm.tracks;
    run.mean_error_px = a.mean_error_px;
    run.aligned = mm.aligned;
    std::vector<double> eh, ev;
    for (std::size_t i = 0; i < n_img; ++i) {
      if (!mm.registered[i] || !uses_priors(c, o) || !c.has_prior(i)) continue;
      ++run.prior_images;
      if (!mm.aligned) continue;
      double C[3];
      camera_centre(&mm.cams.poses[7 * i], C);
      const double* P = &c.prior_position[3 * i];
      eh.push_back(std::hypot(C[0] - P[0], C[1] - P[1]));
      ev.push_back(std::abs(C[2] - P[2]));
      run.prior_inliers += prior_distance(c, o, i, C) < o.prior_inlier_sigmas;
    }
    run.prior_error_horizontal = median(eh);
    run.prior_error_vertical = median(ev);
    // Per image (D151): the points of this model it sees, and its reprojection error.
    run.image_points_seen.assign(n_img, 0);
    std::vector<std::int32_t> seen;
    for (std::size_t i = 0; i < n_img; ++i) {
      seen.clear();
      for (std::int64_t k : mm.image_obs[i])
        if (!mm.obs_dead[k] && mm.has_point[mm.obs_track[k]]) seen.push_back(mm.obs_track[k]);
      std::sort(seen.begin(), seen.end());
      run.image_points_seen[i] = static_cast<std::int32_t>(std::unique(seen.begin(), seen.end()) - seen.begin());
    }
    run.image_error_px.assign(n_img, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> sum(n_img, 0.0);
    std::vector<std::size_t> count(n_img, 0);
    const jba::Problem& pr = run.problem;
    for (std::size_t k = 0; k < pr.obs_image.size(); ++k) {
      const int image = run.image_of[static_cast<std::size_t>(pr.obs_image[k])];
      const double e = reprojection_error(mm, image, &pr.obs_xy[2 * k], &pr.points[3 * static_cast<std::size_t>(pr.obs_point[k])]);
      if (e < 0.0) continue;
      sum[static_cast<std::size_t>(image)] += e;
      ++count[static_cast<std::size_t>(image)];
    }
    for (std::size_t i = 0; i < n_img; ++i)
      if (mm.registered[i] && count[i]) run.image_error_px[i] = sum[i] / static_cast<double>(count[i]);
    run.time.total = seconds_since(t_start);
  };
  auto timed_points = [&](bool split) {
    const auto t = Clock::now();
    update_points(m, o, split);
    run.time.triangulate += seconds_since(t);
  };
  auto timed_adjust = [&](int iterations) {
    const auto t = Clock::now();
    // Into the priors' frame as soon as enough registered images have one, and refitted
    // each time their number has grown by half (D150): an early alignment from a few
    // images is corrected by the later ones.
    if (uses_priors(c, o)) {
      int with = 0;
      for (std::size_t i = 0; i < n_img; ++i) with += m.registered[i] && c.has_prior(i);
      if (!m.aligned || with >= m.aligned_with + std::max(3, m.aligned_with / 2)) align_to_priors(m, o);
    }
    Adjusted a = adjust(m, o, iterations, on_iteration);
    ++run.adjustments;
    run.time.adjust += seconds_since(t);
    return a;
  };
  std::vector<std::int32_t> seen_tracks;
  // The points an image sees: distinct tracks with a point. An image can hold two
  // keypoints of one track (a conflicted track); both go to PnP, which keeps one, but
  // the point counts once in the ordering and in the inlier-ratio test.
  auto points_seen = [&](std::size_t image) {
    seen_tracks.clear();
    for (std::int64_t k : m.image_obs[image])
      if (!m.obs_dead[k] && m.has_point[m.obs_track[k]]) seen_tracks.push_back(m.obs_track[k]);
    std::sort(seen_tracks.begin(), seen_tracks.end());
    return static_cast<int>(std::unique(seen_tracks.begin(), seen_tracks.end()) - seen_tracks.begin());
  };

  // One model from the first pair `first`: steps 3 to 5. Leaves its map in `m` and its
  // last adjustment in `last`; false when poselib-gpu fails (run.error says why) or the
  // run is stopped (`stopped`).
  Adjusted last;
  // Per image, the inliers of its last rejected PnP in the model being grown (D151).
  std::vector<std::int32_t> rejected(n_img, -1);
  // A step done: kept for on_step, shown to on_progress. False when the run is stopped.
  auto step = [&](std::vector<Step>* steps, const char* stage, const std::vector<std::int32_t>& added) {
    if (stopped) return false;
    report(steps, stage, added, m, last);
    if (!on_progress) return true;
    Snapshot s;
    s.stage = stage;
    s.added = added;
    s.poses = &m.cams.poses;
    s.registered = &m.registered;
    s.points = last.problem.points;
    s.point_track = last.track_of;
    s.mean_error_px = last.mean_error_px;
    s.aligned = m.aligned;
    return progress(stage, &s);
  };
  auto grow = [&](int first, std::vector<Step>* steps) -> bool {
    // -- 3. First pair, first points, first adjustment. The first image is the origin;
    // the second is the relative pose as returned (camera a to camera b is then world to
    // camera b), whose unit translation fixes the scale of the map.
    const int fa = c.pairs[first].a, fb = c.pairs[first].b;
    rejected.assign(n_img, -1);
    const double identity[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::copy(identity, identity + 7, &m.cams.poses[7 * static_cast<std::size_t>(fa)]);
    std::copy(run.pair_results[first].pose, run.pair_results[first].pose + 7,
              &m.cams.poses[7 * static_cast<std::size_t>(fb)]);
    m.registered[fa] = m.registered[fb] = 1;
    timed_points(false);
    last = timed_adjust(o.local_iterations);
    if (!step(steps, "first pair", {fa, fb})) return false;

    // -- 4. Register the remaining images. Those that see the most triangulated points go
    // first; up to pnp_batch are resected in one call, all against the same map.
    std::vector<int> failed_with(n_img, -1);  // points an image saw when its resection last failed
    for (;;) {
      timed_points(true);
      std::vector<std::pair<int, int>> seen;  // (points seen, image)
      for (std::size_t i = 0; i < n_img; ++i) {
        if (m.registered[i]) continue;
        const int n = points_seen(i);
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

      const auto tp = Clock::now();
      std::vector<std::uint32_t> off = {0}, cam;
      std::vector<double> xy, xyz;
      for (std::size_t k = 0; k < n_cand; ++k) {
        const int image = seen[k].second;
        cam.push_back(static_cast<std::uint32_t>(c.block(static_cast<std::size_t>(image))));
        for (std::int64_t ob : m.image_obs[image]) {
          const std::int32_t t = m.obs_track[ob];
          if (m.obs_dead[ob] || !m.has_point[t]) continue;
          const double* kp = m.keypoints.at(image, m.tracks.keypoint[ob]);
          xy.insert(xy.end(), kp, kp + 2);
          xyz.insert(xyz.end(), &m.X[3 * t], &m.X[3 * t] + 3);
        }
        off.push_back(static_cast<std::uint32_t>(xy.size() / 2));
      }
      const std::vector<jpo_camera> map_cameras = to_jpo(m.cams.model, m.cams.intrinsics);
      jpo_resections res;
      std::memset(&res, 0, sizeof(res));
      res.n_images = static_cast<std::uint32_t>(n_cand);
      res.n_cameras = static_cast<std::uint32_t>(map_cameras.size());
      res.offsets = off.data();
      res.xy = xy.data();
      res.xyz = xyz.data();
      res.cameras = map_cameras.data();
      res.camera = cam.data();
      std::vector<jpo_pose_result> out(n_cand);
      jpo_ransac_options pnp_ransac;
      jpo_ransac_options_init(&pnp_ransac);
      pnp_ransac.max_error_px = o.pnp_max_error_px;
      if (jpo_absolute_pose(ctx_, &res, &pnp_ransac, out.data(), nullptr) != 0) {
        run.error = std::string("absolute pose: ") + jpo_last_error();
        return false;
      }
      ++run.pnp_calls;
      run.pnp_problems += static_cast<int>(n_cand);
      run.time.pnp += seconds_since(tp);

      std::vector<std::int32_t> added;
      for (std::size_t k = 0; k < n_cand; ++k) {
        const int image = seen[k].second;
        const int n = seen[k].first;  // distinct points, not correspondences
        const bool ok = out[k].status == JPO_OK && out[k].n_inliers >= o.min_pnp_inliers &&
                        out[k].n_inliers >= o.min_pnp_inlier_ratio * static_cast<double>(n);
        if (!ok) {
          failed_with[image] = seen[k].first;
          rejected[static_cast<std::size_t>(image)] = out[k].status == JPO_OK ? static_cast<std::int32_t>(out[k].n_inliers) : 0;
          ++run.pnp_rejected;
          continue;
        }
        std::copy(out[k].pose, out[k].pose + 7, &m.cams.poses[7 * static_cast<std::size_t>(image)]);
        m.registered[image] = 1;
        added.push_back(image);
      }
      if (added.empty()) continue;  // the rejected images wait until the map grows; others get their turn
      timed_points(false);
      last = timed_adjust(o.local_iterations);
      if (!step(steps, "register", added)) return false;
    }

    // -- 5. Final adjustment: complete the tracks, adjust, filter, and adjust what is left.
    // An aligned model is refitted to all its priors first (D150).
    if (m.aligned && uses_priors(c, o)) align_to_priors(m, o);
    timed_points(true);
    last = timed_adjust(o.final_iterations);
    if (stopped) return false;
    last = timed_adjust(o.final_iterations);
    return step(steps, "final", {});
  };

  // Models, as COLMAP makes them: when a model stops growing with images left over, a
  // new one starts from a first pair of images in no model yet (on ETH3D's office the
  // first model held 7 of 26 images, the second all 26). The model with the most
  // images is kept, then the one with the most points.
  std::vector<char> in_a_model(n_img, 0);
  Map best_map;
  Adjusted best_last;
  std::vector<std::int32_t> best_rejected;
  std::vector<Step> best_steps;
  std::size_t best_images = 0;
  int best_first = -1;
  for (int model = 0; model < o.max_models; ++model) {
    const int first = choose_first_pair(c, run, o, in_a_model);
    if (first < 0) break;
    if (model > 0) m = fresh;
    model_index = model;
    std::vector<Step> steps;
    if (!grow(first, on_step ? &steps : nullptr)) {
      if (stopped) {
        cancel();
        return run;
      }
      finish(m, std::move(last));
      return run;
    }
    ++run.models;
    std::size_t images = 0;
    for (std::size_t i = 0; i < n_img; ++i) {
      images += m.registered[i] != 0;
      if (m.registered[i]) in_a_model[i] = 1;
    }
    Reconstruction::Model summary;
    summary.registered = m.registered;
    summary.poses = m.cams.poses;
    summary.points = last.problem.points;
    summary.observations = last.problem.num_observations();
    summary.mean_error_px = last.mean_error_px;
    summary.first_pair = first;
    summary.aligned = m.aligned;
    run.all_models.push_back(std::move(summary));
    if (images > best_images || (images == best_images && last.problem.num_points() > best_last.problem.num_points())) {
      best_images = images;
      best_first = first;
      run.kept_model = static_cast<int>(run.all_models.size()) - 1;
      best_map = std::move(m);
      best_last = std::move(last);
      best_rejected = rejected;
      best_steps = std::move(steps);
    }
    if (best_images == n_img) break;
  }
  if (best_first < 0) {
    run.error = "no pair is fit to start from";
    run.time.total = seconds_since(t_start);
    return run;
  }
  run.first_a = c.pairs[best_first].a;
  run.first_b = c.pairs[best_first].b;
  run.first_inliers = run.pair_results[best_first].n_inliers;
  run.first_angle_deg = run.pair_results[best_first].median_tri_angle_deg;
  for (const Step& st : best_steps) {
    Snapshot s;
    s.stage = st.stage;
    s.added = st.added;
    s.poses = &st.poses;
    s.registered = &st.registered;
    s.points = st.points;
    s.point_track = st.point_track;
    s.mean_error_px = st.mean_error_px;
    s.aligned = st.aligned;
    on_step(s);
  }
  finish(best_map, std::move(best_last));
  run.image_pnp_inliers = std::move(best_rejected);
  return run;
}

}  // namespace demo
