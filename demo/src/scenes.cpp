#include "scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "geometry.h"
#include "jba/camera_model.h"

namespace demo {
namespace {

void cross(const double a[3], const double b[3], double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
double norm3(const double a[3]) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

// A camera at C looking at `target`, image y along the world's y (down), turned by a
// small random rotation so no two cameras are perfectly aligned.
void add_look_at(Scene& s, Rng& rng, const double C[3], const double target[3], double jitter_deg) {
  double z[3] = {target[0] - C[0], target[1] - C[1], target[2] - C[2]};
  const double zn = norm3(z);
  for (double& v : z) v /= zn;
  const double down[3] = {0.0, 1.0, 0.0};
  double x[3], y[3];
  cross(down, z, x);
  const double xn = norm3(x);
  for (double& v : x) v /= xn;
  cross(z, x, y);
  const double R0[9] = {x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]};
  double J[9], R[9];
  random_rotation(rng, jitter_deg, J);
  mat3_mul(J, R0, R);
  double pose[7];
  rotmat_to_quat(R, pose);
  for (int r = 0; r < 3; ++r) pose[4 + r] = -(R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2]);
  s.poses.insert(s.poses.end(), pose, pose + 7);
}

void set_camera(Scene& s, Rng& rng, jba::CameraModel model, double focal_over_width, double k = 0.0) {
  const double f = focal_over_width * s.width;
  const double cx = 0.5 * s.width + rng.uniform(-15.0, 15.0), cy = 0.5 * s.height + rng.uniform(-15.0, 15.0);
  s.capture.model = model;
  if (model == jba::CameraModel::Pinhole) s.capture.intrinsics = {f, f * (1.0 + rng.uniform(-0.003, 0.003)), cx, cy};
  else s.capture.intrinsics = {f, cx, cy, k};  // SimpleRadial
}

// Keypoints: every point that projects inside the image (and faces the camera, when
// the scene has normals) is detected with probability 0.85, at its projection plus noise.
void observe(Scene& s, Rng& rng, const SceneOptions& o, const std::vector<double>& normals, double max_view_deg) {
  const std::size_t n_img = s.poses.size() / 7, n_pts = s.points.size() / 3;
  s.capture.keypoints.assign(n_img, {});
  s.kp_point.assign(n_img, {});
  const double cos_max = std::cos(max_view_deg * kPi / 180.0);
  for (std::size_t i = 0; i < n_img; ++i) {
    const double* pose = &s.poses[7 * i];
    double R[9], C[3];
    quat_to_rotmat(pose, R);
    camera_centre(pose, C);
    for (std::size_t j = 0; j < n_pts; ++j) {
      const double* X = &s.points[3 * j];
      double Y[3], u, v;
      apply(R, pose + 4, X, Y);
      if (!(Y[2] > 0.1) || !jba::project(s.capture.model, s.capture.intrinsics.data(), Y, &u, &v)) continue;
      if (u < 0.0 || v < 0.0 || u > s.width - 1.0 || v > s.height - 1.0) continue;
      if (!normals.empty()) {
        const double* n = &normals[3 * j];
        const double d[3] = {C[0] - X[0], C[1] - X[1], C[2] - X[2]};
        if ((d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / norm3(d) < cos_max) continue;
      }
      if (rng.uniform() > 0.85) continue;
      s.capture.keypoints[i].push_back(u + o.noise_px * rng.normal());
      s.capture.keypoints[i].push_back(v + o.noise_px * rng.normal());
      s.kp_point[i].push_back(static_cast<std::int32_t>(j));
    }
  }
}

// Matches of every pair. Overlapping pairs: one match per shared point, then a share
// of them get a wrong keypoint in b (one this pair does not otherwise use, so matches
// stay one to one as a mutual nearest-neighbour matcher gives them). Pairs that share
// no point: up to `false_pairs` of them get 60 to 140 random matches.
void match(Scene& s, Rng& rng, const SceneOptions& o, int false_pairs) {
  struct Draft {
    Pair p;
    std::vector<std::uint8_t> correct;
    int shared = 0;
  };
  const int n_img = static_cast<int>(s.poses.size() / 7);
  const std::size_t n_pts = s.points.size() / 3;
  std::vector<std::vector<int>> kp_of(n_img, std::vector<int>(n_pts, -1));  // point -> keypoint
  for (int i = 0; i < n_img; ++i)
    for (std::size_t k = 0; k < s.kp_point[i].size(); ++k) kp_of[i][s.kp_point[i][k]] = static_cast<int>(k);
  std::vector<Draft> drafts;
  std::vector<std::pair<int, int>> disjoint;
  for (int a = 0; a < n_img; ++a)
    for (int b = a + 1; b < n_img; ++b) {
      Draft d;
      d.p.a = a;
      d.p.b = b;
      std::vector<char> used_b(s.kp_point[b].size(), 0);
      for (std::size_t k = 0; k < s.kp_point[a].size(); ++k) {
        const int kb = kp_of[b][s.kp_point[a][k]];
        if (kb < 0) continue;
        d.p.ka.push_back(static_cast<std::int32_t>(k));
        d.p.kb.push_back(kb);
        used_b[kb] = 1;
      }
      d.shared = static_cast<int>(d.p.ka.size());
      if (d.shared == 0) {
        disjoint.emplace_back(a, b);
        continue;
      }
      d.correct.assign(d.p.ka.size(), 1);
      std::vector<int> free_b;
      for (std::size_t k = 0; k < used_b.size(); ++k)
        if (!used_b[k]) free_b.push_back(static_cast<int>(k));
      std::vector<int> rotated;  // no unused keypoint left: these swap among themselves
      for (std::size_t k = 0; k < d.p.ka.size(); ++k) {
        if (!(rng.uniform() < o.wrong_fraction)) continue;
        if (!free_b.empty()) {
          const std::size_t pick = rng.next() % free_b.size();
          d.p.kb[k] = free_b[pick];
          free_b[pick] = free_b.back();
          free_b.pop_back();
          d.correct[k] = 0;
        } else {
          rotated.push_back(static_cast<int>(k));
        }
      }
      if (rotated.size() >= 2) {
        const std::int32_t head = d.p.kb[rotated[0]];
        for (std::size_t r = 0; r + 1 < rotated.size(); ++r) d.p.kb[rotated[r]] = d.p.kb[rotated[r + 1]];
        d.p.kb[rotated.back()] = head;
        for (int k : rotated) d.correct[k] = 0;
      }
      drafts.push_back(std::move(d));
    }
  for (std::size_t i = disjoint.size(); i > 1; --i) std::swap(disjoint[i - 1], disjoint[rng.next() % i]);
  for (std::size_t k = 0; k < disjoint.size() && k < static_cast<std::size_t>(std::max(false_pairs, 0)); ++k) {
    Draft d;
    d.p.a = disjoint[k].first;
    d.p.b = disjoint[k].second;
    const std::size_t na = s.kp_point[d.p.a].size(), nb = s.kp_point[d.p.b].size();
    const std::size_t want = std::min<std::size_t>(60 + rng.next() % 81, std::min(na, nb));
    std::vector<int> ia(na), ib(nb);
    for (std::size_t q = 0; q < na; ++q) ia[q] = static_cast<int>(q);
    for (std::size_t q = 0; q < nb; ++q) ib[q] = static_cast<int>(q);
    for (std::size_t q = 0; q < want; ++q) {
      std::swap(ia[q], ia[q + rng.next() % (na - q)]);
      std::swap(ib[q], ib[q + rng.next() % (nb - q)]);
      d.p.ka.push_back(ia[q]);
      d.p.kb.push_back(ib[q]);
    }
    d.correct.assign(want, 0);
    drafts.push_back(std::move(d));
  }
  std::sort(drafts.begin(), drafts.end(),
            [](const Draft& x, const Draft& y) { return x.p.a != y.p.a ? x.p.a < y.p.a : x.p.b < y.p.b; });
  for (Draft& d : drafts) {
    s.capture.pairs.push_back(std::move(d.p));
    s.correct.push_back(std::move(d.correct));
    s.shared.push_back(d.shared);
  }
}

// Orbit: 16 cameras on a 220 degree arc around a bumpy ball of 1500 surface points. A
// point is seen only from its own side, so cameras far apart on the arc share nothing.
Scene make_orbit(const SceneOptions& o) {
  Scene s;
  s.name = "orbit";
  Rng rng(101 + o.seed);
  set_camera(s, rng, jba::CameraModel::Pinhole, 0.9);
  std::vector<double> normals;
  for (int j = 0; j < 1500; ++j) {
    double n[3] = {rng.normal(), rng.normal(), rng.normal()};
    const double nn = norm3(n);
    for (double& v : n) v /= nn;
    const double r = 1.5 * (1.0 + 0.12 * std::sin(5.0 * n[0]) * std::cos(4.0 * n[1]) + 0.05 * rng.normal());
    for (int d = 0; d < 3; ++d) {
      s.points.push_back(r * n[d]);
      normals.push_back(n[d]);
    }
  }
  const int n_cam = 16;
  for (int i = 0; i < n_cam; ++i) {
    const double ang = (-110.0 + 220.0 * i / (n_cam - 1)) * kPi / 180.0;
    const double radius = 5.0 + rng.uniform(-0.4, 0.4);
    const double C[3] = {radius * std::sin(ang), rng.uniform(-0.6, 0.6), -radius * std::cos(ang)};
    const double target[3] = {rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)};
    add_look_at(s, rng, C, target, 3.0);
  }
  observe(s, rng, o, normals, 65.0);
  match(s, rng, o, o.false_pairs >= 0 ? o.false_pairs : 10);
  return s;
}

// Walk along a wall: 14 cameras a metre apart, moving sideways and a little toward and
// away from a wall six metres off, flat except for shallow relief. SIMPLE_RADIAL.
Scene make_wall(const SceneOptions& o) {
  Scene s;
  s.name = "wall";
  Rng rng(202 + o.seed);
  set_camera(s, rng, jba::CameraModel::SimpleRadial, 0.85, -0.06);
  for (int j = 0; j < 3200; ++j) {
    const double x = rng.uniform(-6.0, 19.0), y = rng.uniform(-3.5, 3.5);
    double z = 6.0 + 0.15 * std::sin(0.9 * x) * std::cos(1.3 * y) + 0.03 * rng.normal();
    if (rng.uniform() < 0.1) z -= rng.uniform(0.2, 1.0);  // something standing in front of the wall
    s.points.insert(s.points.end(), {x, y, z});
  }
  const int n_cam = 14;
  for (int i = 0; i < n_cam; ++i) {
    const double C[3] = {1.0 * i + rng.uniform(-0.1, 0.1), rng.uniform(-0.15, 0.15), 0.8 * std::sin(0.7 * i)};
    const double target[3] = {C[0] + rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3), 6.0};
    add_look_at(s, rng, C, target, 2.0);
  }
  observe(s, rng, o, {}, 180.0);
  match(s, rng, o, o.false_pairs >= 0 ? o.false_pairs : 10);
  return s;
}

// Grid: two rows of six cameras 20 cm apart, fanned out by 14 degrees a column, in
// front of a deep volume of points. A thirteenth image is taken from the place of
// image 3 after turning by 3 degrees: with image 3 it makes a pair with many good
// matches and no baseline, the pair a first-pair rule must not pick.
Scene make_grid(const SceneOptions& o) {
  Scene s;
  s.name = "grid";
  Rng rng(303 + o.seed);
  set_camera(s, rng, jba::CameraModel::Pinhole, 0.85);
  for (int j = 0; j < 3000; ++j) {
    const double x = rng.uniform(-10.0, 10.0), y = rng.uniform(-3.5, 3.5), z = rng.uniform(4.5, 8.5);
    s.points.insert(s.points.end(), {x, y, z});
  }
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 6; ++c) {
      const double C[3] = {0.2 * (c - 2.5) + rng.uniform(-0.02, 0.02), 0.3 * (r - 0.5) + rng.uniform(-0.02, 0.02),
                           rng.uniform(-0.03, 0.03)};
      const double yaw = 14.0 * (c - 2.5) * kPi / 180.0;
      const double target[3] = {C[0] + 6.0 * std::sin(yaw), C[1] + rng.uniform(-0.2, 0.2), C[2] + 6.0 * std::cos(yaw)};
      add_look_at(s, rng, C, target, 1.5);
    }
  double C[3];
  camera_centre(&s.poses[7 * 3], C);
  for (double& v : C) v += rng.uniform(-0.006, 0.006);
  const double yaw = (14.0 * 0.5 + 3.0) * kPi / 180.0;
  const double target[3] = {C[0] + 6.0 * std::sin(yaw), C[1] + 0.1, C[2] + 6.0 * std::cos(yaw)};
  add_look_at(s, rng, C, target, 1.0);
  observe(s, rng, o, {}, 180.0);
  match(s, rng, o, o.false_pairs >= 0 ? o.false_pairs : 8);
  return s;
}

}  // namespace

const std::vector<std::string>& scene_names() {
  static const std::vector<std::string> names = {"orbit", "wall", "grid"};
  return names;
}

CaptureFile to_capture_file(const Scene& s) {
  CaptureFile f;
  f.capture = s.capture;
  const std::size_t n_img = s.capture.num_images();
  for (std::size_t i = 0; i < n_img; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "%s_%03zu.jpg", s.name.c_str(), i);
    f.names.push_back(name);
  }
  f.image_size = {static_cast<std::int32_t>(s.width), static_cast<std::int32_t>(s.height)};
  f.truth_registered.assign(n_img, 1);
  f.truth_poses = s.poses;
  Reference& r = f.reference;
  r.registered.assign(n_img, 1);
  r.poses = s.poses;
  std::vector<int> views(s.points.size() / 3, 0), index(s.points.size() / 3, -1);
  for (const auto& kp : s.kp_point)
    for (int j : kp) ++views[static_cast<std::size_t>(j)];
  for (std::size_t j = 0; j < views.size(); ++j)
    if (views[j] >= 2) {
      index[j] = static_cast<int>(r.points.size() / 3);
      r.points.insert(r.points.end(), &s.points[3 * j], &s.points[3 * j] + 3);
    }
  for (std::size_t i = 0; i < n_img; ++i)
    for (std::size_t k = 0; k < s.kp_point[i].size(); ++k) {
      const int j = index[static_cast<std::size_t>(s.kp_point[i][k])];
      if (j < 0) continue;
      r.obs_image.push_back(static_cast<std::int32_t>(i));
      r.obs_point.push_back(j);
      r.obs_keypoint.push_back(static_cast<std::int32_t>(k));
    }
  return f;
}

Scene make_scene(const std::string& name, const SceneOptions& options) {
  Scene s;
  if (name == "orbit") s = make_orbit(options);
  else if (name == "wall") s = make_wall(options);
  else if (name == "grid") s = make_grid(options);
  else throw std::invalid_argument("unknown scene " + name + " (orbit, wall, grid)");
  s.noise_px = options.noise_px;
  return s;
}

}  // namespace demo
