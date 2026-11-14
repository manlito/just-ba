#include "scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include "jba/geometry.h"
#include "jba/initialize.h"
#include "synthetic.h"

namespace jba_test {
namespace {

constexpr double kPi = 3.141592653589793;
constexpr double kWidth = 640, kHeight = 480;

// World: x east, y down (gravity), z north; camera height at y = 0.
struct Face {
  double origin[3], u[3], v[3];  // points origin + a u + b v, a, b in [0, 1]
  double normal[3];              // side it is seen from
  int room = -1;                 // House: which room's side (-1 any)
};

double dot(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
void cross(const double* a, const double* b, double* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
void normalize(double* v) {
  const double n = std::sqrt(dot(v, v));
  for (int a = 0; a < 3; ++a) v[a] /= n;
}

// Inside faces of a box [x0, x1] x [y0, y1] x [z0, z1] (normals inward), or outside
// faces (normals outward).
void box_faces(const double* lo, const double* hi, bool inside, int room, bool skip_east, std::vector<Face>& out) {
  const double s = inside ? 1 : -1;
  auto add = [&](double ox, double oy, double oz, double ux, double uy, double uz, double vx, double vy, double vz,
                 double nx, double ny, double nz) {
    out.push_back({{ox, oy, oz}, {ux, uy, uz}, {vx, vy, vz}, {s * nx, s * ny, s * nz}, room});
  };
  const double dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
  add(lo[0], lo[1], lo[2], 0, dy, 0, 0, 0, dz, 1, 0, 0);       // west wall, x = x0
  if (!skip_east) add(hi[0], lo[1], lo[2], 0, dy, 0, 0, 0, dz, -1, 0, 0);  // east wall
  add(lo[0], lo[1], lo[2], dx, 0, 0, 0, dy, 0, 0, 0, 1);       // south wall, z = z0
  add(lo[0], lo[1], hi[2], dx, 0, 0, 0, dy, 0, 0, 0, -1);      // north wall
  if (inside) {
    add(lo[0], lo[1], lo[2], dx, 0, 0, 0, 0, dz, 0, 1, 0);     // ceiling, y = y0 (seen from below)
    add(lo[0], hi[1], lo[2], dx, 0, 0, 0, 0, dz, 0, -1, 0);    // floor
  }
}

double face_area(const Face& f) {
  double c[3];
  cross(f.u, f.v, c);
  return std::sqrt(dot(c, c));
}

// Camera at C looking along `look` (horizontal-ish), image y along world y (down).
void camera_pose(const double* C, const double* look_in, double pitch, double roll, double* pose) {
  double z[3] = {look_in[0], look_in[1] + std::sin(pitch), look_in[2]};
  normalize(z);
  double down[3] = {std::sin(roll) * z[2], 1, -std::sin(roll) * z[0]};
  double x[3];
  cross(down, z, x);
  normalize(x);
  double y[3];
  cross(z, x, y);
  // Rows of R are the camera axes in world coordinates.
  const double R[9] = {x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]};
  // Quaternion from rotation matrix.
  double q[4];
  const double tr = R[0] + R[4] + R[8];
  if (tr > 0) {
    const double s = std::sqrt(tr + 1) * 2;
    q[0] = s / 4;
    q[1] = (R[7] - R[5]) / s;
    q[2] = (R[2] - R[6]) / s;
    q[3] = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const double s = std::sqrt(1 + R[0] - R[4] - R[8]) * 2;
    q[0] = (R[7] - R[5]) / s;
    q[1] = s / 4;
    q[2] = (R[1] + R[3]) / s;
    q[3] = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const double s = std::sqrt(1 + R[4] - R[0] - R[8]) * 2;
    q[0] = (R[2] - R[6]) / s;
    q[1] = (R[1] + R[3]) / s;
    q[2] = s / 4;
    q[3] = (R[5] + R[7]) / s;
  } else {
    const double s = std::sqrt(1 + R[8] - R[0] - R[4]) * 2;
    q[0] = (R[3] - R[1]) / s;
    q[1] = (R[2] + R[6]) / s;
    q[2] = (R[5] + R[7]) / s;
    q[3] = s / 4;
  }
  jba::quat_normalize(q);
  std::copy_n(q, 4, pose);
  double RC[3];
  jba::quat_rotate(q, C, RC);
  for (int a = 0; a < 3; ++a) pose[4 + a] = -RC[a];
}

void centre_of(const double* pose, double* C) {
  const double c[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  jba::quat_rotate(c, pose + 4, C);
  for (int a = 0; a < 3; ++a) C[a] = -C[a];
}

// Segment from a to b crosses the plane x = x0 outside the door (House), or passes
// through the building box (Exterior).
bool blocked(SceneKind kind, const double* a, const double* b) {
  if (kind == SceneKind::House) {
    const double x0 = 3;
    if ((a[0] - x0) * (b[0] - x0) >= 0) return false;
    const double t = (x0 - a[0]) / (b[0] - a[0]);
    const double y = a[1] + t * (b[1] - a[1]), z = a[2] + t * (b[2] - a[2]);
    const bool in_door = std::abs(z) < 0.6 && y > -0.9;
    return !in_door;
  }
  if (kind == SceneKind::Exterior) {
    // Slab test against the box shrunk slightly, so wall points are not self-blocked.
    const double lo[3] = {-3.95, -2.95, -2.95}, hi[3] = {3.95, 1.45, 2.95};
    double t0 = 0, t1 = 1;
    for (int k = 0; k < 3; ++k) {
      const double d = b[k] - a[k];
      if (std::abs(d) < 1e-12) {
        if (a[k] < lo[k] || a[k] > hi[k]) return false;
        continue;
      }
      double ta = (lo[k] - a[k]) / d, tb = (hi[k] - a[k]) / d;
      if (ta > tb) std::swap(ta, tb);
      t0 = std::max(t0, ta);
      t1 = std::min(t1, tb);
      if (t0 > t1) return false;
    }
    return true;
  }
  return false;
}

}  // namespace

Scene make_scene(const SceneOptions& o) {
  std::mt19937_64 rng(o.seed);
  Uniform u01(0, 1), u(-1, 1);
  Normal n(0, 1);
  Scene sc;
  jba::Problem& truth = sc.truth;
  truth.model = o.model;
  truth.intrinsics.resize(static_cast<std::size_t>(jba::num_params(o.model)));
  typical_intrinsics(o.model, truth.intrinsics.data());

  // Surfaces.
  std::vector<Face> faces;
  if (o.kind == SceneKind::Room) {
    const double lo[3] = {-3, -1.2, -2.5}, hi[3] = {3, 1.5, 2.5};
    box_faces(lo, hi, true, -1, false, faces);
  } else if (o.kind == SceneKind::House) {
    const double loA[3] = {-3, -1.2, -2.5}, hiA[3] = {3, 1.5, 2.5};
    const double loB[3] = {3, -1.2, -2.5}, hiB[3] = {8, 1.5, 2.5};
    box_faces(loA, hiA, true, 0, true, faces);
    box_faces(loB, hiB, true, 1, false, faces);
    // The shared wall seen from room A (east wall of A) is the west wall of B's
    // list's mirror: add it explicitly for A; B's west wall is already in B.
    faces.push_back({{3, -1.2, -2.5}, {0, 2.7, 0}, {0, 0, 5}, {-1, 0, 0}, 0});
  } else {
    const double lo[3] = {-4, -3, -3}, hi[3] = {4, 1.5, 3};
    box_faces(lo, hi, false, -1, false, faces);
    // Ground around the building.
    faces.push_back({{-10, 1.5, -9}, {20, 0, 0}, {0, 0, 18}, {0, -1, 0}, -1});
  }
  std::vector<double> area(faces.size());
  for (std::size_t f = 0; f < faces.size(); ++f) area[f] = face_area(faces[f]);
  const double total_area = std::accumulate(area.begin(), area.end(), 0.0);
  std::vector<int> point_face;
  for (int j = 0; j < o.points; ++j) {
    double r = u01(rng) * total_area;
    std::size_t f = 0;
    while (f + 1 < faces.size() && r > area[f]) r -= area[f++];
    const Face& F = faces[f];
    const double a = u01(rng), b = u01(rng);
    double X[3];
    for (int c = 0; c < 3; ++c) X[c] = F.origin[c] + a * F.u[c] + b * F.v[c];
    if (o.kind == SceneKind::House && std::abs(X[0] - 3) < 1e-9 && std::abs(X[2]) < 0.6 && X[1] > -0.9) {
      --j;  // no points in the doorway
      continue;
    }
    if (o.kind == SceneKind::Exterior && f + 1 == faces.size() && X[0] > -4.2 && X[0] < 4.2 && X[2] > -3.2 &&
        X[2] < 3.2) {
      --j;  // no ground under the building
      continue;
    }
    truth.points.insert(truth.points.end(), X, X + 3);
    point_face.push_back(static_cast<int>(f));
  }

  // Camera path, in capture order.
  const int N = o.images;
  std::vector<double> path_poses;
  for (int s = 0; s < N; ++s) {
    double C[3], look[3];
    const double t = static_cast<double>(s) / N;
    if (o.kind == SceneKind::Room) {
      const double a = 2 * kPi * t;
      C[0] = 1.3 * std::cos(a);
      C[1] = 0.05 * std::sin(7 * a);
      C[2] = 1.0 * std::sin(a);
      const double yaw = a + 0.6 * std::sin(3 * a);  // look outward, swinging
      look[0] = std::cos(yaw);
      look[1] = 0;
      look[2] = std::sin(yaw);
    } else if (o.kind == SceneKind::House) {
      // Loop in A, through the door, loop in B.
      const int half = N / 2;
      if (s < half) {
        const double a = 2 * kPi * s / half;
        C[0] = 1.2 * std::cos(a);
        C[2] = 1.0 * std::sin(a);
        const double yaw = a + 0.5 * std::sin(2 * a);
        look[0] = std::cos(yaw);
        look[2] = std::sin(yaw);
      } else {
        const double a = 2 * kPi * (s - half) / (N - half) + kPi;
        C[0] = 5.5 + 1.0 * std::cos(a);
        C[2] = 1.0 * std::sin(a);
        const double yaw = a + 0.5 * std::sin(2 * a);
        look[0] = std::cos(yaw);
        look[2] = std::sin(yaw);
      }
      C[1] = 0.03 * std::sin(11.0 * s);
      look[1] = 0;
    } else {
      const double a = 2 * kPi * t;
      C[0] = 8.5 * std::cos(a);
      C[1] = -0.2 + 0.05 * std::sin(5 * a);
      C[2] = 7.0 * std::sin(a);
      double to_centre[3] = {-C[0], 0, -C[2]};
      normalize(to_centre);
      const double yaw = std::atan2(to_centre[2], to_centre[0]) + 0.25 * std::sin(4 * a);
      look[0] = std::cos(yaw);
      look[1] = 0;
      look[2] = std::sin(yaw);
    }
    double pose[7];
    camera_pose(C, look, 0.08 * std::sin(5.0 * s) + (o.kind == SceneKind::Exterior ? -0.12 : 0), 0.03 * std::sin(3.0 * s), pose);
    path_poses.insert(path_poses.end(), pose, pose + 7);
  }

  // Frame order: capture order, or shuffled.
  sc.capture_order.resize(static_cast<std::size_t>(N));
  std::iota(sc.capture_order.begin(), sc.capture_order.end(), 0);
  if (o.shuffle_frames) {
    for (int s = N - 1; s > 0; --s) std::swap(sc.capture_order[static_cast<std::size_t>(s)], sc.capture_order[rng() % static_cast<std::uint64_t>(s + 1)]);
  }
  truth.poses.assign(7 * static_cast<std::size_t>(N), 0);
  for (int s = 0; s < N; ++s)
    std::copy_n(&path_poses[7 * static_cast<std::size_t>(s)], 7, &truth.poses[7 * static_cast<std::size_t>(sc.capture_order[static_cast<std::size_t>(s)])]);

  // Visibility and keypoints.
  const std::size_t M = truth.num_points();
  std::vector<std::vector<std::int32_t>> kp_point(static_cast<std::size_t>(N));  // point of each keypoint
  std::vector<std::vector<double>> kp_xy(static_cast<std::size_t>(N));
  std::vector<std::vector<std::pair<int, int>>> seen_by(M);  // (image, keypoint)
  for (int i = 0; i < N; ++i) {
    const double* pose = &truth.poses[7 * static_cast<std::size_t>(i)];
    double C[3];
    centre_of(pose, C);
    std::vector<std::pair<std::int32_t, std::array<double, 2>>> vis;
    for (std::size_t j = 0; j < M; ++j) {
      const double* X = &truth.points[3 * j];
      const Face& F = faces[static_cast<std::size_t>(point_face[j])];
      const double toC[3] = {C[0] - X[0], C[1] - X[1], C[2] - X[2]};
      if (dot(F.normal, toC) <= 0.05 * std::sqrt(dot(toC, toC))) continue;
      if (dot(toC, toC) > 14 * 14) continue;
      if (blocked(o.kind, C, X)) continue;
      if (o.kind == SceneKind::House && F.room >= 0 && ((F.room == 0) != (C[0] < 3))) continue;
      double pc[3];
      jba::quat_rotate(pose, X, pc);
      for (int c = 0; c < 3; ++c) pc[c] += pose[4 + c];
      if (pc[2] < 0.3) continue;
      double px, py;
      if (!jba::project(o.model, truth.intrinsics.data(), pc, &px, &py)) continue;
      if (px < 0 || px >= kWidth || py < 0 || py >= kHeight) continue;
      vis.push_back({static_cast<std::int32_t>(j), {px + o.pixel_noise * n(rng), py + o.pixel_noise * n(rng)}});
    }
    // Keypoint order is not point order.
    for (std::size_t q = vis.size(); q > 1; --q) std::swap(vis[q - 1], vis[rng() % q]);
    for (std::size_t q = 0; q < vis.size(); ++q) {
      kp_point[static_cast<std::size_t>(i)].push_back(vis[q].first);
      kp_xy[static_cast<std::size_t>(i)].insert(kp_xy[static_cast<std::size_t>(i)].end(), vis[q].second.begin(), vis[q].second.end());
      seen_by[static_cast<std::size_t>(vis[q].first)].push_back({i, static_cast<int>(q)});
    }
  }
  jba::Keypoints& K = sc.input.keypoints;
  K.begin.assign(1, 0);
  for (int i = 0; i < N; ++i) {
    K.xy.insert(K.xy.end(), kp_xy[static_cast<std::size_t>(i)].begin(), kp_xy[static_cast<std::size_t>(i)].end());
    K.begin.push_back(static_cast<std::int64_t>(K.xy.size() / 2));
  }

  // Matches between image pairs that share enough points.
  std::map<std::pair<int, int>, std::vector<std::array<int, 2>>> pair_kps;
  for (std::size_t j = 0; j < M; ++j) {
    auto& s = seen_by[j];
    for (std::size_t a = 0; a < s.size(); ++a)
      for (std::size_t b = 0; b < s.size(); ++b) {
        if (s[a].first >= s[b].first) continue;
        pair_kps[{s[a].first, s[b].first}].push_back({s[a].second, s[b].second});
      }
  }
  for (const auto& [pr, list] : pair_kps) {
    if (static_cast<int>(list.size()) < o.match_min_shared) continue;
    const auto nb = static_cast<std::uint64_t>(kp_point[static_cast<std::size_t>(pr.second)].size());
    for (const auto& m : list) {
      int kb = m[1];
      if (u01(rng) < o.wrong_matches) kb = static_cast<int>(rng() % nb);
      sc.input.matches.insert(sc.input.matches.end(), {pr.first, m[0], pr.second, kb});
    }
  }

  // Tracker trajectory: drift as a transform D (applied to camera-to-world poses),
  // updated about the current tracked position.
  sc.tracker_poses.assign(truth.poses.size(), 0);
  double Dq[4] = {1, 0, 0, 0}, Dt[3] = {0, 0, 0};
  std::vector<int> jump_before(static_cast<std::size_t>(N), 0);  // jumps between step s-1 and s
  for (int s = 0; s < N; ++s) {
    const double* T = &path_poses[7 * static_cast<std::size_t>(s)];
    if (s > 0) {
      double C0[3], C1[3];
      centre_of(&path_poses[7 * static_cast<std::size_t>(s - 1)], C0);
      centre_of(T, C1);
      const double dist = std::sqrt((C1[0] - C0[0]) * (C1[0] - C0[0]) + (C1[1] - C0[1]) * (C1[1] - C0[1]) + (C1[2] - C0[2]) * (C1[2] - C0[2]));
      const double sq = std::sqrt(dist);
      double rot[3] = {o.drift_tilt_per_m * sq * n(rng), o.drift_heading_per_m * sq * n(rng), o.drift_tilt_per_m * sq * n(rng)};
      double dt[3] = {o.drift_trans_per_m * sq * n(rng), o.drift_trans_per_m * sq * n(rng), o.drift_trans_per_m * sq * n(rng)};
      jump_before[static_cast<std::size_t>(s)] = jump_before[static_cast<std::size_t>(s - 1)];
      if (u01(rng) < o.jump_probability) {
        rot[1] += (u01(rng) < 0.5 ? -1 : 1) * (0.05 + 0.15 * u01(rng));
        for (double& v : dt) v += 0.6 * u(rng);
        jump_before[static_cast<std::size_t>(s)] += 1;
      }
      // Pivot: the tracked position of the previous frame.
      double P[3], RC0[3];
      jba::quat_rotate(Dq, C0, RC0);
      for (int c = 0; c < 3; ++c) P[c] = RC0[c] + Dt[c];
      double dq[4], nq[4];
      jba::quat_exp(rot, dq);
      jba::quat_multiply(dq, Dq, nq);
      jba::quat_normalize(nq);
      double rel[3], Rrel[3];
      for (int c = 0; c < 3; ++c) rel[c] = Dt[c] - P[c];
      jba::quat_rotate(dq, rel, Rrel);
      for (int c = 0; c < 3; ++c) Dt[c] = Rrel[c] + P[c] + dt[c];
      std::copy_n(nq, 4, Dq);
    }
    // Tracked camera-to-world: R' = R_D R_c2w, C' = R_D C + t_D.
    double C[3], RC[3];
    centre_of(T, C);
    jba::quat_rotate(Dq, C, RC);
    double Ct[3] = {RC[0] + Dt[0], RC[1] + Dt[1], RC[2] + Dt[2]};
    const double Dc[4] = {Dq[0], -Dq[1], -Dq[2], -Dq[3]};
    double q[4];
    jba::quat_multiply(T, Dc, q);  // R_w2c' = R_w2c R_D^T
    jba::quat_normalize(q);
    double* out = &sc.tracker_poses[7 * static_cast<std::size_t>(sc.capture_order[static_cast<std::size_t>(s)])];
    std::copy_n(q, 4, out);
    double Rq[3];
    jba::quat_rotate(q, Ct, Rq);
    for (int c = 0; c < 3; ++c) out[4 + c] = -Rq[c];
  }

  // Scene for refine: camera, tracker poses as starting poses, priors.
  jba::Problem& scene = sc.input.scene;
  scene.model = o.model;
  scene.intrinsics = truth.intrinsics;
  scene.poses = sc.tracker_poses;
  if (o.relative_priors) {
    std::vector<std::int32_t> pairs;
    std::vector<char> spans_jump;
    for (int s = 0; s < N; ++s)
      for (int d = 1; d <= o.prior_neighbours && s + d < N; ++d) {
        const bool jump = jump_before[static_cast<std::size_t>(s + d)] != jump_before[static_cast<std::size_t>(s)];
        if (jump && !(u01(rng) < o.bad_priors_kept)) continue;  // the caller's cleanup removed it
        pairs.insert(pairs.end(), {sc.capture_order[static_cast<std::size_t>(s)], sc.capture_order[static_cast<std::size_t>(s + d)]});
        spans_jump.push_back(jump);
      }
    jba::add_relative_priors_from_poses(scene, sc.tracker_poses, pairs);
    for (std::size_t k = 0; k < scene.num_relative_priors(); ++k) {
      const double step[6] = {o.prior_noise_rot * n(rng), o.prior_noise_rot * n(rng), o.prior_noise_rot * n(rng),
                              o.prior_noise_trans * n(rng), o.prior_noise_trans * n(rng), o.prior_noise_trans * n(rng)};
      double moved[7];
      jba::pose_retract(&scene.rel_pose[7 * k], step, moved);
      std::copy_n(moved, 7, &scene.rel_pose[7 * k]);
      bool bad = spans_jump[k];
      if (u01(rng) < o.bad_prior_fraction) {
        // A fixed-size error in a random direction.
        double d[6], nr = 0, nt = 0;
        for (double& v : d) v = n(rng);
        for (int c = 0; c < 3; ++c) nr += d[c] * d[c], nt += d[3 + c] * d[3 + c];
        const double ar = 0.05 + 0.15 * u01(rng), at = 0.1 + 0.3 * u01(rng);
        double big[6];
        for (int c = 0; c < 3; ++c) big[c] = ar * d[c] / std::sqrt(nr), big[3 + c] = at * d[3 + c] / std::sqrt(nt);
        jba::pose_retract(&scene.rel_pose[7 * k], big, moved);
        std::copy_n(moved, 7, &scene.rel_pose[7 * k]);
        bad = true;
      }
      if (bad) sc.bad_relative.push_back(static_cast<std::int32_t>(k));
    }
  }
  if (o.global_priors) {
    double S[8] = {1, 0, 0, 0, 0, 0, 0, 1};
    if (o.global_relation != jba::FrameRelation::Identity) {
      random_rotation(rng, 2.5, S);
      for (int c = 0; c < 3; ++c) S[4 + c] = 4 * u(rng);
      if (o.global_relation == jba::FrameRelation::Similarity) S[7] = 0.2 + 2 * u01(rng);
    }
    sc.truth_global_transform.assign(S, S + 8);
    for (int i = 0; i < N; ++i) {
      const double* T = &truth.poses[7 * static_cast<std::size_t>(i)];
      double g[7], Rt[3];
      jba::quat_multiply(T, S, g);
      jba::quat_normalize(g);
      jba::quat_rotate(T, S + 4, Rt);
      for (int c = 0; c < 3; ++c) g[4 + c] = (Rt[c] + T[4 + c]) / S[7];
      const bool outlier = u01(rng) < o.global_outliers;
      const double sr = outlier ? 0.3 : o.global_noise_rot, sp = outlier ? 1.0 : o.global_noise_pos;
      // Noise on rotation, and on the centre in the source's units.
      double C[3];
      centre_of(g, C);
      double rot[3] = {sr * n(rng), sr * n(rng), sr * n(rng)}, dq[4], q[4];
      jba::quat_exp(rot, dq);
      jba::quat_multiply(dq, g, q);
      jba::quat_normalize(q);
      for (double& v : C) v += sp * n(rng);
      std::copy_n(q, 4, g);
      double RC[3];
      jba::quat_rotate(q, C, RC);
      for (int c = 0; c < 3; ++c) g[4 + c] = -RC[c];
      scene.glob_image.push_back(i);
      scene.glob_pose.insert(scene.glob_pose.end(), g, g + 7);
      if (outlier) sc.bad_global.push_back(i);
    }
    scene.set_relation = {o.global_relation};
  }
  return sc;
}

PoseError pose_error(const std::vector<double>& est, const std::vector<double>& truth, bool with_scale,
                     const std::vector<char>& use) {
  const std::size_t n = truth.size() / 7;
  std::vector<double> from, to, rots;
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < n; ++i) {
    if (!use.empty() && !use[i]) continue;
    double Ce[3], Ct[3];
    centre_of(&est[7 * i], Ce);
    centre_of(&truth[7 * i], Ct);
    from.insert(from.end(), Ce, Ce + 3);
    to.insert(to.end(), Ct, Ct + 3);
    const double* qt = &truth[7 * i];
    const double qtc[4] = {qt[0], -qt[1], -qt[2], -qt[3]};
    double r[4];
    jba::quat_multiply(qtc, &est[7 * i], r);
    rots.insert(rots.end(), r, r + 4);
    idx.push_back(i);
  }
  PoseError e;
  if (idx.empty()) return e;
  // Alignment rotation from the camera rotations (precise), then scale and
  // translation from the centres. A rotation from the centres alone would be poor
  // for a small loop: 1 cm errors over 1 m tilt it by about half a degree.
  double G[8];
  jba::fit_similarity(from, to, &rots, with_scale, G);
  e.scale = G[7];
  std::vector<double> dc, dr;
  for (std::size_t q = 0; q < idx.size(); ++q) {
    const std::size_t i = idx[q];
    double Rc[3];
    jba::quat_rotate(G, &from[3 * q], Rc);
    double d2 = 0;
    for (int c = 0; c < 3; ++c) {
      const double d = G[7] * Rc[c] + G[4 + c] - to[3 * q + static_cast<std::size_t>(c)];
      d2 += d * d;
    }
    dc.push_back(std::sqrt(d2));
    // Rotation: R_e R_G^T against R_t.
    const double Gc[4] = {G[0], -G[1], -G[2], -G[3]};
    double qa[4];
    jba::quat_multiply(&est[7 * i], Gc, qa);
    const double* qt = &truth[7 * i];
    const double d = std::min(1.0, std::abs(qa[0] * qt[0] + qa[1] * qt[1] + qa[2] * qt[2] + qa[3] * qt[3]));
    dr.push_back(2 * std::acos(d));
  }
  auto med = [](std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  e.median_centre = med(dc);
  e.max_centre = *std::max_element(dc.begin(), dc.end());
  e.median_rotation = med(dr);
  e.max_rotation = *std::max_element(dr.begin(), dr.end());
  return e;
}

}  // namespace jba_test
