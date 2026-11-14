#include "jba/tracks.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "jba/geometry.h"

namespace jba {
namespace {

struct UnionFind {
  std::vector<std::int64_t> parent;
  explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), std::int64_t(0)); }
  std::int64_t find(std::int64_t a) {
    while (parent[static_cast<std::size_t>(a)] != a) {
      parent[static_cast<std::size_t>(a)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
      a = parent[static_cast<std::size_t>(a)];
    }
    return a;
  }
  void join(std::int64_t a, std::int64_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (a < b) parent[static_cast<std::size_t>(b)] = a;
    else parent[static_cast<std::size_t>(a)] = b;
  }
};

struct Obs {
  std::int32_t image, keypoint;
};

void emit(Tracks& t, std::vector<Obs> obs, int max_views) {
  std::sort(obs.begin(), obs.end(), [](const Obs& a, const Obs& b) {
    return a.image != b.image ? a.image < b.image : a.keypoint < b.keypoint;
  });
  if (static_cast<int>(obs.size()) > max_views) obs.resize(static_cast<std::size_t>(max_views));
  for (const Obs& o : obs) {
    t.image.push_back(o.image);
    t.keypoint.push_back(o.keypoint);
  }
  t.begin.push_back(static_cast<std::int64_t>(t.image.size()));
}

// Two rays agree: their closest points lie in front of both cameras, and ray b is
// within max_err of the epipolar plane of ray a. A near-zero baseline (a pure
// rotation) only gets the direction test.
bool epipolar_ok(const Ray& ra, const Ray& rb, double max_err) {
  const double base[3] = {rb.C[0] - ra.C[0], rb.C[1] - ra.C[1], rb.C[2] - ra.C[2]};
  const double bl = std::sqrt(base[0] * base[0] + base[1] * base[1] + base[2] * base[2]);
  const double dab = ra.d[0] * rb.d[0] + ra.d[1] * rb.d[1] + ra.d[2] * rb.d[2];
  if (bl < 1e-9) return dab > std::cos(max_err);
  double n[3] = {base[1] * ra.d[2] - base[2] * ra.d[1], base[2] * ra.d[0] - base[0] * ra.d[2],
                 base[0] * ra.d[1] - base[1] * ra.d[0]};
  const double nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (nn > 1e-12 * bl) {
    const double s = std::abs(n[0] * rb.d[0] + n[1] * rb.d[1] + n[2] * rb.d[2]) / nn;
    if (s > std::sin(max_err)) return false;
  }
  // Closest points: C_a + s d_a and C_b + t d_b with s, t > 0.
  const double e = ra.d[0] * base[0] + ra.d[1] * base[1] + ra.d[2] * base[2];
  const double f = rb.d[0] * base[0] + rb.d[1] * base[1] + rb.d[2] * base[2];
  const double den = 1 - dab * dab;
  if (den < 1e-12) return true;  // parallel rays: a point at infinity, harmless
  const double sa = (e - dab * f) / den, tb = (dab * e - f) / den;
  return sa > 0 && tb > 0;
}

}  // namespace

bool keypoint_ray(const Cameras& c, std::int32_t image, const double* xy, Ray& ray) {
  const int np = num_params(c.model);
  const std::size_t block = c.image_intrinsics.empty() ? 0 : static_cast<std::size_t>(c.image_intrinsics[static_cast<std::size_t>(image)]);
  double b[3];
  if (!unproject(c.model, &c.intrinsics[block * static_cast<std::size_t>(np)], xy[0], xy[1], b)) return false;
  const double* pose = &c.poses[7 * static_cast<std::size_t>(image)];
  const double conj[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  quat_rotate(conj, pose + 4, ray.C);
  for (double& v : ray.C) v = -v;
  quat_rotate(conj, b, ray.d);
  return true;
}

Tracks build_tracks(const Keypoints& kp, const std::vector<std::int32_t>& matches, const TrackOptions& o,
                    const Cameras* cameras, TrackStats* stats) {
  if (matches.size() % 4 != 0) throw std::invalid_argument("matches must hold 4 values per match");
  const std::size_t n_img = kp.num_images();
  const auto total = static_cast<std::size_t>(kp.begin.empty() ? 0 : kp.begin.back());
  auto id = [&](std::int32_t image, std::int32_t k) -> std::int64_t {
    if (image < 0 || static_cast<std::size_t>(image) >= n_img) throw std::invalid_argument("match refers to a missing image");
    const auto b = kp.begin[static_cast<std::size_t>(image)], e = kp.begin[static_cast<std::size_t>(image) + 1];
    if (k < 0 || b + k >= e) throw std::invalid_argument("match refers to a missing keypoint");
    return b + k;
  };
  TrackStats st;
  std::vector<std::int32_t> image_of(total);
  for (std::size_t i = 0; i < n_img; ++i)
    for (auto g = kp.begin[i]; g < kp.begin[i + 1]; ++g) image_of[static_cast<std::size_t>(g)] = static_cast<std::int32_t>(i);

  const bool geometric = cameras != nullptr && o.conflict_policy == ConflictPolicy::Repair;
  const double max_err = o.triangulation.max_angle_error_deg * 3.141592653589793 / 180;

  // Rays of every matched keypoint, once, when the poses are known.
  std::vector<Ray> ray_of;
  std::vector<char> ray_ok;
  if (geometric) {
    ray_of.resize(total);
    ray_ok.assign(total, 0);
    for (std::size_t m = 0; m < matches.size(); m += 4) {
      for (int side = 0; side < 2; ++side) {
        const auto g = static_cast<std::size_t>(id(matches[m + 2 * static_cast<std::size_t>(side)], matches[m + 2 * static_cast<std::size_t>(side) + 1]));
        if (ray_ok[g]) continue;
        ray_ok[g] = keypoint_ray(*cameras, image_of[g], &kp.xy[2 * g], ray_of[g]) ? 1 : 2;
      }
    }
  }

  UnionFind uf(total);
  std::vector<char> used(total, 0);
  for (std::size_t m = 0; m < matches.size(); m += 4) {
    const auto a = id(matches[m], matches[m + 1]), b = id(matches[m + 2], matches[m + 3]);
    if (matches[m] == matches[m + 2]) continue;  // a match inside one image is meaningless
    // With poses: the two rays must meet in front of both cameras, within the
    // inlier angle of the epipolar plane. Removes most wrong matches before they
    // can fuse tracks into one large component.
    if (geometric) {
      const auto ga = static_cast<std::size_t>(a), gb = static_cast<std::size_t>(b);
      if (ray_ok[ga] != 1 || ray_ok[gb] != 1) {
        ++st.rejected_matches;
        continue;
      }
      if (!epipolar_ok(ray_of[ga], ray_of[gb], max_err)) {
        ++st.rejected_matches;
        continue;
      }
    }
    uf.join(a, b);
    used[static_cast<std::size_t>(a)] = used[static_cast<std::size_t>(b)] = 1;
  }

  // Components in order of their smallest keypoint id.
  std::vector<std::vector<Obs>> comps;
  std::vector<std::int64_t> comp_of(total, -1);
  for (std::size_t g = 0; g < total; ++g) {
    if (!used[g]) continue;
    const auto root = static_cast<std::size_t>(uf.find(static_cast<std::int64_t>(g)));
    if (comp_of[root] < 0) {
      comp_of[root] = static_cast<std::int64_t>(comps.size());
      comps.emplace_back();
    }
    comps[static_cast<std::size_t>(comp_of[root])].push_back(
        {image_of[g], static_cast<std::int32_t>(static_cast<std::int64_t>(g) - kp.begin[static_cast<std::size_t>(image_of[g])])});
  }

  st.components = comps.size();
  Tracks out;
  TriangulationOptions tri = o.triangulation;
  tri.min_views = std::max(2, o.min_views);

  for (auto& comp : comps) {
    std::sort(comp.begin(), comp.end(), [](const Obs& a, const Obs& b) {
      return a.image != b.image ? a.image < b.image : a.keypoint < b.keypoint;
    });
    bool clean = true;
    for (std::size_t q = 1; q < comp.size(); ++q) clean = clean && comp[q].image != comp[q - 1].image;
    if (clean) ++st.clean;
    else ++st.conflicted;

    if (clean && !(geometric && o.check_clean)) {
      if (static_cast<int>(comp.size()) >= o.min_views) emit(out, comp, o.max_views);
      else st.dropped_observations += comp.size();
      continue;
    }
    if (!geometric) {
      st.dropped_observations += comp.size();
      continue;
    }

    // Stage 2: repeated robust triangulation; each round's inliers become a track.
    std::vector<Obs> rest;
    std::vector<Ray> rays;
    std::vector<std::int32_t> keys;
    for (const Obs& ob : comp) {
      Ray r;
      if (!keypoint_ray(*cameras, ob.image, kp.at(ob.image, ob.keypoint), r)) continue;
      rest.push_back(ob);
      rays.push_back(r);
      keys.push_back(ob.image);
    }
    std::size_t produced = 0;
    while (static_cast<int>(rest.size()) >= tri.min_views) {
      const auto res = triangulate_robust(rays, tri, &keys);
      if (!res.ok) break;
      std::vector<Obs> track, rest2;
      std::vector<Ray> rays2;
      std::vector<std::int32_t> keys2;
      for (std::size_t q = 0; q < rest.size(); ++q) {
        if (res.inlier[q]) {
          track.push_back(rest[q]);
        } else {
          rest2.push_back(rest[q]);
          rays2.push_back(rays[q]);
          keys2.push_back(keys[q]);
        }
      }
      emit(out, track, o.max_views);
      produced += track.size();
      rest.swap(rest2);
      rays.swap(rays2);
      keys.swap(keys2);
    }
    st.dropped_observations += comp.size() - produced;
  }
  st.tracks = out.size();
  if (stats) *stats = st;
  return out;
}

}  // namespace jba
