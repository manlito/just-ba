#include "jba/tracks.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

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

// Angle (radians) of ray b from the epipolar plane of ray a, or +inf when the rays
// do not meet in front of both cameras. A near-zero baseline (a pure rotation) gets
// the angle between the directions.
double epipolar_error(const Ray& ra, const Ray& rb) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const double base[3] = {rb.C[0] - ra.C[0], rb.C[1] - ra.C[1], rb.C[2] - ra.C[2]};
  const double bl = std::sqrt(base[0] * base[0] + base[1] * base[1] + base[2] * base[2]);
  const double dab = ra.d[0] * rb.d[0] + ra.d[1] * rb.d[1] + ra.d[2] * rb.d[2];
  if (bl < 1e-9) return std::acos(std::clamp(dab, -1.0, 1.0));
  double err = 0;
  double n[3] = {base[1] * ra.d[2] - base[2] * ra.d[1], base[2] * ra.d[0] - base[0] * ra.d[2],
                 base[0] * ra.d[1] - base[1] * ra.d[0]};
  const double nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (nn > 1e-12 * bl) err = std::asin(std::min(1.0, std::abs(n[0] * rb.d[0] + n[1] * rb.d[1] + n[2] * rb.d[2]) / nn));
  // Closest points: C_a + s d_a and C_b + t d_b with s, t > 0.
  const double e = ra.d[0] * base[0] + ra.d[1] * base[1] + ra.d[2] * base[2];
  const double f = rb.d[0] * base[0] + rb.d[1] * base[1] + rb.d[2] * base[2];
  const double den = 1 - dab * dab;
  if (den < 1e-12) return err;  // parallel rays: a point at infinity, harmless
  const double sa = (e - dab * f) / den, tb = (dab * e - f) / den;
  return sa > 0 && tb > 0 ? err : kInf;
}

// Union-find whose sets carry their sorted images: join only when they share none.
struct GreedyMerge {
  UnionFind uf;
  std::vector<std::vector<std::int32_t>> images;  // per root
  GreedyMerge(std::size_t n, const std::vector<std::int32_t>& image_of) : uf(n), images(n) {
    for (std::size_t g = 0; g < n; ++g) images[g] = {image_of[g]};
  }
  // Returns false when the two sets share an image (the match is refused).
  bool join(std::int64_t a, std::int64_t b) {
    a = uf.find(a);
    b = uf.find(b);
    if (a == b) return true;
    auto& ia = images[static_cast<std::size_t>(a)];
    auto& ib = images[static_cast<std::size_t>(b)];
    const auto& small = ia.size() < ib.size() ? ia : ib;
    const auto& large = ia.size() < ib.size() ? ib : ia;
    for (std::int32_t i : small)
      if (std::binary_search(large.begin(), large.end(), i)) return false;
    std::vector<std::int32_t> merged;
    merged.reserve(ia.size() + ib.size());
    std::merge(ia.begin(), ia.end(), ib.begin(), ib.end(), std::back_inserter(merged));
    uf.join(a, b);
    const auto root = static_cast<std::size_t>(uf.find(a));
    ia.clear();
    ia.shrink_to_fit();
    ib.clear();
    ib.shrink_to_fit();
    images[root] = std::move(merged);
    return true;
  }
};

// Runs fn(job) for job in [0, n) on up to `threads` threads (0: all hardware threads).
void parallel_jobs(std::size_t n, int threads, const std::function<void(std::size_t)>& fn) {
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const auto count = static_cast<std::size_t>(threads) < n ? static_cast<std::size_t>(threads) : n;
  if (count <= 1) {
    for (std::size_t j = 0; j < n; ++j) fn(j);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  auto work = [&] {
    for (std::size_t j; (j = next.fetch_add(1)) < n;) {
      try {
        fn(j);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) error = std::current_exception();
      }
    }
  };
  std::vector<std::thread> pool;
  for (std::size_t t = 1; t < count; ++t) pool.emplace_back(work);
  work();
  for (auto& t : pool) t.join();
  if (error) std::rethrow_exception(error);
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
                    const Cameras* cameras, TrackStats* stats, const std::vector<double>* scores) {
  if (matches.size() % 4 != 0) throw std::invalid_argument("matches must hold 4 values per match");
  const std::size_t n_match = matches.size() / 4;
  if (scores && !scores->empty() && scores->size() != n_match)
    throw std::invalid_argument("scores must hold one value per match");
  if (scores && scores->empty()) scores = nullptr;
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

  const bool geometric = cameras != nullptr && o.conflict_policy != ConflictPolicy::Drop;
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

  // Matches that pass: keypoint ids and the greedy order key (smaller first).
  struct Edge {
    std::int64_t a, b;
    double key;
  };
  std::vector<Edge> edges;
  edges.reserve(n_match);
  for (std::size_t m = 0; m < n_match; ++m) {
    const auto a = id(matches[4 * m], matches[4 * m + 1]), b = id(matches[4 * m + 2], matches[4 * m + 3]);
    if (matches[4 * m] == matches[4 * m + 2]) continue;  // a match inside one image is meaningless
    double key = 0;
    // With poses: the two rays must meet in front of both cameras, within the
    // inlier angle of the epipolar plane. Removes most wrong matches before they
    // can fuse tracks into one large component.
    if (geometric) {
      const auto ga = static_cast<std::size_t>(a), gb = static_cast<std::size_t>(b);
      const double err = ray_ok[ga] == 1 && ray_ok[gb] == 1 ? epipolar_error(ray_of[ga], ray_of[gb])
                                                             : std::numeric_limits<double>::infinity();
      if (!(err <= max_err)) {
        ++st.rejected_matches;
        continue;
      }
      key = err;
    }
    if (scores) key = -(*scores)[m];
    edges.push_back({a, b, key});
  }
  const bool ordered = scores != nullptr || geometric;
  auto greedy_order = [&](std::vector<std::size_t>& idx) {
    if (ordered)
      std::stable_sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) { return edges[x].key < edges[y].key; });
  };

  UnionFind uf(total);
  std::vector<char> used(total, 0);
  if (o.conflict_policy == ConflictPolicy::Greedy) {
    std::vector<std::size_t> idx(edges.size());
    std::iota(idx.begin(), idx.end(), std::size_t(0));
    greedy_order(idx);
    GreedyMerge gm(total, image_of);
    for (std::size_t e : idx) {
      if (!gm.join(edges[e].a, edges[e].b)) {
        ++st.refused_matches;
        continue;
      }
      used[static_cast<std::size_t>(edges[e].a)] = used[static_cast<std::size_t>(edges[e].b)] = 1;
    }
    uf = std::move(gm.uf);
  } else {
    for (const Edge& e : edges) {
      uf.join(e.a, e.b);
      used[static_cast<std::size_t>(e.a)] = used[static_cast<std::size_t>(e.b)] = 1;
    }
  }

  // Components in order of their smallest keypoint id.
  std::vector<std::vector<Obs>> comps;
  std::vector<std::int64_t> comp_of(total, -1);  // per root
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

  // Classify; with Repair, split oversized conflicted components greedily. Each
  // entry of `work` goes to stage 2 (or straight out) and keeps its place in the
  // output order.
  struct Work {
    std::vector<Obs> obs;
    bool clean;
    bool split;  // a piece of an oversized component: always checked
  };
  std::vector<Work> work;
  work.reserve(comps.size());
  std::vector<char> oversized(comps.size(), 0), comp_clean(comps.size(), 0);
  bool any_oversized = false;
  for (std::size_t c = 0; c < comps.size(); ++c) {
    auto& comp = comps[c];
    std::sort(comp.begin(), comp.end(), [](const Obs& a, const Obs& b) {
      return a.image != b.image ? a.image < b.image : a.keypoint < b.keypoint;
    });
    std::size_t views = comp.empty() ? 0 : 1;
    for (std::size_t q = 1; q < comp.size(); ++q) views += comp[q].image != comp[q - 1].image;
    if (comp.size() > st.largest_component) {
      st.largest_component = comp.size();
      st.largest_component_images = views;
    }
    const bool clean = views == comp.size();
    comp_clean[c] = clean;
    if (clean) ++st.clean;
    else ++st.conflicted;
    if (!clean && geometric && o.conflict_policy == ConflictPolicy::Repair && o.max_repair_observations > 0 &&
        comp.size() > static_cast<std::size_t>(o.max_repair_observations)) {
      oversized[c] = 1;
      any_oversized = true;
      ++st.split_components;
    }
  }
  // Pieces of the oversized components, by a greedy merge over their own matches.
  std::vector<std::vector<std::vector<Obs>>> pieces(comps.size());
  if (any_oversized) {
    std::vector<std::size_t> idx;
    for (std::size_t e = 0; e < edges.size(); ++e)
      if (oversized[static_cast<std::size_t>(comp_of[static_cast<std::size_t>(uf.find(edges[e].a))])]) idx.push_back(e);
    greedy_order(idx);
    GreedyMerge gm(total, image_of);
    for (std::size_t e : idx)
      if (!gm.join(edges[e].a, edges[e].b)) ++st.refused_matches;
    std::vector<std::int64_t> piece_of(total, -1);  // per greedy root
    for (std::size_t c = 0; c < comps.size(); ++c) {
      if (!oversized[c]) continue;
      for (const Obs& ob : comps[c]) {
        const auto g = kp.begin[static_cast<std::size_t>(ob.image)] + ob.keypoint;
        const auto root = static_cast<std::size_t>(gm.uf.find(g));
        if (piece_of[root] < 0) {
          piece_of[root] = static_cast<std::int64_t>(pieces[c].size());
          pieces[c].emplace_back();
        }
        pieces[c][static_cast<std::size_t>(piece_of[root])].push_back(ob);
      }
    }
  }
  for (std::size_t c = 0; c < comps.size(); ++c) {
    if (!oversized[c]) {
      work.push_back({std::move(comps[c]), comp_clean[c] != 0, false});
      continue;
    }
    for (auto& piece : pieces[c]) work.push_back({std::move(piece), true, true});  // sorted already
  }
  comps.clear();

  TriangulationOptions tri = o.triangulation;
  tri.min_views = std::max(2, o.min_views);
  struct Result {
    std::vector<std::vector<Obs>> tracks;
    std::size_t dropped = 0;
  };
  std::vector<Result> results(work.size());
  auto stage2 = [&](std::size_t w) {
    const Work& job = work[w];
    Result& res = results[w];
    const auto& comp = job.obs;
    if (job.clean && !job.split && !(geometric && o.check_clean)) {
      if (static_cast<int>(comp.size()) >= o.min_views) res.tracks.push_back(comp);
      else res.dropped = comp.size();
      return;
    }
    if (!geometric || static_cast<int>(comp.size()) < tri.min_views) {
      res.dropped = comp.size();
      return;
    }
    // Stage 2: repeated robust triangulation; each round's inliers become a track.
    std::vector<Obs> rest;
    std::vector<Ray> rays;
    std::vector<std::int32_t> keys;
    for (const Obs& ob : comp) {
      const auto g = static_cast<std::size_t>(kp.begin[static_cast<std::size_t>(ob.image)] + ob.keypoint);
      if (ray_ok[g] != 1) continue;
      rest.push_back(ob);
      rays.push_back(ray_of[g]);
      keys.push_back(ob.image);
    }
    std::size_t produced = 0;
    while (static_cast<int>(rest.size()) >= tri.min_views) {
      const auto r = triangulate_robust(rays, tri, &keys);
      if (!r.ok) break;
      std::vector<Obs> track, rest2;
      std::vector<Ray> rays2;
      std::vector<std::int32_t> keys2;
      for (std::size_t q = 0; q < rest.size(); ++q) {
        if (r.inlier[q]) {
          track.push_back(rest[q]);
        } else {
          rest2.push_back(rest[q]);
          rays2.push_back(rays[q]);
          keys2.push_back(keys[q]);
        }
      }
      produced += track.size();
      res.tracks.push_back(std::move(track));
      rest.swap(rest2);
      rays.swap(rays2);
      keys.swap(keys2);
    }
    res.dropped = comp.size() - produced;
  };
  parallel_jobs(work.size(), o.num_threads, stage2);

  Tracks out;
  for (auto& r : results) {
    for (auto& t : r.tracks) emit(out, std::move(t), o.max_views);
    st.dropped_observations += r.dropped;
  }
  st.tracks = out.size();
  if (stats) *stats = st;
  return out;
}

}  // namespace jba
