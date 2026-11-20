#include "jba/refine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

#include "jba/geometry.h"
#include "jba/triangulation.h"

namespace jba {
namespace {

constexpr double kPi = 3.141592653589793;

// Keeps the priors whose images are all active; records where each came from.
void keep_active_priors(Problem& p, const std::vector<char>& active, std::vector<std::int32_t>& rel_index,
                        std::vector<std::int32_t>& glob_index, std::vector<std::int32_t>& grav_index) {
  Problem q = p;
  q.rel_images.clear();
  q.rel_pose.clear();
  q.rel_sigma.clear();
  q.rel_sqrt_info.clear();
  q.glob_image.clear();
  q.glob_set.clear();
  q.glob_pose.clear();
  q.glob_sigma.clear();
  q.glob_sqrt_info.clear();
  q.grav_image.clear();
  q.grav_dir.clear();
  q.grav_sigma.clear();
  rel_index.clear();
  glob_index.clear();
  grav_index.clear();
  for (std::size_t k = 0; k < p.num_relative_priors(); ++k) {
    const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
    if (!active[i] || !active[j]) continue;
    rel_index.push_back(static_cast<std::int32_t>(k));
    q.rel_images.insert(q.rel_images.end(), {p.rel_images[2 * k], p.rel_images[2 * k + 1]});
    q.rel_pose.insert(q.rel_pose.end(), &p.rel_pose[7 * k], &p.rel_pose[7 * k] + 7);
    if (!p.rel_sigma.empty()) q.rel_sigma.insert(q.rel_sigma.end(), &p.rel_sigma[2 * k], &p.rel_sigma[2 * k] + 2);
    if (!p.rel_sqrt_info.empty())
      q.rel_sqrt_info.insert(q.rel_sqrt_info.end(), &p.rel_sqrt_info[36 * k], &p.rel_sqrt_info[36 * k] + 36);
  }
  for (std::size_t k = 0; k < p.num_global_priors(); ++k) {
    if (!active[static_cast<std::size_t>(p.glob_image[k])]) continue;
    glob_index.push_back(static_cast<std::int32_t>(k));
    q.glob_image.push_back(p.glob_image[k]);
    if (!p.glob_set.empty()) q.glob_set.push_back(p.glob_set[k]);
    q.glob_pose.insert(q.glob_pose.end(), &p.glob_pose[7 * k], &p.glob_pose[7 * k] + 7);
    if (!p.glob_sigma.empty()) q.glob_sigma.insert(q.glob_sigma.end(), &p.glob_sigma[2 * k], &p.glob_sigma[2 * k] + 2);
    if (!p.glob_sqrt_info.empty())
      q.glob_sqrt_info.insert(q.glob_sqrt_info.end(), &p.glob_sqrt_info[36 * k], &p.glob_sqrt_info[36 * k] + 36);
  }
  for (std::size_t k = 0; k < p.num_gravity_priors(); ++k) {
    if (!active[static_cast<std::size_t>(p.grav_image[k])]) continue;
    grav_index.push_back(static_cast<std::int32_t>(k));
    q.grav_image.push_back(p.grav_image[k]);
    q.grav_dir.insert(q.grav_dir.end(), &p.grav_dir[3 * k], &p.grav_dir[3 * k] + 3);
    if (!p.grav_sigma.empty()) q.grav_sigma.push_back(p.grav_sigma[k]);
  }
  // Prior sets stay as they are, so set indices are unchanged.
  p = std::move(q);
}

void centre(const double* pose, double* C) {
  const double c[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  quat_rotate(c, pose + 4, C);
  for (int a = 0; a < 3; ++a) C[a] = -C[a];
}

// Removes observations with large error or behind the camera, then points left with
// fewer than two observations or a small triangulation angle. Returns the number of
// observations removed.
std::size_t filter(Problem& p, std::vector<std::int32_t>& obs_keypoint, const SolverOptions& so,
                   const RefineOptions& o) {
  const ResidualReport rep = evaluate(p, so);
  const std::size_t n_obs = p.num_observations(), n_pts = p.num_points();
  std::vector<char> keep(n_obs);
  std::vector<int> count(n_pts, 0);
  for (std::size_t k = 0; k < n_obs; ++k) {
    keep[k] = rep.in_front[k] && rep.reprojection_error[k] <= o.max_reprojection_error;
    if (keep[k]) ++count[static_cast<std::size_t>(p.obs_point[k])];
  }
  // Triangulation angle of each point over its kept observations.
  std::vector<std::vector<std::size_t>> by_point(n_pts);
  for (std::size_t k = 0; k < n_obs; ++k)
    if (keep[k]) by_point[static_cast<std::size_t>(p.obs_point[k])].push_back(k);
  std::vector<char> point_ok(n_pts, 0);
  for (std::size_t j = 0; j < n_pts; ++j) {
    if (count[j] < 2) continue;
    std::vector<Ray> rays;
    for (std::size_t k : by_point[j]) {
      Ray r;
      centre(&p.poses[7 * static_cast<std::size_t>(p.obs_image[k])], r.C);
      rays.push_back(r);
    }
    point_ok[j] = triangulation_angle(rays.data(), rays.size(), &p.points[3 * j]) * 180 / kPi >= o.min_tri_angle_deg;
  }
  // Rebuild with the survivors.
  std::vector<std::int32_t> new_index(n_pts, -1);
  std::vector<double> points;
  for (std::size_t j = 0; j < n_pts; ++j) {
    if (!point_ok[j]) continue;
    new_index[j] = static_cast<std::int32_t>(points.size() / 3);
    points.insert(points.end(), &p.points[3 * j], &p.points[3 * j] + 3);
  }
  Problem q = p;
  q.points = std::move(points);
  q.point_fixed.clear();
  q.obs_image.clear();
  q.obs_point.clear();
  q.obs_xy.clear();
  q.obs_sigma.clear();
  q.obs_depth.clear();
  q.obs_depth_sigma.clear();
  std::vector<std::int32_t> kp;
  std::size_t removed = 0;
  for (std::size_t k = 0; k < n_obs; ++k) {
    const auto j = new_index[static_cast<std::size_t>(p.obs_point[k])];
    if (!keep[k] || j < 0) {
      ++removed;
      continue;
    }
    q.obs_image.push_back(p.obs_image[k]);
    q.obs_point.push_back(j);
    q.obs_xy.insert(q.obs_xy.end(), {p.obs_xy[2 * k], p.obs_xy[2 * k + 1]});
    if (!p.obs_sigma.empty()) q.obs_sigma.push_back(p.obs_sigma[k]);
    if (!p.obs_depth.empty()) q.obs_depth.push_back(p.obs_depth[k]);
    if (!p.obs_depth_sigma.empty()) q.obs_depth_sigma.push_back(p.obs_depth_sigma[k]);
    kp.push_back(obs_keypoint[k]);
  }
  p = std::move(q);
  obs_keypoint = std::move(kp);
  return removed;
}

// Starting depth corrections for images whose correction is free and was not given:
// from the triangulated points, the median ratio (scale only), median difference
// (offset only), or a line fit refit without residuals over three median absolute
// deviations (both).
void initialize_depth_correction(Problem& p) {
  const std::size_t n_img = p.num_images();
  if (p.depth_correction_free.empty() || p.obs_depth.empty()) return;
  p.depth_correction.assign(2 * n_img, 0);
  for (std::size_t i = 0; i < n_img; ++i) p.depth_correction[2 * i] = 1;
  std::vector<std::vector<std::array<double, 2>>> pairs(n_img);  // (reading, solved depth)
  for (std::size_t k = 0; k < p.num_observations(); ++k) {
    if (std::isnan(p.obs_depth[k])) continue;
    const auto i = static_cast<std::size_t>(p.obs_image[k]);
    if (p.depth_correction_free[i] == 0) continue;
    const double* pose = &p.poses[7 * i];
    double pc[3];
    quat_rotate(pose, &p.points[3 * static_cast<std::size_t>(p.obs_point[k])], pc);
    for (int a = 0; a < 3; ++a) pc[a] += pose[4 + a];
    const double z = p.depth_kind == DepthKind::Range ? std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]) : pc[2];
    if (z > 0) pairs[i].push_back({p.obs_depth[k], z});
  }
  auto median = [](std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  for (std::size_t i = 0; i < n_img; ++i) {
    const auto& pr = pairs[i];
    const std::uint8_t mask = p.depth_correction_free[i];
    if (pr.size() < 3) continue;
    double a = 1, b = 0;
    std::vector<double> v;
    if (mask == kDepthScale) {
      for (const auto& [d, z] : pr) v.push_back(z / d);
      a = median(v);
    } else if (mask == kDepthOffset) {
      for (const auto& [d, z] : pr) v.push_back(z - d);
      b = median(v);
    } else {
      std::vector<char> use(pr.size(), 1);
      for (int pass = 0; pass < 2; ++pass) {
        double sw = 0, sd = 0, sz = 0, sdd = 0, sdz = 0;
        for (std::size_t q = 0; q < pr.size(); ++q) {
          if (!use[q]) continue;
          sw += 1, sd += pr[q][0], sz += pr[q][1], sdd += pr[q][0] * pr[q][0], sdz += pr[q][0] * pr[q][1];
        }
        const double det = sw * sdd - sd * sd;
        if (!(sw >= 3 && det > 1e-12 * sw * sdd)) break;
        a = (sw * sdz - sd * sz) / det;
        b = (sz - a * sd) / sw;
        std::vector<double> res;
        for (const auto& [d, z] : pr) res.push_back(std::abs(z - (a * d + b)));
        const double mad = median(res);
        for (std::size_t q = 0; q < pr.size(); ++q) use[q] = res[q] <= 3 * mad + 1e-12;
      }
      if (!(a > 0)) a = 1, b = 0;
    }
    p.depth_correction[2 * i] = a;
    p.depth_correction[2 * i + 1] = b;
  }
}

}  // namespace

RefineResult refine(const RefineInput& in, const RefineOptions& o) {
  const std::size_t n_img = in.keypoints.num_images();
  if (n_img == 0) throw std::invalid_argument("refine: no images (keypoints.begin is empty)");
  if (!in.keypoint_sigma.empty() &&
      in.keypoint_sigma.size() != static_cast<std::size_t>(in.keypoints.begin.back()))
    throw std::invalid_argument("refine: keypoint_sigma must have one value per keypoint");
  const auto n_kp = static_cast<std::size_t>(in.keypoints.begin.back());
  if (!in.keypoint_depth.empty() && in.keypoint_depth.size() != n_kp)
    throw std::invalid_argument("refine: keypoint_depth must have one value per keypoint");
  if (!in.keypoint_depth_sigma.empty() && in.keypoint_depth_sigma.size() != in.keypoint_depth.size())
    throw std::invalid_argument("refine: keypoint_depth_sigma must have one value per keypoint, with keypoint_depth");

  Problem base = in.scene;
  base.points.clear();
  base.point_fixed.clear();
  base.obs_image.clear();
  base.obs_point.clear();
  base.obs_xy.clear();
  base.obs_sigma.clear();
  base.obs_depth.clear();
  base.obs_depth_sigma.clear();
  base.image_fixed.clear();
  if (base.poses.empty()) {
    for (std::size_t i = 0; i < n_img; ++i) base.poses.insert(base.poses.end(), {1, 0, 0, 0, 0, 0, 0});
  }
  if (base.num_images() != n_img) throw std::invalid_argument("refine: scene.poses must have one pose per image");
  if (auto err = validate(base); !err.empty()) throw std::invalid_argument("refine: invalid scene: " + err);
  // The intrinsics prior anchors to the caller's values in every round (D161), not to
  // the previous round's result.
  if (!base.intrinsics_prior_sigma.empty() && base.intrinsics_prior.empty()) base.intrinsics_prior = base.intrinsics;

  RefineResult out;
  const PoseInitialization init = initialize_poses(base, o.solver.prior_defaults);
  base.poses = init.poses;
  if (base.num_prior_sets() > 0) base.set_transform = init.set_transform;
  out.group = init.group;
  out.num_groups = init.num_groups;

  std::vector<char> active(n_img, 0);
  for (std::size_t i = 0; i < n_img; ++i) active[i] = init.group[i] >= 0 && (!o.largest_group_only || init.group[i] == 0);
  std::vector<std::int32_t> rel_index, glob_index, grav_index;
  keep_active_priors(base, active, rel_index, glob_index, grav_index);

  const bool have_priors = base.num_relative_priors() + base.num_global_priors() + base.num_gravity_priors() > 0;
  if (o.prior_solve && have_priors) {
    out.summaries.push_back(solve(base, o.solver));
  }

  // Manhattan prior: assign, test support, and remove heading drift before
  // triangulating. Segments of images left out are never assigned.
  base.line_axis.clear();
  base.manhattan_frame.clear();
  auto assign = [&](const double* frame, double angle) {
    ManhattanOptions mo = o.manhattan;
    mo.max_angle_deg = angle;
    ManhattanAssignment a = assign_lines(base, mo, frame);
    for (std::size_t k = 0; k < a.axis.size(); ++k) {
      if (active[static_cast<std::size_t>(base.line_image[k])] || a.axis[k] < 0) continue;
      --a.assigned[a.axis[k]];
      a.axis[k] = -1;
    }
    apply_assignment(base, a, mo.dof);
    return a;
  };
  double line_angle = o.manhattan.max_angle_deg;
  if (o.use_manhattan && base.num_lines() > 0) {
    out.manhattan = assign(nullptr, line_angle);
    if (out.manhattan.accepted && o.prior_solve && have_priors) out.summaries.push_back(solve(base, o.solver));
  }

  Problem problem = base;
  std::vector<std::int32_t> obs_keypoint;
  const int rounds = std::max(1, o.rounds);
  for (int round = 0; round < rounds; ++round) {
    if (round > 0 && base.has_manhattan()) {
      line_angle = std::max(o.manhattan_min_angle_deg, 0.5 * line_angle);
      const std::vector<double> frame = base.manhattan_frame;
      out.manhattan = assign(frame.data(), line_angle);
    }
    TrackOptions topt = o.tracks;
    topt.triangulation.max_angle_error_deg = std::max(0.25, o.tracks.triangulation.max_angle_error_deg * std::pow(0.5, round));
    Cameras cams{base.model, base.intrinsics, base.image_intrinsics, base.poses};
    Tracks tracks;
    if (in.tracks.size() > 0) {
      tracks = in.tracks;
    } else {
      tracks = build_tracks(in.keypoints, in.matches, topt, &cams, &out.track_stats);
    }

    // One point per track: observations on active images of the track's main
    // group, triangulated robustly from the current poses.
    problem = base;
    obs_keypoint.clear();
    for (std::size_t t = 0; t < tracks.size(); ++t) {
      std::map<int, int> per_group;
      for (auto q = tracks.begin[t]; q < tracks.begin[t + 1]; ++q) {
        const auto i = static_cast<std::size_t>(tracks.image[static_cast<std::size_t>(q)]);
        if (active[i]) ++per_group[init.group[i]];
      }
      if (per_group.empty()) continue;
      int g = per_group.begin()->first, best = 0;
      for (const auto& [grp, c] : per_group)
        if (c > best) g = grp, best = c;
      std::vector<Ray> rays;
      std::vector<std::int32_t> keys, imgs, kps;
      for (auto q = tracks.begin[t]; q < tracks.begin[t + 1]; ++q) {
        const auto i = tracks.image[static_cast<std::size_t>(q)];
        const auto k = tracks.keypoint[static_cast<std::size_t>(q)];
        if (!active[static_cast<std::size_t>(i)] || init.group[static_cast<std::size_t>(i)] != g) continue;
        Ray r;
        if (!keypoint_ray(cams, i, in.keypoints.at(i, k), r)) continue;
        rays.push_back(r);
        keys.push_back(i);
        imgs.push_back(i);
        kps.push_back(k);
      }
      TriangulationOptions tri = topt.triangulation;
      tri.min_views = std::max(2, topt.min_views);
      tri.min_tri_angle_deg = o.min_tri_angle_deg;
      const auto res = triangulate_robust(rays, tri, &keys);
      if (!res.ok) continue;
      const auto j = static_cast<std::int32_t>(problem.num_points());
      problem.points.insert(problem.points.end(), res.X, res.X + 3);
      for (std::size_t q = 0; q < rays.size(); ++q) {
        if (!res.inlier[q]) continue;
        problem.obs_image.push_back(imgs[q]);
        problem.obs_point.push_back(j);
        const double* xy = in.keypoints.at(imgs[q], kps[q]);
        problem.obs_xy.insert(problem.obs_xy.end(), {xy[0], xy[1]});
        const auto kp_index = static_cast<std::size_t>(in.keypoints.begin[static_cast<std::size_t>(imgs[q])] + kps[q]);
        if (!in.keypoint_sigma.empty()) problem.obs_sigma.push_back(in.keypoint_sigma[kp_index]);
        if (!in.keypoint_depth.empty()) {
          const double d = in.keypoint_depth[kp_index];
          problem.obs_depth.push_back(d > 0 && std::isfinite(d) ? d : std::nan(""));
          if (!in.keypoint_depth_sigma.empty()) {
            const double s = in.keypoint_depth_sigma[kp_index];
            problem.obs_depth_sigma.push_back(s > 0 && std::isfinite(s) ? s : 1.0);
          }
        }
        obs_keypoint.push_back(kps[q]);
      }
    }
    if (problem.num_points() == 0 && problem.num_relative_priors() + problem.num_global_priors() == 0)
      throw std::runtime_error("refine: no point could be triangulated and there are no priors");
    if (round == 0 && in.scene.depth_correction.empty()) initialize_depth_correction(problem);

    out.summaries.push_back(solve(problem, o.solver));
    out.rejected_observations = filter(problem, obs_keypoint, o.solver, o);
    if (out.rejected_observations > 0) out.summaries.push_back(solve(problem, o.solver));
    base.poses = problem.poses;
    base.intrinsics = problem.intrinsics;
    base.set_transform = problem.set_transform;
    base.manhattan_frame = problem.manhattan_frame;
    base.depth_correction = problem.depth_correction;
  }
  if (out.manhattan.accepted) out.manhattan.frame = problem.manhattan_frame;

  out.num_observations.assign(n_img, 0);
  for (const auto i : problem.obs_image) ++out.num_observations[static_cast<std::size_t>(i)];
  out.refined.assign(n_img, 0);
  for (std::size_t i = 0; i < n_img; ++i)
    out.refined[i] = active[i] && out.num_observations[i] >= std::max(0, o.min_observations);
  out.report = evaluate(problem, o.solver, o.downweight_threshold);
  // Report prior indices in the caller's numbering.
  for (auto& k : out.report.downweighted_relative) k = rel_index[static_cast<std::size_t>(k)];
  for (auto& k : out.report.downweighted_global) k = glob_index[static_cast<std::size_t>(k)];
  for (auto& k : out.report.downweighted_gravity) k = grav_index[static_cast<std::size_t>(k)];
  store_losses(problem, o.solver);  // so the solved problem replays with them (D80)
  out.problem = std::move(problem);
  out.obs_keypoint = std::move(obs_keypoint);
  out.relative_prior_index = std::move(rel_index);
  out.global_prior_index = std::move(glob_index);
  out.gravity_prior_index = std::move(grav_index);
  return out;
}

}  // namespace jba
