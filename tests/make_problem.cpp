// Writes a synthetic problem for the parity tests and the benchmark, with perturbed
// starting values.
//
//   jba_make_problem <out.jba> [--model NAME] [--images N] [--points N] [--seed N] [--track-length N]
//                    [--noise PX] [--outliers FRACTION] [--free-intrinsics MASK]
//                    [--per-image-intrinsics] [--fix-images N] [--fix-points N] [--sigma PX]
//                    [--relative-priors N] [--global-priors] [--global-relation identity|rigid|similarity]
//                    [--prior-noise ROT,TRANS] [--prior-outliers FRACTION] [--prior-sigma ROT,TRANS]
//                    [--gravity-priors] [--gravity-noise RAD] [--gravity-outliers FRACTION] [--gravity-sigma RAD]
//                    [--lines N] [--line-dof 1|3] [--line-outliers FRACTION] [--prior-groups 2]
//                    [--depth FRACTION] [--depth-kind z|range] [--depth-form linear|inverse]
//                    [--depth-free MASK] [--depth-outliers FRACTION] [--far-points FRACTION]
//                    [--intrinsics-prior FOCAL_FRACTION,PP_PX,DISTORTION] [--intrinsics-prior-at-start]
//                    [--intrinsics-groups G] [--intrinsics-group-sigma F,PP,DIST,GROUP_F,GROUP_PP,GROUP_DIST]
//                    [--intrinsics-group-at-start]
//
// MASK is the IntrinsicsGroup bit mask: 1 focal, 2 principal point, 4 distortion.
// --intrinsics-prior: a prior on the intrinsics (D161, set_intrinsics_prior) toward
// the true values, which the free groups then start away from; inf leaves a group
// without one. With --intrinsics-prior-at-start the target is left empty, so it is
// the (perturbed) starting values.
// --intrinsics-groups G: the hierarchical prior (D165, set_intrinsics_groups), block c
// in group c % G (with --per-image-intrinsics, image i's block in group i % G), with
// the ties' and the targets' sigmas of --intrinsics-group-sigma (default
// 0.005,inf,0.001,0.05,inf,inf), the target the true values; with
// --intrinsics-group-at-start the target is left empty, the members' starting mean.
// --track-length N: each point is seen by N neighbouring images on the ring only (a
// walked capture); by default every image sees every point.
// Priors come from the true poses with noise: relative priors tie each image to
// its N next neighbours on the ring; global priors are in a frame related to the
// working frame by a random transform of the given kind. Outliers are gross errors
// (0.3 to 1 rad, 0.5 to 2 m). Gravity priors: every image, measured from the true
// pose with noise; gravity in the working frame is a fixed oblique direction, so no
// coordinate axis is special; outliers are 0.2 to 0.5 rad off. Lines: N segments in
// the point cloud along the axes of a random Manhattan frame (vertical axis up for
// one degree of freedom), assigned by the truth except a fraction given a wrong
// axis, seen by every image that has both endpoints in front; the frame starts
// 0.02 rad off.
// Not the synthetic generator of document 05 (milestone 2).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "jba/geometry.h"
#include "jba/priors.h"
#include "jba/problem_io.h"
#include "synthetic.h"

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <out.jba> [options]\n", argv[0]);
      return 2;
    }
    const std::string out = argv[1];
    jba::CameraModel model = jba::CameraModel::SimpleRadial;
    int images = 12, points = 300, fix_images = 0, fix_points = 0, free_mask = 0, track_length = 0;
    double far_points = 0;
    std::uint64_t seed = 5;
    double noise = 0.5, outliers = 0, sigma = 0;
    bool per_image = false, global_priors = false, gravity_priors = false;
    double gravity_noise = 0.003, gravity_outliers = 0, gravity_sigma = 0;
    int lines = 0, line_dof = 1, prior_groups = 1, depth_free = 0;
    double depth_fraction = 0, depth_outliers = 0;
    jba::DepthKind depth_kind = jba::DepthKind::Z;
    jba::DepthForm depth_form = jba::DepthForm::Linear;
    double line_outliers = 0;
    int relative_priors = 0;
    jba::FrameRelation relation = jba::FrameRelation::Identity;
    double prior_noise[2] = {0.002, 0.005}, prior_sigma[2] = {0, 0}, prior_outliers = 0;
    double intr_prior[3] = {0, 0, 0};  // focal fraction, principal point, distortion; 0: no prior
    bool intr_prior_at_start = false;
    int intr_groups = 0;
    double group_sigma[6] = {0.005, INFINITY, 0.001, 0.05, INFINITY, INFINITY};
    bool group_at_start = false;
    auto pair_of = [](const std::string& v, double* out) {
      const auto comma = v.find(',');
      if (comma == std::string::npos) throw std::invalid_argument("expected ROT,TRANS: " + v);
      out[0] = std::stod(v.substr(0, comma));
      out[1] = std::stod(v.substr(comma + 1));
    };
    for (int a = 2; a < argc; ++a) {
      const std::string key = argv[a];
      if (key == "--per-image-intrinsics") {
        per_image = true;
        continue;
      }
      if (key == "--global-priors") {
        global_priors = true;
        continue;
      }
      if (key == "--gravity-priors") {
        gravity_priors = true;
        continue;
      }
      if (key == "--intrinsics-prior-at-start") {
        intr_prior_at_start = true;
        continue;
      }
      if (key == "--intrinsics-group-at-start") {
        group_at_start = true;
        continue;
      }
      if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      const std::string val = argv[++a];
      if (key == "--model") {
        auto m = jba::model_from_name(val);
        if (!m) throw std::invalid_argument("unknown model " + val);
        model = *m;
      } else if (key == "--images") images = std::stoi(val);
      else if (key == "--points") points = std::stoi(val);
      else if (key == "--seed") seed = std::stoull(val);
      else if (key == "--track-length") track_length = std::stoi(val);
      else if (key == "--far-points") far_points = std::stod(val);
      else if (key == "--noise") noise = std::stod(val);
      else if (key == "--outliers") outliers = std::stod(val);
      else if (key == "--free-intrinsics") free_mask = std::stoi(val);
      else if (key == "--fix-images") fix_images = std::stoi(val);
      else if (key == "--fix-points") fix_points = std::stoi(val);
      else if (key == "--sigma") sigma = std::stod(val);
      else if (key == "--relative-priors") relative_priors = std::stoi(val);
      else if (key == "--prior-noise") pair_of(val, prior_noise);
      else if (key == "--prior-sigma") pair_of(val, prior_sigma);
      else if (key == "--prior-outliers") prior_outliers = std::stod(val);
      else if (key == "--gravity-noise") gravity_noise = std::stod(val);
      else if (key == "--gravity-outliers") gravity_outliers = std::stod(val);
      else if (key == "--gravity-sigma") gravity_sigma = std::stod(val);
      else if (key == "--lines") lines = std::stoi(val);
      else if (key == "--line-dof") line_dof = std::stoi(val);
      else if (key == "--prior-groups") prior_groups = std::stoi(val);
      else if (key == "--depth") depth_fraction = std::stod(val);
      else if (key == "--depth-free") depth_free = std::stoi(val);
      else if (key == "--intrinsics-prior") {
        std::size_t at = 0;
        for (int c = 0; c < 3; ++c) {
          const auto comma = c < 2 ? val.find(',', at) : std::string::npos;
          if (c < 2 && comma == std::string::npos) throw std::invalid_argument("expected F,PP,DIST: " + val);
          intr_prior[c] = std::stod(val.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
          at = comma + 1;
        }
      }
      else if (key == "--intrinsics-groups") intr_groups = std::stoi(val);
      else if (key == "--intrinsics-group-sigma") {
        std::size_t at = 0;
        for (int c = 0; c < 6; ++c) {
          const auto comma = c < 5 ? val.find(',', at) : std::string::npos;
          if (c < 5 && comma == std::string::npos) throw std::invalid_argument("expected six sigmas: " + val);
          group_sigma[c] = std::stod(val.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
          at = comma + 1;
        }
      }
      else if (key == "--depth-outliers") depth_outliers = std::stod(val);
      else if (key == "--depth-kind") depth_kind = val == "range" ? jba::DepthKind::Range : jba::DepthKind::Z;
      else if (key == "--depth-form") depth_form = val == "inverse" ? jba::DepthForm::Inverse : jba::DepthForm::Linear;
      else if (key == "--line-outliers") line_outliers = std::stod(val);
      else if (key == "--global-relation") {
        if (val == "identity") relation = jba::FrameRelation::Identity;
        else if (val == "rigid") relation = jba::FrameRelation::Rigid;
        else if (val == "similarity") relation = jba::FrameRelation::Similarity;
        else throw std::invalid_argument("unknown relation " + val);
      }
      else throw std::invalid_argument("unknown option " + key);
    }

    auto p = jba_test::random_problem(model, images, points, seed, noise, track_length, far_points);
    std::mt19937_64 rng(seed + 1000);
    jba_test::Normal n(0, 1);
    jba_test::Uniform u(0, 1);

    // Gross outliers: move a fraction of observations by 20 to 60 pixels.
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      if (u(rng) < outliers) {
        const double angle = 6.283185307179586 * u(rng), len = 20 + 40 * u(rng);
        p.obs_xy[2 * k] += len * std::cos(angle);
        p.obs_xy[2 * k + 1] += len * std::sin(angle);
      }
    }
    if (sigma > 0) p.obs_sigma.assign(p.num_observations(), sigma);

    // Priors from the true poses, with noise and gross outliers.
    const std::vector<double> truth = p.poses;
    auto noisy = [&](double* pose) {
      const bool gross = u(rng) < prior_outliers;
      const double sr = gross ? 0.3 + 0.7 * u(rng) : prior_noise[0];
      const double st = gross ? 0.5 + 1.5 * u(rng) : prior_noise[1];
      double step[6];
      if (gross) {
        // Fixed magnitude in a random direction.
        double d[6];
        double nr = 0, nt = 0;
        for (int c = 0; c < 6; ++c) d[c] = n(rng);
        for (int c = 0; c < 3; ++c) nr += d[c] * d[c], nt += d[3 + c] * d[3 + c];
        for (int c = 0; c < 3; ++c) step[c] = sr * d[c] / std::sqrt(nr), step[3 + c] = st * d[3 + c] / std::sqrt(nt);
      } else {
        for (int c = 0; c < 3; ++c) step[c] = sr * n(rng), step[3 + c] = st * n(rng);
      }
      double moved[7];
      jba::pose_retract(pose, step, moved);
      std::copy_n(moved, 7, pose);
    };
    if (relative_priors > 0) {
      std::vector<std::int32_t> pairs;
      for (int i = 0; i < images; ++i)
        for (int d = 1; d <= relative_priors && d < images; ++d) {
          const int j = (i + d) % images;
          // --prior-groups 2: no prior between the two halves of the ring.
          if (prior_groups == 2 && (2 * i < images) != (2 * j < images)) continue;
          pairs.insert(pairs.end(), {i, j});
        }
      jba::add_relative_priors_from_poses(p, truth, pairs);
      for (std::size_t k = 0; k < p.num_relative_priors(); ++k) noisy(&p.rel_pose[7 * k]);
      if (prior_sigma[0] > 0) {
        p.rel_sigma.clear();
        for (std::size_t k = 0; k < p.num_relative_priors(); ++k) p.rel_sigma.insert(p.rel_sigma.end(), {prior_sigma[0], prior_sigma[1]});
      }
    }
    if (global_priors) {
      // The set's frame: X_working = s R X_set + t, so a world-to-camera pose in the
      // set's frame is (R_i R, (R_i t + t_i) / s).
      double S[8] = {1, 0, 0, 0, 0, 0, 0, 1};
      if (relation != jba::FrameRelation::Identity) {
        jba_test::random_rotation(rng, 2.5, S);
        for (int c = 0; c < 3; ++c) S[4 + c] = 3 * (2 * u(rng) - 1);
        if (relation == jba::FrameRelation::Similarity) S[7] = 0.3 + 2 * u(rng);
      }
      for (int i = 0; i < images; ++i) {
        const double* T = &truth[7 * static_cast<std::size_t>(i)];
        double g[7], Rt[3];
        jba::quat_multiply(T, S, g);
        jba::quat_normalize(g);
        jba::quat_rotate(T, S + 4, Rt);
        for (int c = 0; c < 3; ++c) g[4 + c] = (Rt[c] + T[4 + c]) / S[7];
        noisy(g);
        p.glob_image.push_back(i);
        p.glob_pose.insert(p.glob_pose.end(), g, g + 7);
      }
      p.set_relation = {relation};
      if (prior_sigma[0] > 0) {
        for (int i = 0; i < images; ++i) p.glob_sigma.insert(p.glob_sigma.end(), {prior_sigma[0], prior_sigma[1]});
      }
      if (relation != jba::FrameRelation::Identity) {
        // Start near the truth, as after the alignment that builds starting poses.
        const double step[6] = {0.05 * n(rng), 0.05 * n(rng), 0.05 * n(rng), 0.1 * n(rng), 0.1 * n(rng), 0.1 * n(rng)};
        double start[8];
        jba::pose_retract(S, step, start);
        start[7] = relation == jba::FrameRelation::Similarity ? S[7] * (1 + 0.05 * n(rng)) : 1.0;
        p.set_transform.assign(start, start + 8);
      }
    }

    if (gravity_priors) {
      p.gravity_world = {0.2, 0.9, -0.3};
      const double norm = std::sqrt(0.04 + 0.81 + 0.09);
      for (double& v : p.gravity_world) v /= norm;
      for (int i = 0; i < images; ++i) {
        const double* T = &truth[7 * static_cast<std::size_t>(i)];
        const bool gross = u(rng) < gravity_outliers;
        double rot[3] = {n(rng), n(rng), n(rng)};
        const double len = std::sqrt(rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2]);
        const double angle = gross ? 0.2 + 0.3 * u(rng) : gravity_noise * len;
        for (double& v : rot) v *= angle / len;
        double dq[4], q[4], g[3];
        jba::quat_exp(rot, dq);
        jba::quat_multiply(dq, T, q);
        jba::quat_rotate(q, p.gravity_world.data(), g);
        const double gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        p.grav_image.push_back(i);
        for (double v : g) p.grav_dir.push_back(v / gn);
        if (gravity_sigma > 0) p.grav_sigma.push_back(gravity_sigma);
      }
    }

    if (lines > 0) {
      // The true frame: columns e0, e1, e2 = up for one degree of freedom.
      double g[3];
      jba::gravity_direction(p, g);
      double Rm[9];
      if (line_dof == 1) {
        const double up[3] = {-g[0], -g[1], -g[2]};
        double x[3] = {up[1], -up[0], 0};
        if (std::abs(up[2]) > 0.9) x[0] = 1, x[1] = 0;
        double xu = x[0] * up[0] + x[1] * up[1] + x[2] * up[2];
        for (int c = 0; c < 3; ++c) x[c] -= xu * up[c];
        const double xn = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        for (double& v : x) v /= xn;
        const double y[3] = {up[1] * x[2] - up[2] * x[1], up[2] * x[0] - up[0] * x[2], up[0] * x[1] - up[1] * x[0]};
        const double psi = 1.5 * u(rng);
        double e0[3];
        for (int c = 0; c < 3; ++c) e0[c] = std::cos(psi) * x[c] + std::sin(psi) * y[c];
        const double e1[3] = {up[1] * e0[2] - up[2] * e0[1], up[2] * e0[0] - up[0] * e0[2], up[0] * e0[1] - up[1] * e0[0]};
        for (int r = 0; r < 3; ++r) {
          Rm[3 * r] = e0[r];
          Rm[3 * r + 1] = e1[r];
          Rm[3 * r + 2] = up[r];
        }
      } else {
        double q[4];
        jba_test::random_rotation(rng, 3.0, q);
        for (int c = 0; c < 3; ++c) {
          double e[3] = {0, 0, 0}, d[3];
          e[c] = 1;
          jba::quat_rotate(q, e, d);
          for (int r = 0; r < 3; ++r) Rm[3 * r + c] = d[r];
        }
      }
      double qm[4];
      jba::quat_from_matrix(Rm, qm);
      const double* K = p.intrinsics.data();
      for (int s = 0; s < lines; ++s) {
        const int axis = static_cast<int>(rng() % 3);
        const double half = 0.2 + 0.4 * u(rng);
        double X0[3], X1[3];
        for (int c = 0; c < 3; ++c) {
          const double m = 2 * u(rng) - 1;
          X0[c] = m - half * Rm[3 * c + axis];
          X1[c] = m + half * Rm[3 * c + axis];
        }
        int label = axis;
        if (u(rng) < line_outliers) label = (axis + 1 + static_cast<int>(rng() % 2)) % 3;
        for (int i = 0; i < images; ++i) {
          const double* T = &truth[7 * static_cast<std::size_t>(i)];
          double xy[4];
          bool ok = true;
          for (int e = 0; e < 2 && ok; ++e) {
            double pc[3];
            jba::quat_rotate(T, e == 0 ? X0 : X1, pc);
            for (int c = 0; c < 3; ++c) pc[c] += T[4 + c];
            ok = pc[2] > 0.5 && jba::project(model, K, pc, &xy[2 * e], &xy[2 * e + 1]);
          }
          if (!ok) continue;
          p.line_image.push_back(i);
          for (double v : xy) p.line_xy.push_back(v + noise * n(rng));
          p.line_axis.push_back(label);
        }
      }
      // Start 0.02 rad off, about up for one degree of freedom.
      double d[3];
      if (line_dof == 1) {
        for (int c = 0; c < 3; ++c) d[c] = -0.02 * g[c];
      } else {
        for (double& v : d) v = 0.012 * n(rng);
      }
      double dq[4];
      jba::quat_exp(d, dq);
      p.manhattan_frame.resize(4);
      jba::quat_multiply(dq, qm, p.manhattan_frame.data());
      jba::quat_normalize(p.manhattan_frame.data());
      p.manhattan_dof = line_dof;
    }

    if (depth_fraction > 0) {
      // Readings from the true depth through a per-image correction error (when that
      // part is free), with noise growing with depth, and gross outliers.
      std::vector<double> a(static_cast<std::size_t>(images), 1.0), b(static_cast<std::size_t>(images), 0.0);
      for (int i = 0; i < images; ++i) {
        if (depth_free & jba::kDepthScale) a[static_cast<std::size_t>(i)] = 1 + 0.05 * n(rng);
        if (depth_free & jba::kDepthOffset) b[static_cast<std::size_t>(i)] = 0.05 * n(rng);
      }
      p.obs_depth.assign(p.num_observations(), std::nan(""));
      for (std::size_t k = 0; k < p.num_observations(); ++k) {
        if (!(u(rng) < depth_fraction)) continue;
        const auto i = static_cast<std::size_t>(p.obs_image[k]);
        const double* T = &truth[7 * i];
        double pc[3];
        jba::quat_rotate(T, &p.points[3 * static_cast<std::size_t>(p.obs_point[k])], pc);
        for (int c = 0; c < 3; ++c) pc[c] += T[4 + c];
        const double z = depth_kind == jba::DepthKind::Range ? std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]) : pc[2];
        double d = (z - b[i]) / a[i] + (0.005 + 0.005 * z) * n(rng);
        if (u(rng) < depth_outliers) d *= 1.3 + 0.5 * u(rng);
        p.obs_depth[k] = d;
      }
      p.depth_kind = depth_kind;
      p.depth_form = depth_form;
      if (depth_free != 0) p.depth_correction_free.assign(p.num_images(), static_cast<std::uint8_t>(depth_free));
    }

    if (fix_images > 0) {
      p.image_fixed.assign(p.num_images(), 0);
      for (int i = 0; i < fix_images && i < images; ++i) p.image_fixed[static_cast<std::size_t>(i)] = 1;
    }
    if (fix_points > 0) {
      p.point_fixed.assign(p.num_points(), 0);
      for (int j = 0; j < fix_points && j < points; ++j) p.point_fixed[static_cast<std::size_t>(3 * j % points)] = 1;
    }

    // Perturb what is free.
    for (std::size_t j = 0; j < p.num_points(); ++j) {
      if (!p.point_fixed.empty() && p.point_fixed[j]) continue;
      for (int a = 0; a < 3; ++a) p.points[3 * j + static_cast<std::size_t>(a)] += 0.02 * n(rng);
    }
    for (std::size_t i = 0; i < p.num_images(); ++i) {
      if (!p.image_fixed.empty() && p.image_fixed[i]) continue;
      double* pose = &p.poses[7 * i];
      const double step[6] = {0.01 * n(rng), 0.01 * n(rng), 0.01 * n(rng), 0.02 * n(rng), 0.02 * n(rng), 0.02 * n(rng)};
      double moved[7];
      jba::pose_retract(pose, step, moved);
      for (int a = 0; a < 7; ++a) pose[a] = moved[a];
    }
    const int np = jba::num_params(model);
    if (per_image) {
      const std::vector<double> block = p.intrinsics;
      p.intrinsics.clear();
      for (int i = 0; i < images; ++i) p.intrinsics.insert(p.intrinsics.end(), block.begin(), block.end());
      p.image_intrinsics.resize(static_cast<std::size_t>(images));
      for (int i = 0; i < images; ++i) p.image_intrinsics[static_cast<std::size_t>(i)] = i;
    }
    if (intr_prior[0] > 0) {
      jba::set_intrinsics_prior(p, intr_prior[0], intr_prior[1], intr_prior[2]);
      if (intr_prior_at_start) p.intrinsics_prior.clear();
    }
    if (intr_groups > 0) {
      std::vector<std::int32_t> group(p.num_intrinsics_blocks());
      for (std::size_t c = 0; c < group.size(); ++c) group[c] = static_cast<std::int32_t>(c % static_cast<std::size_t>(intr_groups));
      jba::set_intrinsics_groups(p, group, group_sigma[0], group_sigma[1], group_sigma[2], group_sigma[3], group_sigma[4],
                                 group_sigma[5]);
      if (group_at_start) p.intrinsics_group_prior.clear();
    }
    if (free_mask != 0) {
      p.intrinsics_free.assign(p.num_intrinsics_blocks(), static_cast<std::uint8_t>(free_mask));
      for (std::size_t c = 0; c < p.num_intrinsics_blocks(); ++c) {
        for (int a = 0; a < np; ++a) {
          double& v = p.intrinsics[c * static_cast<std::size_t>(np) + static_cast<std::size_t>(a)];
          const auto g = jba::intrinsics_group(model, a);
          if (!(free_mask & g)) continue;
          if (g == jba::kFocal) v *= 1 + 0.01 * n(rng);
          else if (g == jba::kPrincipalPoint) v += 2 * n(rng);
          else v += 0.005 * n(rng);
        }
      }
    }
    jba::write_problem(out, p);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_make_problem: %s\n", e.what());
    return 1;
  }
}
