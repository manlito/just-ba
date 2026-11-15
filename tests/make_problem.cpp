// Writes a synthetic problem for the parity tests and the benchmark, with perturbed
// starting values.
//
//   jba_make_problem <out.jba> [--model NAME] [--images N] [--points N] [--seed N]
//                    [--noise PX] [--outliers FRACTION] [--free-intrinsics MASK]
//                    [--per-image-intrinsics] [--fix-images N] [--fix-points N] [--sigma PX]
//                    [--relative-priors N] [--global-priors] [--global-relation identity|rigid|similarity]
//                    [--prior-noise ROT,TRANS] [--prior-outliers FRACTION] [--prior-sigma ROT,TRANS]
//                    [--gravity-priors] [--gravity-noise RAD] [--gravity-outliers FRACTION] [--gravity-sigma RAD]
//
// MASK is the IntrinsicsGroup bit mask: 1 focal, 2 principal point, 4 distortion.
// Priors come from the true poses with noise: relative priors tie each image to
// its N next neighbours on the ring; global priors are in a frame related to the
// working frame by a random transform of the given kind. Outliers are gross errors
// (0.3 to 1 rad, 0.5 to 2 m). Gravity priors: every image, measured from the true
// pose with noise; gravity in the working frame is a fixed oblique direction, so no
// coordinate axis is special; outliers are 0.2 to 0.5 rad off.
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
    int images = 12, points = 300, fix_images = 0, fix_points = 0, free_mask = 0;
    std::uint64_t seed = 5;
    double noise = 0.5, outliers = 0, sigma = 0;
    bool per_image = false, global_priors = false, gravity_priors = false;
    double gravity_noise = 0.003, gravity_outliers = 0, gravity_sigma = 0;
    int relative_priors = 0;
    jba::FrameRelation relation = jba::FrameRelation::Identity;
    double prior_noise[2] = {0.002, 0.005}, prior_sigma[2] = {0, 0}, prior_outliers = 0;
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
      if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      const std::string val = argv[++a];
      if (key == "--model") {
        auto m = jba::model_from_name(val);
        if (!m) throw std::invalid_argument("unknown model " + val);
        model = *m;
      } else if (key == "--images") images = std::stoi(val);
      else if (key == "--points") points = std::stoi(val);
      else if (key == "--seed") seed = std::stoull(val);
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
      else if (key == "--global-relation") {
        if (val == "identity") relation = jba::FrameRelation::Identity;
        else if (val == "rigid") relation = jba::FrameRelation::Rigid;
        else if (val == "similarity") relation = jba::FrameRelation::Similarity;
        else throw std::invalid_argument("unknown relation " + val);
      }
      else throw std::invalid_argument("unknown option " + key);
    }

    auto p = jba_test::random_problem(model, images, points, seed, noise);
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
        for (int d = 1; d <= relative_priors && d < images; ++d) pairs.insert(pairs.end(), {i, (i + d) % images});
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
