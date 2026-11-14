#pragma once

// Small random problems for tests. Not the synthetic generator of document 05
// (milestone 2); just enough geometry to exercise residuals and file formats.

#include <cmath>
#include <random>

#include "jba/camera_model.h"
#include "jba/geometry.h"
#include "jba/problem.h"

namespace jba_test {

// Random numbers that are the same on every platform. std::mt19937_64 is fully
// specified, but std::uniform_real_distribution and std::normal_distribution are
// not: libc++ and libstdc++ give different values. These use only the raw engine
// output and arithmetic.
struct Uniform {
  double lo, hi;
  Uniform(double a = 0, double b = 1) : lo(a), hi(b) {}
  double operator()(std::mt19937_64& g) const {
    return lo + (hi - lo) * static_cast<double>(g() >> 11) * 0x1.0p-53;
  }
};

// Approximately normal: the sum of twelve uniforms minus six (Irwin-Hall). Plenty
// for test noise, and free of transcendental functions.
struct Normal {
  double mean, sigma;
  Normal(double m = 0, double s = 1) : mean(m), sigma(s) {}
  double operator()(std::mt19937_64& g) const {
    double z = -6;
    for (int i = 0; i < 12; ++i) z += static_cast<double>(g() >> 11) * 0x1.0p-53;
    return mean + sigma * z;
  }
};

inline void typical_intrinsics(jba::CameraModel model, double* k) {
  using M = jba::CameraModel;
  switch (model) {
    case M::SimplePinhole: k[0] = 520; k[1] = 320; k[2] = 240; break;
    case M::Pinhole: k[0] = 515; k[1] = 525; k[2] = 318; k[3] = 243; break;
    case M::SimpleRadial: k[0] = 520; k[1] = 320; k[2] = 240; k[3] = -0.08; break;
    case M::Radial: k[0] = 520; k[1] = 320; k[2] = 240; k[3] = -0.08; k[4] = 0.01; break;
    default: break;
  }
}

// Random unit quaternion with rotation angle up to max_angle.
inline void random_rotation(std::mt19937_64& rng, double max_angle, double* q) {
  jba_test::Normal n(0, 1);
  jba_test::Uniform u(0, max_angle);
  double axis[3] = {n(rng), n(rng), n(rng)};
  const double len = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  const double angle = u(rng);
  double v[3] = {axis[0] / len * angle, axis[1] / len * angle, axis[2] / len * angle};
  jba::quat_exp(v, q);
}

// World point that lands at camera coordinates p_cam for the given pose.
inline void point_from_camera(const double* pose, const double* p_cam, double* X) {
  const double conj[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  const double d[3] = {p_cam[0] - pose[4], p_cam[1] - pose[5], p_cam[2] - pose[6]};
  jba::quat_rotate(conj, d, X);
}

// Hand-written projection, independent of the generated code. COLMAP formulas.
inline void reference_project(jba::CameraModel model, const double* k, const double* pc, double* px,
                              double* py) {
  const double x = pc[0] / pc[2], y = pc[1] / pc[2];
  switch (model) {
    case jba::CameraModel::SimplePinhole: *px = k[0] * x + k[1]; *py = k[0] * y + k[2]; break;
    case jba::CameraModel::Pinhole: *px = k[0] * x + k[2]; *py = k[1] * y + k[3]; break;
    case jba::CameraModel::SimpleRadial: {
      const double rad = 1 + k[3] * (x * x + y * y);
      *px = k[0] * x * rad + k[1]; *py = k[0] * y * rad + k[2]; break;
    }
    case jba::CameraModel::Radial: {
      const double r2 = x * x + y * y, rad = 1 + k[3] * r2 + k[4] * r2 * r2;
      *px = k[0] * x * rad + k[1]; *py = k[0] * y * rad + k[2]; break;
    }
    default: *px = *py = 0; break;
  }
}

// A scene of `images` cameras around the origin looking roughly at a cloud of
// `points` points, each point seen by every camera with pixel noise.
inline jba::Problem random_problem(jba::CameraModel model, int images, int points, std::uint64_t seed,
                                   double pixel_noise = 0.5) {
  std::mt19937_64 rng(seed);
  jba_test::Uniform u(-1, 1);
  jba_test::Normal noise(0, pixel_noise);
  jba::Problem p;
  p.model = model;
  p.intrinsics.resize(static_cast<std::size_t>(jba::num_params(model)));
  typical_intrinsics(model, p.intrinsics.data());
  for (int i = 0; i < images; ++i) {
    double q[4];
    random_rotation(rng, 0.3, q);
    // Camera centre on a ring at distance 6, then t = -R C.
    const double a = 2 * 3.141592653589793 * i / images;
    const double C[3] = {6 * std::sin(a) * 0.3, 0.2 * u(rng), -6 + 0.3 * u(rng)};
    double RC[3];
    jba::quat_rotate(q, C, RC);
    p.poses.insert(p.poses.end(), {q[0], q[1], q[2], q[3], -RC[0], -RC[1], -RC[2]});
  }
  for (int j = 0; j < points; ++j) p.points.insert(p.points.end(), {u(rng), u(rng), u(rng)});
  for (int i = 0; i < images; ++i) {
    const double* pose = &p.poses[7 * static_cast<std::size_t>(i)];
    for (int j = 0; j < points; ++j) {
      const double* X = &p.points[3 * static_cast<std::size_t>(j)];
      double pc[3];
      jba::quat_rotate(pose, X, pc);
      for (int c = 0; c < 3; ++c) pc[c] += pose[4 + c];
      if (pc[2] < 0.5) continue;
      double px, py;
      reference_project(model, p.intrinsics.data(), pc, &px, &py);
      p.obs_image.push_back(i);
      p.obs_point.push_back(j);
      p.obs_xy.push_back(px + noise(rng));
      p.obs_xy.push_back(py + noise(rng));
    }
  }
  return p;
}

}  // namespace jba_test
