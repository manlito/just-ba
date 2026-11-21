// Level 1 check for the generated direction and depth residuals (document 05): the
// gravity prior, the Manhattan line residual and the depth term.
// - residual values agree with formulas written here from geometry.h helpers;
// - Jacobians agree with central finite differences on the manifold: poses by
//   R <- Exp(dtheta) R, t <- t + dt; the Manhattan frame by R_M <- Exp(dphi) R_M;
// - edge cases: exact agreement, gravity measured upside down.
// The Manhattan line residual (D163) is checked per camera model: its value against
// a brute-force minimum over the circle of the endpoints' pixel distances to the
// circle's image (a curve under distortion), and its intrinsics columns against
// finite differences that undistort the endpoints again at each step.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "fd.h"
#include "jba/cpu/depth.h"
#include "jba/camera_model.h"
#include "jba/cpu/priors.h"
#include "jba/geometry.h"
#include "synthetic.h"

namespace {

jba_test::FdStats stats;
double worst_value = 0;

void random_unit(std::mt19937_64& rng, double* v) {
  jba_test::Normal n(0, 1);
  double len = 0;
  for (int a = 0; a < 3; ++a) {
    v[a] = n(rng);
    len += v[a] * v[a];
  }
  for (int a = 0; a < 3; ++a) v[a] /= std::sqrt(len);
}

void random_pose(std::mt19937_64& rng, double* pose) {
  jba_test::random_rotation(rng, 3.0, pose);
  jba_test::Normal n(0, 2);
  for (int a = 4; a < 7; ++a) pose[a] = n(rng);
}

std::vector<double> gravity_reference(const double* pose, const double* gw, const double* gc) {
  double Rg[3];
  jba::quat_rotate(pose, gw, Rg);
  return {Rg[0] - gc[0], Rg[1] - gc[1], Rg[2] - gc[2]};
}

void check_gravity(const double* pose, const double* gw, const double* gc) {
  double r[3], J[18];
  jba::cpu::gravity_prior<double>(pose, gw, gc, r, J);
  const auto ref = gravity_reference(pose, gw, gc);
  for (int a = 0; a < 3; ++a) {
    worst_value = std::max(worst_value, std::abs(r[a] - ref[static_cast<std::size_t>(a)]));
    EXPECT(std::abs(r[a] - ref[static_cast<std::size_t>(a)]) < 1e-14);
  }
  for (int c = 0; c < 6; ++c) {
    const auto fd = jba_test::fd_derivative([&](double h) {
      double step[6] = {0, 0, 0, 0, 0, 0}, moved[7];
      step[c] = h;
      jba::pose_retract(pose, step, moved);
      return gravity_reference(moved, gw, gc);
    });
    jba_test::check_fd_column(stats, "gravity J_pose", c, fd, J, 6);
  }
}


// Intrinsics for a 640 x 480 image, with distortion where the model has some.
std::vector<double> test_intrinsics(jba::CameraModel m, int trial) {
  const double f = 450 + 10 * (trial % 7), cx = 318 + trial % 5, cy = 243 - trial % 3;
  switch (m) {
    case jba::CameraModel::SimplePinhole: return {f, cx, cy};
    case jba::CameraModel::Pinhole: return {f, 1.02 * f, cx, cy};
    case jba::CameraModel::SimpleRadial: return {f, cx, cy, -0.12};
    case jba::CameraModel::Radial: return {f, cx, cy, -0.2, 0.05};
    case jba::CameraModel::OpenCV: return {f, 1.01 * f, cx, cy, -0.25, 0.08, 0.002, -0.001};
    case jba::CameraModel::OpenCVFisheye: return {0.7 * f, 0.7 * f, cx, cy, 0.05, -0.02, 0.01, -0.003};
  }
  return {};
}

double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
void cross3(const double* a, const double* b, double* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
void normalize3(double* a) {
  const double n = std::sqrt(dot3(a, a));
  for (int c = 0; c < 3; ++c) a[c] /= n;
}

// Golden-section minimum of f on [a, b].
template <typename F>
double golden_min(F f, double a, double b, int iters = 80) {
  const double g = 0.5 * (std::sqrt(5.0) - 1);
  double c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  for (int it = 0; it < iters; ++it) {
    if (fc < fd) {
      b = d, d = c, fd = fc, c = b - g * (b - a), fc = f(c);
    } else {
      a = c, c = d, fc = fd, d = a + g * (b - a), fd = f(d);
    }
  }
  return std::min(fc, fd);
}

// Brute force: the smallest sum of squared pixel distances of the two endpoints to
// the image of a great circle through d, over the circle; its square root.
double line_brute_force(jba::CameraModel m, const double* K, const double* d, const double* xy) {
  double b[2][3];
  for (int j = 0; j < 2; ++j) jba::unproject(m, K, xy[2 * j], xy[2 * j + 1], b[j]);
  double e1[3], e2[3], n0[3];
  cross3(b[0], b[1], n0);  // the segment's own circle, near the best one
  // Basis of the normals to d: e1 the projection of n0, e2 = d x e1.
  const double nd = dot3(n0, d);
  for (int c = 0; c < 3; ++c) e1[c] = n0[c] - nd * d[c];
  normalize3(e1);
  cross3(d, e1, e2);
  auto dist2 = [&](const double* mvec, int j) {
    // Points of the circle near b_j: c(t) = cos t c0 + sin t (m x c0).
    double c0[3], w[3];
    const double mb = dot3(mvec, b[j]);
    for (int c = 0; c < 3; ++c) c0[c] = b[j][c] - mb * mvec[c];
    normalize3(c0);
    cross3(mvec, c0, w);
    return golden_min([&](double t) {
      double p[3], x, y;
      for (int c = 0; c < 3; ++c) p[c] = std::cos(t) * c0[c] + std::sin(t) * w[c];
      if (!jba::project(m, K, p, &x, &y)) return 1e30;
      return (x - xy[2 * j]) * (x - xy[2 * j]) + (y - xy[2 * j + 1]) * (y - xy[2 * j + 1]);
    }, -0.2, 0.2, 120);
  };
  const double best = golden_min([&](double phi) {
    double mvec[3];
    for (int c = 0; c < 3; ++c) mvec[c] = std::cos(phi) * e1[c] + std::sin(phi) * e2[c];
    return dist2(mvec, 0) + dist2(mvec, 1);
  }, -0.5, 0.5, 120);
  return std::sqrt(best);
}
}  // namespace

int main() {
  std::mt19937_64 rng(31);

  // Gravity: random poses and directions; measured near the truth and far from it.
  for (int trial = 0; trial < 200; ++trial) {
    double pose[7], gw[3], gc[3];
    random_pose(rng, pose);
    random_unit(rng, gw);
    if (trial % 2 == 0) {
      random_unit(rng, gc);
    } else {
      jba::quat_rotate(pose, gw, gc);
      gc[0] += 0.01;
      const double n = std::sqrt(gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]);
      for (double& v : gc) v /= n;
    }
    check_gravity(pose, gw, gc);
  }
  {
    // Exact agreement gives zero; upside down gives |r| = 2, not 0.
    double pose[7], gw[3] = {0, -1, 0}, gc[3], r[3];
    random_pose(rng, pose);
    jba::quat_rotate(pose, gw, gc);
    jba::cpu::gravity_prior<double>(pose, gw, gc, r, static_cast<double*>(nullptr));
    for (double v : r) EXPECT(std::abs(v) < 1e-15);
    for (double& v : gc) v = -v;
    jba::cpu::gravity_prior<double>(pose, gw, gc, r, static_cast<double*>(nullptr));
    EXPECT_NEAR(std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), 2.0, 1e-14);
  }

  // Manhattan line (D163), every camera model: value against the brute force, and
  // pose, frame and intrinsics columns.
  constexpr jba::CameraModel kModels[] = {jba::CameraModel::SimplePinhole, jba::CameraModel::Pinhole,
                                          jba::CameraModel::SimpleRadial,  jba::CameraModel::Radial,
                                          jba::CameraModel::OpenCV,        jba::CameraModel::OpenCVFisheye};
  double worst_brute = 0;
  for (const jba::CameraModel m : kModels) {
    const int np = jba::num_params(m);
    int done = 0;
    for (int trial = 0; done < 40 && trial < 400; ++trial) {
      std::vector<double> K = test_intrinsics(m, trial);
      double pose[7], frame[4], axis[3] = {0, 0, 0};
      random_pose(rng, pose);
      jba_test::random_rotation(rng, 3.0, frame);
      axis[trial % 3] = 1;
      // A 3D segment along the axis, in front of the camera, nudged off it by up to
      // 3 pixels at each end.
      double dm[3], d[3];
      jba::quat_rotate(frame, axis, dm);
      jba::quat_rotate(pose, dm, d);
      jba_test::Normal nrm(0, 1);
      const double X0[3] = {0.6 * nrm(rng), 0.45 * nrm(rng), 3.0 + 0.5 * nrm(rng)};
      const double half = 0.3 + 0.4 * std::abs(nrm(rng));
      double xy[4];
      bool ok = true;
      for (int j = 0; j < 2 && ok; ++j) {
        const double sgn = j ? 1 : -1, X[3] = {X0[0] + sgn * half * d[0], X0[1] + sgn * half * d[1], X0[2] + sgn * half * d[2]};
        ok = X[2] > 0.5 && jba::project(m, K.data(), X, &xy[2 * j], &xy[2 * j + 1]) && xy[2 * j] > 20 &&
             xy[2 * j] < 620 && xy[2 * j + 1] > 20 && xy[2 * j + 1] < 460;
      }
      if (!ok || std::hypot(xy[2] - xy[0], xy[3] - xy[1]) < 40) continue;
      for (double& v : xy) v += 1.5 * nrm(rng);
      double uv[4];
      if (!jba::undistort(m, K.data(), xy[0], xy[1], uv) || !jba::undistort(m, K.data(), xy[2], xy[3], uv + 2)) continue;
      ++done;
      double r, Jp[6], Jm[3], Jk[8];
      jba::cpu::manhattan_line<double>(m, pose, frame, axis, uv, uv + 2, K.data(), &r, Jp, Jm, Jk);
      // The value: within 0.2 % (or 0.002 px) of the brute force; 0.099 % at worst.
      const double bf = line_brute_force(m, K.data(), d, xy);
      const double err = std::abs(std::abs(r) - bf) / std::max(bf, 1.0);
      worst_brute = std::max(worst_brute, err);
      if (!(err < 0.002)) {
        std::fprintf(stderr, "line value (%s): residual %.9g, brute force %.9g\n", std::string(jba::model_name(m)).c_str(), r, bf);
        ++jba_test::failures;
      }
      auto ref = [&](const double* q, const double* f, const double* k) {
        double u[4], res = 0;
        if (!jba::undistort(m, k, xy[0], xy[1], u) || !jba::undistort(m, k, xy[2], xy[3], u + 2)) return std::vector<double>{NAN};
        double* none = nullptr;
        jba::cpu::manhattan_line<double>(m, q, f, axis, u, u + 2, k, &res, none, none, none);
        return std::vector<double>{res};
      };
      for (int c = 0; c < 6; ++c) {
        const auto fd = jba_test::fd_derivative([&](double h) {
          double step[6] = {0, 0, 0, 0, 0, 0}, moved[7];
          step[c] = h;
          jba::pose_retract(pose, step, moved);
          return ref(moved, frame, K.data());
        });
        jba_test::check_fd_column(stats, "line J_pose", c, fd, Jp, 6);
      }
      for (int c = 0; c < 3; ++c) {
        const auto fd = jba_test::fd_derivative([&](double h) {
          double v[3] = {0, 0, 0}, dq[4], moved[4];
          v[c] = h;
          jba::quat_exp(v, dq);
          jba::quat_multiply(dq, frame, moved);
          return ref(pose, moved, K.data());
        });
        jba_test::check_fd_column(stats, "line J_frame", c, fd, Jm, 3);
      }
      for (int c = 0; c < np; ++c) {
        const auto fd = jba_test::fd_derivative([&](double h) {
          std::vector<double> k = K;
          k[static_cast<std::size_t>(c)] += h;
          return ref(pose, frame, k.data());
        });
        const double Jc = Jk[c];
        jba_test::check_fd_column(stats, "line J_intrinsics", 0, fd, &Jc, 1);
      }
    }
    EXPECT(done == 40);
  }
  std::printf("line residual against the brute force: worst relative error %.3g\n", worst_brute);
  {
    // An endpoint exactly at the vanishing point: zero, with zero Jacobians, not NaN.
    const std::vector<double> K = test_intrinsics(jba::CameraModel::Pinhole, 0);
    const double pose[7] = {1, 0, 0, 0, 0, 0, 0}, frame[4] = {1, 0, 0, 0}, axis[3] = {0, 0, 1};
    const double uv[4] = {0, 0, 0.1, 0.05};
    double r, Jp[6], Jm[3], Jk[4];
    jba::cpu::manhattan_line<double>(jba::CameraModel::Pinhole, pose, frame, axis, uv, uv + 2, K.data(), &r, Jp, Jm, Jk);
    EXPECT(r == 0);
    for (double v : Jp) EXPECT(v == 0);
    for (double v : Jk) EXPECT(v == 0);
  }

  // Depth: every kind and form; pose, point and correction columns.
  for (int trial = 0; trial < 200; ++trial) {
    const auto kind = trial % 2 ? jba::DepthKind::Range : jba::DepthKind::Z;
    const auto form = (trial / 2) % 2 ? jba::DepthForm::Inverse : jba::DepthForm::Linear;
    double pose[7], X[3], corr[2] = {0.9 + 0.2 * (trial % 7) / 7.0, 0.1 - 0.03 * (trial % 5)};
    random_pose(rng, pose);
    const double pc[3] = {0.3 * (trial % 5 - 2), 0.2 * (trial % 3 - 1), 1.0 + 0.04 * trial};
    double v[3] = {pc[0] - pose[4], pc[1] - pose[5], pc[2] - pose[6]};
    const double qc[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
    jba::quat_rotate(qc, v, X);
    const double reading = 1.5 + 0.03 * trial;
    double r, Jp[6], Jx[3], Jc[2];
    jba::cpu::depth_residual<double>(kind, form, pose, X, reading, corr, &r, Jp, Jx, Jc);
    auto ref = [&](const double* q, const double* P, const double* c) {
      double p[3];
      jba::quat_rotate(q, P, p);
      for (int a = 0; a < 3; ++a) p[a] += q[4 + a];
      const double z = kind == jba::DepthKind::Range ? std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) : p[2];
      return std::vector<double>{form == jba::DepthForm::Inverse ? c[0] / (z - c[1]) - 1 / reading
                                                                 : (z - c[1]) / c[0] - reading};
    };
    EXPECT(std::abs(r - ref(pose, X, corr)[0]) < 1e-13);
    for (int c = 0; c < 6; ++c) {
      const auto fd = jba_test::fd_derivative([&](double h) {
        double step[6] = {0, 0, 0, 0, 0, 0}, moved[7];
        step[c] = h;
        jba::pose_retract(pose, step, moved);
        return ref(moved, X, corr);
      });
      jba_test::check_fd_column(stats, "depth J_pose", c, fd, Jp, 6);
    }
    for (int c = 0; c < 3; ++c) {
      const auto fd = jba_test::fd_derivative([&](double h) {
        double Y[3] = {X[0], X[1], X[2]};
        Y[c] += h;
        return ref(pose, Y, corr);
      });
      jba_test::check_fd_column(stats, "depth J_point", c, fd, Jx, 3);
    }
    for (int c = 0; c < 2; ++c) {
      const auto fd = jba_test::fd_derivative([&](double h) {
        double k[2] = {corr[0], corr[1]};
        k[c] += h;
        return ref(pose, X, k);
      });
      jba_test::check_fd_column(stats, "depth J_correction", c, fd, Jc, 2);
    }
  }

  std::printf("worst Jacobian column error %.3g, worst value error %.3g\n", stats.worst, worst_value);
  return TEST_MAIN_RESULT();
}
