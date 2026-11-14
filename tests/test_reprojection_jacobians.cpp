// Level 1 check for the generated reprojection residual (document 05):
// - residual values agree with a hand-written projection;
// - Jacobians agree with central finite differences (Richardson-extrapolated),
//   taken on the manifold: R <- Exp(dtheta) R, t <- t + dt;
// - optional outputs can be skipped;
// - edge cases: identity rotation, rotation near pi, point on the optical axis,
//   points very near and very far.

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "check.h"
#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "synthetic.h"

using jba::CameraModel;

namespace {

struct Case {
  double pose[7];
  double point[3];
  double intrinsics[8];
  double observed[2];
};

void residual_of(CameraModel model, const Case& c, double* r) {
  jba::cpu::reprojection<double>(model, c.pose, c.pose + 4, c.point, c.intrinsics, c.observed, r, nullptr,
                                 nullptr, nullptr);
}

// Central difference with one Richardson step: error O(h^4).
std::array<double, 2> derivative(const std::function<void(double, double*)>& f, double h) {
  auto central = [&](double step) {
    double a[2], b[2];
    f(step, a);
    f(-step, b);
    return std::array<double, 2>{(a[0] - b[0]) / (2 * step), (a[1] - b[1]) / (2 * step)};
  };
  const auto d1 = central(h), d2 = central(h / 2);
  return {(4 * d2[0] - d1[0]) / 3, (4 * d2[1] - d1[1]) / 3};
}

double worst = 0;

void check_column(const char* what, int col, const std::array<double, 2>& fd, const double* J, int stride,
                  double scale) {
  for (int row = 0; row < 2; ++row) {
    const double analytic = J[row * stride + col];
    const double err = std::abs(analytic - fd[row]) / std::max(1.0, scale);
    worst = std::max(worst, err);
    if (!(err < 1e-6)) {
      std::fprintf(stderr, "%s[%d][%d]: analytic %.12g, finite difference %.12g\n", what, row, col, analytic,
                   fd[row]);
      ++jba_test::failures;
    }
  }
}

void check_case(CameraModel model, const Case& c) {
  const int np = jba::num_params(model);
  double r[2], Jp[12], Jx[6], Jk[16];
  jba::cpu::reprojection<double>(model, c.pose, c.pose + 4, c.point, c.intrinsics, c.observed, r, Jp, Jx, Jk);

  // Residual against the independent projection.
  double pc[3];
  jba::quat_rotate(c.pose, c.point, pc);
  for (int i = 0; i < 3; ++i) pc[i] += c.pose[4 + i];
  double px, py;
  jba_test::reference_project(model, c.intrinsics, pc, &px, &py);
  const double rscale = std::max({1.0, std::abs(px), std::abs(py)});
  EXPECT_NEAR(r[0], px - c.observed[0], 1e-12 * rscale);
  EXPECT_NEAR(r[1], py - c.observed[1], 1e-12 * rscale);

  // Optional outputs skipped: same residual.
  double r_only[2];
  residual_of(model, c, r_only);
  EXPECT(r_only[0] == r[0] && r_only[1] == r[1]);

  // Scale of each Jacobian, so tolerances are relative.
  auto max_abs = [](const double* v, int n) {
    double m = 0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::abs(v[i]));
    return m;
  };
  const double sp = max_abs(Jp, 12), sx = max_abs(Jx, 6), sk = max_abs(Jk, 2 * np);

  for (int a = 0; a < 6; ++a) {
    auto f = [&](double h, double* out) {
      Case d = c;
      double step[6] = {0, 0, 0, 0, 0, 0};
      step[a] = h;
      jba::pose_retract(c.pose, step, d.pose);
      residual_of(model, d, out);
    };
    check_column("J_pose", a, derivative(f, 1e-4), Jp, 6, sp);
  }
  for (int a = 0; a < 3; ++a) {
    auto f = [&](double h, double* out) {
      Case d = c;
      d.point[a] += h;
      residual_of(model, d, out);
    };
    check_column("J_point", a, derivative(f, 1e-4 * std::max(1.0, std::abs(c.point[a]))), Jx, 3, sx);
  }
  for (int a = 0; a < np; ++a) {
    auto f = [&](double h, double* out) {
      Case d = c;
      d.intrinsics[a] += h;
      residual_of(model, d, out);
    };
    check_column("J_intrinsics", a, derivative(f, 1e-4 * std::max(1.0, std::abs(c.intrinsics[a]))), Jk, np, sk);
  }
}

// Builds a case whose point sits at camera coordinates p_cam.
Case make_case(CameraModel model, const double* q, const double* t, const double* p_cam, double noise_x,
               double noise_y) {
  Case c{};
  for (int i = 0; i < 4; ++i) c.pose[i] = q[i];
  for (int i = 0; i < 3; ++i) c.pose[4 + i] = t[i];
  jba_test::typical_intrinsics(model, c.intrinsics);
  jba_test::point_from_camera(c.pose, p_cam, c.point);
  double px, py;
  jba_test::reference_project(model, c.intrinsics, p_cam, &px, &py);
  c.observed[0] = px + noise_x;
  c.observed[1] = py + noise_y;
  return c;
}

}  // namespace

int main() {
  const CameraModel models[] = {CameraModel::SimplePinhole, CameraModel::Pinhole, CameraModel::SimpleRadial,
                                CameraModel::Radial,        CameraModel::OpenCV,  CameraModel::OpenCVFisheye};
  std::mt19937_64 rng(42);
  jba_test::Uniform u(-1, 1);
  int cases = 0;

  for (CameraModel model : models) {
    // Random cases.
    for (int n = 0; n < 200; ++n) {
      double q[4];
      jba_test::random_rotation(rng, 3.1, q);
      const double t[3] = {3 * u(rng), 3 * u(rng), 3 * u(rng)};
      const double z = 1 + 9 * (u(rng) + 1) / 2;
      const double p_cam[3] = {0.6 * z * u(rng), 0.45 * z * u(rng), z};
      check_case(model, make_case(model, q, t, p_cam, 2 * u(rng), 2 * u(rng)));
      ++cases;
    }

    // Edge cases.
    const double identity[4] = {1, 0, 0, 0};
    double near_pi[4];
    const double v_pi[3] = {0, 3.141592653589793 - 1e-6, 0};
    jba::quat_exp(v_pi, near_pi);
    const double zero[3] = {0, 0, 0};
    const double t1[3] = {0.1, -0.2, 0.3};
    const double on_axis[3] = {0, 0, 4};
    const double off_axis[3] = {1.5, -1, 3};
    const double near_cam[3] = {0.002, -0.001, 0.01};
    const double far_away[3] = {300, -200, 1000};
    const double near_axis[3] = {3e-9, -2e-9, 2};  // fisheye: just above its small-radius switch
    const double wide[3] = {2.5, 1.8, 1};          // 72 degrees off axis: fisheye territory
    for (const double* q : {identity, static_cast<const double*>(near_pi)}) {
      for (const double* pc : {on_axis, off_axis, near_cam, far_away, near_axis, wide}) {
        check_case(model, make_case(model, q, zero, pc, 0, 0));
        check_case(model, make_case(model, q, t1, pc, 0.7, -0.3));
        cases += 2;
      }
    }
  }

  // The generated code also instantiates in single precision.
  {
    const float q[4] = {1, 0, 0, 0}, t[3] = {0, 0, 0}, X[3] = {0.1f, 0.2f, 2}, k[5] = {500, 320, 240, -0.1f, 0.01f},
                o[2] = {340, 290};
    float r[2], Jp[12];
    jba::cpu::reprojection<float>(CameraModel::Radial, q, t, X, k, o, r, Jp, nullptr, nullptr);
    EXPECT(std::isfinite(r[0]) && std::isfinite(Jp[0]));
  }

  std::printf("checked %d cases, worst relative Jacobian error %.3g\n", cases, worst);
  return TEST_MAIN_RESULT();
}
