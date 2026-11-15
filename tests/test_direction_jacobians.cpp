// Level 1 check for the generated direction and depth residuals (document 05): the
// gravity prior, the Manhattan line residual and the depth term.
// - residual values agree with formulas written here from geometry.h helpers;
// - Jacobians agree with central finite differences on the manifold: poses by
//   R <- Exp(dtheta) R, t <- t + dt; the Manhattan frame by R_M <- Exp(dphi) R_M;
// - edge cases: exact agreement, gravity measured upside down.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "fd.h"
#include "jba/cpu/depth.h"
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

  // Manhattan line: pose and frame columns.
  for (int trial = 0; trial < 200; ++trial) {
    double pose[7], frame[4], n[3], axis[3] = {0, 0, 0};
    random_pose(rng, pose);
    jba_test::random_rotation(rng, 3.0, frame);
    random_unit(rng, n);
    axis[trial % 3] = 1;
    double r, Jp[6], Jm[3];
    jba::cpu::manhattan_line<double>(pose, n, frame, axis, &r, Jp, Jm);
    auto ref = [&](const double* q, const double* f) {
      double d[3], Rd[3];
      jba::quat_rotate(f, axis, d);
      jba::quat_rotate(q, d, Rd);
      return std::vector<double>{n[0] * Rd[0] + n[1] * Rd[1] + n[2] * Rd[2]};
    };
    EXPECT(std::abs(r - ref(pose, frame)[0]) < 1e-14);
    for (int c = 0; c < 6; ++c) {
      const auto fd = jba_test::fd_derivative([&](double h) {
        double step[6] = {0, 0, 0, 0, 0, 0}, moved[7];
        step[c] = h;
        jba::pose_retract(pose, step, moved);
        return ref(moved, frame);
      });
      jba_test::check_fd_column(stats, "line J_pose", c, fd, Jp, 6);
    }
    for (int c = 0; c < 3; ++c) {
      const auto fd = jba_test::fd_derivative([&](double h) {
        double v[3] = {0, 0, 0}, dq[4], moved[4];
        v[c] = h;
        jba::quat_exp(v, dq);
        jba::quat_multiply(dq, frame, moved);
        return ref(pose, moved);
      });
      jba_test::check_fd_column(stats, "line J_frame", c, fd, Jm, 3);
    }
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
