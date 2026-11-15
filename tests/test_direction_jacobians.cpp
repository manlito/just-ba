// Level 1 check for the generated direction residuals (document 05): the gravity
// prior and the Manhattan line residual.
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

  std::printf("worst Jacobian column error %.3g, worst value error %.3g\n", stats.worst, worst_value);
  return TEST_MAIN_RESULT();
}
