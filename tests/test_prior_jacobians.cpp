// Level 1 check for the generated pose prior residuals (document 05):
// - residual values agree with formulas written here from geometry.h helpers;
// - Jacobians agree with central finite differences (Richardson-extrapolated) on the
//   manifold: poses and the set's rigid part by R <- Exp(dtheta) R, t <- t + dt, the
//   set's scale by s <- s exp(ds);
// - a quaternion and its negative give the same residual;
// - edge cases: priors that agree exactly (zero rotation error), rotation errors
//   near pi, large translations.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

#include "check.h"
#include "jba/cpu/priors.h"
#include "jba/geometry.h"
#include "jba/priors.h"
#include "synthetic.h"

namespace {

using Vec6 = std::array<double, 6>;

// Rotation vector of a unit quaternion, either sign.
void log_quat(const double* q, double* v) {
  const double n = std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double sign = q[0] < 0 ? -1 : 1;
  const double f = n < 1e-300 ? 2 * sign : 2 * std::atan2(n, std::abs(q[0])) * sign / n;
  for (int a = 0; a < 3; ++a) v[a] = f * q[1 + a];
}

void conj(const double* q, double* c) {
  c[0] = q[0];
  c[1] = -q[1];
  c[2] = -q[2];
  c[3] = -q[3];
}

Vec6 relative_reference(const double* pi, const double* pj, const double* prior) {
  double qic[4], qji[4], Rti[3], qpc[4], qe[4];
  conj(pi, qic);
  jba::quat_multiply(pj, qic, qji);
  jba::quat_rotate(qji, pi + 4, Rti);
  conj(prior, qpc);
  jba::quat_multiply(qpc, qji, qe);
  Vec6 r;
  log_quat(qe, r.data());
  const double d[3] = {pj[4] - Rti[0] - prior[4], pj[5] - Rti[1] - prior[5], pj[6] - Rti[2] - prior[6]};
  jba::quat_rotate(qpc, d, r.data() + 3);
  return r;
}

Vec6 global_reference(const double* pose, const double* T, const double* qp, const double* Cp) {
  double q1[4], qpc[4], qe[4];
  conj(qp, qpc);
  jba::quat_multiply(pose, T, q1);
  jba::quat_multiply(q1, qpc, qe);
  Vec6 r;
  log_quat(qe, r.data());
  double qc[4], C[3];
  conj(pose, qc);
  jba::quat_rotate(qc, pose + 4, C);
  const double d[3] = {-C[0] - T[4], -C[1] - T[5], -C[2] - T[6]};
  double Tc[4], local[3];
  conj(T, Tc);
  jba::quat_rotate(Tc, d, local);
  for (int a = 0; a < 3; ++a) r[3 + a] = local[a] / T[7] - Cp[a];
  return r;
}

Vec6 derivative(const std::function<Vec6(double)>& f, double h) {
  auto central = [&](double step) {
    const Vec6 a = f(step), b = f(-step);
    Vec6 d;
    for (int i = 0; i < 6; ++i) d[i] = (a[i] - b[i]) / (2 * step);
    return d;
  };
  const Vec6 d1 = central(h), d2 = central(h / 2);
  Vec6 out;
  for (int i = 0; i < 6; ++i) out[i] = (4 * d2[i] - d1[i]) / 3;
  return out;
}

double worst = 0, worst_value = 0;

void check_column(const char* what, int col, const Vec6& fd, const double* J, int stride) {
  double scale = 1;
  for (int row = 0; row < 6; ++row) scale = std::max(scale, std::abs(J[row * stride + col]));
  for (int row = 0; row < 6; ++row) {
    const double err = std::abs(J[row * stride + col] - fd[row]) / scale;
    worst = std::max(worst, err);
    if (!(err < 1e-6)) {
      std::fprintf(stderr, "%s[%d][%d]: analytic %.12g, finite difference %.12g\n", what, row, col,
                   J[row * stride + col], fd[row]);
      ++jba_test::failures;
    }
  }
}

void expect_close(const double* a, const Vec6& b) {
  double scale = 1;
  for (double v : b) scale = std::max(scale, std::abs(v));
  for (int i = 0; i < 6; ++i) {
    const double e = std::abs(a[i] - b[i]) / scale;
    worst_value = std::max(worst_value, e);
    EXPECT(e < 1e-12);
  }
}

Vec6 as_vec(const double* r) {
  Vec6 v;
  std::copy_n(r, 6, v.begin());
  return v;
}

void check_relative(const double* pi, const double* pj, const double* prior) {
  double r[6], Ji[36], Jj[36];
  jba::cpu::relative_prior<double>(pi, pj, prior, r, Ji, Jj);
  expect_close(r, relative_reference(pi, pj, prior));
  double r2[6];
  jba::cpu::relative_prior<double>(pi, pj, prior, r2, nullptr, nullptr);
  for (int a = 0; a < 6; ++a) EXPECT(r2[a] == r[a]);
  // Double cover.
  double ni[7], nprior[7];
  std::copy_n(pi, 7, ni);
  std::copy_n(prior, 7, nprior);
  for (int a = 0; a < 4; ++a) ni[a] = -ni[a], nprior[a] = -nprior[a];
  jba::cpu::relative_prior<double>(ni, pj, nprior, r2, nullptr, nullptr);
  expect_close(r2, as_vec(r));

  for (int which = 0; which < 2; ++which) {
    for (int a = 0; a < 6; ++a) {
      auto f = [&](double h) {
        double step[6] = {0, 0, 0, 0, 0, 0}, moved[7], out[6];
        step[a] = h;
        jba::pose_retract(which == 0 ? pi : pj, step, moved);
        jba::cpu::relative_prior<double>(which == 0 ? moved : pi, which == 0 ? pj : moved, prior, out, nullptr,
                                         nullptr);
        return as_vec(out);
      };
      check_column(which == 0 ? "J_i" : "J_j", a, derivative(f, 1e-5), which == 0 ? Ji : Jj, 6);
    }
  }
}

void check_global(const double* pose, const double* T, const double* qp, const double* Cp) {
  double r[6], Jp[36], Jr[36], Js[6];
  jba::cpu::global_prior<double>(pose, T, qp, Cp, r, Jp, Jr, Js);
  expect_close(r, global_reference(pose, T, qp, Cp));
  double npose[7], nqp[4], r2[6];
  std::copy_n(pose, 7, npose);
  for (int a = 0; a < 4; ++a) npose[a] = -npose[a], nqp[a] = -qp[a];
  jba::cpu::global_prior<double>(npose, T, nqp, Cp, r2, nullptr, nullptr, nullptr);
  expect_close(r2, as_vec(r));

  for (int a = 0; a < 6; ++a) {
    auto fp = [&](double h) {
      double step[6] = {0, 0, 0, 0, 0, 0}, moved[7], out[6];
      step[a] = h;
      jba::pose_retract(pose, step, moved);
      jba::cpu::global_prior<double>(moved, T, qp, Cp, out, nullptr, nullptr, nullptr);
      return as_vec(out);
    };
    check_column("J_pose", a, derivative(fp, 1e-5), Jp, 6);
    auto fr = [&](double h) {
      double step[6] = {0, 0, 0, 0, 0, 0}, moved[8], out[6];
      step[a] = h;
      jba::pose_retract(T, step, moved);
      moved[7] = T[7];
      jba::cpu::global_prior<double>(pose, moved, qp, Cp, out, nullptr, nullptr, nullptr);
      return as_vec(out);
    };
    check_column("J_rigid", a, derivative(fr, 1e-5), Jr, 6);
  }
  auto fs = [&](double h) {
    double moved[8], out[6];
    std::copy_n(T, 8, moved);
    moved[7] = T[7] * std::exp(h);
    jba::cpu::global_prior<double>(pose, moved, qp, Cp, out, nullptr, nullptr, nullptr);
    return as_vec(out);
  };
  check_column("J_scale", 0, derivative(fs, 1e-5), Js, 1);
}

void random_pose(std::mt19937_64& rng, double max_angle, double trans, double* pose) {
  jba_test::Uniform u(-1, 1);
  jba_test::random_rotation(rng, max_angle, pose);
  for (int a = 0; a < 3; ++a) pose[4 + a] = trans * u(rng);
}

}  // namespace

int main() {
  std::mt19937_64 rng(7);
  jba_test::Uniform u(-1, 1);
  int cases = 0;

  // Relative priors: random poses, priors near the true motion and far from it.
  for (int n = 0; n < 300; ++n) {
    double pi[7], pj[7], prior[7], noise[7];
    random_pose(rng, 3.1, 5, pi);
    random_pose(rng, 3.1, 5, pj);
    const double far = n % 3 == 0 ? 3.0 : 0.05;
    random_pose(rng, far, far, noise);
    // prior = noise * (T_j T_i^-1)
    double qic[4], qji[4], Rti[3];
    conj(pi, qic);
    jba::quat_multiply(pj, qic, qji);
    jba::quat_rotate(qji, pi + 4, Rti);
    jba::quat_multiply(noise, qji, prior);
    jba::quat_normalize(prior);
    for (int a = 0; a < 3; ++a) prior[4 + a] = pj[4 + a] - Rti[a] + noise[4 + a];
    check_relative(pi, pj, prior);
    ++cases;
  }
  // Exact agreement: zero rotation error takes the small-angle branch.
  {
    double pi[7], pj[7], prior[7];
    random_pose(rng, 1, 2, pi);
    random_pose(rng, 1, 2, pj);
    std::vector<double> poses(pi, pi + 7);
    poses.insert(poses.end(), pj, pj + 7);
    jba::Problem p;
    jba::add_relative_priors_from_poses(p, poses, {0, 1});
    std::copy_n(p.rel_pose.data(), 7, prior);
    double r[6];
    jba::cpu::relative_prior<double>(pi, pj, prior, r, nullptr, nullptr);
    for (double v : r) EXPECT(std::abs(v) < 1e-14);
    check_relative(pi, pj, prior);
    ++cases;
  }

  // Global priors: identity sets, rigid sets, similarity sets.
  for (int n = 0; n < 300; ++n) {
    double pose[7], T[8], qp[4], Cp[3];
    random_pose(rng, 3.1, 5, pose);
    const int kind = n % 3;
    if (kind == 0) {
      const double id[8] = {1, 0, 0, 0, 0, 0, 0, 1};
      std::copy_n(id, 8, T);
    } else {
      random_pose(rng, 3.1, 10, T);
      T[7] = kind == 1 ? 1.0 : std::exp(1.5 * u(rng));
    }
    jba_test::random_rotation(rng, 3.1, qp);
    for (double& v : Cp) v = 4 * u(rng);
    check_global(pose, T, qp, Cp);
    ++cases;
  }
  // Rotation error near pi.
  {
    double pose[7] = {1, 0, 0, 0, 0.3, -0.2, 1}, T[8] = {1, 0, 0, 0, 0, 0, 0, 1}, qp[4], Cp[3] = {0, 0, 0};
    const double v[3] = {0, 0, 3.141592653589793 - 1e-4};
    jba::quat_exp(v, qp);
    check_global(pose, T, qp, Cp);
    ++cases;
  }

  // Single precision instantiates.
  {
    const float pi[7] = {1, 0, 0, 0, 0, 0, 0}, pj[7] = {1, 0, 0, 0, 1, 0, 0}, prior[7] = {1, 0, 0, 0, 0.9f, 0, 0};
    float r[6], Ji[36];
    jba::cpu::relative_prior<float>(pi, pj, prior, r, Ji, nullptr);
    EXPECT(std::isfinite(r[3]) && std::isfinite(Ji[0]));
  }

  std::printf("checked %d prior cases, worst residual difference %.3g, worst relative Jacobian error %.3g\n", cases,
              worst_value, worst);
  return TEST_MAIN_RESULT();
}
