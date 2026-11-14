// Level 1 against Ceres (document 05): for every camera model, residuals and
// Jacobians of the generated code agree with Ceres's automatic differentiation of
// the independent functor, mapped to the same tangent space by the manifold's
// PlusJacobian. Robust losses agree with Ceres's loss functions.

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>

#include "../check.h"
#include "../synthetic.h"
#include "ceres_reference.h"
#include "jba/cpu/priors.h"
#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "jba/loss.h"

using jba::CameraModel;

int main() {
  std::mt19937_64 rng(17);
  jba_test::Uniform u(-1, 1);
  double worst = 0;
  jba_ceres::LeftRotationVectorQuaternion manifold;

  for (CameraModel model : {CameraModel::SimplePinhole, CameraModel::Pinhole, CameraModel::SimpleRadial,
                            CameraModel::Radial, CameraModel::OpenCV, CameraModel::OpenCVFisheye}) {
    const int np = jba::num_params(model);
    for (int n = 0; n < 300; ++n) {
      double pose[7], X[3], K[8], obs[2];
      jba_test::random_rotation(rng, 3.1, pose);
      for (int a = 0; a < 3; ++a) pose[4 + a] = 2 * u(rng);
      const double z = 1 + 4 * (u(rng) + 1);
      const double pc[3] = {0.5 * z * u(rng), 0.4 * z * u(rng), z};
      jba_test::point_from_camera(pose, pc, X);
      jba_test::typical_intrinsics(model, K);
      obs[0] = 300 + 100 * u(rng);
      obs[1] = 250 + 100 * u(rng);
      const double inv_sigma = 1 / (0.5 + (u(rng) + 1));

      // Ceres: ambient Jacobians of the independent functor.
      std::unique_ptr<ceres::CostFunction> cost(jba_ceres::make_reprojection_cost(model, obs[0], obs[1], inv_sigma));
      const double* params[3] = {pose, X, K};
      double rc[2], Jpose7[14], JX[6], JK[16];
      double* jac[3] = {Jpose7, JX, JK};
      EXPECT(cost->Evaluate(params, rc, jac));
      // Tangent: rotation through the quaternion manifold, translation as is.
      double P[12];
      manifold.PlusJacobian(pose, P);
      double Jt[12];
      for (int row = 0; row < 2; ++row) {
        for (int c = 0; c < 3; ++c) {
          double s = 0;
          for (int a = 0; a < 4; ++a) s += Jpose7[7 * row + a] * P[3 * a + c];
          Jt[6 * row + c] = s;
          Jt[6 * row + 3 + c] = Jpose7[7 * row + 4 + c];
        }
      }

      // just-ba: generated code, weighted the same way.
      double r[2], Jp[12], Jx[6], Jk[16];
      jba::cpu::reprojection<double>(model, pose, pose + 4, X, K, obs, r, Jp, Jx, Jk);
      for (double& v : r) v *= inv_sigma;
      for (double& v : Jp) v *= inv_sigma;
      for (double& v : Jx) v *= inv_sigma;
      for (int a = 0; a < 2 * np; ++a) Jk[a] *= inv_sigma;

      auto cmp = [&](const double* a, const double* b, int len) {
        double scale = 1;
        for (int i = 0; i < len; ++i) scale = std::max(scale, std::abs(b[i]));
        for (int i = 0; i < len; ++i) {
          const double e = std::abs(a[i] - b[i]) / scale;
          worst = std::max(worst, e);
          EXPECT(e < 1e-12);
        }
      };
      cmp(r, rc, 2);
      cmp(Jp, Jt, 12);
      cmp(Jx, JX, 6);
      cmp(Jk, JK, 2 * np);
    }
  }
  std::printf("reprojection vs Ceres autodiff: worst relative difference %.3g\n", worst);

  // Pose priors: whitened residuals and Jacobians against the independent functors.
  // Ambient pose Jacobians (6 x 7) map to the tangent through the quaternion
  // manifold for columns 0..3 and directly for 4..6.
  auto to_tangent = [&](const double* pose, const double* J7, double* J6) {
    double P[12];
    manifold.PlusJacobian(pose, P);
    for (int row = 0; row < 6; ++row) {
      for (int c = 0; c < 3; ++c) {
        double v = 0;
        for (int a = 0; a < 4; ++a) v += J7[7 * row + a] * P[3 * a + c];
        J6[6 * row + c] = v;
        J6[6 * row + 3 + c] = J7[7 * row + 4 + c];
      }
    }
  };
  auto whiten = [](const double* W, double* r, double* J, int cols) {
    double r2[6], J2[36];
    for (int a = 0; a < 6; ++a) {
      r2[a] = 0;
      for (int c = 0; c < 6; ++c) r2[a] += W[6 * a + c] * r[c];
      for (int k = 0; k < cols; ++k) {
        J2[cols * a + k] = 0;
        for (int c = 0; c < 6; ++c) J2[cols * a + k] += W[6 * a + c] * J[cols * c + k];
      }
    }
    std::copy_n(r2, 6, r);
    std::copy_n(J2, 6 * cols, J);
  };
  double worst_prior = 0;
  auto cmp_prior = [&](const double* a, const double* b, int len) {
    double scale = 1;
    for (int i = 0; i < len; ++i) scale = std::max(scale, std::abs(b[i]));
    for (int i = 0; i < len; ++i) {
      const double e = std::abs(a[i] - b[i]) / scale;
      worst_prior = std::max(worst_prior, e);
      EXPECT(e < 1e-12);
    }
  };
  for (int n = 0; n < 300; ++n) {
    double pi[7], pj[7], prior[7], W[36];
    for (double* pose : {pi, pj, prior}) {
      jba_test::random_rotation(rng, 3.1, pose);
      for (int a = 0; a < 3; ++a) pose[4 + a] = 3 * u(rng);
    }
    for (int a = 0; a < 36; ++a) W[a] = (a % 7 == 0 ? 2.0 : 0.0) + 0.3 * u(rng);
    std::unique_ptr<ceres::CostFunction> cost(jba_ceres::make_relative_prior_cost(prior, W));
    const double* params[2] = {pi, pj};
    double rc[6], Ji7[42], Jj7[42];
    double* jac[2] = {Ji7, Jj7};
    EXPECT(cost->Evaluate(params, rc, jac));
    double Jic[36], Jjc[36];
    to_tangent(pi, Ji7, Jic);
    to_tangent(pj, Jj7, Jjc);
    double r[6], Ji[36], Jj[36];
    jba::cpu::relative_prior<double>(pi, pj, prior, r, Ji, Jj);
    double rj[6];
    std::copy_n(r, 6, rj);
    whiten(W, r, Ji, 6);
    whiten(W, rj, Jj, 6);
    cmp_prior(r, rc, 6);
    cmp_prior(Ji, Jic, 36);
    cmp_prior(Jj, Jjc, 36);
  }
  for (int n = 0; n < 300; ++n) {
    const bool identity = n % 2 == 0;
    double pose[7], T[8], qp[4], Cp[3], W[36];
    jba_test::random_rotation(rng, 3.1, pose);
    for (int a = 0; a < 3; ++a) pose[4 + a] = 3 * u(rng);
    if (identity) {
      const double id[8] = {1, 0, 0, 0, 0, 0, 0, 1};
      std::copy_n(id, 8, T);
    } else {
      jba_test::random_rotation(rng, 3.1, T);
      for (int a = 0; a < 3; ++a) T[4 + a] = 5 * u(rng);
      T[7] = std::exp(u(rng));
    }
    jba_test::random_rotation(rng, 3.1, qp);
    for (double& v : Cp) v = 3 * u(rng);
    for (int a = 0; a < 36; ++a) W[a] = (a % 7 == 0 ? 2.0 : 0.0) + 0.3 * u(rng);
    std::unique_ptr<ceres::CostFunction> cost(jba_ceres::make_global_prior_cost(qp, Cp, W, identity));
    double rc[6], Jp7[42], Jr7[42], Js1[6];
    if (identity) {
      const double* params[1] = {pose};
      double* jac[1] = {Jp7};
      EXPECT(cost->Evaluate(params, rc, jac));
    } else {
      const double* params[3] = {pose, T, T + 7};
      double* jac[3] = {Jp7, Jr7, Js1};
      EXPECT(cost->Evaluate(params, rc, jac));
    }
    double r[6], Jp[36], Jr[36], Js[6];
    jba::cpu::global_prior<double>(pose, T, qp, Cp, r, Jp, Jr, Js);
    double r1[6], r2[6];
    std::copy_n(r, 6, r1);
    std::copy_n(r, 6, r2);
    whiten(W, r, Jp, 6);
    whiten(W, r1, Jr, 6);
    whiten(W, r2, Js, 1);
    double Jpc[36];
    to_tangent(pose, Jp7, Jpc);
    cmp_prior(r, rc, 6);
    cmp_prior(Jp, Jpc, 36);
    if (!identity) {
      double Jrc[36], Jsc[6];
      to_tangent(T, Jr7, Jrc);
      for (int row = 0; row < 6; ++row) Jsc[row] = Js1[row] * T[7];  // LogScale: d s / d(ds) = s
      cmp_prior(Jr, Jrc, 36);
      cmp_prior(Js, Jsc, 6);
    }
  }
  std::printf("pose priors vs Ceres autodiff: worst relative difference %.3g\n", worst_prior);

  // Losses: same rho, rho', rho'' as Ceres's (Huber, Cauchy) and as the harness's
  // Geman-McClure.
  double worst_loss = 0;
  for (jba::LossType type : {jba::LossType::Huber, jba::LossType::Cauchy, jba::LossType::GemanMcClure}) {
    for (double scale : {0.5, 1.0, 3.0}) {
      const jba::Loss loss{type, scale};
      std::unique_ptr<ceres::LossFunction> ref(jba_ceres::make_loss(loss));
      for (double s : {0.0, 1e-6, 0.1, 0.24, 0.25, 0.26, 1.0, 8.9, 9.1, 100.0, 1e6}) {
        double a[3], b[3];
        jba::evaluate_loss(loss, s, a);
        ref->Evaluate(s, b);
        for (int i = 0; i < 3; ++i) {
          // Ceres clamps rho' away from zero; ours does not need to.
          const double e = std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i]));
          worst_loss = std::max(worst_loss, e);
          EXPECT(e < 1e-14);
        }
      }
    }
  }
  std::printf("losses vs Ceres: worst relative difference %.3g\n", worst_loss);
  return TEST_MAIN_RESULT();
}
