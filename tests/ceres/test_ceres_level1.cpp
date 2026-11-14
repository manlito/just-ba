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
#include "jba/cpu/reprojection.h"
#include "jba/loss.h"

using jba::CameraModel;

int main() {
  std::mt19937_64 rng(17);
  std::uniform_real_distribution<double> u(-1, 1);
  double worst = 0;
  jba_ceres::LeftRotationVectorQuaternion manifold;

  for (CameraModel model : {CameraModel::SimplePinhole, CameraModel::Pinhole, CameraModel::SimpleRadial,
                            CameraModel::Radial}) {
    const int np = jba::num_params(model);
    for (int n = 0; n < 300; ++n) {
      double pose[7], X[3], K[5], obs[2];
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
      double rc[2], Jpose7[14], JX[6], JK[10];
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
      double r[2], Jp[12], Jx[6], Jk[10];
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
