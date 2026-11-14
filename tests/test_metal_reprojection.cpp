// Metal spike check: the generated Metal kernels give the same residuals and
// Jacobians as the generated CPU code.
//
// Two comparisons per model:
// - against the CPU code instantiated in float: same arithmetic, so differences come
//   only from operation order and fused multiply-add;
// - against the CPU code in double: the single-precision error itself.
// Residuals are compared in absolute pixels, the rest relative to the largest
// entry. Each model runs twice: raw pixels, and pixels relative to the principal
// point (part of the conditioning of document 03), to measure what that buys.
// Measured on an M5 Pro: it buys little (about 1.0e-4 px down to 0.8e-4 px), because
// the floor is float rounding of pixel values of a few hundred, not the offset.
// Skips (exit 77) when no Metal device is present.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "jba/cpu/reprojection.h"
#include "jba/metal/reprojection.h"
#include "synthetic.h"

using jba::CameraModel;

namespace {

struct Errors {
  double vs_float = 0;
  double vs_double = 0;
};

// Error relative to the largest entry, or absolute when `absolute` is set.
void compare(const float* gpu, const float* cpu_f, const double* cpu_d, int n, Errors& e, bool absolute = false) {
  double scale = 1;
  if (!absolute) for (int i = 0; i < n; ++i) scale = std::max(scale, std::abs(cpu_d[i]));
  for (int i = 0; i < n; ++i) {
    e.vs_float = std::max(e.vs_float, std::abs(double(gpu[i]) - double(cpu_f[i])) / scale);
    e.vs_double = std::max(e.vs_double, std::abs(double(gpu[i]) - cpu_d[i]) / scale);
  }
}

}  // namespace

int main() {
  if (!jba::metal::available()) {
    std::printf("no Metal device; skipped\n");
    return 77;
  }
  std::printf("device: %s\n", jba::metal::device_name().c_str());

  for (CameraModel model : {CameraModel::SimplePinhole, CameraModel::Pinhole, CameraModel::SimpleRadial,
                            CameraModel::Radial})
  for (bool conditioned : {false, true}) {
    auto p = jba_test::random_problem(model, 40, 2500, 11 + static_cast<int>(model));
    if (conditioned) {
      // Principal point to the origin: shift observations by (cx, cy).
      const int cx = model == CameraModel::Pinhole ? 2 : 1;
      for (std::size_t k = 0; k < p.num_observations(); ++k) {
        p.obs_xy[2 * k] -= p.intrinsics[cx];
        p.obs_xy[2 * k + 1] -= p.intrinsics[cx + 1];
      }
      p.intrinsics[cx] = p.intrinsics[cx + 1] = 0;
    }
    const auto gpu = jba::metal::evaluate_reprojection(p);
    const int np = jba::num_params(model);
    const std::size_t n = p.num_observations();

    Errors res, jp, jx, jk;
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t i = static_cast<std::size_t>(p.obs_image[k]);
      const double* pose = &p.poses[7 * i];
      const double* X = &p.points[3 * static_cast<std::size_t>(p.obs_point[k])];
      const double* K = p.intrinsics.data();
      const double* o = &p.obs_xy[2 * k];
      double rd[2], Jpd[12], Jxd[6], Jkd[10];
      jba::cpu::reprojection<double>(model, pose, pose + 4, X, K, o, rd, Jpd, Jxd, Jkd);

      float posef[7], Xf[3], Kf[5], of[2];
      for (int a = 0; a < 7; ++a) posef[a] = float(pose[a]);
      for (int a = 0; a < 3; ++a) Xf[a] = float(X[a]);
      for (int a = 0; a < np; ++a) Kf[a] = float(K[a]);
      for (int a = 0; a < 2; ++a) of[a] = float(o[a]);
      float rf[2], Jpf[12], Jxf[6], Jkf[10];
      jba::cpu::reprojection<float>(model, posef, posef + 4, Xf, Kf, of, rf, Jpf, Jxf, Jkf);

      compare(&gpu.residual[2 * k], rf, rd, 2, res, /*absolute=*/true);
      compare(&gpu.J_pose[12 * k], Jpf, Jpd, 12, jp);
      compare(&gpu.J_point[6 * k], Jxf, Jxd, 6, jx);
      compare(&gpu.J_intrinsics[2 * np * k], Jkf, Jkd, 2 * np, jk);
    }
    std::printf("%-15s %-11s %6zu obs | vs CPU float: residual %.2g px, J_pose %.2g, J_point %.2g, "
                "J_intr %.2g | vs CPU double: residual %.2g px, J_pose %.2g, J_point %.2g, J_intr %.2g\n",
                std::string(jba::model_name(model)).c_str(), conditioned ? "conditioned" : "raw", n, res.vs_float,
                jp.vs_float, jx.vs_float, jk.vs_float, res.vs_double, jp.vs_double, jx.vs_double, jk.vs_double);
    // Same arithmetic in float: agreement to a few ulps.
    for (const Errors* e : {&res, &jp, &jx, &jk}) EXPECT(e->vs_float < 1e-5);
    // Single precision against double: Jacobians to a few ulps of their largest
    // entry; residuals to a small fraction of a pixel (far below keypoint noise).
    for (const Errors* e : {&jp, &jx, &jk}) EXPECT(e->vs_double < 1e-5);
    EXPECT(res.vs_double < 5e-4);
  }
  return TEST_MAIN_RESULT();
}
