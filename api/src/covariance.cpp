#include "jba/covariance.h"

#include <numeric>
#include <stdexcept>

#include "jba/cpu/backend.h"
#include "jba/geometry.h"

namespace jba {

Covariance covariance(const Problem& p, const SolverOptions& o, const CovarianceOptions& co) {
  if (auto err = validate(p); !err.empty()) throw std::invalid_argument("invalid problem: " + err);
  Covariance out;
  out.gauge = o.gauge == GaugeMode::Auto ? choose_gauge(p) : Gauge{};
  cpu::CpuBackend backend(p, o, out.gauge);
  if (co.all_points) {
    out.points.resize(p.num_points());
    std::iota(out.points.begin(), out.points.end(), 0);
  } else {
    out.points = co.points;
  }
  for (auto j : out.points)
    if (j < 0 || static_cast<std::size_t>(j) >= p.num_points()) throw std::invalid_argument("covariance: no point " + std::to_string(j));
  std::vector<int> pts(out.points.begin(), out.points.end());
  backend.covariance(pts, co.max_unknowns, out.pose, out.point);

  // Camera centre: dC = -R^T [t]x dtheta - R^T dt, so Cov_C = A Cov A^T.
  const std::size_t n = p.num_images();
  out.centre.assign(9 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double* q = &p.poses[7 * i];
    const double* t = q + 4;
    const double qc[4] = {q[0], -q[1], -q[2], -q[3]};
    double A[18];
    for (int c = 0; c < 3; ++c) {
      // Column c of -R^T [t]x is -R^T (t x e_c); of -R^T is -R^T e_c.
      double e[3] = {0, 0, 0};
      e[c] = 1;
      const double txe[3] = {t[1] * e[2] - t[2] * e[1], t[2] * e[0] - t[0] * e[2], t[0] * e[1] - t[1] * e[0]};
      double a[3], b[3];
      quat_rotate(qc, txe, a);
      quat_rotate(qc, e, b);
      for (int r = 0; r < 3; ++r) {
        A[6 * r + c] = -a[r];
        A[6 * r + 3 + c] = -b[r];
      }
    }
    const double* S = &out.pose[36 * i];
    double AS[18];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 6; ++c) {
        double v = 0;
        for (int k = 0; k < 6; ++k) v += A[6 * r + k] * S[6 * k + c];
        AS[6 * r + c] = v;
      }
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        double v = 0;
        for (int k = 0; k < 6; ++k) v += AS[6 * r + k] * A[6 * c + k];
        out.centre[9 * i + static_cast<std::size_t>(3 * r + c)] = v;
      }
  }
  return out;
}

}  // namespace jba
