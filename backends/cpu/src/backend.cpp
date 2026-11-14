#include "jba/cpu/backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "jba/loss.h"
#include "parallel.h"

namespace jba::cpu {
namespace {

constexpr double kMinDiagonal = 1e-6;  // Ceres's LM diagonal clamp
constexpr double kMaxDiagonal = 1e32;
constexpr int kResidualResetPeriod = 10;  // Ceres's conjugate gradients default
constexpr int kMaxParams = 8;

// In-place inverse of a symmetric positive definite n x n matrix (row-major) by
// Cholesky. Returns false if it is not positive definite.
bool spd_inverse(double* A, int n) {
  double L[kMaxParams > 6 ? kMaxParams * kMaxParams : 36] = {};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = A[i * n + j];
      for (int k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
      if (i == j) {
        if (!(s > 0) || !std::isfinite(s)) return false;
        L[i * n + i] = std::sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  // Inverse of L, then A^-1 = L^-T L^-1.
  double Li[kMaxParams * kMaxParams] = {};
  for (int i = 0; i < n; ++i) {
    Li[i * n + i] = 1 / L[i * n + i];
    for (int j = 0; j < i; ++j) {
      double s = 0;
      for (int k = j; k < i; ++k) s -= L[i * n + k] * Li[k * n + j];
      Li[i * n + j] = s / L[i * n + i];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = 0;
      for (int k = i; k < n; ++k) s += Li[k * n + i] * Li[k * n + j];
      A[i * n + j] = A[j * n + i] = s;
    }
  }
  return true;
}

// Triggs correction of residual and Jacobian columns for a robust loss, as in Ceres
// (corrector.cc): with rho'' <= 0 it reduces to scaling by sqrt(rho').
struct Corrector {
  double sqrt_rho1 = 1, residual_scaling = 1, alpha_sq_norm = 0;
  Corrector(double sq_norm, const double rho[3]) {
    sqrt_rho1 = std::sqrt(rho[1]);
    if (sq_norm == 0 || rho[2] <= 0) {
      residual_scaling = sqrt_rho1;
      alpha_sq_norm = 0;
      return;
    }
    const double D = 1 + 2 * sq_norm * rho[2] / rho[1];
    const double alpha = 1 - std::sqrt(D);
    residual_scaling = sqrt_rho1 / (1 - alpha);
    alpha_sq_norm = alpha / sq_norm;
  }
  // J is 2 x cols, row-major; r is the uncorrected residual.
  void jacobian(const double* r, double* J, int cols) const {
    if (alpha_sq_norm == 0) {
      for (int i = 0; i < 2 * cols; ++i) J[i] *= sqrt_rho1;
      return;
    }
    for (int c = 0; c < cols; ++c) {
      const double rJ = r[0] * J[c] + r[1] * J[cols + c];
      J[c] = sqrt_rho1 * (J[c] - alpha_sq_norm * r[0] * rJ);
      J[cols + c] = sqrt_rho1 * (J[cols + c] - alpha_sq_norm * r[1] * rJ);
    }
  }
};

}  // namespace

struct CpuBackend::Impl {
  CameraModel model;
  int np;
  Loss loss;
  ThreadPool pool;

  // Parameters: current and candidate.
  std::vector<double> poses, points, intr;
  std::vector<double> c_poses, c_points, c_intr;

  // Variables. f = cameras side (pose and intrinsics blocks), e = points.
  struct FBlock {
    int offset;
    int size;
    int pinv_offset;
    bool is_pose;
    int index;  // image or intrinsics block
  };
  std::vector<FBlock> fblocks;
  std::vector<int> pose_off, intr_off, point_off;  // -1 when fixed
  std::vector<std::uint8_t> pose_fixed_cols;      // bit per tangent column
  std::vector<std::uint16_t> intr_fixed_cols;     // bit per parameter
  std::size_t nf = 0, ne = 0;

  // Observations, sorted by point then image.
  std::size_t nobs = 0;
  std::vector<std::int32_t> o_img, o_pt;
  std::vector<double> o_xy, o_w;
  std::vector<int> o_pose, o_intr;  // f offsets, -1 if fixed
  std::vector<std::size_t> pt_begin;
  std::vector<std::size_t> img_begin, img_list, ib_begin, ib_list;
  std::vector<std::int32_t> image_block;

  // Linearization.
  std::vector<double> r, Jp, Jx, Jk;
  std::vector<double> f_scale, e_scale, g_f, g_e, diag_f, diag_e;
  bool have_scaling = false;
  double gmax = 0;

  // Step.
  std::vector<double> d2_f, d2_e, Cinv, Pinv, t, dx_f, dx_e;

  Impl(const Problem& p, const SolverOptions& o, const Gauge& gauge)
      : model(p.model), np(num_params(p.model)), loss(o.loss), pool(o.num_threads) {
    poses = p.poses;
    points = p.points;
    intr = p.intrinsics;
    const std::size_t n_img = p.num_images(), n_pts = p.num_points(), n_blk = p.num_intrinsics_blocks();
    image_block.resize(n_img);
    for (std::size_t i = 0; i < n_img; ++i) image_block[i] = p.intrinsics_of_image(i);

    // Which blocks are observed at all.
    std::vector<char> img_seen(n_img, 0), pt_seen(n_pts, 0), blk_seen(n_blk, 0);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      img_seen[p.obs_image[k]] = 1;
      pt_seen[p.obs_point[k]] = 1;
      blk_seen[image_block[p.obs_image[k]]] = 1;
    }

    // Pose variables.
    pose_off.assign(n_img, -1);
    pose_fixed_cols.assign(n_img, 0);
    int off = 0;
    for (std::size_t i = 0; i < n_img; ++i) {
      const bool fixed = (!p.image_fixed.empty() && p.image_fixed[i]) || static_cast<int>(i) == gauge.reference_image;
      if (fixed || !img_seen[i]) continue;
      pose_off[i] = off;
      if (static_cast<int>(i) == gauge.scale_image) pose_fixed_cols[i] = std::uint8_t(1u << (3 + gauge.scale_coordinate));
      fblocks.push_back({off, 6, 0, true, static_cast<int>(i)});
      off += 6;
    }
    // Intrinsics variables.
    intr_off.assign(n_blk, -1);
    intr_fixed_cols.assign(n_blk, 0);
    for (std::size_t c = 0; c < n_blk; ++c) {
      const std::uint8_t mask = p.intrinsics_free.empty() ? 0 : p.intrinsics_free[c];
      if (mask == 0 || !blk_seen[c]) continue;
      std::uint16_t fixed_cols = 0;
      for (int a = 0; a < np; ++a) {
        if (!(mask & intrinsics_group(model, a))) fixed_cols |= std::uint16_t(1u << a);
      }
      if (fixed_cols == (1u << np) - 1) continue;
      intr_off[c] = off;
      intr_fixed_cols[c] = fixed_cols;
      fblocks.push_back({off, np, 0, false, static_cast<int>(c)});
      off += np;
    }
    nf = static_cast<std::size_t>(off);
    int pinv = 0;
    for (auto& b : fblocks) {
      b.pinv_offset = pinv;
      pinv += b.size * b.size;
    }
    Pinv.assign(static_cast<std::size_t>(pinv), 0.0);

    // Point variables.
    point_off.assign(n_pts, -1);
    int poff = 0;
    for (std::size_t j = 0; j < n_pts; ++j) {
      const bool fixed = !p.point_fixed.empty() && p.point_fixed[j];
      if (fixed || !pt_seen[j]) continue;
      point_off[j] = poff;
      poff += 3;
    }
    ne = static_cast<std::size_t>(poff);

    // Observations in point-major order.
    nobs = p.num_observations();
    std::vector<std::size_t> order(nobs);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      if (p.obs_point[a] != p.obs_point[b]) return p.obs_point[a] < p.obs_point[b];
      return p.obs_image[a] < p.obs_image[b];
    });
    o_img.resize(nobs);
    o_pt.resize(nobs);
    o_xy.resize(2 * nobs);
    o_w.resize(nobs);
    o_pose.resize(nobs);
    o_intr.resize(nobs);
    for (std::size_t k = 0; k < nobs; ++k) {
      const std::size_t s = order[k];
      o_img[k] = p.obs_image[s];
      o_pt[k] = p.obs_point[s];
      o_xy[2 * k] = p.obs_xy[2 * s];
      o_xy[2 * k + 1] = p.obs_xy[2 * s + 1];
      o_w[k] = p.obs_sigma.empty() ? 1.0 : 1.0 / p.obs_sigma[s];
      o_pose[k] = pose_off[o_img[k]];
      o_intr[k] = intr_off[image_block[o_img[k]]];
    }
    pt_begin.assign(n_pts + 1, 0);
    for (std::size_t k = 0; k < nobs; ++k) ++pt_begin[o_pt[k] + 1];
    for (std::size_t j = 0; j < n_pts; ++j) pt_begin[j + 1] += pt_begin[j];
    build_csr(n_img, [&](std::size_t k) { return static_cast<std::size_t>(o_img[k]); }, img_begin, img_list);
    build_csr(n_blk, [&](std::size_t k) { return static_cast<std::size_t>(image_block[o_img[k]]); }, ib_begin,
              ib_list);

    r.assign(2 * nobs, 0);
    Jp.assign(12 * nobs, 0);
    Jx.assign(6 * nobs, 0);
    Jk.assign(2 * static_cast<std::size_t>(np) * nobs, 0);
    t.assign(2 * nobs, 0);
    f_scale.assign(nf, 1);
    e_scale.assign(ne, 1);
    g_f.assign(nf, 0);
    g_e.assign(ne, 0);
    diag_f.assign(nf, 0);
    diag_e.assign(ne, 0);
    d2_f.assign(nf, 0);
    d2_e.assign(ne, 0);
    dx_f.assign(nf, 0);
    dx_e.assign(ne, 0);
    Cinv.assign(9 * n_pts, 0);
  }

  // CSR lists of observation indices grouped by key, each list in increasing order.
  template <typename Key>
  void build_csr(std::size_t n, Key key, std::vector<std::size_t>& begin, std::vector<std::size_t>& list) {
    begin.assign(n + 1, 0);
    for (std::size_t k = 0; k < nobs; ++k) ++begin[key(k) + 1];
    for (std::size_t i = 0; i < n; ++i) begin[i + 1] += begin[i];
    list.resize(nobs);
    std::vector<std::size_t> fill(begin.begin(), begin.end() - 1);
    for (std::size_t k = 0; k < nobs; ++k) list[fill[key(k)]++] = k;
  }

  // ---------------------------------------------------------------------------

  double cost_of(const std::vector<double>& P, const std::vector<double>& X, const std::vector<double>& K) {
    return 0.5 * pool.sum(nobs, [&](std::size_t k) {
      const std::size_t i = static_cast<std::size_t>(o_img[k]);
      const double* pose = &P[7 * i];
      double res[2];
      reprojection<double>(model, pose, pose + 4, &X[3 * static_cast<std::size_t>(o_pt[k])],
                           &K[static_cast<std::size_t>(np) * static_cast<std::size_t>(image_block[i])], &o_xy[2 * k],
                           res, nullptr, nullptr, nullptr);
      const double s = (res[0] * res[0] + res[1] * res[1]) * o_w[k] * o_w[k];
      double rho[3];
      evaluate_loss(loss, s, rho);
      return rho[0];
    });
  }

  void linearize(bool compute_scaling) {
    pool.parallel_for(nobs, [&](std::size_t k0, std::size_t k1) {
      for (std::size_t k = k0; k < k1; ++k) {
        const std::size_t i = static_cast<std::size_t>(o_img[k]);
        const double* pose = &poses[7 * i];
        double* rk = &r[2 * k];
        double* jp = &Jp[12 * k];
        double* jx = &Jx[6 * k];
        double* jk = &Jk[2 * static_cast<std::size_t>(np) * k];
        reprojection<double>(model, pose, pose + 4, &points[3 * static_cast<std::size_t>(o_pt[k])],
                             &intr[static_cast<std::size_t>(np) * static_cast<std::size_t>(image_block[i])],
                             &o_xy[2 * k], rk, jp, jx, jk);
        const double w = o_w[k];
        rk[0] *= w;
        rk[1] *= w;
        for (int a = 0; a < 12; ++a) jp[a] *= w;
        for (int a = 0; a < 6; ++a) jx[a] *= w;
        for (int a = 0; a < 2 * np; ++a) jk[a] *= w;
        if (loss.type != LossType::None) {
          const double s = rk[0] * rk[0] + rk[1] * rk[1];
          double rho[3];
          evaluate_loss(loss, s, rho);
          const Corrector corr(s, rho);
          corr.jacobian(rk, jp, 6);
          corr.jacobian(rk, jx, 3);
          corr.jacobian(rk, jk, np);
          rk[0] *= corr.residual_scaling;
          rk[1] *= corr.residual_scaling;
        }
        // Columns that are not variables are zero.
        const std::uint8_t pmask = o_pose[k] < 0 ? 0x3f : pose_fixed_cols[i];
        for (int c = 0; c < 6; ++c) {
          if (pmask & (1u << c)) jp[c] = jp[6 + c] = 0;
        }
        const std::uint16_t kmask =
            o_intr[k] < 0 ? std::uint16_t(0xffff) : intr_fixed_cols[static_cast<std::size_t>(image_block[i])];
        for (int c = 0; c < np; ++c) {
          if (kmask & (1u << c)) jk[c] = jk[np + c] = 0;
        }
        if (point_off[o_pt[k]] < 0) {
          for (int a = 0; a < 6; ++a) jx[a] = 0;
        }
      }
    });

    if (compute_scaling) {
      // Ceres: scale = 1 / (1 + column norm), computed once from the first Jacobian.
      column_norms_and_gradient(false);
      for (std::size_t c = 0; c < nf; ++c) f_scale[c] = 1 / (1 + std::sqrt(diag_f[c]));
      for (std::size_t c = 0; c < ne; ++c) e_scale[c] = 1 / (1 + std::sqrt(diag_e[c]));
      have_scaling = true;
    }
    if (have_scaling) {
      pool.parallel_for(nobs, [&](std::size_t k0, std::size_t k1) {
        for (std::size_t k = k0; k < k1; ++k) {
          if (o_pose[k] >= 0) {
            for (int c = 0; c < 6; ++c) {
              Jp[12 * k + c] *= f_scale[o_pose[k] + c];
              Jp[12 * k + 6 + c] *= f_scale[o_pose[k] + c];
            }
          }
          if (o_intr[k] >= 0) {
            double* jk = &Jk[2 * static_cast<std::size_t>(np) * k];
            for (int c = 0; c < np; ++c) {
              jk[c] *= f_scale[o_intr[k] + c];
              jk[np + c] *= f_scale[o_intr[k] + c];
            }
          }
          const int e = point_off[o_pt[k]];
          if (e >= 0) {
            for (int c = 0; c < 3; ++c) {
              Jx[6 * k + c] *= e_scale[e + c];
              Jx[6 * k + 3 + c] *= e_scale[e + c];
            }
          }
        }
      });
    }
    column_norms_and_gradient(true);

    double m = 0;
    for (std::size_t c = 0; c < nf; ++c) m = std::max(m, std::abs(g_f[c] / f_scale[c]));
    for (std::size_t c = 0; c < ne; ++c) m = std::max(m, std::abs(g_e[c] / e_scale[c]));
    gmax = m;
  }

  // diag = squared column norms of J; g = J^T r when `gradient` is set.
  void column_norms_and_gradient(bool gradient) {
    pool.parallel_for(
        fblocks.size(),
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const FBlock& fb = fblocks[b];
            double d[kMaxParams] = {}, g[kMaxParams] = {};
            const auto& begin = fb.is_pose ? img_begin : ib_begin;
            const auto& list = fb.is_pose ? img_list : ib_list;
            for (std::size_t q = begin[fb.index]; q < begin[fb.index + 1]; ++q) {
              const std::size_t k = list[q];
              const double* J = fb.is_pose ? &Jp[12 * k] : &Jk[2 * static_cast<std::size_t>(np) * k];
              const int n = fb.size;
              for (int c = 0; c < n; ++c) {
                d[c] += J[c] * J[c] + J[n + c] * J[n + c];
                g[c] += J[c] * r[2 * k] + J[n + c] * r[2 * k + 1];
              }
            }
            for (int c = 0; c < fb.size; ++c) {
              diag_f[fb.offset + c] = d[c];
              if (gradient) g_f[fb.offset + c] = g[c];
            }
          }
        },
        1);
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j) {
        const int e = point_off[j];
        if (e < 0) continue;
        double d[3] = {}, g[3] = {};
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          const double* J = &Jx[6 * k];
          for (int c = 0; c < 3; ++c) {
            d[c] += J[c] * J[c] + J[3 + c] * J[3 + c];
            g[c] += J[c] * r[2 * k] + J[3 + c] * r[2 * k + 1];
          }
        }
        for (int c = 0; c < 3; ++c) {
          diag_e[e + c] = d[c];
          if (gradient) g_e[e + c] = g[c];
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Products with the camera-side Jacobian.

  // t_k = J_f,k x for every observation.
  void jf_times(const double* x, std::size_t k, double* out) const {
    out[0] = out[1] = 0;
    if (o_pose[k] >= 0) {
      const double* J = &Jp[12 * k];
      const double* v = x + o_pose[k];
      for (int c = 0; c < 6; ++c) {
        out[0] += J[c] * v[c];
        out[1] += J[6 + c] * v[c];
      }
    }
    if (o_intr[k] >= 0) {
      const double* J = &Jk[2 * static_cast<std::size_t>(np) * k];
      const double* v = x + o_intr[k];
      for (int c = 0; c < np; ++c) {
        out[0] += J[c] * v[c];
        out[1] += J[np + c] * v[c];
      }
    }
  }

  // y = J_f^T t (gathered per camera-side block).
  void jft_times_t(double* y) {
    pool.parallel_for(
        fblocks.size(),
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const FBlock& fb = fblocks[b];
            double acc[kMaxParams] = {};
            const auto& begin = fb.is_pose ? img_begin : ib_begin;
            const auto& list = fb.is_pose ? img_list : ib_list;
            const int n = fb.size;
            for (std::size_t q = begin[fb.index]; q < begin[fb.index + 1]; ++q) {
              const std::size_t k = list[q];
              const double* J = fb.is_pose ? &Jp[12 * k] : &Jk[2 * static_cast<std::size_t>(np) * k];
              for (int c = 0; c < n; ++c) acc[c] += J[c] * t[2 * k] + J[n + c] * t[2 * k + 1];
            }
            for (int c = 0; c < n; ++c) y[fb.offset + c] = acc[c];
          }
        },
        1);
  }

  // y = S x, S = J_f^T J_f + D_f^2 - J_f^T J_e (J_e^T J_e + D_e^2)^-1 J_e^T J_f.
  void apply_schur(const double* x, double* y) {
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j) {
        double u[3] = {0, 0, 0};
        const int e = point_off[j];
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          jf_times(x, k, &t[2 * k]);
          if (e >= 0) {
            const double* J = &Jx[6 * k];
            for (int c = 0; c < 3; ++c) u[c] += J[c] * t[2 * k] + J[3 + c] * t[2 * k + 1];
          }
        }
        if (e < 0) continue;
        const double* Ci = &Cinv[9 * j];
        double w[3];
        for (int a = 0; a < 3; ++a) w[a] = Ci[3 * a] * u[0] + Ci[3 * a + 1] * u[1] + Ci[3 * a + 2] * u[2];
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          const double* J = &Jx[6 * k];
          t[2 * k] -= J[0] * w[0] + J[1] * w[1] + J[2] * w[2];
          t[2 * k + 1] -= J[3] * w[0] + J[4] * w[1] + J[5] * w[2];
        }
      }
    });
    jft_times_t(y);
    for (std::size_t c = 0; c < nf; ++c) y[c] += d2_f[c] * x[c];
  }

  void apply_preconditioner(const double* x, double* y) const {
    for (const FBlock& fb : fblocks) {
      const double* P = &Pinv[static_cast<std::size_t>(fb.pinv_offset)];
      const int n = fb.size;
      for (int a = 0; a < n; ++a) {
        double s = 0;
        for (int b = 0; b < n; ++b) s += P[a * n + b] * x[fb.offset + b];
        y[fb.offset + a] = s;
      }
    }
  }

  double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return pool.sum(a.size(), [&](std::size_t i) { return a[i] * b[i]; });
  }

  // ---------------------------------------------------------------------------

  bool build_blocks(double radius) {
    for (std::size_t c = 0; c < nf; ++c) d2_f[c] = std::clamp(diag_f[c], kMinDiagonal, kMaxDiagonal) / radius;
    for (std::size_t c = 0; c < ne; ++c) d2_e[c] = std::clamp(diag_e[c], kMinDiagonal, kMaxDiagonal) / radius;

    std::atomic<bool> ok{true};
    // Point blocks C_j = J_e^T J_e + D_e^2, inverted.
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j) {
        const int e = point_off[j];
        if (e < 0) continue;
        double C[9] = {};
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          const double* J = &Jx[6 * k];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) C[3 * a + b] += J[a] * J[b] + J[3 + a] * J[3 + b];
        }
        for (int a = 0; a < 3; ++a) C[4 * a] += d2_e[e + a];
        if (!spd_inverse(C, 3)) ok = false;
        std::copy(C, C + 9, &Cinv[9 * j]);
      }
    });
    if (!ok) return false;

    // Preconditioner: camera-side diagonal blocks of the Schur complement,
    // P_b = B_bb + D_b^2 - sum_j E_bj C_j^-1 E_bj^T, inverted. The observations of
    // a block are in point-major order, so those of one point are consecutive.
    pool.parallel_for(
        fblocks.size(),
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const FBlock& fb = fblocks[b];
            const int n = fb.size;
            double P[kMaxParams * kMaxParams] = {};
            const auto& begin = fb.is_pose ? img_begin : ib_begin;
            const auto& list = fb.is_pose ? img_list : ib_list;
            std::size_t q = begin[fb.index];
            const std::size_t qend = begin[fb.index + 1];
            while (q < qend) {
              const std::int32_t j = o_pt[list[q]];
              double E[kMaxParams * 3] = {};
              for (; q < qend && o_pt[list[q]] == j; ++q) {
                const std::size_t k = list[q];
                const double* J = fb.is_pose ? &Jp[12 * k] : &Jk[2 * static_cast<std::size_t>(np) * k];
                const double* X = &Jx[6 * k];
                for (int a = 0; a < n; ++a) {
                  for (int c = 0; c < n; ++c) P[a * n + c] += J[a] * J[c] + J[n + a] * J[n + c];
                  for (int c = 0; c < 3; ++c) E[3 * a + c] += J[a] * X[c] + J[n + a] * X[3 + c];
                }
              }
              if (point_off[j] < 0) continue;
              const double* Ci = &Cinv[9 * static_cast<std::size_t>(j)];
              double EC[kMaxParams * 3];
              for (int a = 0; a < n; ++a)
                for (int c = 0; c < 3; ++c)
                  EC[3 * a + c] = E[3 * a] * Ci[c] + E[3 * a + 1] * Ci[3 + c] + E[3 * a + 2] * Ci[6 + c];
              for (int a = 0; a < n; ++a)
                for (int c = 0; c < n; ++c)
                  P[a * n + c] -= EC[3 * a] * E[3 * c] + EC[3 * a + 1] * E[3 * c + 1] + EC[3 * a + 2] * E[3 * c + 2];
            }
            for (int a = 0; a < n; ++a) P[a * n + a] += d2_f[fb.offset + a];
            if (!spd_inverse(P, n)) ok = false;
            std::copy(P, P + n * n, &Pinv[static_cast<std::size_t>(fb.pinv_offset)]);
          }
        },
        1);
    return ok;
  }

  // Preconditioned conjugate gradients on S x = b, with Ceres's stopping test
  // (Nash and Sofer: stop when i * (Q_i - Q_{i-1}) / Q_i < eta).
  int pcg(const std::vector<double>& b, std::vector<double>& x, double eta, int min_it, int max_it, bool& ok) {
    ok = true;
    std::fill(x.begin(), x.end(), 0.0);
    if (nf == 0) return 0;
    const double norm_b = std::sqrt(dot(b, b));
    if (norm_b == 0) return 0;
    std::vector<double> res = b, z(nf), p(nf), q(nf), bpr(nf);
    double rho_prev = 1, Q0 = 0;
    int i = 1;
    for (; i <= max_it; ++i) {
      apply_preconditioner(res.data(), z.data());
      const double rho = dot(res, z);
      if (rho == 0) break;  // exact solution
      if (!std::isfinite(rho)) {
        ok = false;
        break;
      }
      if (i == 1) {
        p = z;
      } else {
        const double beta = rho / rho_prev;
        if (!std::isfinite(beta) || beta == 0) {
          ok = false;
          break;
        }
        for (std::size_t c = 0; c < nf; ++c) p[c] = z[c] + beta * p[c];
      }
      apply_schur(p.data(), q.data());
      const double pq = dot(p, q);
      // As in Ceres: no more progress is possible; keep the solution so far.
      if (!(pq > 0) || std::isinf(pq)) break;
      const double alpha = rho / pq;
      for (std::size_t c = 0; c < nf; ++c) x[c] += alpha * p[c];
      if (i % kResidualResetPeriod == 0) {
        apply_schur(x.data(), q.data());
        for (std::size_t c = 0; c < nf; ++c) res[c] = b[c] - q[c];
      } else {
        for (std::size_t c = 0; c < nf; ++c) res[c] -= alpha * q[c];
      }
      for (std::size_t c = 0; c < nf; ++c) bpr[c] = b[c] + res[c];
      const double Q1 = -dot(x, bpr);
      const double zeta = i * (Q1 - Q0) / Q1;
      if (zeta < eta && i >= min_it) break;
      Q0 = Q1;
      rho_prev = rho;
    }
    return std::min(i, max_it);
  }

  StepResult compute_step(double radius, double eta, int min_it, int max_it) {
    StepResult out;
    if (!build_blocks(radius)) return out;

    // Right-hand side: b = -g_f + J_f^T J_e C^-1 g_e.
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j) {
        const int e = point_off[j];
        double w[3] = {0, 0, 0};
        if (e >= 0) {
          const double* Ci = &Cinv[9 * j];
          for (int a = 0; a < 3; ++a)
            w[a] = Ci[3 * a] * g_e[e] + Ci[3 * a + 1] * g_e[e + 1] + Ci[3 * a + 2] * g_e[e + 2];
        }
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          const double* J = &Jx[6 * k];
          t[2 * k] = J[0] * w[0] + J[1] * w[1] + J[2] * w[2];
          t[2 * k + 1] = J[3] * w[0] + J[4] * w[1] + J[5] * w[2];
        }
      }
    });
    std::vector<double> b(nf);
    jft_times_t(b.data());
    for (std::size_t c = 0; c < nf; ++c) b[c] -= g_f[c];

    bool ok = true;
    out.linear_iterations = pcg(b, dx_f, eta, min_it, max_it, ok);
    if (!ok) return out;

    // Back substitution: dx_e = -C^-1 (g_e + J_e^T J_f dx_f).
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j) {
        const int e = point_off[j];
        double u[3] = {0, 0, 0};
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          jf_times(dx_f.data(), k, &t[2 * k]);
          if (e < 0) continue;
          const double* J = &Jx[6 * k];
          for (int c = 0; c < 3; ++c) u[c] += J[c] * t[2 * k] + J[3 + c] * t[2 * k + 1];
        }
        if (e < 0) continue;
        for (int c = 0; c < 3; ++c) u[c] += g_e[e + c];
        const double* Ci = &Cinv[9 * j];
        for (int a = 0; a < 3; ++a) dx_e[e + a] = -(Ci[3 * a] * u[0] + Ci[3 * a + 1] * u[1] + Ci[3 * a + 2] * u[2]);
      }
    });

    // Predicted decrease: -m . (r + m / 2), m = J dx (t already holds J_f dx_f).
    out.model_cost_change = -pool.sum(nobs, [&](std::size_t k) {
      double m0 = t[2 * k], m1 = t[2 * k + 1];
      const int e = point_off[o_pt[k]];
      if (e >= 0) {
        const double* J = &Jx[6 * k];
        m0 += J[0] * dx_e[e] + J[1] * dx_e[e + 1] + J[2] * dx_e[e + 2];
        m1 += J[3] * dx_e[e] + J[4] * dx_e[e + 1] + J[5] * dx_e[e + 2];
      }
      return m0 * (r[2 * k] + m0 / 2) + m1 * (r[2 * k + 1] + m1 / 2);
    });

    // Candidate: unscale the step and retract.
    c_poses = poses;
    c_points = points;
    c_intr = intr;
    // The step norm is taken in stored coordinates (candidate minus current, the
    // quaternion included), as Ceres does for its parameter tolerance.
    double sq = 0;
    auto add_difference = [&sq](const double* a, const double* b, int n) {
      for (int c = 0; c < n; ++c) sq += (a[c] - b[c]) * (a[c] - b[c]);
    };
    for (const FBlock& fb : fblocks) {
      double delta[kMaxParams];
      for (int c = 0; c < fb.size; ++c) delta[c] = f_scale[fb.offset + c] * dx_f[fb.offset + c];
      if (fb.is_pose) {
        const double* cur = &poses[7 * static_cast<std::size_t>(fb.index)];
        double* cand = &c_poses[7 * static_cast<std::size_t>(fb.index)];
        pose_retract(cur, delta, cand);
        add_difference(cand, cur, 7);
      } else {
        const std::size_t at = static_cast<std::size_t>(np) * static_cast<std::size_t>(fb.index);
        double* k = &c_intr[at];
        for (int c = 0; c < fb.size; ++c) k[c] += delta[c];
        add_difference(k, &intr[at], fb.size);
      }
    }
    for (std::size_t j = 0; j < point_off.size(); ++j) {
      const int e = point_off[j];
      if (e < 0) continue;
      for (int c = 0; c < 3; ++c) {
        // Two statements on purpose: one expression would let the compiler fuse the
        // multiply and add, which changes rounding and with it the iteration history.
        const double d = e_scale[e + c] * dx_e[e + c];
        c_points[3 * j + c] += d;
      }
      add_difference(&c_points[3 * j], &points[3 * j], 3);
    }
    out.step_norm = std::sqrt(sq);
    out.valid = std::isfinite(out.model_cost_change) && out.model_cost_change > 0 && std::isfinite(out.step_norm);
    return out;
  }

  double parameter_norm() const {
    double s = 0;
    for (const FBlock& fb : fblocks) {
      if (fb.is_pose) {
        for (int a = 0; a < 7; ++a) s += poses[7 * static_cast<std::size_t>(fb.index) + a] * poses[7 * static_cast<std::size_t>(fb.index) + a];
      } else {
        for (int a = 0; a < np; ++a) {
          const double v = intr[static_cast<std::size_t>(np) * static_cast<std::size_t>(fb.index) + a];
          s += v * v;
        }
      }
    }
    for (std::size_t j = 0; j < point_off.size(); ++j) {
      if (point_off[j] < 0) continue;
      for (int a = 0; a < 3; ++a) s += points[3 * j + a] * points[3 * j + a];
    }
    return std::sqrt(s);
  }
};

CpuBackend::CpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : impl_(std::make_unique<Impl>(problem, options, gauge)) {}

CpuBackend::~CpuBackend() = default;

double CpuBackend::cost() { return impl_->cost_of(impl_->poses, impl_->points, impl_->intr); }

void CpuBackend::linearize(bool compute_scaling) { impl_->linearize(compute_scaling); }

double CpuBackend::gradient_max_norm() const { return impl_->gmax; }

double CpuBackend::parameter_norm() const { return impl_->parameter_norm(); }

StepResult CpuBackend::compute_step(double radius, double eta, int min_linear_iterations, int max_linear_iterations) {
  return impl_->compute_step(radius, eta, min_linear_iterations, max_linear_iterations);
}

bool CpuBackend::candidate_cost(double* cost) {
  *cost = impl_->cost_of(impl_->c_poses, impl_->c_points, impl_->c_intr);
  return std::isfinite(*cost);
}

void CpuBackend::accept_candidate() {
  impl_->poses.swap(impl_->c_poses);
  impl_->points.swap(impl_->c_points);
  impl_->intr.swap(impl_->c_intr);
}

void CpuBackend::write_back(Problem& problem) const {
  problem.poses = impl_->poses;
  problem.points = impl_->points;
  problem.intrinsics = impl_->intr;
}

}  // namespace jba::cpu
