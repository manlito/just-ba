#include "jba/cpu/backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/cpu/depth.h"
#include "jba/cpu/reprojection.h"
#include "jba/cpu/system.h"
#include "jba/loss.h"

namespace jba::cpu {
namespace {

constexpr double kMinDiagonal = 1e-6;  // Ceres's LM diagonal clamp
constexpr double kMaxDiagonal = 1e32;
constexpr int kResidualResetPeriod = 10;  // Ceres's conjugate gradients default
// Observations per gather unit (see Impl::units). A pose block rarely has this many, so
// it stays one unit; a shared intrinsics block, which sees every observation, does not.
constexpr std::size_t kGatherUnit = 16384;
constexpr int kUnitFloats = kMaxParams * kMaxParams;  // the widest per-block sum: a preconditioner block

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

}  // namespace

struct CpuBackend::Impl : System {
  // Depth rows of the observations with a reading: residual and Jacobian blocks
  // (pose 6, point 3, block 2).
  std::vector<double> dr, dJp, dJx, dJc, td;

  // Linearization.
  std::vector<double> r, Jp, Jx, Jk;
  std::vector<double> g_f, g_e, diag_f, diag_e;
  double gmax = 0;

  // Step.
  std::vector<double> d2_f, d2_e, Cinv, Pinv, t, dx_f, dx_e;

  // Gather units: each camera-side block's observations in runs of at least
  // kGatherUnit, extended to the end of a point's observations (the preconditioner
  // sums per point). The per-block sums run in parallel over the units, so a block with
  // far more observations than the others (a shared intrinsics block sees all of them)
  // is no longer one thread's work. A block of one unit is summed exactly as before;
  // the units of a larger one are added in order, so the results do not depend on the
  // number of threads.
  struct Unit {
    std::size_t block, q0, q1;
  };
  std::vector<Unit> units;
  std::vector<std::size_t> block_units;  // per block, its first unit; one past the end last
  std::vector<std::size_t> split_blocks;  // the blocks of more than one unit
  std::vector<double> unit_partial;       // kUnitFloats per unit of a split block

  Impl(const Problem& p, const SolverOptions& o, const Gauge& gauge) : System(p, o, gauge) {
    Pinv.assign(pinv_size, 0.0);
    block_units.push_back(0);
    for (std::size_t b = 0; b < fblocks.size(); ++b) {
      std::size_t q0, q1;
      const std::vector<std::size_t>* list;
      obs_range(fblocks[b], q0, q1, list);
      std::size_t q = q0;
      do {
        std::size_t end = std::min(q1, q + kGatherUnit);
        while (end < q1 && o_pt[(*list)[end]] == o_pt[(*list)[end - 1]]) ++end;
        units.push_back({b, q, end});
        q = end;
      } while (q < q1);
      block_units.push_back(units.size());
      if (block_units[b + 1] - block_units[b] > 1) split_blocks.push_back(b);
    }
    unit_partial.assign(kUnitFloats * units.size(), 0.0);
    if (has_depth) {
      dr.assign(nobs, 0);
      dJp.assign(6 * nobs, 0);
      dJx.assign(3 * nobs, 0);
      dJc.assign(2 * nobs, 0);
      td.assign(nobs, 0);
    }
    r.assign(2 * nobs, 0);
    Jp.assign(12 * nobs, 0);
    Jx.assign(6 * nobs, 0);
    Jk.assign(2 * static_cast<std::size_t>(np) * nobs, 0);
    t.assign(2 * nobs, 0);
    g_f.assign(nf, 0);
    g_e.assign(ne, 0);
    diag_f.assign(nf, 0);
    diag_e.assign(ne, 0);
    d2_f.assign(nf, 0);
    d2_e.assign(ne, 0);
    dx_f.assign(nf, 0);
    dx_e.assign(ne, 0);
    Cinv.assign(9 * point_off.size(), 0);
  }

  void linearize(bool compute_scaling) {
    update_bases();
    pool.parallel_for(nobs, [&](std::size_t k0, std::size_t k1) {
      for (std::size_t k = k0; k < k1; ++k) {
        const std::size_t i = static_cast<std::size_t>(o_img[k]);
        const double* pose = &cur.poses[7 * i];
        double* rk = &r[2 * k];
        double* jp = &Jp[12 * k];
        double* jx = &Jx[6 * k];
        double* jk = &Jk[2 * static_cast<std::size_t>(np) * k];
        reprojection<double>(model, pose, pose + 4, &cur.points[3 * static_cast<std::size_t>(o_pt[k])],
                             &cur.intr[static_cast<std::size_t>(np) * static_cast<std::size_t>(image_block[i])],
                             &o_xy[2 * k], rk, jp, jx, jk);
        if (!basis_image.empty() && basis_of[i] >= 0) apply_basis(&basis[9 * static_cast<std::size_t>(basis_of[i])], jp, 2, 6);
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
        if (has_depth && o_dw[k] != 0) linearize_depth_row(k, pmask, &dr[k], &dJp[6 * k], &dJx[3 * k], &dJc[2 * k]);
      }
    });
    linearize_factors();

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
          if (has_depth && o_dw[k] != 0) {
            if (o_pose[k] >= 0)
              for (int c = 0; c < 6; ++c) dJp[6 * k + c] *= f_scale[o_pose[k] + c];
            if (e >= 0)
              for (int c = 0; c < 3; ++c) dJx[3 * k + c] *= e_scale[e + c];
            if (o_dep[k] >= 0)
              for (int c = 0; c < 2; ++c) dJc[2 * k + c] *= f_scale[o_dep[k] + c];
          }
        }
      });
      scale_factors();
    }
    column_norms_and_gradient(true);

    double m = 0;
    for (std::size_t c = 0; c < nf; ++c) m = std::max(m, std::abs(g_f[c] / f_scale[c]));
    for (std::size_t c = 0; c < ne; ++c) m = std::max(m, std::abs(g_e[c] / e_scale[c]));
    gmax = m;
  }

  // The depth row's Jacobian block for a camera-side block kind, or nullptr when the
  // depth residual does not depend on it (or observation k has no reading).
  const double* depth_row(BlockKind kind, std::size_t k) const {
    if (!has_depth || o_dw[k] == 0) return nullptr;
    if (kind == BlockKind::Pose) return &dJp[6 * k];
    if (kind == BlockKind::Depth) return &dJc[2 * k];
    return nullptr;
  }

  // m = J_factor x for every factor.
  void factors_times(const double* x) {
    for (FactorSet& fs : fsets) {
      pool.parallel_for(fs.count, [&](std::size_t f0, std::size_t f1) {
        for (std::size_t f = f0; f < f1; ++f) {
          const double* J = fs.Jf(f);
          double* m = fs.mf(f);
          for (int row = 0; row < fs.rows; ++row) m[row] = 0;
          for (int slot = 0; slot < fs.nslots; ++slot) {
            const int so = fs.off(f, slot);
            if (so < 0) continue;
            const int n = fs.slot_size[slot], base = fs.slot_base[slot];
            for (int row = 0; row < fs.rows; ++row)
              for (int c = 0; c < n; ++c) m[row] += J[row * fs.cols + base + c] * x[so + c];
          }
        }
      }, fs.grain);
    }
  }

  // Per camera-side block, a sum over its observations, then the rest: unit(u, acc)
  // adds unit u's observations (u.q0 to u.q1 of the block's list) to acc, kUnitFloats
  // values starting at zero; finish(b, acc) takes block b's total.
  template <typename UnitFn, typename FinishFn>
  void gather(UnitFn&& unit, FinishFn&& finish) {
    pool.parallel_for(
        units.size(),
        [&](std::size_t u0, std::size_t u1) {
          for (std::size_t u = u0; u < u1; ++u) {
            const Unit& un = units[u];
            double acc[kUnitFloats] = {};
            unit(un, acc);
            if (block_units[un.block + 1] - block_units[un.block] == 1) finish(un.block, acc);
            else std::copy(acc, acc + kUnitFloats, &unit_partial[kUnitFloats * u]);
          }
        },
        1);
    if (split_blocks.empty()) return;
    pool.parallel_for(
        split_blocks.size(),
        [&](std::size_t s0, std::size_t s1) {
          for (std::size_t s = s0; s < s1; ++s) {
            const std::size_t b = split_blocks[s];
            double acc[kUnitFloats] = {};
            for (std::size_t u = block_units[b]; u < block_units[b + 1]; ++u)
              for (int c = 0; c < kUnitFloats; ++c) acc[c] += unit_partial[kUnitFloats * u + c];
            finish(b, acc);
          }
        },
        1);
  }

  // diag = squared column norms of J; g = J^T r when `gradient` is set.
  void column_norms_and_gradient(bool gradient) {
    gather(
        [&](const Unit& un, double* acc) {
          const FBlock& fb = fblocks[un.block];
          double* d = acc;
          double* g = acc + kMaxParams;
          const int n = fb.size;
          std::size_t q0, q1;
          const std::vector<std::size_t>* list;
          obs_range(fb, q0, q1, list);
          q0 = un.q0;
          q1 = un.q1;
          {
            if (fb.kind != BlockKind::Depth) {
              for (std::size_t q = q0; q < q1; ++q) {
                const std::size_t k = (*list)[q];
                const double* J = fb.kind == BlockKind::Pose ? &Jp[12 * k] : &Jk[2 * static_cast<std::size_t>(np) * k];
                for (int c = 0; c < n; ++c) {
                  d[c] += J[c] * J[c] + J[n + c] * J[n + c];
                  g[c] += J[c] * r[2 * k] + J[n + c] * r[2 * k + 1];
                }
              }
            }
            if (has_depth && fb.kind != BlockKind::Intrinsics) {
              for (std::size_t q = q0; q < q1; ++q) {
                const double* Jd = depth_row(fb.kind, (*list)[q]);
                if (!Jd) continue;
                const double rk = dr[(*list)[q]];
                for (int c = 0; c < n; ++c) {
                  d[c] += Jd[c] * Jd[c];
                  g[c] += Jd[c] * rk;
                }
              }
            }
          }
        },
        [&](std::size_t b, double* acc) {
          const FBlock& fb = fblocks[b];
          double* d = acc;
          double* g = acc + kMaxParams;
          const int n = fb.size;
          {
            for (const FactorSet& fs : fsets) {
              for (std::size_t q = fs.begin[b]; q < fs.begin[b + 1]; ++q) {
                const auto [f, slot] = fs.list[q];
                const double* J = fs.Jf(static_cast<std::size_t>(f));
                const double* rf = fs.rf(static_cast<std::size_t>(f));
                const int base = fs.slot_base[slot];
                for (int row = 0; row < fs.rows; ++row)
                  for (int c = 0; c < n; ++c) {
                    const double v = J[row * fs.cols + base + c];
                    d[c] += v * v;
                    g[c] += v * rf[row];
                  }
              }
            }
            for (int c = 0; c < fb.size; ++c) {
              diag_f[fb.offset + c] = d[c];
              if (gradient) g_f[fb.offset + c] = g[c];
            }
          }
        });
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      const bool depth = has_depth;
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
          if (depth && o_dw[k] != 0) {
            const double* Jd = &dJx[3 * k];
            for (int c = 0; c < 3; ++c) {
              d[c] += Jd[c] * Jd[c];
              g[c] += Jd[c] * dr[k];
            }
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

  // Depth row of observation k times x (camera side): pose and depth block.
  double jf_depth(const double* x, std::size_t k) const {
    double out = 0;
    if (o_pose[k] >= 0) {
      const double* J = &dJp[6 * k];
      const double* v = x + o_pose[k];
      for (int c = 0; c < 6; ++c) out += J[c] * v[c];
    }
    if (o_dep[k] >= 0) {
      const double* J = &dJc[2 * k];
      const double* v = x + o_dep[k];
      out += J[0] * v[0];
      out += J[1] * v[1];
    }
    return out;
  }

  // y = J_f^T t (gathered per camera-side block); with `factors_too`, plus
  // J_factor^T m.
  void jft_times_t(double* y, bool factors_too) {
    gather(
        [&](const Unit& un, double* acc) {
          const FBlock& fb = fblocks[un.block];
          const int n = fb.size;
          std::size_t q0, q1;
          const std::vector<std::size_t>* list;
          obs_range(fb, q0, q1, list);
          q0 = un.q0;
          q1 = un.q1;
          {
            if (fb.kind != BlockKind::Depth) {
              for (std::size_t q = q0; q < q1; ++q) {
                const std::size_t k = (*list)[q];
                const double* J = fb.kind == BlockKind::Pose ? &Jp[12 * k] : &Jk[2 * static_cast<std::size_t>(np) * k];
                for (int c = 0; c < n; ++c) acc[c] += J[c] * t[2 * k] + J[n + c] * t[2 * k + 1];
              }
            }
            if (has_depth && fb.kind != BlockKind::Intrinsics) {
              for (std::size_t q = q0; q < q1; ++q) {
                const std::size_t k = (*list)[q];
                if (const double* Jd = depth_row(fb.kind, k))
                  for (int c = 0; c < n; ++c) acc[c] += Jd[c] * td[k];
              }
            }
          }
        },
        [&](std::size_t b, double* acc) {
          const FBlock& fb = fblocks[b];
          const int n = fb.size;
          {
            if (factors_too) {
              for (const FactorSet& fs : fsets) {
                for (std::size_t q = fs.begin[b]; q < fs.begin[b + 1]; ++q) {
                  const auto [f, slot] = fs.list[q];
                  const double* J = fs.Jf(static_cast<std::size_t>(f));
                  const double* m = fs.mf(static_cast<std::size_t>(f));
                  const int base = fs.slot_base[slot];
                  for (int row = 0; row < fs.rows; ++row)
                    for (int c = 0; c < n; ++c) acc[c] += J[row * fs.cols + base + c] * m[row];
                }
              }
            }
            for (int c = 0; c < n; ++c) y[fb.offset + c] = acc[c];
          }
        });
  }

  // y = S x, S = J_f^T J_f + D_f^2 - J_f^T J_e (J_e^T J_e + D_e^2)^-1 J_e^T J_f.
  void apply_schur(const double* x, double* y) {
    pool.parallel_for(point_off.size(), [&](std::size_t j0, std::size_t j1) {
      const bool depth = has_depth;
      for (std::size_t j = j0; j < j1; ++j) {
        double u[3] = {0, 0, 0};
        const int e = point_off[j];
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          jf_times(x, k, &t[2 * k]);
          if (e >= 0) {
            const double* J = &Jx[6 * k];
            for (int c = 0; c < 3; ++c) u[c] += J[c] * t[2 * k] + J[3 + c] * t[2 * k + 1];
          }
          if (depth && o_dw[k] != 0) {
            td[k] = jf_depth(x, k);
            if (e >= 0)
              for (int c = 0; c < 3; ++c) u[c] += dJx[3 * k + c] * td[k];
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
          if (depth && o_dw[k] != 0) {
            const double* Jd = &dJx[3 * k];
            td[k] -= Jd[0] * w[0] + Jd[1] * w[1] + Jd[2] * w[2];
          }
        }
      }
    });
    factors_times(x);
    jft_times_t(y, true);
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
      const bool depth = has_depth;
      for (std::size_t j = j0; j < j1; ++j) {
        const int e = point_off[j];
        if (e < 0) continue;
        double C[9] = {};
        for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
          const double* J = &Jx[6 * k];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) C[3 * a + b] += J[a] * J[b] + J[3 + a] * J[3 + b];
          if (depth && o_dw[k] != 0) {
            const double* Jd = &dJx[3 * k];
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b) C[3 * a + b] += Jd[a] * Jd[b];
          }
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
    gather(
        [&](const Unit& un, double* P) {
          const FBlock& fb = fblocks[un.block];
          const int n = fb.size;
          const bool depth_here = has_depth && fb.kind != BlockKind::Intrinsics;
          std::size_t q, qend;
          const std::vector<std::size_t>* lp;
          obs_range(fb, q, qend, lp);
          q = un.q0;
          qend = un.q1;
          {
            while (q < qend) {
              const auto& list = *lp;
              const std::int32_t j = o_pt[list[q]];
              double E[kMaxParams * 3] = {};
              for (; q < qend && o_pt[list[q]] == j; ++q) {
                const std::size_t k = list[q];
                if (fb.kind != BlockKind::Depth) {
                  const double* J = fb.kind == BlockKind::Pose ? &Jp[12 * k] : &Jk[2 * static_cast<std::size_t>(np) * k];
                  const double* X = &Jx[6 * k];
                  for (int a = 0; a < n; ++a) {
                    for (int c = 0; c < n; ++c) P[a * n + c] += J[a] * J[c] + J[n + a] * J[n + c];
                    for (int c = 0; c < 3; ++c) E[3 * a + c] += J[a] * X[c] + J[n + a] * X[3 + c];
                  }
                }
                if (!depth_here) continue;
                if (const double* Jd = depth_row(fb.kind, k)) {
                  const double* X = &dJx[3 * k];
                  for (int a = 0; a < n; ++a) {
                    for (int c = 0; c < n; ++c) P[a * n + c] += Jd[a] * Jd[c];
                    for (int c = 0; c < 3; ++c) E[3 * a + c] += Jd[a] * X[c];
                  }
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
          }
        },
        [&](std::size_t b, double* P) {
          const FBlock& fb = fblocks[b];
          const int n = fb.size;
          {
            for (const FactorSet& fs : fsets) {
              for (std::size_t qf = fs.begin[b]; qf < fs.begin[b + 1]; ++qf) {
                const auto [f, slot] = fs.list[qf];
                const double* J = fs.Jf(static_cast<std::size_t>(f));
                const int base = fs.slot_base[slot], cols = fs.cols;
                for (int row = 0; row < fs.rows; ++row)
                  for (int a = 0; a < n; ++a)
                    for (int c = 0; c < n; ++c) P[a * n + c] += J[row * cols + base + a] * J[row * cols + base + c];
              }
            }
            for (int a = 0; a < n; ++a) P[a * n + a] += d2_f[fb.offset + a];
            if (!spd_inverse(P, n)) ok = false;
            std::copy(P, P + n * n, &Pinv[static_cast<std::size_t>(fb.pinv_offset)]);
          }
        });
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
          if (has_depth && o_dw[k] != 0) {
            const double* Jd = &dJx[3 * k];
            td[k] = Jd[0] * w[0] + Jd[1] * w[1] + Jd[2] * w[2];
          }
        }
      }
    });
    std::vector<double> b(nf);
    jft_times_t(b.data(), false);
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
          const bool depth_row_k = has_depth && o_dw[k] != 0;
          if (depth_row_k) td[k] = jf_depth(dx_f.data(), k);
          if (e < 0) continue;
          const double* J = &Jx[6 * k];
          for (int c = 0; c < 3; ++c) u[c] += J[c] * t[2 * k] + J[3 + c] * t[2 * k + 1];
          if (depth_row_k)
            for (int c = 0; c < 3; ++c) u[c] += dJx[3 * k + c] * td[k];
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
    if (has_depth) {
      out.model_cost_change -= pool.sum(nobs, [&](std::size_t k) {
        if (o_dw[k] == 0) return 0.0;
        double m = td[k];
        const int e = point_off[o_pt[k]];
        if (e >= 0) {
          const double* J = &dJx[3 * k];
          m += J[0] * dx_e[e] + J[1] * dx_e[e + 1] + J[2] * dx_e[e + 2];
        }
        return m * (dr[k] + m / 2);
      });
    }
    if (!fsets.empty()) {
      factors_times(dx_f.data());
      for (const FactorSet& fs : fsets) {
        out.model_cost_change -= pool.sum(fs.count, [&](std::size_t f) {
          const double* m = fs.mf(f);
          const double* rf = fs.rf(f);
          double v = 0;
          for (int row = 0; row < fs.rows; ++row) v += m[row] * (rf[row] + m[row] / 2);
          return v;
        });
      }
    }

    out.step_norm = retract(dx_f.data(), dx_e.data());
    out.valid = std::isfinite(out.model_cost_change) && out.model_cost_change > 0 && std::isfinite(out.step_norm);
    return out;
  }
  // ---------------------------------------------------------------------------
  // Covariance.

  // In-place inverse of a dense symmetric positive definite matrix (n x n), by
  // Cholesky; rows run in parallel. False when it is not positive definite, or a
  // pivot falls below 1e-14 of its diagonal entry: the matrix is singular up to
  // rounding (Ceres's covariance uses the same reciprocal condition number).
  bool dense_spd_inverse(std::vector<double>& A, std::size_t n, std::size_t* failed = nullptr) {
    // A = L L^T, L in the lower triangle.
    for (std::size_t j = 0; j < n; ++j) {
      const double diag = A[j * n + j];
      double d = diag;
      for (std::size_t k = 0; k < j; ++k) d -= A[j * n + k] * A[j * n + k];
      if (!(d > 1e-14 * diag) || !std::isfinite(d)) {
        if (failed) *failed = j;
        return false;
      }
      const double ljj = std::sqrt(d);
      A[j * n + j] = ljj;
      pool.parallel_for(n - j - 1, [&](std::size_t a0, std::size_t a1) {
        for (std::size_t a = a0; a < a1; ++a) {
          const std::size_t i = j + 1 + a;
          double v = A[i * n + j];
          for (std::size_t k = 0; k < j; ++k) v -= A[i * n + k] * A[j * n + k];
          A[i * n + j] = v / ljj;
        }
      }, 64);
    }
    // Columns of L^-1, then A^-1 = L^-T L^-1 (lower triangle, mirrored).
    std::vector<double> Li(n * n, 0.0);
    pool.parallel_for(n, [&](std::size_t c0, std::size_t c1) {
      for (std::size_t c = c0; c < c1; ++c) {
        Li[c * n + c] = 1 / A[c * n + c];
        for (std::size_t i = c + 1; i < n; ++i) {
          double v = 0;
          for (std::size_t k = c; k < i; ++k) v -= A[i * n + k] * Li[k * n + c];
          Li[i * n + c] = v / A[i * n + i];
        }
      }
    }, 8);
    pool.parallel_for(n, [&](std::size_t i0, std::size_t i1) {
      for (std::size_t i = i0; i < i1; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
          double v = 0;
          for (std::size_t k = i; k < n; ++k) v += Li[k * n + i] * Li[k * n + j];
          A[i * n + j] = v;
        }
    }, 8);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < i; ++j) A[j * n + i] = A[i * n + j];
    return true;
  }

  // The camera-side columns and Jacobian rows of observation k's residual rows:
  // up to 3 rows (2 reprojection, 1 depth) over pose, intrinsics and depth blocks.
  struct ObsRows {
    int rows = 0;
    int off[3] = {-1, -1, -1}, size[3] = {0, 0, 0};
    double J[3][3 * kMaxParams] = {};  // [block][row * size + col]
    double X[3][3] = {};               // point Jacobian per row
  };
  ObsRows obs_rows(std::size_t k) const {
    ObsRows o;
    const bool depth = has_depth && o_dw[k] != 0;
    o.rows = depth ? 3 : 2;
    if (o_pose[k] >= 0) {
      o.off[0] = o_pose[k];
      o.size[0] = 6;
      for (int c = 0; c < 6; ++c) {
        o.J[0][c] = Jp[12 * k + c];
        o.J[0][6 + c] = Jp[12 * k + 6 + c];
        if (depth) o.J[0][12 + c] = dJp[6 * k + c];
      }
    }
    if (o_intr[k] >= 0) {
      o.off[1] = o_intr[k];
      o.size[1] = np;
      for (int c = 0; c < np; ++c) {
        o.J[1][c] = Jk[2 * static_cast<std::size_t>(np) * k + c];
        o.J[1][np + c] = Jk[2 * static_cast<std::size_t>(np) * k + np + c];
      }
    }
    if (depth && o_dep[k] >= 0) {
      o.off[2] = o_dep[k];
      o.size[2] = 2;
      o.J[2][4] = dJc[2 * k];
      o.J[2][5] = dJc[2 * k + 1];
    }
    for (int c = 0; c < 3; ++c) {
      o.X[0][c] = Jx[6 * k + c];
      o.X[1][c] = Jx[6 * k + 3 + c];
      if (depth) o.X[2][c] = dJx[3 * k + c];
    }
    return o;
  }

  void covariance(const std::vector<int>& want_points, std::size_t max_unknowns, std::vector<double>& pose_cov,
                  std::vector<double>& point_cov) {
    if (nf > max_unknowns)
      throw std::runtime_error("covariance: " + std::to_string(nf) + " camera-side unknowns, more than the limit of " +
                               std::to_string(max_unknowns));
    // Unscaled, robustified Jacobian at the current parameters.
    have_scaling = false;
    std::fill(f_scale.begin(), f_scale.end(), 1.0);
    std::fill(e_scale.begin(), e_scale.end(), 1.0);
    linearize(false);
    const std::size_t n = nf;
    std::vector<double> S(n * n, 0.0);
    // Point blocks C_j = J_e^T J_e (undamped) and their inverses.
    // A free point seen from fewer than two images absorbs its observations entirely:
    // it is left out, which is the limit of marginalising it.
    std::vector<char> point_ok(point_off.size(), 1), point_used(point_off.size(), 1);
    for (std::size_t j = 0; j < point_off.size(); ++j) {
      if (point_off[j] < 0) continue;
      int images = 0;
      for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k)
        if (k == pt_begin[j] || o_img[k] != o_img[k - 1]) ++images;
      if (images < 2) {
        point_used[j] = point_ok[j] = 0;
        continue;
      }
      double C[9] = {};
      for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
        const ObsRows o = obs_rows(k);
        for (int row = 0; row < o.rows; ++row)
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) C[3 * a + b] += o.X[row][a] * o.X[row][b];
      }
      point_ok[j] = spd_inverse(C, 3);
      std::copy(C, C + 9, &Cinv[9 * j]);
    }
    // S = sum over residuals of J_f^T J_f - sum over points E_j C_j^-1 E_j^T.
    for (std::size_t j = 0; j < point_off.size(); ++j) {
      if (!point_used[j]) continue;
      const bool eliminate = point_off[j] >= 0 && point_ok[j];
      // E_j as a list of (column, 3 values).
      std::vector<std::pair<int, std::array<double, 3>>> E;
      for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
        const ObsRows o = obs_rows(k);
        for (int b1 = 0; b1 < 3; ++b1) {
          if (o.off[b1] < 0) continue;
          for (int c1 = 0; c1 < o.size[b1]; ++c1) {
            const auto col = static_cast<std::size_t>(o.off[b1] + c1);
            for (int b2 = 0; b2 < 3; ++b2) {
              if (o.off[b2] < 0) continue;
              for (int c2 = 0; c2 < o.size[b2]; ++c2) {
                double v = 0;
                for (int row = 0; row < o.rows; ++row) v += o.J[b1][row * o.size[b1] + c1] * o.J[b2][row * o.size[b2] + c2];
                S[col * n + static_cast<std::size_t>(o.off[b2] + c2)] += v;
              }
            }
            if (eliminate) {
              std::array<double, 3> e{};
              for (int row = 0; row < o.rows; ++row)
                for (int a = 0; a < 3; ++a) e[static_cast<std::size_t>(a)] += o.J[b1][row * o.size[b1] + c1] * o.X[row][a];
              E.push_back({static_cast<int>(col), e});
            }
          }
        }
      }
      if (!eliminate) continue;
      const double* Ci = &Cinv[9 * j];
      for (const auto& [c1, e1] : E) {
        double w[3];
        for (int a = 0; a < 3; ++a) w[a] = Ci[3 * a] * e1[0] + Ci[3 * a + 1] * e1[1] + Ci[3 * a + 2] * e1[2];
        for (const auto& [c2, e2] : E)
          S[static_cast<std::size_t>(c1) * n + static_cast<std::size_t>(c2)] -= w[0] * e2[0] + w[1] * e2[1] + w[2] * e2[2];
      }
    }
    for (const FactorSet& fs : fsets) {
      for (std::size_t f = 0; f < fs.count; ++f) {
        const double* J = fs.Jf(f);
        for (int s1 = 0; s1 < fs.nslots; ++s1) {
          if (fs.off(f, s1) < 0) continue;
          for (int s2 = 0; s2 < fs.nslots; ++s2) {
            if (fs.off(f, s2) < 0) continue;
            for (int c1 = 0; c1 < fs.slot_size[s1]; ++c1)
              for (int c2 = 0; c2 < fs.slot_size[s2]; ++c2) {
                double v = 0;
                for (int row = 0; row < fs.rows; ++row)
                  v += J[row * fs.cols + fs.slot_base[s1] + c1] * J[row * fs.cols + fs.slot_base[s2] + c2];
                S[static_cast<std::size_t>(fs.off(f, s1) + c1) * n + static_cast<std::size_t>(fs.off(f, s2) + c2)] += v;
              }
          }
        }
      }
    }
    // Held coordinates: a unit diagonal, zeroed again afterwards.
    const std::vector<char> free = free_columns();
    for (std::size_t c = 0; c < n; ++c) {
      if (free[c]) continue;
      for (std::size_t a = 0; a < n; ++a) S[c * n + a] = S[a * n + c] = 0;
      S[c * n + c] = 1;
    }
    std::size_t failed = 0;
    if (n > 0 && !dense_spd_inverse(S, n, &failed)) {
      std::string what = "unknown " + std::to_string(failed);
      for (const FBlock& fb : fblocks) {
        if (static_cast<int>(failed) < fb.offset || static_cast<int>(failed) >= fb.offset + fb.size) continue;
        static constexpr const char* kKinds[] = {"pose of image",   "intrinsics block",          "prior set",
                                                 "prior set scale", "Manhattan frame",           "depth correction of image",
                                                 "mean of intrinsics group"};
        static_assert(sizeof(kKinds) / sizeof(kKinds[0]) == static_cast<std::size_t>(BlockKind::IntrinsicsGroup) + 1,
                      "a name per block kind");
        what = std::string(kKinds[static_cast<int>(fb.kind)]) + " " + std::to_string(fb.index) + ", coordinate " +
               std::to_string(static_cast<int>(failed) - fb.offset);
      }
      throw std::runtime_error("covariance: the system is singular at " + what +
                               "; some freedom is not held (gauge) or not observed");
    }
    for (std::size_t c = 0; c < n; ++c) {
      if (free[c]) continue;
      for (std::size_t a = 0; a < n; ++a) S[c * n + a] = S[a * n + c] = 0;
    }
    // Poses.
    const std::size_t n_img = pose_off.size();
    pose_cov.assign(36 * n_img, 0.0);
    for (std::size_t i = 0; i < n_img; ++i) {
      const int o = pose_off[i];
      if (o < 0) continue;
      double* P = &pose_cov[36 * i];
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
          P[6 * a + b] = S[static_cast<std::size_t>(o + a) * n + static_cast<std::size_t>(o + b)];
      // A pose whose heading the gauge holds (D83) has its rotation in the basis B of
      // heading_basis, dtheta = B delta; back to the pose tangent: P <- T P T^T with
      // T = diag(B, I). Its rotation block is then singular along the vertical.
      if (basis_image.empty() || basis_of[i] < 0) continue;
      const double* B = &basis[9 * static_cast<std::size_t>(basis_of[i])];
      double Q[36];
      for (int a = 0; a < 6; ++a)  // Q = T P
        for (int b = 0; b < 6; ++b)
          Q[6 * a + b] = a < 3 ? B[3 * a] * P[b] + B[3 * a + 1] * P[6 + b] + B[3 * a + 2] * P[12 + b] : P[6 * a + b];
      for (int a = 0; a < 6; ++a)  // P = Q T^T
        for (int b = 0; b < 6; ++b)
          P[6 * a + b] = b < 3 ? Q[6 * a] * B[3 * b] + Q[6 * a + 1] * B[3 * b + 1] + Q[6 * a + 2] * B[3 * b + 2] : Q[6 * a + b];
    }
    // Points: C^-1 + C^-1 E^T Sigma_f E C^-1.
    point_cov.assign(9 * want_points.size(), 0.0);
    for (std::size_t q = 0; q < want_points.size(); ++q) {
      const auto j = static_cast<std::size_t>(want_points[q]);
      if (j >= point_off.size() || point_off[j] < 0) continue;
      double* out = &point_cov[9 * q];
      if (!point_ok[j]) {
        for (int a = 0; a < 9; ++a) out[a] = std::nan("");
        continue;
      }
      std::vector<std::pair<int, std::array<double, 3>>> E;
      for (std::size_t k = pt_begin[j]; k < pt_begin[j + 1]; ++k) {
        const ObsRows o = obs_rows(k);
        for (int b1 = 0; b1 < 3; ++b1) {
          if (o.off[b1] < 0) continue;
          for (int c1 = 0; c1 < o.size[b1]; ++c1) {
            std::array<double, 3> e{};
            for (int row = 0; row < o.rows; ++row)
              for (int a = 0; a < 3; ++a) e[static_cast<std::size_t>(a)] += o.J[b1][row * o.size[b1] + c1] * o.X[row][a];
            E.push_back({o.off[b1] + c1, e});
          }
        }
      }
      // M = E^T Sigma_f E (3 x 3).
      double M[9] = {};
      for (const auto& [c1, e1] : E)
        for (const auto& [c2, e2] : E) {
          const double s = S[static_cast<std::size_t>(c1) * n + static_cast<std::size_t>(c2)];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) M[3 * a + b] += e1[static_cast<std::size_t>(a)] * s * e2[static_cast<std::size_t>(b)];
        }
      const double* Ci = &Cinv[9 * j];
      double T[9];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) T[3 * a + b] = Ci[3 * a] * M[b] + Ci[3 * a + 1] * M[3 + b] + Ci[3 * a + 2] * M[6 + b];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          out[3 * a + b] = Ci[3 * a + b] + T[3 * a] * Ci[b] + T[3 * a + 1] * Ci[3 + b] + T[3 * a + 2] * Ci[6 + b];
    }
  }

};

CpuBackend::CpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : impl_(std::make_unique<Impl>(problem, options, gauge)) {}

CpuBackend::~CpuBackend() = default;

double CpuBackend::cost() { return impl_->cost_of(impl_->cur); }

void CpuBackend::cost_by_type(double* out) { impl_->cost_by_type(out); }

void CpuBackend::covariance(const std::vector<int>& points, std::size_t max_unknowns, std::vector<double>& pose_cov,
                            std::vector<double>& point_cov) {
  impl_->covariance(points, max_unknowns, pose_cov, point_cov);
}

void CpuBackend::linearize(bool compute_scaling) { impl_->linearize(compute_scaling); }

double CpuBackend::gradient_max_norm() const { return impl_->gmax; }

double CpuBackend::parameter_norm() const { return impl_->parameter_norm(); }

StepResult CpuBackend::compute_step(double radius, double eta, int min_linear_iterations, int max_linear_iterations) {
  return impl_->compute_step(radius, eta, min_linear_iterations, max_linear_iterations);
}

bool CpuBackend::candidate_cost(double* cost) {
  *cost = impl_->cost_of(impl_->cand);
  return std::isfinite(*cost);
}

void CpuBackend::accept_candidate() {
  impl_->cur.swap(impl_->cand);
}

void CpuBackend::write_back(Problem& problem) const { impl_->write_back(problem); }

}  // namespace jba::cpu
