// Kernels of the GPU backends (Metal, milestone 4; CUDA, milestone 5; D96).
//
// One source for both. It is compiled after a prelude that defines the macros below
// (backends/metal/kernels/prelude.metal, backends/cuda/kernels/prelude.cuh), the
// generated reprojection code of that target, and the shared structures
// (backends/gpu/internal/jba/gpu/layout.h):
//
//   JBA_KERNEL(name)  a kernel with Args a, State& s, Op op and the indices gid
//                     (thread in the grid), group (group in the grid) and lid
//                     (thread in its group); one-dimensional dispatches only
//   JBA_FUNC          a helper function
//   JBA_DEVICE        device memory;  JBA_THREAD  a thread's own memory
//   JBA_ARGS_REF      a helper's reference to Args
//   JBA_SHARED        memory shared by a group;  JBA_BARRIER()  a group barrier
//   JBA_SET_BAD(s), JBA_GET_BAD(s)  the State's flag for a block that is not
//                     positive definite (an atomic on Metal)
//
// Single precision throughout; every decision is taken on the host in double.
//
// Every accumulation is a gather in a fixed order (document 03, "Accumulation without
// float atomics"): each output is written by one thread, and sums over long lists are
// split into fixed chunks whose partial sums are added in order. Results are the
// same from run to run.
//
// Buffers are reached through one argument structure (Args) of device pointers;
// per-dispatch choices (which vectors, which mode) come in a small Op structure.
// Every kernel checks its own bounds, so a dispatch may round its thread count up.

using namespace jba_gpu;

// ---------------------------------------------------------------------------
// Small helpers.

// Pivots below a share of their diagonal entry (Args::pivot_floor) are raised to it. In single
// precision a block that is positive definite in exact arithmetic (a point seen with
// a weak depth direction, once the damping is small) can lose its smallest pivot to
// rounding; raising it adds a little damping in that direction instead of failing
// the step. The host checks every step against the cost in double.

// In-place inverse of a symmetric positive semi-definite n x n matrix by Cholesky,
// with the pivot floor. False only for a diagonal entry that is not positive or a
// value that is not finite.
JBA_FUNC bool spd_inverse(JBA_THREAD float* A, int n, float floor) {
  float L[kMaxParams * kMaxParams];
  float Li[kMaxParams * kMaxParams];
  for (int i = 0; i < n * n; ++i) L[i] = Li[i] = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      float s = A[i * n + j];
      for (int k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
      if (i == j) {
        const float diag = A[i * n + i];
        if (!(diag > 0) || !isfinite(s)) return false;
        s = max(s, floor * diag);
        L[i * n + i] = sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    Li[i * n + i] = 1 / L[i * n + i];
    for (int j = 0; j < i; ++j) {
      float s = 0;
      for (int k = j; k < i; ++k) s -= L[i * n + k] * Li[k * n + j];
      Li[i * n + j] = s / L[i * n + i];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      float s = 0;
      for (int k = i; k < n; ++k) s += Li[k * n + i] * Li[k * n + j];
      A[i * n + j] = A[j * n + i] = s;
    }
  }
  return true;
}

// rho'(s), rho''(s) of the robust loss (jba/loss.h).
JBA_FUNC void loss_derivatives(int type, float a, float s, JBA_THREAD float& rho1, JBA_THREAD float& rho2) {
  const float a2 = a * a;
  rho1 = 1;
  rho2 = 0;
  if (type == kHuber) {
    if (s > a2) {
      const float r = sqrt(s);
      rho1 = a / r;
      rho2 = -rho1 / (2 * s);
    }
  } else if (type == kCauchy) {
    const float inv = 1 / (1 + s / a2);
    rho1 = inv;
    rho2 = -inv * inv / a2;
  } else if (type == kGemanMcClure) {
    const float d = a2 + s;
    rho1 = a2 * a2 / (d * d);
    rho2 = -2 * rho1 / d;
  }
}

// Ceres's corrector (jba/cpu/system.h) on a two-row residual and its Jacobian blocks.
struct Corrector {
  float sqrt_rho1, residual_scaling, alpha_sq_norm;
};

JBA_FUNC Corrector make_corrector(float sq, float rho1, float rho2) {
  Corrector c;
  c.sqrt_rho1 = sqrt(rho1);
  if (sq == 0 || rho2 <= 0) {
    c.residual_scaling = c.sqrt_rho1;
    c.alpha_sq_norm = 0;
    return c;
  }
  const float D = 1 + 2 * sq * rho2 / rho1;
  const float alpha = 1 - sqrt(D);
  c.residual_scaling = c.sqrt_rho1 / (1 - alpha);
  c.alpha_sq_norm = alpha / sq;
  return c;
}

JBA_FUNC void correct(Corrector c, JBA_THREAD const float* r, JBA_THREAD float* J, int cols) {
  if (c.alpha_sq_norm == 0) {
    for (int i = 0; i < 2 * cols; ++i) J[i] *= c.sqrt_rho1;
    return;
  }
  for (int k = 0; k < cols; ++k) {
    const float rJ = r[0] * J[k] + r[1] * J[cols + k];
    J[k] = c.sqrt_rho1 * (J[k] - c.alpha_sq_norm * r[0] * rJ);
    J[cols + k] = c.sqrt_rho1 * (J[cols + k] - c.alpha_sq_norm * r[1] * rJ);
  }
}

// ---------------------------------------------------------------------------
// Linearization of the reprojection residual, one thread per observation.
//
// The device sees points minus a centre c and translations t + R c, so single
// precision is spent on the scene, not on its offset (document 03). The residual
// itself comes from the host (hres). The generated
// rotation Jacobian is then the one for p = R X' + t'; the tangent is still the
// library's (D46: R <- Exp(dtheta) R, t <- t + dt), whose rotation Jacobian is
// J_theta - J_t x (R c).

#define JBA_LINEARIZE_KERNEL(NAME, NP)                                                    \
  JBA_KERNEL(linearize_##NAME) {                                                         \
    const uint k = gid;                                                                  \
    if (k >= a.nobs) return;                                                             \
    const int i = a.o_img[k];                                                            \
    const int j = a.o_pt[k];                                                             \
    const int blk = a.image_block[i];                                                    \
    float q[4], tt[3], X[3], K[NP], o[2];                                                \
    for (int c = 0; c < 4; ++c) q[c] = a.poses[8 * i + c];                               \
    for (int c = 0; c < 3; ++c) tt[c] = a.poses[8 * i + 4 + c];                          \
    for (int c = 0; c < 3; ++c) X[c] = a.points[3 * j + c];                              \
    for (int c = 0; c < NP; ++c) K[c] = a.intr[NP * blk + c];                            \
    o[0] = a.o_xy[2 * k];                                                                \
    o[1] = a.o_xy[2 * k + 1];                                                            \
    float r[2], jp[12], jx[6], jk[2 * NP];                                               \
    jba_generated::reprojection_##NAME<float>(q, tt, X, K, o, r, jp, jx, jk);            \
    const float v0 = a.rc[4 * i], v1 = a.rc[4 * i + 1], v2 = a.rc[4 * i + 2];            \
    for (int row = 0; row < 2; ++row) {                                                  \
      JBA_THREAD float* J = jp + 6 * row;                                                    \
      J[0] -= J[4] * v2 - J[5] * v1;                                                     \
      J[1] -= J[5] * v0 - J[3] * v2;                                                     \
      J[2] -= J[3] * v1 - J[4] * v0;                                                     \
    }                                                                                    \
    const int bi = a.basis_of[i];                                                        \
    if (bi >= 0) {                                                                       \
      JBA_DEVICE const float* B = a.basis + 9 * bi;                                          \
      for (int row = 0; row < 2; ++row) {                                                \
        JBA_THREAD float* J = jp + 6 * row;                                                  \
        const float j0 = J[0], j1 = J[1], j2 = J[2];                                     \
        for (int c = 0; c < 3; ++c) J[c] = j0 * B[c] + j1 * B[3 + c] + j2 * B[6 + c];    \
      }                                                                                  \
    }                                                                                    \
    /* The whitened residual from the host, evaluated in double and rounded after */  \
    /* the subtraction: it sets the gradient's accuracy (document 03, Precision). */      \
    const float w = a.o_w[k];                                                            \
    r[0] = a.hres[2 * k];                                                                \
    r[1] = a.hres[2 * k + 1];                                                            \
    for (int c = 0; c < 12; ++c) jp[c] *= w;                                             \
    for (int c = 0; c < 6; ++c) jx[c] *= w;                                              \
    for (int c = 0; c < 2 * NP; ++c) jk[c] *= w;                                         \
    if (a.loss_type != kNone) {                                                          \
      const float s = r[0] * r[0] + r[1] * r[1];                                         \
      float rho1, rho2;                                                                  \
      loss_derivatives(a.loss_type, a.loss_scale, s, rho1, rho2);                        \
      const Corrector corr = make_corrector(s, rho1, rho2);                              \
      correct(corr, r, jp, 6);                                                           \
      correct(corr, r, jx, 3);                                                           \
      correct(corr, r, jk, NP);                                                          \
      r[0] *= corr.residual_scaling;                                                     \
      r[1] *= corr.residual_scaling;                                                     \
    }                                                                                    \
    const uint pmask = a.pose_mask[i];                                                   \
    for (int c = 0; c < 6; ++c)                                                          \
      if (pmask & (1u << c)) jp[c] = jp[6 + c] = 0;                                      \
    if (a.point_off[j] < 0)                                                              \
      for (int c = 0; c < 6; ++c) jx[c] = 0;                                             \
    a.r[2 * k] = r[0];                                                                   \
    a.r[2 * k + 1] = r[1];                                                               \
    for (int c = 0; c < 12; ++c) a.Jp[12 * k + c] = jp[c];                               \
    for (int c = 0; c < 6; ++c) a.Jx[6 * k + c] = jx[c];                                 \
    if (a.has_intr) {                                                                    \
      const uint kmask = a.intr_mask[blk];                                               \
      for (int c = 0; c < NP; ++c)                                                       \
        if (kmask & (1u << c)) jk[c] = jk[NP + c] = 0;                                   \
      for (int c = 0; c < 2 * NP; ++c) a.Jk[2 * NP * k + c] = jk[c];                     \
    }                                                                                    \
  }

JBA_LINEARIZE_KERNEL(simple_pinhole, 3)
JBA_LINEARIZE_KERNEL(pinhole, 4)
JBA_LINEARIZE_KERNEL(simple_radial, 4)
JBA_LINEARIZE_KERNEL(radial, 5)
JBA_LINEARIZE_KERNEL(opencv, 8)
JBA_LINEARIZE_KERNEL(opencv_fisheye, 8)

// Jacobian column scaling, once the scale is known: every observation's blocks.
JBA_KERNEL(scale_observations) {
  const uint k = gid;
  if (k >= a.nobs) return;
  const int po = a.o_pose[k], io = a.o_intr[k], e = a.point_off[a.o_pt[k]];
  const bool depth = a.has_depth && a.o_dw[k] != 0;
  if (po >= 0) {
    for (int c = 0; c < 6; ++c) {
      const float s = a.f_scale[po + c];
      a.Jp[12 * k + c] *= s;
      a.Jp[12 * k + 6 + c] *= s;
      if (depth) a.dJp[6 * k + c] *= s;
    }
  }
  if (io >= 0 && a.has_intr) {
    const int np = int(a.np);
    for (int c = 0; c < np; ++c) {
      const float s = a.f_scale[io + c];
      a.Jk[2 * np * k + c] *= s;
      a.Jk[2 * np * k + np + c] *= s;
    }
  }
  if (e >= 0) {
    for (int c = 0; c < 3; ++c) {
      const float s = a.e_scale[e + c];
      a.Jx[6 * k + c] *= s;
      a.Jx[6 * k + 3 + c] *= s;
      if (depth) a.dJx[3 * k + c] *= s;
    }
  }
  if (depth && a.o_dep[k] >= 0)
    for (int c = 0; c < 2; ++c) a.dJc[2 * k + c] *= a.f_scale[a.o_dep[k] + c];
}

JBA_KERNEL(scale_factors) {
  const uint f = gid;
  if (f >= a.nfac) return;
  const Factor F = a.factors[f];
  JBA_DEVICE float* J = a.fJ + F.joff;
  for (int s = 0; s < F.nslots; ++s) {
    if (F.off[s] < 0) continue;
    for (int row = 0; row < F.rows; ++row)
      for (int c = 0; c < F.size[s]; ++c) J[row * F.cols + F.base[s] + c] *= a.f_scale[F.off[s] + c];
  }
}

// ---------------------------------------------------------------------------
// Camera-side rows of an observation for one block kind: up to 3 rows (2
// reprojection, 1 depth), n columns each, with the point Jacobian rows.

struct Rows {
  int n;
  float J[3][kMaxParams];
  float X[3][3];
  float v[3];  // the vector the rows multiply: residual or t
};

// `use_t`: take v from (t, td) instead of (r, dr).
JBA_FUNC Rows rows_of(JBA_ARGS_REF a, int kind, uint k, bool use_t) {
  Rows o;
  o.n = 0;
  const bool depth = a.has_depth && a.o_dw[k] != 0;
  if (kind == kPose) {
    for (int row = 0; row < 2; ++row)
      for (int c = 0; c < 6; ++c) o.J[row][c] = a.Jp[12 * k + 6 * row + c];
    o.n = 2;
    if (depth) {
      for (int c = 0; c < 6; ++c) o.J[2][c] = a.dJp[6 * k + c];
      o.n = 3;
    }
  } else if (kind == kIntrinsics) {
    const int np = int(a.np);
    for (int row = 0; row < 2; ++row)
      for (int c = 0; c < np; ++c) o.J[row][c] = a.Jk[2 * np * k + np * row + c];
    o.n = 2;
  } else if (kind == kDepth && depth) {
    // The depth row only; put it in row 2 so X and v line up.
    o.J[0][0] = o.J[0][1] = o.J[1][0] = o.J[1][1] = 0;
    o.J[2][0] = a.dJc[2 * k];
    o.J[2][1] = a.dJc[2 * k + 1];
    o.n = 3;
  }
  for (int c = 0; c < 3; ++c) {
    o.X[0][c] = a.Jx[6 * k + c];
    o.X[1][c] = a.Jx[6 * k + 3 + c];
    o.X[2][c] = depth ? a.dJx[3 * k + c] : 0;
  }
  if (use_t) {
    o.v[0] = a.t[2 * k];
    o.v[1] = a.t[2 * k + 1];
    o.v[2] = depth ? a.td[k] : 0;
  } else {
    o.v[0] = a.r[2 * k];
    o.v[1] = a.r[2 * k + 1];
    o.v[2] = depth ? a.dr[k] : 0;
  }
  return o;
}

// Partial sums of one chunk of a camera-side block's observations.
// mode 0: squared column norms (0..7) and J^T r (8..15).
// mode 1: J^T t (0..7).
// mode 2: the preconditioner block, sum over the chunk's points of
//         J^T J - E C^-1 E^T (n x n, row-major).
JBA_KERNEL(chunk_partial) {
  const uint c = gid;
  if (c >= a.nchunks || (op.iter > 0 && s.done)) return;
  const Chunk ch = a.chunks[c];
  const FBlockDesc fb = a.fblocks[ch.block];
  const int n = fb.size;
  JBA_DEVICE const uint* list = ch.list == 0 ? a.img_list : a.ib_list;
  float acc[kChunkFloats];
  for (int i = 0; i < kChunkFloats; ++i) acc[i] = 0;
  if (op.mode == 0 || op.mode == 1) {
    const bool use_t = op.mode == 1;
    for (uint qi = ch.q0; qi < ch.q1; ++qi) {
      const Rows o = rows_of(a, fb.kind, list[qi], use_t);
      const int r0 = fb.kind == kDepth ? 2 : 0;
      for (int row = r0; row < o.n; ++row)
        for (int col = 0; col < n; ++col) {
          const float v = o.J[row][col];
          if (op.mode == 0) {
            acc[col] += v * v;
            acc[8 + col] += v * o.v[row];
          } else {
            acc[col] += v * o.v[row];
          }
        }
    }
  } else {
    uint qi = ch.q0;
    while (qi < ch.q1) {
      const int j = a.o_pt[list[qi]];
      float E[kMaxParams * 3];
      for (int i = 0; i < 3 * n; ++i) E[i] = 0;
      for (; qi < ch.q1 && a.o_pt[list[qi]] == j; ++qi) {
        const Rows o = rows_of(a, fb.kind, list[qi], false);
        const int r0 = fb.kind == kDepth ? 2 : 0;
        for (int row = r0; row < o.n; ++row)
          for (int x = 0; x < n; ++x) {
            for (int y = 0; y < n; ++y) acc[x * n + y] += o.J[row][x] * o.J[row][y];
            for (int y = 0; y < 3; ++y) E[3 * x + y] += o.J[row][x] * o.X[row][y];
          }
      }
      if (a.point_off[j] < 0) continue;
      JBA_DEVICE const float* Ci = a.Cinv + 9 * j;
      for (int x = 0; x < n; ++x) {
        float EC[3];
        for (int y = 0; y < 3; ++y) EC[y] = E[3 * x] * Ci[y] + E[3 * x + 1] * Ci[3 + y] + E[3 * x + 2] * Ci[6 + y];
        for (int y = 0; y < n; ++y) acc[x * n + y] -= EC[0] * E[3 * y] + EC[1] * E[3 * y + 1] + EC[2] * E[3 * y + 2];
      }
    }
  }
  JBA_DEVICE float* out = a.partial + kChunkFloats * c;
  for (int i = 0; i < kChunkFloats; ++i) out[i] = acc[i];
}

// Per camera-side block: its chunks' partial sums in order, plus its factors.
// mode 0: diag_f and g_f.
// mode 1: y = sum J^T t, plus J_factor^T m and d2 x (the Schur product).
// mode 2: Pinv = (preconditioner block + factors' J^T J + d2)^-1.
// mode 3: b = sum J^T t - g_f (right-hand side; no factors).
JBA_KERNEL(block_finish) {
  const uint b = gid;
  if (b >= a.nfb) return;
  if ((op.mode == 1) && s.done) return;
  const FBlockDesc fb = a.fblocks[b];
  const int n = fb.size;
  float acc[kChunkFloats];
  for (int i = 0; i < kChunkFloats; ++i) acc[i] = 0;
  const int width = op.mode == 0 ? 16 : (op.mode == 2 ? n * n : n);
  for (uint c = a.block_chunks[b]; c < a.block_chunks[b + 1]; ++c) {
    JBA_DEVICE const float* part = a.partial + kChunkFloats * c;
    if (op.mode == 0) {
      for (int i = 0; i < n; ++i) {
        acc[i] += part[i];
        acc[8 + i] += part[8 + i];
      }
    } else {
      for (int i = 0; i < width; ++i) acc[i] += part[i];
    }
  }
  if (op.mode != 3) {
    for (uint e = a.fb_fac_begin[b]; e < a.fb_fac_begin[b + 1]; ++e) {
      const uint2 fs = a.fb_fac[e];
      const Factor F = a.factors[fs.x];
      const int base = F.base[fs.y];
      JBA_DEVICE const float* J = a.fJ + F.joff;
      for (int row = 0; row < F.rows; ++row) {
        if (op.mode == 0) {
          const float rr = a.fr[F.roff + row];
          for (int c = 0; c < n; ++c) {
            const float v = J[row * F.cols + base + c];
            acc[c] += v * v;
            acc[8 + c] += v * rr;
          }
        } else if (op.mode == 1) {
          const float m = a.fm[F.roff + row];
          for (int c = 0; c < n; ++c) acc[c] += J[row * F.cols + base + c] * m;
        } else {
          for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y) acc[x * n + y] += J[row * F.cols + base + x] * J[row * F.cols + base + y];
        }
      }
    }
  }
  const int off = fb.offset;
  if (op.mode == 0) {
    for (int c = 0; c < n; ++c) {
      a.diag_f[off + c] = acc[c];
      a.g_f[off + c] = acc[8 + c];
    }
  } else if (op.mode == 1) {
    for (int c = 0; c < n; ++c) op.y[off + c] = acc[c] + a.d2_f[off + c] * op.x[off + c];
  } else if (op.mode == 3) {
    for (int c = 0; c < n; ++c) a.b[off + c] = acc[c] - a.g_f[off + c];
  } else {
    for (int c = 0; c < n; ++c) acc[c * n + c] += a.d2_f[off + c];
    if (!spd_inverse(acc, n, a.pivot_floor)) JBA_SET_BAD(s);
    JBA_DEVICE float* P = a.Pinv + fb.pinv_offset;
    for (int i = 0; i < n * n; ++i) P[i] = acc[i];
  }
}

// Per camera-side block, one threadgroup of kGatherThreads: thread i takes the
// block's observations i, i + T, ... in order, then a fixed tree adds the threads'
// sums; thread 0 adds the factors. Deterministic, and it hides memory latency that
// one thread per block cannot.
// mode 0: diag_f and g_f (squared column norms, J^T r).
// mode 1: y = J^T t + J_factor^T m + d2 x (the Schur product; skipped once done).
// mode 3: b = J^T t - g_f (right-hand side; no factors).
JBA_KERNEL(block_gather) {
  const uint b = group;
  const uint i = lid;
  JBA_SHARED float buf[kGatherThreads * 16];
  if (op.mode == 1 && s.done) return;
  const FBlockDesc fb = a.fblocks[b];
  const int n = fb.size;
  const int width = op.mode == 0 ? 16 : 8;
  float acc[16];
  for (int c = 0; c < 16; ++c) acc[c] = 0;
  const uint c0 = a.block_chunks[b], c1 = a.block_chunks[b + 1];
  if (c0 < c1) {
    const Chunk first = a.chunks[c0];
    const uint q0 = first.q0, q1 = a.chunks[c1 - 1].q1;
    JBA_DEVICE const uint* list = first.list == 0 ? a.img_list : a.ib_list;
    const bool use_t = op.mode != 0;
    JBA_DEVICE const float* v2 = use_t ? a.t : a.r;
    JBA_DEVICE const float* v1 = use_t ? a.td : a.dr;
    const int np = int(a.np);
    for (uint q = q0 + i; q < q1; q += kGatherThreads) {
      const uint k = list[q];
      const float r0 = v2[2 * k], r1 = v2[2 * k + 1];
      if (fb.kind == kPose) {
        JBA_DEVICE const float* J = a.Jp + 12 * k;
        for (int c = 0; c < 6; ++c) {
          const float j0 = J[c], j1 = J[6 + c];
          if (op.mode == 0) {
            acc[c] += j0 * j0 + j1 * j1;
            acc[8 + c] += j0 * r0 + j1 * r1;
          } else {
            acc[c] += j0 * r0 + j1 * r1;
          }
        }
        if (a.has_depth && a.o_dw[k] != 0) {
          const float rd = v1[k];
          for (int c = 0; c < 6; ++c) {
            const float jd = a.dJp[6 * k + c];
            if (op.mode == 0) {
              acc[c] += jd * jd;
              acc[8 + c] += jd * rd;
            } else {
              acc[c] += jd * rd;
            }
          }
        }
      } else if (fb.kind == kIntrinsics) {
        JBA_DEVICE const float* J = a.Jk + 2 * np * k;
        for (int c = 0; c < np; ++c) {
          const float j0 = J[c], j1 = J[np + c];
          if (op.mode == 0) {
            acc[c] += j0 * j0 + j1 * j1;
            acc[8 + c] += j0 * r0 + j1 * r1;
          } else {
            acc[c] += j0 * r0 + j1 * r1;
          }
        }
      } else if (fb.kind == kDepth && a.has_depth && a.o_dw[k] != 0) {
        const float rd = v1[k];
        for (int c = 0; c < 2; ++c) {
          const float jd = a.dJc[2 * k + c];
          if (op.mode == 0) {
            acc[c] += jd * jd;
            acc[8 + c] += jd * rd;
          } else {
            acc[c] += jd * rd;
          }
        }
      }
    }
  }
  for (int c = 0; c < width; ++c) buf[c * kGatherThreads + i] = acc[c];
  JBA_BARRIER();
  for (uint w = kGatherThreads / 2; w > 0; w /= 2) {
    if (i < w)
      for (int c = 0; c < width; ++c) buf[c * kGatherThreads + i] += buf[c * kGatherThreads + i + w];
    JBA_BARRIER();
  }
  if (i != 0) return;
  for (int c = 0; c < width; ++c) acc[c] = buf[c * kGatherThreads];
  if (op.mode != 3) {
    for (uint e = a.fb_fac_begin[b]; e < a.fb_fac_begin[b + 1]; ++e) {
      const uint2 fs = a.fb_fac[e];
      const Factor F = a.factors[fs.x];
      const int base = F.base[fs.y];
      JBA_DEVICE const float* J = a.fJ + F.joff;
      for (int row = 0; row < F.rows; ++row) {
        const float m = op.mode == 0 ? a.fr[F.roff + row] : a.fm[F.roff + row];
        for (int c = 0; c < n; ++c) {
          const float v = J[row * F.cols + base + c];
          if (op.mode == 0) {
            acc[c] += v * v;
            acc[8 + c] += v * m;
          } else {
            acc[c] += v * m;
          }
        }
      }
    }
  }
  const int off = fb.offset;
  if (op.mode == 0) {
    for (int c = 0; c < n; ++c) {
      a.diag_f[off + c] = acc[c];
      a.g_f[off + c] = acc[8 + c];
    }
  } else if (op.mode == 1) {
    for (int c = 0; c < n; ++c) op.y[off + c] = acc[c] + a.d2_f[off + c] * op.x[off + c];
  } else {
    for (int c = 0; c < n; ++c) a.b[off + c] = acc[c] - a.g_f[off + c];
  }
}

// The reduced camera system: S_ab = sum of J_a^T J_b over observations and factors
// - sum over points of E_aj C_j^-1 E_bj^T, plus D^2 on a diagonal tile. One
// threadgroup per chunk of a tile's entries: threads take entries i, i + T, ... in
// order and a fixed tree adds their sums (schur_tiles); then each tile adds its
// chunks in order (tile_finish).
JBA_KERNEL(schur_tiles) {
  const uint tc = group;
  const uint i = lid;
  JBA_SHARED float buf[kTileThreads * kChunkFloats];
  const TileChunk TC = a.tile_chunks[tc];
  const Tile T = a.tiles[TC.tile];
  const FBlockDesc A = a.fblocks[T.a], B = a.fblocks[T.b];
  const int na = A.size, nb = B.size, nv = na * nb;
  float acc[kChunkFloats];
  for (int c = 0; c < kChunkFloats; ++c) acc[c] = 0;
  for (uint e = TC.e0 + i; e < TC.e1; e += kTileThreads) {
    const TileEntry E = a.entries[e];
    const int ra = int(E.info & 3), rb = int((E.info >> 2) & 3);
    if ((E.info >> 4) & 1) {
      const Factor F = a.factors[E.ka];
      JBA_DEVICE const float* J = a.fJ + F.joff;
      const int ba = F.base[ra], bb = F.base[rb];
      for (int row = 0; row < F.rows; ++row)
        for (int x = 0; x < na; ++x)
          for (int y = 0; y < nb; ++y) acc[x * nb + y] += J[row * F.cols + ba + x] * J[row * F.cols + bb + y];
      continue;
    }
    const int kinds[3] = {kPose, kIntrinsics, kDepth};
    const Rows oa = rows_of(a, kinds[ra], E.ka, false);
    const Rows ob = rows_of(a, kinds[rb], E.kb, false);
    const int a0 = ra == 2 ? 2 : 0, b0 = rb == 2 ? 2 : 0;
    if (E.ka == E.kb) {
      for (int row = max(a0, b0); row < min(oa.n, ob.n); ++row)
        for (int x = 0; x < na; ++x)
          for (int y = 0; y < nb; ++y) acc[x * nb + y] += oa.J[row][x] * ob.J[row][y];
    }
    const int j = a.o_pt[E.ka];
    if (a.point_off[j] < 0) continue;
    // G = E_a C^-1 (na x 3), E_b (nb x 3); acc -= G E_b^T.
    float Ea[kMaxParams * 3], Eb[kMaxParams * 3];
    for (int c = 0; c < 3 * kMaxParams; ++c) Ea[c] = Eb[c] = 0;
    for (int row = a0; row < oa.n; ++row)
      for (int x = 0; x < na; ++x)
        for (int c = 0; c < 3; ++c) Ea[3 * x + c] += oa.J[row][x] * oa.X[row][c];
    for (int row = b0; row < ob.n; ++row)
      for (int y = 0; y < nb; ++y)
        for (int c = 0; c < 3; ++c) Eb[3 * y + c] += ob.J[row][y] * ob.X[row][c];
    JBA_DEVICE const float* Ci = a.Cinv + 9 * j;
    for (int x = 0; x < na; ++x) {
      float G[3];
      for (int c = 0; c < 3; ++c) G[c] = Ea[3 * x] * Ci[c] + Ea[3 * x + 1] * Ci[3 + c] + Ea[3 * x + 2] * Ci[6 + c];
      for (int y = 0; y < nb; ++y) acc[x * nb + y] -= G[0] * Eb[3 * y] + G[1] * Eb[3 * y + 1] + G[2] * Eb[3 * y + 2];
    }
  }
  for (int c = 0; c < nv; ++c) buf[c * kTileThreads + i] = acc[c];
  JBA_BARRIER();
  for (uint w = kTileThreads / 2; w > 0; w /= 2) {
    if (i < w)
      for (int c = 0; c < nv; ++c) buf[c * kTileThreads + i] += buf[c * kTileThreads + i + w];
    JBA_BARRIER();
  }
  for (int c = int(i); c < nv; c += int(kTileThreads)) a.tile_partial[kChunkFloats * tc + c] = buf[c * kTileThreads];
}

// A tile's values: its chunks' sums in order, with D^2 on a diagonal tile. One
// thread per value.
JBA_KERNEL(tile_finish) {
  const uint id = gid;
  const uint t = id / kChunkFloats, c = id % kChunkFloats;
  if (t >= a.ntiles) return;
  const Tile T = a.tiles[t];
  const FBlockDesc A = a.fblocks[T.a], B = a.fblocks[T.b];
  const int nb = B.size;
  if (int(c) >= A.size * nb) return;
  float v = 0;
  for (uint k = T.c0; k < T.c1; ++k) v += a.tile_partial[kChunkFloats * k + c];
  if (T.a == T.b && int(c) / nb == int(c) % nb) v += a.d2_f[A.offset + int(c) / nb];
  a.tile_vals[T.voff + c] = v;
}

// Point side of the column norms and the gradient.
JBA_KERNEL(point_colnorm) {
  const uint j = gid;
  if (j >= a.npts) return;
  const int e = a.point_off[j];
  if (e < 0) return;
  float d[3] = {0, 0, 0}, g[3] = {0, 0, 0};
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    for (int c = 0; c < 3; ++c) {
      const float j0 = a.Jx[6 * k + c], j1 = a.Jx[6 * k + 3 + c];
      d[c] += j0 * j0 + j1 * j1;
      g[c] += j0 * a.r[2 * k] + j1 * a.r[2 * k + 1];
    }
    if (a.has_depth && a.o_dw[k] != 0) {
      for (int c = 0; c < 3; ++c) {
        const float jd = a.dJx[3 * k + c];
        d[c] += jd * jd;
        g[c] += jd * a.dr[k];
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    a.diag_e[e + c] = d[c];
    a.g_e[e + c] = g[c];
  }
}

// Levenberg-Marquardt damping: D^2 = clamp(diag, 1e-6, 1e32) / radius (Ceres).
JBA_KERNEL(damping) {
  const uint c = gid;
  if (c < a.nf) a.d2_f[c] = clamp(a.diag_f[c], 1e-6f, 1e32f) / op.radius;
  else if (c < a.nf + a.ne) a.d2_e[c - a.nf] = clamp(a.diag_e[c - a.nf], 1e-6f, 1e32f) / op.radius;
}

// Point blocks C_j = J_e^T J_e + D_e^2, inverted.
JBA_KERNEL(point_blocks) {
  const uint j = gid;
  if (j >= a.npts) return;
  const int e = a.point_off[j];
  if (e < 0) return;
  float C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    JBA_DEVICE const float* J = a.Jx + 6 * k;
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y) C[3 * x + y] += J[x] * J[y] + J[3 + x] * J[3 + y];
    if (a.has_depth && a.o_dw[k] != 0) {
      JBA_DEVICE const float* D = a.dJx + 3 * k;
      for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) C[3 * x + y] += D[x] * D[y];
    }
  }
  for (int x = 0; x < 3; ++x) C[4 * x] += a.d2_e[e + x];
  if (!spd_inverse(C, 3, a.pivot_floor)) JBA_SET_BAD(s);
  for (int i = 0; i < 9; ++i) a.Cinv[9 * j + i] = C[i];
}

// Camera-side rows of observation k times x: t_k (2 values) and td_k.
JBA_FUNC void jf_times(JBA_ARGS_REF a, JBA_DEVICE const float* x, uint k, JBA_THREAD float* t, JBA_THREAD float& td) {
  t[0] = t[1] = 0;
  td = 0;
  const int po = a.o_pose[k], io = a.o_intr[k];
  if (po >= 0) {
    JBA_DEVICE const float* J = a.Jp + 12 * k;
    for (int c = 0; c < 6; ++c) {
      t[0] += J[c] * x[po + c];
      t[1] += J[6 + c] * x[po + c];
    }
  }
  if (io >= 0 && a.has_intr) {
    const int np = int(a.np);
    JBA_DEVICE const float* J = a.Jk + 2 * np * k;
    for (int c = 0; c < np; ++c) {
      t[0] += J[c] * x[io + c];
      t[1] += J[np + c] * x[io + c];
    }
  }
  if (a.has_depth && a.o_dw[k] != 0) {
    if (po >= 0)
      for (int c = 0; c < 6; ++c) td += a.dJp[6 * k + c] * x[po + c];
    const int dof = a.o_dep[k];
    if (dof >= 0) td += a.dJc[2 * k] * x[dof] + a.dJc[2 * k + 1] * x[dof + 1];
  }
}

// Per point, the part of the Schur product before the camera gather:
// t = J_f x - J_e C^-1 J_e^T J_f x for its observations.
JBA_KERNEL(schur_points) {
  const uint j = gid;
  if (j >= a.npts || s.done) return;
  const int e = a.point_off[j];
  float u[3] = {0, 0, 0};
  const bool depth = a.has_depth;
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    float t[2], td;
    jf_times(a, op.x, k, t, td);
    a.t[2 * k] = t[0];
    a.t[2 * k + 1] = t[1];
    if (depth) a.td[k] = td;
    if (e < 0) continue;
    JBA_DEVICE const float* J = a.Jx + 6 * k;
    for (int c = 0; c < 3; ++c) u[c] += J[c] * t[0] + J[3 + c] * t[1];
    if (depth && a.o_dw[k] != 0)
      for (int c = 0; c < 3; ++c) u[c] += a.dJx[3 * k + c] * td;
  }
  if (e < 0) return;
  JBA_DEVICE const float* Ci = a.Cinv + 9 * j;
  float w[3];
  for (int x = 0; x < 3; ++x) w[x] = Ci[3 * x] * u[0] + Ci[3 * x + 1] * u[1] + Ci[3 * x + 2] * u[2];
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    JBA_DEVICE const float* J = a.Jx + 6 * k;
    a.t[2 * k] -= J[0] * w[0] + J[1] * w[1] + J[2] * w[2];
    a.t[2 * k + 1] -= J[3] * w[0] + J[4] * w[1] + J[5] * w[2];
    if (depth && a.o_dw[k] != 0) a.td[k] -= a.dJx[3 * k] * w[0] + a.dJx[3 * k + 1] * w[1] + a.dJx[3 * k + 2] * w[2];
  }
}

// m = J_factor x for every factor.
JBA_KERNEL(factors_times) {
  const uint f = gid;
  if (f >= a.nfac || (op.mode == 1 && s.done)) return;
  const Factor F = a.factors[f];
  JBA_DEVICE const float* J = a.fJ + F.joff;
  for (int row = 0; row < F.rows; ++row) {
    float m = 0;
    for (int sl = 0; sl < F.nslots; ++sl) {
      if (F.off[sl] < 0) continue;
      for (int c = 0; c < F.size[sl]; ++c) m += J[row * F.cols + F.base[sl] + c] * op.x[F.off[sl] + c];
    }
    a.fm[F.roff + row] = m;
  }
}

// Right-hand side, point part: t = J_e C^-1 g_e.
JBA_KERNEL(rhs_points) {
  const uint j = gid;
  if (j >= a.npts) return;
  const int e = a.point_off[j];
  float w[3] = {0, 0, 0};
  if (e >= 0) {
    JBA_DEVICE const float* Ci = a.Cinv + 9 * j;
    for (int x = 0; x < 3; ++x) w[x] = Ci[3 * x] * a.g_e[e] + Ci[3 * x + 1] * a.g_e[e + 1] + Ci[3 * x + 2] * a.g_e[e + 2];
  }
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    JBA_DEVICE const float* J = a.Jx + 6 * k;
    a.t[2 * k] = J[0] * w[0] + J[1] * w[1] + J[2] * w[2];
    a.t[2 * k + 1] = J[3] * w[0] + J[4] * w[1] + J[5] * w[2];
    if (a.has_depth) a.td[k] = a.dJx[3 * k] * w[0] + a.dJx[3 * k + 1] * w[1] + a.dJx[3 * k + 2] * w[2];
  }
}

// Back substitution dx_e = -C^-1 (g_e + J_e^T J_f dx_f), keeping t = J_f dx_f for the
// predicted decrease; then this point's share of the predicted decrease,
// sum over its rows of m (r + m / 2) with m = J dx.
JBA_KERNEL(back_substitute) {
  const uint j = gid;
  if (j >= a.npts) return;
  const int e = a.point_off[j];
  const bool depth = a.has_depth;
  float u[3] = {0, 0, 0};
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    float t[2], td;
    jf_times(a, a.x, k, t, td);
    a.t[2 * k] = t[0];
    a.t[2 * k + 1] = t[1];
    if (depth) a.td[k] = td;
    if (e < 0) continue;
    JBA_DEVICE const float* J = a.Jx + 6 * k;
    for (int c = 0; c < 3; ++c) u[c] += J[c] * t[0] + J[3 + c] * t[1];
    if (depth && a.o_dw[k] != 0)
      for (int c = 0; c < 3; ++c) u[c] += a.dJx[3 * k + c] * td;
  }
  float dx[3] = {0, 0, 0};
  if (e >= 0) {
    for (int c = 0; c < 3; ++c) u[c] += a.g_e[e + c];
    JBA_DEVICE const float* Ci = a.Cinv + 9 * j;
    for (int x = 0; x < 3; ++x) dx[x] = -(Ci[3 * x] * u[0] + Ci[3 * x + 1] * u[1] + Ci[3 * x + 2] * u[2]);
    for (int x = 0; x < 3; ++x) a.dx_e[e + x] = dx[x];
  }
  float v = 0, v_abs = 0;
  for (uint k = a.pt_begin[j]; k < a.pt_begin[j + 1]; ++k) {
    float m0 = a.t[2 * k], m1 = a.t[2 * k + 1];
    JBA_DEVICE const float* J = a.Jx + 6 * k;
    m0 += J[0] * dx[0] + J[1] * dx[1] + J[2] * dx[2];
    m1 += J[3] * dx[0] + J[4] * dx[1] + J[5] * dx[2];
    const float v0 = m0 * (a.r[2 * k] + m0 / 2), v1 = m1 * (a.r[2 * k + 1] + m1 / 2);
    v += v0 + v1;
    v_abs += fabs(v0) + fabs(v1);
    if (depth && a.o_dw[k] != 0) {
      const float m = a.td[k] + a.dJx[3 * k] * dx[0] + a.dJx[3 * k + 1] * dx[1] + a.dJx[3 * k + 2] * dx[2];
      const float vd = m * (a.dr[k] + m / 2);
      v += vd;
      v_abs += fabs(vd);
    }
  }
  a.vals[j] = v;
  a.vals[a.npts + a.nfac + j] = v_abs;
}

// The factors' share of the predicted decrease (fm holds J_factor dx_f).
JBA_KERNEL(factor_model) {
  const uint f = gid;
  if (f >= a.nfac) return;
  const Factor F = a.factors[f];
  float v = 0, v_abs = 0;
  for (int row = 0; row < F.rows; ++row) {
    const float m = a.fm[F.roff + row];
    const float vr = m * (a.fr[F.roff + row] + m / 2);
    v += vr;
    v_abs += fabs(vr);
  }
  a.vals[a.npts + f] = v;
  a.vals[2 * a.npts + a.nfac + f] = v_abs;
}

// ---------------------------------------------------------------------------
// Reductions: one threadgroup of kReduceThreads; thread i adds elements i, i + T,
// ... in order, then a fixed tree. Deterministic for a given n.

// mode 0: sum x; 1: sum x * y; 2: sum x * (b + res).
JBA_KERNEL(reduce) {
  const uint i = lid;
  JBA_SHARED float buf[kReduceThreads];
  if (op.iter > 0 && s.done) return;
  float acc = 0;
  for (uint c = i; c < op.n; c += kReduceThreads) {
    if (op.mode == 0) acc += op.x[c];
    else if (op.mode == 1) acc += op.x[c] * op.y[c];
    else acc += op.x[c] * (a.b[c] + a.res[c]);
  }
  buf[i] = acc;
  JBA_BARRIER();
  for (uint w = kReduceThreads / 2; w > 0; w /= 2) {
    if (i < w) buf[i] += buf[i + w];
    JBA_BARRIER();
  }
  if (i == 0) s.slot[op.slot] = buf[0];
}

// ---------------------------------------------------------------------------
// Preconditioned conjugate gradients on S x = b with Ceres's stopping test
// (jba/cpu backend.cpp, pcg()). The host encodes iterations in batches; once a
// test ends the loop, `done` is set and every later kernel returns at once.

// Start: x = 0, res = b.
JBA_KERNEL(pcg_vectors_init) {
  const uint c = gid;
  if (c >= a.nf) return;
  a.x[c] = 0;
  a.res[c] = a.b[c];
}

JBA_KERNEL(pcg_init) {
  const uint tid = gid;
  if (tid != 0) return;
  s.rho_prev = 1;
  s.Q0 = 0;
  s.iterations = 0;
  s.failed = 0;
  s.done = 0;
  if (JBA_GET_BAD(s) != 0) {
    s.failed = 1;
    s.done = 1;
  } else if (a.nf == 0 || s.slot[kSlotBB] == 0) {
    s.done = 1;
  }
}

// z = P^-1 res, per camera-side block.
JBA_KERNEL(pcg_precondition) {
  const uint b = gid;
  if (b >= a.nfb || s.done) return;
  const FBlockDesc fb = a.fblocks[b];
  JBA_DEVICE const float* P = a.Pinv + fb.pinv_offset;
  for (int x = 0; x < fb.size; ++x) {
    float v = 0;
    for (int y = 0; y < fb.size; ++y) v += P[x * fb.size + y] * a.res[fb.offset + y];
    a.z[fb.offset + x] = v;
  }
}

// After rho = res . z: beta, or the end of the loop.
JBA_KERNEL(pcg_rho) {
  const uint tid = gid;
  if (tid != 0 || s.done) return;
  const float rho = s.slot[kSlotRho];
  s.rho = rho;
  if (rho == 0) {  // exact solution
    s.iterations = op.iter;
    s.done = 1;
    return;
  }
  if (!isfinite(rho)) {
    s.iterations = op.iter;
    s.failed = 1;
    s.done = 1;
    return;
  }
  if (op.iter == 1) {
    s.beta = 0;
    return;
  }
  const float beta = rho / s.rho_prev;
  if (!isfinite(beta) || beta == 0) {
    s.iterations = op.iter;
    s.failed = 1;
    s.done = 1;
    return;
  }
  s.beta = beta;
}

JBA_KERNEL(pcg_direction) {
  const uint c = gid;
  if (c >= a.nf || s.done) return;
  a.p[c] = op.iter == 1 ? a.z[c] : a.z[c] + s.beta * a.p[c];
}

// After pq = p . q: alpha, or the end of the loop (no more progress possible; the
// solution so far is kept, as in Ceres).
JBA_KERNEL(pcg_alpha) {
  const uint tid = gid;
  if (tid != 0 || s.done) return;
  const float pq = s.slot[kSlotPQ];
  if (!(pq > 0) || isinf(pq)) {
    s.iterations = op.iter;
    s.done = 1;
    return;
  }
  s.alpha = s.rho / pq;
}

// x += alpha p; res -= alpha q unless this iteration resets the residual.
JBA_KERNEL(pcg_update) {
  const uint c = gid;
  if (c >= a.nf || s.done) return;
  a.x[c] += s.alpha * a.p[c];
  if (op.mode == 0) a.res[c] -= s.alpha * a.q[c];
}

// Residual reset: res = b - q, q = S x.
JBA_KERNEL(pcg_reset) {
  const uint c = gid;
  if (c >= a.nf || s.done) return;
  a.res[c] = a.b[c] - a.q[c];
}

// After Q1 = -x . (b + res): the quadratic-model test.
JBA_KERNEL(pcg_test) {
  const uint tid = gid;
  if (tid != 0 || s.done) return;
  const float Q1 = -s.slot[kSlotQ];
  const float zeta = op.iter * (Q1 - s.Q0) / Q1;
  s.iterations = op.iter;
  if (zeta < s.eta && op.iter >= s.min_it) {
    s.done = 1;
    return;
  }
  s.Q0 = Q1;
  s.rho_prev = s.rho;
  if (op.iter >= s.max_it) s.done = 1;
}
