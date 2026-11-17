// WGSL port of the GPU backends' kernels (backends/gpu/kernels/solver.inc, D96), for
// the WebGPU backend (exploration). Same arithmetic and the same order of additions in
// the source, same structures; what differs is how memory is reached. Dawn compiles the
// MSL in Metal's relaxed math mode, which allows contraction and reassociation, so the
// results agree with Metal's within rounding, not bit for bit.
//
// WGSL has no pointers. Every buffer of the shared host logic is a range of one
// storage buffer, the heap H (32-bit words); the device hands out byte offsets into
// it as "device addresses" (backends/webgpu/src/backend.cpp), so Args holds, for each
// of solver.inc's pointers, the word index where that array starts. Floats are
// bitcast from the words; structures (Chunk, Factor, Tile...) are read field by
// field at their C layout. One uniform Args per solve, one uniform Op per dispatch
// (dynamic offset).
//
// Kernels are functions `<name>_body(gid, group, lid)`; the host appends one entry
// point per kernel with its group size and the 2-D grid that WebGPU's 65,535 groups
// per dimension need. The six linearize kernels are kernels/linearize.wgsl,
// instantiated by the host per camera model.
//
// WGSL may assume there are no NaNs or infinities, so finiteness is tested on the
// bits. A barrier must be reached by every thread of a group: kernels that stop early
// once the conjugate gradients are done (`done`) still run their barriers.

struct Args {
  // Word indices into H, in solver.inc's (layout.h's) order.
  o_img: u32, o_pt: u32, o_xy: u32, o_w: u32, o_pose: u32, o_intr: u32, o_dw: u32, o_dep: u32,
  pt_begin: u32, point_off: u32, image_block: u32, pose_off: u32, intr_off: u32,
  pose_mask: u32,  // bytes, one per image
  intr_mask: u32,  // 16 bits per block
  basis_of: u32,
  img_list: u32, ib_list: u32, chunks: u32, block_chunks: u32, fblocks: u32, factors: u32,
  fb_fac_begin: u32, fb_fac: u32,
  poses: u32, rc: u32, points: u32, intr: u32, basis: u32, f_scale: u32, e_scale: u32, hres: u32,
  r: u32, Jp: u32, Jx: u32, Jk: u32, dr: u32, dJp: u32, dJx: u32, dJc: u32, fr: u32, fJ: u32, fm: u32,
  diag_f: u32, g_f: u32, diag_e: u32, g_e: u32,
  d2_f: u32, d2_e: u32, Cinv: u32, Pinv: u32, t: u32, td: u32, partial: u32, vals: u32,
  b: u32, x: u32, res: u32, z: u32, p: u32, q: u32, dx_e: u32,
  tiles: u32, tile_chunks: u32, entries: u32, tile_partial: u32, tile_vals: u32, Eo: u32, Go: u32,
  gunits: u32, gsplits: u32, gpartial: u32,
  // Sizes and options.
  nobs: u32, npts: u32, nf: u32, ne: u32, nfb: u32, nchunks: u32, nfac: u32, np: u32, ntiles: u32,
  ntile_chunks: u32, eo_dep: u32, eo_intr: u32, ngunits: u32, ngsplits: u32,
  has_depth: i32, has_intr: i32, loss_type: i32,
  loss_scale: f32, pivot_floor: f32,
  // Not in layout.h: where the State is.
  state: u32,
}

struct Op {
  x: u32,  // word indices
  y: u32,
  n: u32,
  mode: i32,
  slot: i32,
  iter: i32,
  radius: f32,
  pad: f32,
}

@group(0) @binding(0) var<storage, read_write> H: array<u32>;
@group(0) @binding(1) var<uniform> a: Args;
@group(0) @binding(2) var<uniform> op: Op;

const kMaxParams: i32 = 8;
const kChunkFloats: u32 = 64u;
const kGatherThreads: u32 = 256u;
const kTileThreads: u32 = 64u;
const kReduceThreads: u32 = 256u;
const kNoSlot: u32 = 0xffffffffu;

const kPose: i32 = 0;
const kIntrinsics: i32 = 1;
const kDepth: i32 = 5;
const kNone: i32 = 0;
const kHuber: i32 = 1;
const kCauchy: i32 = 2;
const kGemanMcClure: i32 = 3;
const kSlotBB: u32 = 0u;
const kSlotRho: u32 = 1u;
const kSlotPQ: u32 = 2u;
const kSlotQ: u32 = 3u;

// State's fields, as words (layout.h).
const S_SLOT: u32 = 0u;
const S_RHO: u32 = 8u;
const S_RHO_PREV: u32 = 9u;
const S_PQ: u32 = 10u;
const S_ALPHA: u32 = 11u;
const S_BETA: u32 = 12u;
const S_Q0: u32 = 13u;
const S_ETA: u32 = 14u;
const S_MIN_IT: u32 = 16u;
const S_MAX_IT: u32 = 17u;
const S_DONE: u32 = 18u;
const S_FAILED: u32 = 19u;
const S_ITERATIONS: u32 = 20u;
const S_BAD: u32 = 24u;

// ---------------------------------------------------------------------------
// Memory.

fn ld(i: u32) -> f32 { return bitcast<f32>(H[i]); }
fn st(i: u32, v: f32) { H[i] = bitcast<u32>(v); }
fn ldi(i: u32) -> i32 { return bitcast<i32>(H[i]); }
fn ldu(i: u32) -> u32 { return H[i]; }
fn ldu8(w: u32, i: u32) -> u32 { return (H[w + i / 4u] >> (8u * (i % 4u))) & 0xffu; }
fn ldu16(w: u32, i: u32) -> u32 { return (H[w + i / 2u] >> (16u * (i % 2u))) & 0xffffu; }

fn sf(k: u32) -> f32 { return ld(a.state + k); }
fn sf_set(k: u32, v: f32) { st(a.state + k, v); }
fn si(k: u32) -> i32 { return ldi(a.state + k); }
fn si_set(k: u32, v: i32) { H[a.state + k] = bitcast<u32>(v); }
fn done() -> bool { return si(S_DONE) != 0; }
// Every writer stores 1, so a plain store does what solver.inc's atomic does.
fn set_bad() { si_set(S_BAD, 1); }

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn is_inf(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) == 0x7f800000u; }

struct FBlockDesc {
  offset: i32,
  size: i32,
  pinv_offset: i32,
  kind: i32,
}

fn fblock(b: u32) -> FBlockDesc {
  let w = a.fblocks + 4u * b;
  return FBlockDesc(ldi(w), ldi(w + 1u), ldi(w + 2u), ldi(w + 3u));
}

struct Chunk {
  block: u32,
  list: u32,
  q0: u32,
  q1: u32,
}

fn chunk(c: u32) -> Chunk {
  let w = a.chunks + 4u * c;
  return Chunk(ldu(w), ldu(w + 1u), ldu(w + 2u), ldu(w + 3u));
}

struct Factor {
  rows: i32,
  cols: i32,
  joff: i32,
  roff: i32,
  nslots: i32,
  off: array<i32, 3>,
  base: array<i32, 3>,
  size: array<i32, 3>,
}

fn factor(f: u32) -> Factor {
  let w = a.factors + 14u * f;
  var F: Factor;
  F.rows = ldi(w);
  F.cols = ldi(w + 1u);
  F.joff = ldi(w + 2u);
  F.roff = ldi(w + 3u);
  F.nslots = ldi(w + 4u);
  for (var s = 0u; s < 3u; s++) {
    F.off[s] = ldi(w + 5u + s);
    F.base[s] = ldi(w + 8u + s);
    F.size[s] = ldi(w + 11u + s);
  }
  return F;
}

// Index helpers: a word index plus a signed offset known to be valid.
fn at(w: u32, i: i32) -> u32 { return w + u32(i); }

// ---------------------------------------------------------------------------
// Small helpers (solver.inc).

// In-place inverse of a symmetric positive semi-definite n x n matrix by Cholesky,
// with the pivot floor; with use_d2, a diagonal entry that is not positive takes the
// damping d2 (words from d2) as its pivot (D103).
fn spd_inverse(A: ptr<function, array<f32, 64>>, n: i32, pf: f32, d2: u32, use_d2: bool) -> bool {
  var L: array<f32, 64>;
  var Li: array<f32, 64>;
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      var s = (*A)[i * n + j];
      for (var k = 0; k < j; k++) { s -= L[i * n + k] * L[j * n + k]; }
      if (i == j) {
        let diag = (*A)[i * n + i];
        if (!finite(s) || !finite(diag)) { return false; }
        if (diag > 0.0) {
          s = max(s, pf * diag);
        } else if (use_d2 && ld(at(d2, i)) > 0.0) {
          s = max(s, ld(at(d2, i)));
        } else {
          return false;
        }
        L[i * n + i] = sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  for (var i = 0; i < n; i++) {
    Li[i * n + i] = 1.0 / L[i * n + i];
    for (var j = 0; j < i; j++) {
      var s = 0.0;
      for (var k = j; k < i; k++) { s -= L[i * n + k] * Li[k * n + j]; }
      Li[i * n + j] = s / L[i * n + i];
    }
  }
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      var s = 0.0;
      for (var k = i; k < n; k++) { s += Li[k * n + i] * Li[k * n + j]; }
      (*A)[i * n + j] = s;
      (*A)[j * n + i] = s;
    }
  }
  return true;
}

// (rho'(s), rho''(s)) of the robust loss.
fn loss_derivatives(kind: i32, sc: f32, s: f32) -> vec2f {
  let a2 = sc * sc;
  var rho1 = 1.0;
  var rho2 = 0.0;
  if (kind == kHuber) {
    if (s > a2) {
      let r = sqrt(s);
      rho1 = sc / r;
      rho2 = -rho1 / (2.0 * s);
    }
  } else if (kind == kCauchy) {
    let inv = 1.0 / (1.0 + s / a2);
    rho1 = inv;
    rho2 = -inv * inv / a2;
  } else if (kind == kGemanMcClure) {
    let d = a2 + s;
    rho1 = a2 * a2 / (d * d);
    rho2 = -2.0 * rho1 / d;
  }
  return vec2f(rho1, rho2);
}

struct Corrector {
  sqrt_rho1: f32,
  residual_scaling: f32,
  alpha_sq_norm: f32,
}

fn make_corrector(sq: f32, rho1: f32, rho2: f32) -> Corrector {
  var c: Corrector;
  c.sqrt_rho1 = sqrt(rho1);
  if (sq == 0.0 || rho2 <= 0.0) {
    c.residual_scaling = c.sqrt_rho1;
    c.alpha_sq_norm = 0.0;
    return c;
  }
  let D = 1.0 + 2.0 * sq * rho2 / rho1;
  let alpha = 1.0 - sqrt(D);
  c.residual_scaling = c.sqrt_rho1 / (1.0 - alpha);
  c.alpha_sq_norm = alpha / sq;
  return c;
}

// correct() of solver.inc for each Jacobian array (WGSL pointers have one type each).
fn correct12(c: Corrector, r0: f32, r1: f32, J: ptr<function, array<f32, 12>>) {
  if (c.alpha_sq_norm == 0.0) {
    for (var i = 0; i < 12; i++) { (*J)[i] *= c.sqrt_rho1; }
    return;
  }
  for (var k = 0; k < 6; k++) {
    let rJ = r0 * (*J)[k] + r1 * (*J)[6 + k];
    (*J)[k] = c.sqrt_rho1 * ((*J)[k] - c.alpha_sq_norm * r0 * rJ);
    (*J)[6 + k] = c.sqrt_rho1 * ((*J)[6 + k] - c.alpha_sq_norm * r1 * rJ);
  }
}

fn correct6(c: Corrector, r0: f32, r1: f32, J: ptr<function, array<f32, 6>>) {
  if (c.alpha_sq_norm == 0.0) {
    for (var i = 0; i < 6; i++) { (*J)[i] *= c.sqrt_rho1; }
    return;
  }
  for (var k = 0; k < 3; k++) {
    let rJ = r0 * (*J)[k] + r1 * (*J)[3 + k];
    (*J)[k] = c.sqrt_rho1 * ((*J)[k] - c.alpha_sq_norm * r0 * rJ);
    (*J)[3 + k] = c.sqrt_rho1 * ((*J)[3 + k] - c.alpha_sq_norm * r1 * rJ);
  }
}

fn correct16(c: Corrector, r0: f32, r1: f32, J: ptr<function, array<f32, 16>>, cols: i32) {
  if (c.alpha_sq_norm == 0.0) {
    for (var i = 0; i < 2 * cols; i++) { (*J)[i] *= c.sqrt_rho1; }
    return;
  }
  for (var k = 0; k < cols; k++) {
    let rJ = r0 * (*J)[k] + r1 * (*J)[cols + k];
    (*J)[k] = c.sqrt_rho1 * ((*J)[k] - c.alpha_sq_norm * r0 * rJ);
    (*J)[cols + k] = c.sqrt_rho1 * ((*J)[cols + k] - c.alpha_sq_norm * r1 * rJ);
  }
}

fn has_depth_row(k: u32) -> bool { return a.has_depth != 0 && ld(a.o_dw + k) != 0.0; }

// ---------------------------------------------------------------------------
// Jacobian column scaling.

fn scale_observations_body(gid: u32, group: u32, lid: u32) {
  let k = gid;
  if (k >= a.nobs) { return; }
  let po = ldi(a.o_pose + k);
  let io = ldi(a.o_intr + k);
  let e = ldi(a.point_off + u32(ldi(a.o_pt + k)));
  let depth = has_depth_row(k);
  if (po >= 0) {
    for (var c = 0u; c < 6u; c++) {
      let s = ld(at(a.f_scale, po) + c);
      st(a.Jp + 12u * k + c, ld(a.Jp + 12u * k + c) * s);
      st(a.Jp + 12u * k + 6u + c, ld(a.Jp + 12u * k + 6u + c) * s);
      if (depth) { st(a.dJp + 6u * k + c, ld(a.dJp + 6u * k + c) * s); }
    }
  }
  if (io >= 0 && a.has_intr != 0) {
    let np = a.np;
    for (var c = 0u; c < np; c++) {
      let s = ld(at(a.f_scale, io) + c);
      st(a.Jk + 2u * np * k + c, ld(a.Jk + 2u * np * k + c) * s);
      st(a.Jk + 2u * np * k + np + c, ld(a.Jk + 2u * np * k + np + c) * s);
    }
  }
  if (e >= 0) {
    for (var c = 0u; c < 3u; c++) {
      let s = ld(at(a.e_scale, e) + c);
      st(a.Jx + 6u * k + c, ld(a.Jx + 6u * k + c) * s);
      st(a.Jx + 6u * k + 3u + c, ld(a.Jx + 6u * k + 3u + c) * s);
      if (depth) { st(a.dJx + 3u * k + c, ld(a.dJx + 3u * k + c) * s); }
    }
  }
  if (depth && ldi(a.o_dep + k) >= 0) {
    let dof = ldi(a.o_dep + k);
    for (var c = 0u; c < 2u; c++) { st(a.dJc + 2u * k + c, ld(a.dJc + 2u * k + c) * ld(at(a.f_scale, dof) + c)); }
  }
}

fn scale_factors_body(gid: u32, group: u32, lid: u32) {
  let f = gid;
  if (f >= a.nfac) { return; }
  var F = factor(f);
  let J = at(a.fJ, F.joff);
  for (var s = 0; s < F.nslots; s++) {
    if (F.off[s] < 0) { continue; }
    for (var row = 0; row < F.rows; row++) {
      for (var c = 0; c < F.size[s]; c++) {
        let w = at(J, row * F.cols + F.base[s] + c);
        st(w, ld(w) * ld(at(a.f_scale, F.off[s] + c)));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Camera-side rows of an observation for one block kind.

struct Rows {
  n: i32,
  J: array<array<f32, 8>, 3>,
  X: array<array<f32, 3>, 3>,
  v: array<f32, 3>,
}

fn rows_of(kind: i32, k: u32, use_t: bool) -> Rows {
  var o: Rows;
  o.n = 0;
  let depth = has_depth_row(k);
  if (kind == kPose) {
    for (var row = 0u; row < 2u; row++) {
      for (var c = 0u; c < 6u; c++) { o.J[row][c] = ld(a.Jp + 12u * k + 6u * row + c); }
    }
    o.n = 2;
    if (depth) {
      for (var c = 0u; c < 6u; c++) { o.J[2][c] = ld(a.dJp + 6u * k + c); }
      o.n = 3;
    }
  } else if (kind == kIntrinsics) {
    let np = a.np;
    for (var row = 0u; row < 2u; row++) {
      for (var c = 0u; c < np; c++) { o.J[row][c] = ld(a.Jk + 2u * np * k + np * row + c); }
    }
    o.n = 2;
  } else if (kind == kDepth && depth) {
    o.J[2][0] = ld(a.dJc + 2u * k);
    o.J[2][1] = ld(a.dJc + 2u * k + 1u);
    o.n = 3;
  }
  for (var c = 0u; c < 3u; c++) {
    o.X[0][c] = ld(a.Jx + 6u * k + c);
    o.X[1][c] = ld(a.Jx + 6u * k + 3u + c);
    o.X[2][c] = select(0.0, ld(a.dJx + 3u * k + c), depth);
  }
  if (use_t) {
    o.v[0] = ld(a.t + 2u * k);
    o.v[1] = ld(a.t + 2u * k + 1u);
    o.v[2] = select(0.0, ld(a.td + k), depth);
  } else {
    o.v[0] = ld(a.r + 2u * k);
    o.v[1] = ld(a.r + 2u * k + 1u);
    o.v[2] = select(0.0, ld(a.dr + k), depth);
  }
  return o;
}

// Partial sums of one chunk of a camera-side block's observations (solver.inc).
fn chunk_partial_body(gid: u32, group: u32, lid: u32) {
  let c = gid;
  if (c >= a.nchunks || (op.iter > 0 && done())) { return; }
  let ch = chunk(c);
  let fb = fblock(ch.block);
  let n = fb.size;
  let list = select(a.ib_list, a.img_list, ch.list == 0u);
  var acc: array<f32, 64>;
  if (op.mode == 0 || op.mode == 1) {
    let use_t = op.mode == 1;
    for (var qi = ch.q0; qi < ch.q1; qi++) {
      var o = rows_of(fb.kind, ldu(list + qi), use_t);
      let r0 = select(0, 2, fb.kind == kDepth);
      for (var row = r0; row < o.n; row++) {
        for (var col = 0; col < n; col++) {
          let v = o.J[row][col];
          if (op.mode == 0) {
            acc[col] += v * v;
            acc[8 + col] += v * o.v[row];
          } else {
            acc[col] += v * o.v[row];
          }
        }
      }
    }
  } else {
    var qi = ch.q0;
    while (qi < ch.q1) {
      let j = ldi(a.o_pt + ldu(list + qi));
      var E: array<f32, 24>;
      for (; qi < ch.q1 && ldi(a.o_pt + ldu(list + qi)) == j; qi++) {
        var o = rows_of(fb.kind, ldu(list + qi), false);
        let r0 = select(0, 2, fb.kind == kDepth);
        for (var row = r0; row < o.n; row++) {
          for (var x = 0; x < n; x++) {
            for (var y = 0; y < n; y++) { acc[x * n + y] += o.J[row][x] * o.J[row][y]; }
            for (var y = 0; y < 3; y++) { E[3 * x + y] += o.J[row][x] * o.X[row][y]; }
          }
        }
      }
      if (ldi(a.point_off + u32(j)) < 0) { continue; }
      let Ci = a.Cinv + 9u * u32(j);
      for (var x = 0; x < n; x++) {
        var EC: array<f32, 3>;
        for (var y = 0u; y < 3u; y++) {
          EC[y] = E[3 * x] * ld(Ci + y) + E[3 * x + 1] * ld(Ci + 3u + y) + E[3 * x + 2] * ld(Ci + 6u + y);
        }
        for (var y = 0; y < n; y++) {
          acc[x * n + y] -= EC[0] * E[3 * y] + EC[1] * E[3 * y + 1] + EC[2] * E[3 * y + 2];
        }
      }
    }
  }
  let out = a.partial + kChunkFloats * c;
  for (var i = 0u; i < kChunkFloats; i++) { st(out + i, acc[i]); }
}

// Per camera-side block: its chunks' partial sums in order, plus its factors
// (solver.inc's modes 0 to 3).
fn block_finish_body(gid: u32, group: u32, lid: u32) {
  let b = gid;
  if (b >= a.nfb) { return; }
  if (op.mode == 1 && done()) { return; }
  let fb = fblock(b);
  let n = fb.size;
  var acc: array<f32, 64>;
  let width = select(select(n, n * n, op.mode == 2), 16, op.mode == 0);
  for (var c = ldu(a.block_chunks + b); c < ldu(a.block_chunks + b + 1u); c++) {
    let part = a.partial + kChunkFloats * c;
    if (op.mode == 0) {
      for (var i = 0; i < n; i++) {
        acc[i] += ld(at(part, i));
        acc[8 + i] += ld(at(part, 8 + i));
      }
    } else {
      for (var i = 0; i < width; i++) { acc[i] += ld(at(part, i)); }
    }
  }
  if (op.mode != 3) {
    for (var e = ldu(a.fb_fac_begin + b); e < ldu(a.fb_fac_begin + b + 1u); e++) {
      var F = factor(ldu(a.fb_fac + 2u * e));
      let base = F.base[ldu(a.fb_fac + 2u * e + 1u)];
      let J = at(a.fJ, F.joff);
      for (var row = 0; row < F.rows; row++) {
        if (op.mode == 0) {
          let rr = ld(at(a.fr, F.roff + row));
          for (var c = 0; c < n; c++) {
            let v = ld(at(J, row * F.cols + base + c));
            acc[c] += v * v;
            acc[8 + c] += v * rr;
          }
        } else if (op.mode == 1) {
          let m = ld(at(a.fm, F.roff + row));
          for (var c = 0; c < n; c++) { acc[c] += ld(at(J, row * F.cols + base + c)) * m; }
        } else {
          for (var x = 0; x < n; x++) {
            for (var y = 0; y < n; y++) {
              acc[x * n + y] += ld(at(J, row * F.cols + base + x)) * ld(at(J, row * F.cols + base + y));
            }
          }
        }
      }
    }
  }
  let off = fb.offset;
  if (op.mode == 0) {
    for (var c = 0; c < n; c++) {
      st(at(a.diag_f, off + c), acc[c]);
      st(at(a.g_f, off + c), acc[8 + c]);
    }
  } else if (op.mode == 1) {
    for (var c = 0; c < n; c++) {
      st(at(op.y, off + c), acc[c] + ld(at(a.d2_f, off + c)) * ld(at(op.x, off + c)));
    }
  } else if (op.mode == 3) {
    for (var c = 0; c < n; c++) { st(at(a.b, off + c), acc[c] - ld(at(a.g_f, off + c))); }
  } else {
    for (var c = 0; c < n; c++) { acc[c * n + c] += ld(at(a.d2_f, off + c)); }
    if (!spd_inverse(&acc, n, a.pivot_floor, at(a.d2_f, off), true)) { set_bad(); }
    let P = at(a.Pinv, fb.pinv_offset);
    for (var i = 0; i < n * n; i++) { st(at(P, i), acc[i]); }
  }
}

// Shared by block_gather (256 x 16 floats) and schur_tiles (64 x 64): 16 KiB, the
// WebGPU default limit of group memory.
var<workgroup> gbuf: array<f32, 4096>;
var<workgroup> rbuf: array<f32, 256>;

// One level of block_gather's fixed tree.
fn gather_step(i: u32, w: u32, width: u32) {
  if (i < w) {
    for (var c = 0u; c < width; c++) { gbuf[c * kGatherThreads + i] += gbuf[c * kGatherThreads + i + w]; }
  }
}

// The end of a camera-side block's gather (solver.inc's gather_output).
fn gather_output(b: u32, acc_in: array<f32, 16>) {
  var acc = acc_in;
  let fb = fblock(b);
  let n = fb.size;
  if (op.mode != 3) {
    for (var e = ldu(a.fb_fac_begin + b); e < ldu(a.fb_fac_begin + b + 1u); e++) {
      var F = factor(ldu(a.fb_fac + 2u * e));
      let base = F.base[ldu(a.fb_fac + 2u * e + 1u)];
      let J = at(a.fJ, F.joff);
      for (var row = 0; row < F.rows; row++) {
        let m = select(ld(at(a.fm, F.roff + row)), ld(at(a.fr, F.roff + row)), op.mode == 0);
        for (var c = 0; c < n; c++) {
          let v = ld(at(J, row * F.cols + base + c));
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
  let off = fb.offset;
  if (op.mode == 0) {
    for (var c = 0; c < n; c++) {
      st(at(a.diag_f, off + c), acc[c]);
      st(at(a.g_f, off + c), acc[8 + c]);
    }
  } else if (op.mode == 1) {
    for (var c = 0; c < n; c++) {
      st(at(op.y, off + c), acc[c] + ld(at(a.d2_f, off + c)) * ld(at(op.x, off + c)));
    }
  } else {
    for (var c = 0; c < n; c++) { st(at(a.b, off + c), acc[c] - ld(at(a.g_f, off + c))); }
  }
}

// Per gather unit, one group of kGatherThreads (solver.inc's block_gather).
fn block_gather_body(gid: u32, group: u32, lid: u32) {
  let uw = a.gunits + 4u * group;
  let b = ldu(uw);
  let slot = ldu(uw + 3u);
  let i = lid;
  let skip = op.mode == 1 && done();
  let fb = fblock(b);
  let n = fb.size;
  let width = select(8u, 16u, op.mode == 0);
  var acc: array<f32, 16>;
  let c0 = ldu(a.block_chunks + b);
  let c1 = ldu(a.block_chunks + b + 1u);
  if (!skip && c0 < c1) {
    let first = chunk(c0);
    let q0 = ldu(uw + 1u);
    let q1 = ldu(uw + 2u);
    let list = select(a.ib_list, a.img_list, first.list == 0u);
    let use_t = op.mode != 0;
    let v2 = select(a.r, a.t, use_t);
    let v1 = select(a.dr, a.td, use_t);
    let np = a.np;
    for (var q = q0 + i; q < q1; q += kGatherThreads) {
      let k = ldu(list + q);
      let r0 = ld(v2 + 2u * k);
      let r1 = ld(v2 + 2u * k + 1u);
      if (fb.kind == kPose) {
        let J = a.Jp + 12u * k;
        for (var c = 0u; c < 6u; c++) {
          let j0 = ld(J + c);
          let j1 = ld(J + 6u + c);
          if (op.mode == 0) {
            acc[c] += j0 * j0 + j1 * j1;
            acc[8u + c] += j0 * r0 + j1 * r1;
          } else {
            acc[c] += j0 * r0 + j1 * r1;
          }
        }
        if (has_depth_row(k)) {
          let rd = ld(v1 + k);
          for (var c = 0u; c < 6u; c++) {
            let jd = ld(a.dJp + 6u * k + c);
            if (op.mode == 0) {
              acc[c] += jd * jd;
              acc[8u + c] += jd * rd;
            } else {
              acc[c] += jd * rd;
            }
          }
        }
      } else if (fb.kind == kIntrinsics) {
        let J = a.Jk + 2u * np * k;
        // Columns 1, 2, ..., np - 1, then 0: the same sums as columns in order. In order,
        // Apple's shader compiler or GPU now and then drops one product of the first
        // pass, so repeated solves differ (document 16); this order avoids it.
        for (var cc = 0u; cc < np; cc++) {
          let c = (cc + 1u) % np;
          let j0 = ld(J + c);
          let j1 = ld(J + np + c);
          if (op.mode == 0) {
            acc[c] += j0 * j0 + j1 * j1;
            acc[8u + c] += j0 * r0 + j1 * r1;
          } else {
            acc[c] += j0 * r0 + j1 * r1;
          }
        }
      } else if (fb.kind == kDepth && has_depth_row(k)) {
        let rd = ld(v1 + k);
        for (var c = 0u; c < 2u; c++) {
          let jd = ld(a.dJc + 2u * k + c);
          if (op.mode == 0) {
            acc[c] += jd * jd;
            acc[8u + c] += jd * rd;
          } else {
            acc[c] += jd * rd;
          }
        }
      }
    }
  }
  for (var c = 0u; c < width; c++) { gbuf[c * kGatherThreads + i] = acc[c]; }
  workgroupBarrier();
  for (var w = kGatherThreads / 2u; w > 0u; w /= 2u) {
    gather_step(i, w, width);
    workgroupBarrier();
  }
  if (i != 0u || skip) { return; }
  for (var c = 0u; c < width; c++) { acc[c] = gbuf[c * kGatherThreads]; }
  if (slot != kNoSlot) {
    for (var c = 0u; c < width; c++) { st(a.gpartial + 16u * slot + c, acc[c]); }
    return;
  }
  gather_output(b, acc);
}

// Per split block: its units' sums in order (solver.inc's gather_finish).
fn gather_finish_body(gid: u32, group: u32, lid: u32) {
  if (gid >= a.ngsplits || (op.mode == 1 && done())) { return; }
  let sw = a.gsplits + 4u * gid;
  let width = select(8u, 16u, op.mode == 0);
  var acc: array<f32, 16>;
  for (var u = ldu(sw + 1u); u < ldu(sw + 2u); u++) {
    for (var c = 0u; c < width; c++) { acc[c] += ld(a.gpartial + 16u * u + c); }
  }
  gather_output(ldu(sw), acc);
}

// Strategy B: where an observation's E and G for one role are (Args::Eo).
fn schur_e_offset(role: i32, p: u32) -> u32 {
  if (role == 0) { return 18u * p; }
  if (role == 2) { return a.eo_dep + 6u * p; }
  return a.eo_intr + 3u * a.np * p;
}

fn store_schur_e(kind: i32, k: u32, n: i32, out: u32) {
  var o = rows_of(kind, k, false);
  let r0 = select(0, 2, kind == kDepth);
  var E: array<f32, 24>;
  for (var row = r0; row < o.n; row++) {
    for (var x = 0; x < n; x++) {
      for (var c = 0; c < 3; c++) { E[3 * x + c] += o.J[row][x] * o.X[row][c]; }
    }
  }
  for (var c = 0; c < 3 * n; c++) { st(at(out, c), E[c]); }
}

// Whether place p of ib_list starts a point's run of observations in one intrinsics
// block (solver.inc's intr_run_start).
fn intr_run_start(p: u32) -> bool {
  if (p == 0u) { return true; }
  let k = ldu(a.ib_list + p);
  let kp = ldu(a.ib_list + p - 1u);
  return ldi(a.o_pt + kp) != ldi(a.o_pt + k) || ldi(a.o_intr + kp) != ldi(a.o_intr + k);
}

fn obs_schur_e_body(gid: u32, group: u32, lid: u32) {
  let intr = gid >= a.nobs;
  let p = select(gid, gid - a.nobs, intr);
  if (p >= a.nobs || (intr && a.has_intr == 0)) { return; }
  let k = select(ldu(a.img_list + p), ldu(a.ib_list + p), intr);
  let j = ldi(a.o_pt + k);
  if (ldi(a.point_off + u32(j)) < 0) { return; }
  if (intr) {
    // The run's sum at its first place, added in order.
    if (ldi(a.o_intr + k) < 0 || !intr_run_start(p)) { return; }
    let n = i32(a.np);
    var E: array<f32, 24>;
    for (var q = p; q < a.nobs; q++) {
      let kq = ldu(a.ib_list + q);
      if (ldi(a.o_pt + kq) != j || ldi(a.o_intr + kq) != ldi(a.o_intr + k)) { break; }
      var o = rows_of(kIntrinsics, kq, false);
      for (var row = 0; row < o.n; row++) {
        for (var x = 0; x < n; x++) {
          for (var c = 0; c < 3; c++) { E[3 * x + c] += o.J[row][x] * o.X[row][c]; }
        }
      }
    }
    let out = a.Eo + schur_e_offset(1, p);
    for (var c = 0; c < 3 * n; c++) { st(at(out, c), E[c]); }
    return;
  }
  if (ldi(a.o_pose + k) >= 0) { store_schur_e(kPose, k, 6, a.Eo + schur_e_offset(0, p)); }
  if (has_depth_row(k) && ldi(a.o_dep + k) >= 0) { store_schur_e(kDepth, k, 2, a.Eo + schur_e_offset(2, p)); }
}

fn store_schur_g(E: u32, Ci: u32, n: u32, out: u32) {
  for (var x = 0u; x < n; x++) {
    for (var c = 0u; c < 3u; c++) {
      st(out + 3u * x + c,
         ld(E + 3u * x) * ld(Ci + c) + ld(E + 3u * x + 1u) * ld(Ci + 3u + c) + ld(E + 3u * x + 2u) * ld(Ci + 6u + c));
    }
  }
}

fn obs_schur_g_body(gid: u32, group: u32, lid: u32) {
  let intr = gid >= a.nobs;
  let p = select(gid, gid - a.nobs, intr);
  if (p >= a.nobs || (intr && a.has_intr == 0)) { return; }
  let k = select(ldu(a.img_list + p), ldu(a.ib_list + p), intr);
  let j = ldi(a.o_pt + k);
  if (ldi(a.point_off + u32(j)) < 0) { return; }
  let Ci = a.Cinv + 9u * u32(j);
  if (intr) {
    if (ldi(a.o_intr + k) >= 0 && intr_run_start(p)) {  // G of the run's summed E
      let o = schur_e_offset(1, p);
      store_schur_g(a.Eo + o, Ci, a.np, a.Go + o);
    }
    return;
  }
  if (ldi(a.o_pose + k) >= 0) {
    let o = schur_e_offset(0, p);
    store_schur_g(a.Eo + o, Ci, 6u, a.Go + o);
  }
  if (has_depth_row(k) && ldi(a.o_dep + k) >= 0) {
    let o = schur_e_offset(2, p);
    store_schur_g(a.Eo + o, Ci, 2u, a.Go + o);
  }
}

// One thread's sums over a tile chunk's entries i, i + T, ... (solver.inc's
// tile_sums, without its compile-time shapes: 6 x 6 is tile_sums66 below, 6 x 4 and
// 4 x 4 take this one, with the same additions in the same order).
fn tile_sums(e0: u32, e1: u32, i: u32, na: i32, nb: i32) -> array<f32, 64> {
  var acc: array<f32, 64>;
  for (var e = e0 + i; e < e1; e += kTileThreads) {
    let ka = ldu(a.entries + 3u * e);
    let kb = ldu(a.entries + 3u * e + 1u);
    let info = ldu(a.entries + 3u * e + 2u);
    let ra = i32(info & 3u);
    let rb = i32((info >> 2u) & 3u);
    if (((info >> 4u) & 1u) != 0u) {
      var F = factor(ka);
      let J = at(a.fJ, F.joff);
      let ba = F.base[ra];
      let bb = F.base[rb];
      for (var row = 0; row < F.rows; row++) {
        for (var x = 0; x < na; x++) {
          for (var y = 0; y < nb; y++) {
            acc[x * nb + y] += ld(at(J, row * F.cols + ba + x)) * ld(at(J, row * F.cols + bb + y));
          }
        }
      }
      continue;
    }
    let kinds = array<i32, 3>(kPose, kIntrinsics, kDepth);
    let a0 = select(0, 2, ra == 2);
    let b0 = select(0, 2, rb == 2);
    if (((info >> 5u) & 1u) != 0u) {
      let k = select(ldu(a.img_list + ka), ldu(a.ib_list + ka), ra == 1);
      var oa = rows_of(kinds[ra], k, false);
      var ob = rows_of(kinds[rb], k, false);
      for (var row = max(a0, b0); row < min(oa.n, ob.n); row++) {
        for (var x = 0; x < na; x++) {
          for (var y = 0; y < nb; y++) { acc[x * nb + y] += oa.J[row][x] * ob.J[row][y]; }
        }
      }
    }
    if (((info >> 7u) & 1u) != 0u) {  // the own terms of the intrinsics run that ka starts
      let k0 = ldu(a.ib_list + ka);
      for (var q = ka; q < a.nobs; q++) {
        let k = ldu(a.ib_list + q);
        if (ldi(a.o_pt + k) != ldi(a.o_pt + k0) || ldi(a.o_intr + k) != ldi(a.o_intr + k0)) { break; }
        var o = rows_of(kIntrinsics, k, false);
        for (var row = 0; row < o.n; row++) {
          for (var x = 0; x < na; x++) {
            for (var y = 0; y < nb; y++) { acc[x * nb + y] += o.J[row][x] * o.J[row][y]; }
          }
        }
      }
    }
    if (((info >> 6u) & 1u) == 0u) { continue; }
    let G = a.Go + schur_e_offset(ra, ka);
    let Eb = a.Eo + schur_e_offset(rb, kb);
    for (var x = 0; x < na; x++) {
      for (var y = 0; y < nb; y++) {
        acc[x * nb + y] -= ld(at(G, 3 * x)) * ld(at(Eb, 3 * y)) + ld(at(G, 3 * x + 1)) * ld(at(Eb, 3 * y + 1)) +
                           ld(at(G, 3 * x + 2)) * ld(at(Eb, 3 * y + 2));
      }
    }
  }
  return acc;
}

// The pose by pose case with its 6 x 6 shape fixed, as solver.inc's tile_sums<6, 6>:
// constant loop bounds, so the sums can stay in registers. The same additions in the
// same order as tile_sums.
fn tile_sums66(e0: u32, e1: u32, i: u32) -> array<f32, 36> {
  var acc: array<f32, 36>;
  for (var e = e0 + i; e < e1; e += kTileThreads) {
    let ka = ldu(a.entries + 3u * e);
    let kb = ldu(a.entries + 3u * e + 1u);
    let info = ldu(a.entries + 3u * e + 2u);
    let ra = i32(info & 3u);
    let rb = i32((info >> 2u) & 3u);
    if (((info >> 4u) & 1u) != 0u) {
      var F = factor(ka);
      let J = at(a.fJ, F.joff);
      let ba = F.base[ra];
      let bb = F.base[rb];
      for (var row = 0; row < F.rows; row++) {
        for (var x = 0; x < 6; x++) {
          for (var y = 0; y < 6; y++) {
            acc[x * 6 + y] += ld(at(J, row * F.cols + ba + x)) * ld(at(J, row * F.cols + bb + y));
          }
        }
      }
      continue;
    }
    let kinds = array<i32, 3>(kPose, kIntrinsics, kDepth);
    let a0 = select(0, 2, ra == 2);
    let b0 = select(0, 2, rb == 2);
    if (((info >> 5u) & 1u) != 0u) {
      let k = select(ldu(a.img_list + ka), ldu(a.ib_list + ka), ra == 1);
      var oa = rows_of(kinds[ra], k, false);
      var ob = rows_of(kinds[rb], k, false);
      for (var row = max(a0, b0); row < min(oa.n, ob.n); row++) {
        for (var x = 0; x < 6; x++) {
          for (var y = 0; y < 6; y++) { acc[x * 6 + y] += oa.J[row][x] * ob.J[row][y]; }
        }
      }
    }
    if (((info >> 6u) & 1u) == 0u) { continue; }
    let G = a.Go + schur_e_offset(ra, ka);
    let Eb = a.Eo + schur_e_offset(rb, kb);
    var g: array<f32, 18>;
    var eb: array<f32, 18>;
    for (var c = 0u; c < 18u; c++) {
      g[c] = ld(G + c);
      eb[c] = ld(Eb + c);
    }
    for (var x = 0; x < 6; x++) {
      for (var y = 0; y < 6; y++) {
        acc[x * 6 + y] -= g[3 * x] * eb[3 * y] + g[3 * x + 1] * eb[3 * y + 1] + g[3 * x + 2] * eb[3 * y + 2];
      }
    }
  }
  return acc;
}

fn schur_tiles_body(gid: u32, group: u32, lid: u32) {
  let tc = group;
  let i = lid;
  let w0 = a.tile_chunks + 3u * tc;
  let tile = ldu(w0);
  let e0 = ldu(w0 + 1u);
  let e1 = ldu(w0 + 2u);
  let ta = ldu(a.tiles + 5u * tile);
  let tb = ldu(a.tiles + 5u * tile + 1u);
  let na = fblock(ta).size;
  let nb = fblock(tb).size;
  let nv = u32(na * nb);
  var acc: array<f32, 64>;
  if (na == 6 && nb == 6) {
    let a66 = tile_sums66(e0, e1, i);
    for (var c = 0u; c < 36u; c++) { acc[c] = a66[c]; }
  } else {
    acc = tile_sums(e0, e1, i, na, nb);
  }
  for (var c = 0u; c < nv; c++) { gbuf[c * kTileThreads + i] = acc[c]; }
  workgroupBarrier();
  for (var w = kTileThreads / 2u; w > 0u; w /= 2u) {
    if (i < w) {
      for (var c = 0u; c < nv; c++) { gbuf[c * kTileThreads + i] += gbuf[c * kTileThreads + i + w]; }
    }
    workgroupBarrier();
  }
  for (var c = i; c < nv; c += kTileThreads) { st(a.tile_partial + kChunkFloats * tc + c, gbuf[c * kTileThreads]); }
}

fn tile_finish_body(gid: u32, group: u32, lid: u32) {
  let id = gid;
  let t = id / kChunkFloats;
  let c = id % kChunkFloats;
  if (t >= a.ntiles) { return; }
  let w = a.tiles + 5u * t;
  let ta = ldu(w);
  let tb = ldu(w + 1u);
  let A = fblock(ta);
  let nb = fblock(tb).size;
  if (i32(c) >= A.size * nb) { return; }
  var v = 0.0;
  for (var k = ldu(w + 2u); k < ldu(w + 3u); k++) { v += ld(a.tile_partial + kChunkFloats * k + c); }
  if (ta == tb && i32(c) / nb == i32(c) % nb) { v += ld(at(a.d2_f, A.offset + i32(c) / nb)); }
  st(ldu(w + 4u) + a.tile_vals + c, v);
}

// ---------------------------------------------------------------------------
// Point side.

fn point_colnorm_body(gid: u32, group: u32, lid: u32) {
  let j = gid;
  if (j >= a.npts) { return; }
  let e = ldi(a.point_off + j);
  if (e < 0) { return; }
  var d = array<f32, 3>(0.0, 0.0, 0.0);
  var g = array<f32, 3>(0.0, 0.0, 0.0);
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    for (var c = 0u; c < 3u; c++) {
      let j0 = ld(a.Jx + 6u * k + c);
      let j1 = ld(a.Jx + 6u * k + 3u + c);
      d[c] += j0 * j0 + j1 * j1;
      g[c] += j0 * ld(a.r + 2u * k) + j1 * ld(a.r + 2u * k + 1u);
    }
    if (has_depth_row(k)) {
      for (var c = 0u; c < 3u; c++) {
        let jd = ld(a.dJx + 3u * k + c);
        d[c] += jd * jd;
        g[c] += jd * ld(a.dr + k);
      }
    }
  }
  for (var c = 0u; c < 3u; c++) {
    st(at(a.diag_e, e) + c, d[c]);
    st(at(a.g_e, e) + c, g[c]);
  }
}

fn damping_body(gid: u32, group: u32, lid: u32) {
  let c = gid;
  if (c < a.nf) {
    st(a.d2_f + c, clamp(ld(a.diag_f + c), 1e-6f, 1e32f) / op.radius);
  } else if (c < a.nf + a.ne) {
    st(a.d2_e + c - a.nf, clamp(ld(a.diag_e + c - a.nf), 1e-6f, 1e32f) / op.radius);
  }
}

fn point_blocks_body(gid: u32, group: u32, lid: u32) {
  let j = gid;
  if (j >= a.npts) { return; }
  let e = ldi(a.point_off + j);
  if (e < 0) { return; }
  var C: array<f32, 64>;
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    let J = a.Jx + 6u * k;
    for (var x = 0u; x < 3u; x++) {
      for (var y = 0u; y < 3u; y++) {
        C[3u * x + y] += ld(J + x) * ld(J + y) + ld(J + 3u + x) * ld(J + 3u + y);
      }
    }
    if (has_depth_row(k)) {
      let D = a.dJx + 3u * k;
      for (var x = 0u; x < 3u; x++) {
        for (var y = 0u; y < 3u; y++) { C[3u * x + y] += ld(D + x) * ld(D + y); }
      }
    }
  }
  for (var x = 0u; x < 3u; x++) { C[4u * x] += ld(at(a.d2_e, e) + x); }
  if (!spd_inverse(&C, 3, a.pivot_floor, 0u, false)) { set_bad(); }
  for (var i = 0u; i < 9u; i++) { st(a.Cinv + 9u * j + i, C[i]); }
}

// Camera-side rows of observation k times x (word index): (t0, t1, td).
fn jf_times(x: u32, k: u32) -> vec3f {
  var t0 = 0.0;
  var t1 = 0.0;
  var td = 0.0;
  let po = ldi(a.o_pose + k);
  let io = ldi(a.o_intr + k);
  if (po >= 0) {
    let J = a.Jp + 12u * k;
    for (var c = 0u; c < 6u; c++) {
      t0 += ld(J + c) * ld(at(x, po) + c);
      t1 += ld(J + 6u + c) * ld(at(x, po) + c);
    }
  }
  if (io >= 0 && a.has_intr != 0) {
    let np = a.np;
    let J = a.Jk + 2u * np * k;
    for (var c = 0u; c < np; c++) {
      t0 += ld(J + c) * ld(at(x, io) + c);
      t1 += ld(J + np + c) * ld(at(x, io) + c);
    }
  }
  if (has_depth_row(k)) {
    if (po >= 0) {
      for (var c = 0u; c < 6u; c++) { td += ld(a.dJp + 6u * k + c) * ld(at(x, po) + c); }
    }
    let dof = ldi(a.o_dep + k);
    if (dof >= 0) { td += ld(a.dJc + 2u * k) * ld(at(x, dof)) + ld(a.dJc + 2u * k + 1u) * ld(at(x, dof + 1)); }
  }
  return vec3f(t0, t1, td);
}

fn schur_points_body(gid: u32, group: u32, lid: u32) {
  let j = gid;
  if (j >= a.npts || done()) { return; }
  let e = ldi(a.point_off + j);
  var u = vec3f(0.0);
  let depth = a.has_depth != 0;
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    let t = jf_times(op.x, k);
    st(a.t + 2u * k, t.x);
    st(a.t + 2u * k + 1u, t.y);
    if (depth) { st(a.td + k, t.z); }
    if (e < 0) { continue; }
    let J = a.Jx + 6u * k;
    for (var c = 0u; c < 3u; c++) { u[c] += ld(J + c) * t.x + ld(J + 3u + c) * t.y; }
    if (depth && ld(a.o_dw + k) != 0.0) {
      for (var c = 0u; c < 3u; c++) { u[c] += ld(a.dJx + 3u * k + c) * t.z; }
    }
  }
  if (e < 0) { return; }
  let Ci = a.Cinv + 9u * j;
  var w: array<f32, 3>;
  for (var x = 0u; x < 3u; x++) {
    w[x] = ld(Ci + 3u * x) * u[0] + ld(Ci + 3u * x + 1u) * u[1] + ld(Ci + 3u * x + 2u) * u[2];
  }
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    let J = a.Jx + 6u * k;
    st(a.t + 2u * k, ld(a.t + 2u * k) - (ld(J) * w[0] + ld(J + 1u) * w[1] + ld(J + 2u) * w[2]));
    st(a.t + 2u * k + 1u, ld(a.t + 2u * k + 1u) - (ld(J + 3u) * w[0] + ld(J + 4u) * w[1] + ld(J + 5u) * w[2]));
    if (depth && ld(a.o_dw + k) != 0.0) {
      st(a.td + k, ld(a.td + k) - (ld(a.dJx + 3u * k) * w[0] + ld(a.dJx + 3u * k + 1u) * w[1] + ld(a.dJx + 3u * k + 2u) * w[2]));
    }
  }
}

fn factors_times_body(gid: u32, group: u32, lid: u32) {
  let f = gid;
  if (f >= a.nfac || (op.mode == 1 && done())) { return; }
  var F = factor(f);
  let J = at(a.fJ, F.joff);
  for (var row = 0; row < F.rows; row++) {
    var m = 0.0;
    for (var sl = 0; sl < F.nslots; sl++) {
      if (F.off[sl] < 0) { continue; }
      for (var c = 0; c < F.size[sl]; c++) {
        m += ld(at(J, row * F.cols + F.base[sl] + c)) * ld(at(op.x, F.off[sl] + c));
      }
    }
    st(at(a.fm, F.roff + row), m);
  }
}

fn rhs_points_body(gid: u32, group: u32, lid: u32) {
  let j = gid;
  if (j >= a.npts) { return; }
  let e = ldi(a.point_off + j);
  var w = array<f32, 3>(0.0, 0.0, 0.0);
  if (e >= 0) {
    let Ci = a.Cinv + 9u * j;
    let g = at(a.g_e, e);
    for (var x = 0u; x < 3u; x++) {
      w[x] = ld(Ci + 3u * x) * ld(g) + ld(Ci + 3u * x + 1u) * ld(g + 1u) + ld(Ci + 3u * x + 2u) * ld(g + 2u);
    }
  }
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    let J = a.Jx + 6u * k;
    st(a.t + 2u * k, ld(J) * w[0] + ld(J + 1u) * w[1] + ld(J + 2u) * w[2]);
    st(a.t + 2u * k + 1u, ld(J + 3u) * w[0] + ld(J + 4u) * w[1] + ld(J + 5u) * w[2]);
    if (a.has_depth != 0) {
      st(a.td + k, ld(a.dJx + 3u * k) * w[0] + ld(a.dJx + 3u * k + 1u) * w[1] + ld(a.dJx + 3u * k + 2u) * w[2]);
    }
  }
}

fn back_substitute_body(gid: u32, group: u32, lid: u32) {
  let j = gid;
  if (j >= a.npts) { return; }
  let e = ldi(a.point_off + j);
  let depth = a.has_depth != 0;
  var u = vec3f(0.0);
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    let t = jf_times(a.x, k);
    st(a.t + 2u * k, t.x);
    st(a.t + 2u * k + 1u, t.y);
    if (depth) { st(a.td + k, t.z); }
    if (e < 0) { continue; }
    let J = a.Jx + 6u * k;
    for (var c = 0u; c < 3u; c++) { u[c] += ld(J + c) * t.x + ld(J + 3u + c) * t.y; }
    if (depth && ld(a.o_dw + k) != 0.0) {
      for (var c = 0u; c < 3u; c++) { u[c] += ld(a.dJx + 3u * k + c) * t.z; }
    }
  }
  var dx = array<f32, 3>(0.0, 0.0, 0.0);
  if (e >= 0) {
    for (var c = 0u; c < 3u; c++) { u[c] += ld(at(a.g_e, e) + c); }
    let Ci = a.Cinv + 9u * j;
    for (var x = 0u; x < 3u; x++) {
      dx[x] = -(ld(Ci + 3u * x) * u[0] + ld(Ci + 3u * x + 1u) * u[1] + ld(Ci + 3u * x + 2u) * u[2]);
    }
    for (var x = 0u; x < 3u; x++) { st(at(a.dx_e, e) + x, dx[x]); }
  }
  var v = 0.0;
  var v_abs = 0.0;
  for (var k = ldu(a.pt_begin + j); k < ldu(a.pt_begin + j + 1u); k++) {
    var m0 = ld(a.t + 2u * k);
    var m1 = ld(a.t + 2u * k + 1u);
    let J = a.Jx + 6u * k;
    m0 += ld(J) * dx[0] + ld(J + 1u) * dx[1] + ld(J + 2u) * dx[2];
    m1 += ld(J + 3u) * dx[0] + ld(J + 4u) * dx[1] + ld(J + 5u) * dx[2];
    let v0 = m0 * (ld(a.r + 2u * k) + m0 / 2.0);
    let v1 = m1 * (ld(a.r + 2u * k + 1u) + m1 / 2.0);
    v += v0 + v1;
    v_abs += abs(v0) + abs(v1);
    if (depth && ld(a.o_dw + k) != 0.0) {
      let m = ld(a.td + k) + ld(a.dJx + 3u * k) * dx[0] + ld(a.dJx + 3u * k + 1u) * dx[1] + ld(a.dJx + 3u * k + 2u) * dx[2];
      let vd = m * (ld(a.dr + k) + m / 2.0);
      v += vd;
      v_abs += abs(vd);
    }
  }
  st(a.vals + j, v);
  st(a.vals + a.npts + a.nfac + j, v_abs);
}

fn factor_model_body(gid: u32, group: u32, lid: u32) {
  let f = gid;
  if (f >= a.nfac) { return; }
  let F = factor(f);
  var v = 0.0;
  var v_abs = 0.0;
  for (var row = 0; row < F.rows; row++) {
    let m = ld(at(a.fm, F.roff + row));
    let vr = m * (ld(at(a.fr, F.roff + row)) + m / 2.0);
    v += vr;
    v_abs += abs(vr);
  }
  st(a.vals + a.npts + f, v);
  st(a.vals + 2u * a.npts + a.nfac + f, v_abs);
}

// ---------------------------------------------------------------------------
// Reductions: one group of kReduceThreads, a fixed tree.

fn reduce_body(gid: u32, group: u32, lid: u32) {
  let i = lid;
  let skip = op.iter > 0 && done();
  var acc = 0.0;
  if (!skip) {
    for (var c = i; c < op.n; c += kReduceThreads) {
      if (op.mode == 0) {
        acc += ld(op.x + c);
      } else if (op.mode == 1) {
        acc += ld(op.x + c) * ld(op.y + c);
      } else {
        acc += ld(op.x + c) * (ld(a.b + c) + ld(a.res + c));
      }
    }
  }
  rbuf[i] = acc;
  workgroupBarrier();
  for (var w = kReduceThreads / 2u; w > 0u; w /= 2u) {
    if (i < w) { rbuf[i] += rbuf[i + w]; }
    workgroupBarrier();
  }
  if (i == 0u && !skip) { sf_set(S_SLOT + u32(op.slot), rbuf[0]); }
}

// ---------------------------------------------------------------------------
// Preconditioned conjugate gradients with Ceres's stopping test (solver.inc).

fn pcg_vectors_init_body(gid: u32, group: u32, lid: u32) {
  let c = gid;
  if (c >= a.nf) { return; }
  st(a.x + c, 0.0);
  st(a.res + c, ld(a.b + c));
}

fn pcg_init_body(gid: u32, group: u32, lid: u32) {
  if (gid != 0u) { return; }
  sf_set(S_RHO_PREV, 1.0);
  sf_set(S_Q0, 0.0);
  si_set(S_ITERATIONS, 0);
  si_set(S_FAILED, 0);
  si_set(S_DONE, 0);
  if (si(S_BAD) != 0) {
    si_set(S_FAILED, 1);
    si_set(S_DONE, 1);
  } else if (a.nf == 0u || sf(S_SLOT + kSlotBB) == 0.0) {
    si_set(S_DONE, 1);
  }
}

fn pcg_precondition_body(gid: u32, group: u32, lid: u32) {
  let b = gid;
  if (b >= a.nfb || done()) { return; }
  let fb = fblock(b);
  let P = at(a.Pinv, fb.pinv_offset);
  for (var x = 0; x < fb.size; x++) {
    var v = 0.0;
    for (var y = 0; y < fb.size; y++) { v += ld(at(P, x * fb.size + y)) * ld(at(a.res, fb.offset + y)); }
    st(at(a.z, fb.offset + x), v);
  }
}

fn pcg_end(failed: bool) {
  si_set(S_ITERATIONS, op.iter);
  if (failed) { si_set(S_FAILED, 1); }
  si_set(S_DONE, 1);
}

fn pcg_rho_body(gid: u32, group: u32, lid: u32) {
  if (gid != 0u || done()) { return; }
  let rho = sf(S_SLOT + kSlotRho);
  sf_set(S_RHO, rho);
  if (!finite(rho)) {
    pcg_end(true);
    return;
  }
  if (rho == 0.0) {  // exact solution
    pcg_end(false);
    return;
  }
  if (op.iter == 1) {
    sf_set(S_BETA, 0.0);
    return;
  }
  let beta = rho / sf(S_RHO_PREV);
  if (!finite(beta) || beta == 0.0) {
    pcg_end(true);
    return;
  }
  sf_set(S_BETA, beta);
}

fn pcg_direction_body(gid: u32, group: u32, lid: u32) {
  let c = gid;
  if (c >= a.nf || done()) { return; }
  if (op.iter == 1) {
    st(a.p + c, ld(a.z + c));
  } else {
    st(a.p + c, ld(a.z + c) + sf(S_BETA) * ld(a.p + c));
  }
}

fn pcg_alpha_body(gid: u32, group: u32, lid: u32) {
  if (gid != 0u || done()) { return; }
  let pq = sf(S_SLOT + kSlotPQ);
  if (!finite(pq) || !(pq > 0.0)) {
    pcg_end(false);
    return;
  }
  sf_set(S_ALPHA, sf(S_RHO) / pq);
}

fn pcg_update_body(gid: u32, group: u32, lid: u32) {
  let c = gid;
  if (c >= a.nf || done()) { return; }
  let alpha = sf(S_ALPHA);
  st(a.x + c, ld(a.x + c) + alpha * ld(a.p + c));
  if (op.mode == 0) { st(a.res + c, ld(a.res + c) - alpha * ld(a.q + c)); }
}

fn pcg_reset_body(gid: u32, group: u32, lid: u32) {
  let c = gid;
  if (c >= a.nf || done()) { return; }
  st(a.res + c, ld(a.b + c) - ld(a.q + c));
}

fn pcg_test_body(gid: u32, group: u32, lid: u32) {
  if (gid != 0u || done()) { return; }
  let Q1 = -sf(S_SLOT + kSlotQ);
  let zeta = f32(op.iter) * (Q1 - sf(S_Q0)) / Q1;
  si_set(S_ITERATIONS, op.iter);
  if (zeta < sf(S_ETA) && op.iter >= si(S_MIN_IT)) {
    si_set(S_DONE, 1);
    return;
  }
  sf_set(S_Q0, Q1);
  sf_set(S_RHO_PREV, sf(S_RHO));
  if (op.iter >= si(S_MAX_IT)) { si_set(S_DONE, 1); }
}
