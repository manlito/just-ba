// Linearization of the reprojection residual, one thread per observation
// (solver.inc's JBA_LINEARIZE_KERNEL). A template: the host replaces $MODEL with the
// camera model's name (the generated reprojection_$MODEL) and $NP with its number of
// intrinsics, once per model.

fn linearize_$MODEL_body(gid: u32, group: u32, lid: u32) {
  let k = gid;
  if (k >= a.nobs) { return; }
  let i = u32(ldi(a.o_img + k));
  let j = u32(ldi(a.o_pt + k));
  let blk = u32(ldi(a.image_block + i));
  let NP = $NPu;
  var q: array<f32, 4>;
  var tt: array<f32, 3>;
  var X: array<f32, 3>;
  var K: array<f32, 8>;
  var o: array<f32, 2>;
  for (var c = 0u; c < 4u; c++) { q[c] = ld(a.poses + 8u * i + c); }
  for (var c = 0u; c < 3u; c++) { tt[c] = ld(a.poses + 8u * i + 4u + c); }
  for (var c = 0u; c < 3u; c++) { X[c] = ld(a.points + 3u * j + c); }
  for (var c = 0u; c < NP; c++) { K[c] = ld(a.intr + NP * blk + c); }
  o[0] = ld(a.o_xy + 2u * k);
  o[1] = ld(a.o_xy + 2u * k + 1u);
  var r: array<f32, 2>;
  var jp: array<f32, 12>;
  var jx: array<f32, 6>;
  var jk: array<f32, 16>;
  reprojection_$MODEL(q, tt, X, K, o, &r, &jp, &jx, &jk);
  // The device sees t + R c; the library's rotation Jacobian is J_theta - J_t x (R c).
  let v0 = ld(a.rc + 4u * i);
  let v1 = ld(a.rc + 4u * i + 1u);
  let v2 = ld(a.rc + 4u * i + 2u);
  for (var row = 0u; row < 2u; row++) {
    let b = 6u * row;
    jp[b] -= jp[b + 4u] * v2 - jp[b + 5u] * v1;
    jp[b + 1u] -= jp[b + 5u] * v0 - jp[b + 3u] * v2;
    jp[b + 2u] -= jp[b + 3u] * v1 - jp[b + 4u] * v0;
  }
  let bi = ldi(a.basis_of + i);
  if (bi >= 0) {
    let B = a.basis + 9u * u32(bi);
    for (var row = 0u; row < 2u; row++) {
      let b = 6u * row;
      let j0 = jp[b];
      let j1 = jp[b + 1u];
      let j2 = jp[b + 2u];
      for (var c = 0u; c < 3u; c++) { jp[b + c] = j0 * ld(B + c) + j1 * ld(B + 3u + c) + j2 * ld(B + 6u + c); }
    }
  }
  // The whitened residual from the host, evaluated in double (document 03, Precision).
  let w = ld(a.o_w + k);
  r[0] = ld(a.hres + 2u * k);
  r[1] = ld(a.hres + 2u * k + 1u);
  for (var c = 0u; c < 12u; c++) { jp[c] *= w; }
  for (var c = 0u; c < 6u; c++) { jx[c] *= w; }
  for (var c = 0u; c < 2u * NP; c++) { jk[c] *= w; }
  if (a.loss_type != kNone) {
    let s = r[0] * r[0] + r[1] * r[1];
    let rho = loss_derivatives(a.loss_type, a.loss_scale, s);
    let corr = make_corrector(s, rho.x, rho.y);
    correct12(corr, r[0], r[1], &jp);
    correct6(corr, r[0], r[1], &jx);
    correct16(corr, r[0], r[1], &jk, i32(NP));
    r[0] *= corr.residual_scaling;
    r[1] *= corr.residual_scaling;
  }
  let pmask = ldu8(a.pose_mask, i);
  for (var c = 0u; c < 6u; c++) {
    if ((pmask & (1u << c)) != 0u) {
      jp[c] = 0.0;
      jp[6u + c] = 0.0;
    }
  }
  if (ldi(a.point_off + j) < 0) {
    for (var c = 0u; c < 6u; c++) { jx[c] = 0.0; }
  }
  st(a.r + 2u * k, r[0]);
  st(a.r + 2u * k + 1u, r[1]);
  for (var c = 0u; c < 12u; c++) { st(a.Jp + 12u * k + c, jp[c]); }
  for (var c = 0u; c < 6u; c++) { st(a.Jx + 6u * k + c, jx[c]); }
  if (a.has_intr != 0) {
    let kmask = ldu16(a.intr_mask, blk);
    for (var c = 0u; c < NP; c++) {
      if ((kmask & (1u << c)) != 0u) {
        jk[c] = 0.0;
        jk[NP + c] = 0.0;
      }
    }
    for (var c = 0u; c < 2u * NP; c++) { st(a.Jk + 2u * NP * k + c, jk[c]); }
  }
}
