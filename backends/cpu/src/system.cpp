#include "jba/cpu/system.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "jba/cpu/depth.h"
#include "jba/cpu/priors.h"
#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "jba/manhattan.h"
#include "jba/priors.h"

namespace jba::cpu {

System::System(const Problem& p, const SolverOptions& o, const Gauge& gauge)
    : model(p.model), np(num_params(p.model)), loss(o.loss), pool(o.num_threads) {
  cur.poses = p.poses;
  cur.points = p.points;
  cur.intr = p.intrinsics;
  const std::size_t n_img = p.num_images(), n_pts = p.num_points(), n_blk = p.num_intrinsics_blocks();
  const std::size_t n_sets = p.num_prior_sets();
  cur.sets.resize(8 * n_sets);
  for (std::size_t s = 0; s < n_sets; ++s) {
    if (p.set_transform.empty()) std::copy_n(kIdentityTransform, 8, &cur.sets[8 * s]);
    else std::copy_n(&p.set_transform[8 * s], 8, &cur.sets[8 * s]);
  }
  image_block.resize(n_img);
  for (std::size_t i = 0; i < n_img; ++i) image_block[i] = p.intrinsics_of_image(i);

  // Which blocks are used at all.
  std::vector<char> img_seen(n_img, 0), pt_seen(n_pts, 0), blk_seen(n_blk, 0), set_seen(n_sets, 0);
  for (std::size_t k = 0; k < p.num_observations(); ++k) {
    img_seen[p.obs_image[k]] = 1;
    pt_seen[p.obs_point[k]] = 1;
    blk_seen[image_block[p.obs_image[k]]] = 1;
  }
  for (auto i : p.rel_images) img_seen[static_cast<std::size_t>(i)] = 1;
  for (auto i : p.grav_image) img_seen[static_cast<std::size_t>(i)] = 1;
  const bool manhattan = p.has_manhattan();
  if (manhattan) {
    for (std::size_t k = 0; k < p.num_lines(); ++k) {
      if (p.line_axis[k] < 0) continue;
      const auto i = static_cast<std::size_t>(p.line_image[k]);
      img_seen[i] = 1;
    }
    std::copy_n(p.manhattan_frame.data(), 4, cur.frame);
  }
  has_depth = p.num_depth_observations() > 0;
  depth_kind = p.depth_kind;
  depth_form = p.depth_form;
  depth_loss = o.depth_loss;
  cur.depth.resize(2 * n_img);
  for (std::size_t i = 0; i < n_img; ++i) {
    cur.depth[2 * i] = p.depth_correction.empty() ? 1.0 : p.depth_correction[2 * i];
    cur.depth[2 * i + 1] = p.depth_correction.empty() ? 0.0 : p.depth_correction[2 * i + 1];
  }
  std::vector<char> depth_seen(n_img, 0);
  if (has_depth)
    for (std::size_t k = 0; k < p.num_observations(); ++k)
      if (!std::isnan(p.obs_depth[k])) depth_seen[static_cast<std::size_t>(p.obs_image[k])] = 1;
  for (std::size_t k = 0; k < p.num_global_priors(); ++k) {
    img_seen[static_cast<std::size_t>(p.glob_image[k])] = 1;
    set_seen[static_cast<std::size_t>(p.set_of_global_prior(k))] = 1;
  }
  std::vector<char> gauge_fixed(n_img, 0), scale_held(n_sets, 0);
  for (int i : gauge.fixed_images) gauge_fixed[static_cast<std::size_t>(i)] = 1;
  for (int s : gauge.fixed_set_scales) scale_held[static_cast<std::size_t>(s)] = 1;

  // Pose variables.
  pose_off.assign(n_img, -1);
  pose_fixed_cols.assign(n_img, 0);
  for (const auto& [i, c] : gauge.fixed_translations) pose_fixed_cols[static_cast<std::size_t>(i)] |= std::uint8_t(1u << (3 + c));
  // Heading held: translation and the basis's vertical column.
  gravity_direction(p, g_world);
  for (int c = 0; c < 3; ++c) up[c] = -g_world[c];
  basis_of.assign(n_img, -1);
  for (int i : gauge.translation_images) pose_fixed_cols[static_cast<std::size_t>(i)] |= std::uint8_t(0x38);
  for (int i : gauge.heading_images) {
    pose_fixed_cols[static_cast<std::size_t>(i)] |= std::uint8_t(0x3c);
    basis_of[static_cast<std::size_t>(i)] = static_cast<int>(basis_image.size());
    basis_image.push_back(i);
  }
  basis.assign(9 * basis_image.size(), 0);
  int off = 0;
  for (std::size_t i = 0; i < n_img; ++i) {
    const bool fixed = (!p.image_fixed.empty() && p.image_fixed[i]) || gauge_fixed[i];
    if (fixed || !img_seen[i]) continue;
    pose_off[i] = off;
    fblocks.push_back({off, 6, 0, BlockKind::Pose, static_cast<int>(i)});
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
    fblocks.push_back({off, np, 0, BlockKind::Intrinsics, static_cast<int>(c)});
    off += np;
  }
  // Prior set transforms: the rigid part (6), then the scale (1).
  rigid_off.assign(n_sets, -1);
  scale_off.assign(n_sets, -1);
  for (std::size_t s = 0; s < n_sets; ++s) {
    const FrameRelation rel = p.relation_of_set(s);
    if (rel == FrameRelation::Identity || !set_seen[s]) continue;
    rigid_off[s] = off;
    fblocks.push_back({off, 6, 0, BlockKind::SetRigid, static_cast<int>(s)});
    off += 6;
    if (rel == FrameRelation::Similarity && !scale_held[s]) {
      scale_off[s] = off;
      fblocks.push_back({off, 1, 0, BlockKind::SetScale, static_cast<int>(s)});
      off += 1;
    }
  }
  // The Manhattan frame: 1 (heading about the vertical) or 3 tangent values.
  if (manhattan) {
    frame_dof = p.manhattan_dof;
    if (!gauge.manhattan_frame_held) {
      frame_off = off;
      fblocks.push_back({off, frame_dof, 0, BlockKind::Frame, 0});
      off += frame_dof;
    }
  }
  // Depth corrections: (a, b) of images with readings and something free.
  depth_off.assign(n_img, -1);
  depth_fixed_cols.assign(n_img, 0);
  for (std::size_t i = 0; i < n_img; ++i) {
    const std::uint8_t mask = p.depth_correction_free.empty() ? 0 : p.depth_correction_free[i];
    if (mask == 0 || !depth_seen[i]) continue;
    depth_off[i] = off;
    depth_fixed_cols[i] = static_cast<std::uint8_t>((mask & kDepthScale ? 0 : 1) | (mask & kDepthOffset ? 0 : 2));
    fblocks.push_back({off, 2, 0, BlockKind::Depth, static_cast<int>(i)});
    off += 2;
  }
  // The group prior's means (D165): one block per group with a tied parameter, its
  // other columns held. Nothing without groups, so such problems solve as before.
  const IntrinsicsGroupTerms groups = intrinsics_group_terms(p);
  if (groups.any()) {
    group_off.assign(groups.groups, -1);
    group_fixed_cols.assign(groups.groups, 0);
    cur.groups.assign(groups.groups * static_cast<std::size_t>(np), 0.0);
    for (std::size_t g = 0; g < groups.groups; ++g) {
      std::uint16_t fixed_cols = 0;
      for (int a = 0; a < np; ++a) {
        const std::size_t at = static_cast<std::size_t>(np) * g + static_cast<std::size_t>(a);
        if (groups.tie_w[at] == 0) fixed_cols |= std::uint16_t(1u << a);
        else cur.groups[at] = groups.mean[at];
      }
      if (fixed_cols == (1u << np) - 1) continue;
      group_off[g] = off;
      group_fixed_cols[g] = fixed_cols;
      fblocks.push_back({off, np, 0, BlockKind::IntrinsicsGroup, static_cast<int>(g)});
      off += np;
    }
  }
  nf = static_cast<std::size_t>(off);
  int pinv = 0;
  for (auto& b : fblocks) {
    b.pinv_offset = pinv;
    pinv += b.size * b.size;
  }
  pinv_size = static_cast<std::size_t>(pinv);

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
  if (has_depth) {
    o_d.assign(nobs, 0);
    o_dw.assign(nobs, 0);
    o_dep.assign(nobs, -1);
    for (std::size_t k = 0; k < nobs; ++k) {
      const std::size_t s = order[k];
      if (std::isnan(p.obs_depth[s])) continue;
      o_d[k] = p.obs_depth[s];
      o_dw[k] = 1 / depth_residual_sigma(p, s, o.prior_defaults);
      o_dep[k] = depth_off[static_cast<std::size_t>(o_img[k])];
    }
  }
  pt_begin.assign(n_pts + 1, 0);
  for (std::size_t k = 0; k < nobs; ++k) ++pt_begin[o_pt[k] + 1];
  for (std::size_t j = 0; j < n_pts; ++j) pt_begin[j + 1] += pt_begin[j];
  build_csr(n_img, [&](std::size_t k) { return static_cast<std::size_t>(o_img[k]); }, img_begin, img_list);
  build_csr(n_blk, [&](std::size_t k) { return static_cast<std::size_t>(image_block[o_img[k]]); }, ib_begin,
            ib_list);

  build_factors(p, o);

  f_scale.assign(nf, 1);
  e_scale.assign(ne, 1);
}

FactorSet System::make_set(FactorKind kind, int rows, std::initializer_list<int> slot_sizes) {
  FactorSet fs;
  fs.kind = kind;
  fs.rows = rows;
  for (int n : slot_sizes) {
    fs.slot_base[fs.nslots] = fs.cols;
    fs.slot_size[fs.nslots] = n;
    fs.cols += n;
    ++fs.nslots;
  }
  return fs;
}

std::uint8_t System::slot_mask_of(int off, int image) const {
  if (off < 0) return 0xff;  // every column of a block that is not a variable
  return image >= 0 ? pose_fixed_cols[static_cast<std::size_t>(image)] : std::uint8_t(0);
}

void System::add_factor(FactorSet& fs, std::initializer_list<std::array<int, 2>> slots, const Loss& factor_loss) {
  for (const auto& [off, image] : slots) {
    fs.slot_off.push_back(off);
    fs.slot_mask.push_back(slot_mask_of(off, image));
    fs.slot_image.push_back(off >= 0 ? image : -1);
  }
  fs.loss.push_back(factor_loss);
  ++fs.count;
}

void System::build_factors(const Problem& p, const SolverOptions& o) {
  std::vector<double> W_rel, W_glob;
  prior_whitening(p, o.prior_defaults, W_rel, W_glob);
  FactorSet priors = make_set(FactorKind::PosePrior, kPriorRows, {6, 6, 1});
  for (std::size_t k = 0; k < p.num_relative_priors(); ++k) {
    PosePrior f;
    f.relative = true;
    f.a = p.rel_images[2 * k];
    f.b = p.rel_images[2 * k + 1];
    std::copy_n(&W_rel[36 * k], 36, f.W);
    std::copy_n(&p.rel_pose[7 * k], 7, f.prior);
    add_factor(priors, {{pose_off[static_cast<std::size_t>(f.a)], f.a}, {pose_off[static_cast<std::size_t>(f.b)], f.b}, {-1, -1}},
               o.relative_prior_loss);
    pose_priors.push_back(f);
  }
  for (std::size_t k = 0; k < p.num_global_priors(); ++k) {
    PosePrior f;
    f.relative = false;
    f.a = p.glob_image[k];
    f.set = p.set_of_global_prior(k);
    const auto s = static_cast<std::size_t>(f.set);
    f.set_identity = p.relation_of_set(s) == FrameRelation::Identity;
    std::copy_n(&W_glob[36 * k], 36, f.W);
    // Prior rotation, then camera centre C = -R^T t, in the set's frame.
    const double* g = &p.glob_pose[7 * k];
    std::copy_n(g, 4, f.prior);
    const double conj[4] = {g[0], -g[1], -g[2], -g[3]};
    quat_rotate(conj, g + 4, f.prior + 4);
    for (int a = 4; a < 7; ++a) f.prior[a] = -f.prior[a];
    add_factor(priors, {{pose_off[static_cast<std::size_t>(f.a)], f.a}, {rigid_off[s], -1}, {scale_off[s], -1}},
               o.global_prior_loss);
    pose_priors.push_back(f);
  }
  if (priors.count > 0) fsets.push_back(std::move(priors));

  FactorSet gravity = make_set(FactorKind::Gravity, 3, {6});
  for (std::size_t k = 0; k < p.num_gravity_priors(); ++k) {
    GravityPrior f;
    f.image = p.grav_image[k];
    std::copy_n(&p.grav_dir[3 * k], 3, f.g);
    f.w = 1 / (p.grav_sigma.empty() ? o.prior_defaults.gravity_sigma : p.grav_sigma[k]);
    add_factor(gravity, {{pose_off[static_cast<std::size_t>(f.image)], f.image}}, o.gravity_loss);
    gravity_priors.push_back(f);
  }
  if (gravity.count > 0) fsets.push_back(std::move(gravity));

  FactorSet dprior = make_set(FactorKind::DepthPrior, 2, {2});
  for (std::size_t i = 0; i < depth_off.size(); ++i) {
    if (depth_off[i] < 0) continue;
    const std::uint8_t mask = p.depth_correction_free[i];
    DepthPrior f;
    f.image = static_cast<int>(i);
    f.w[0] = mask & kDepthScale ? 1 / o.prior_defaults.depth_scale_sigma : 0.0;
    f.w[1] = mask & kDepthOffset ? 1 / o.prior_defaults.depth_offset_sigma : 0.0;
    if (f.w[0] == 0 && f.w[1] == 0) continue;
    add_factor(dprior, {{depth_off[i], -1}}, Loss{});
    dprior.slot_mask.back() = depth_fixed_cols[i];
    depth_priors.push_back(f);
  }
  if (dprior.count > 0) fsets.push_back(std::move(dprior));

  // Priors on free intrinsics (D161): one factor per variable block with a finite
  // sigma on a free parameter. Nothing is built without sigma, so such problems
  // solve exactly as before.
  if (!p.intrinsics_prior_sigma.empty()) {
    // The target per parameter: the caller's, or for an empty target or a NaN entry
    // the intrinsics given to solve(). Written back for the blocks that get a factor
    // only, so a block that is fixed, unseen or without a finite sigma now is not
    // pinned to its current (perhaps placeholder) values for later solves.
    std::vector<double> written = p.intrinsics_prior;
    if (written.empty()) written.assign(p.intrinsics.size(), std::nan(""));
    FactorSet iprior = make_set(FactorKind::IntrinsicsPrior, np, {np});
    iprior.grain = std::size_t(1) << 20;  // inline
    for (std::size_t c = 0; c < intr_off.size(); ++c) {
      if (intr_off[c] < 0) continue;
      IntrinsicsPrior f;
      f.block = static_cast<int>(c);
      bool any = false;
      for (int a = 0; a < np; ++a) {
        const std::size_t at = static_cast<std::size_t>(np) * c + static_cast<std::size_t>(a);
        f.k0[a] = std::isnan(written[at]) ? p.intrinsics[at] : written[at];
        const double sigma = p.intrinsics_prior_sigma[at];
        const bool free = !(intr_fixed_cols[c] & (1u << a));
        f.w[a] = free && std::isfinite(sigma) ? 1 / sigma : 0.0;
        any = any || f.w[a] != 0;
      }
      if (!any) continue;
      add_factor(iprior, {{intr_off[c], -1}}, Loss{});
      iprior.slot_mask.back() = static_cast<std::uint8_t>(intr_fixed_cols[c]);
      intrinsics_priors.push_back(f);
      std::copy_n(f.k0, np, &written[static_cast<std::size_t>(np) * c]);
    }
    if (iprior.count > 0) {
      fsets.push_back(std::move(iprior));
      intr_prior_target = std::move(written);
    }
  }

  // The group prior (D165): each member tied to its group's mean, the mean tied to its
  // target where it has one.
  if (!group_off.empty()) {
    const IntrinsicsGroupTerms groups = intrinsics_group_terms(p);
    const std::size_t gsize = groups.groups * static_cast<std::size_t>(np);
    group_mean_written = p.intrinsics_group_mean.empty() ? std::vector<double>(gsize, std::nan("")) : p.intrinsics_group_mean;
    group_target_written =
        p.intrinsics_group_prior.empty() ? std::vector<double>(gsize, std::nan("")) : p.intrinsics_group_prior;
    FactorSet ties = make_set(FactorKind::GroupTie, np, {np, np});
    ties.grain = std::size_t(1) << 20;  // inline
    for (std::size_t c = 0; c < groups.member.size(); ++c) {
      if (!groups.member[c]) continue;
      GroupTie f;
      f.block = static_cast<int>(c);
      f.group = p.intrinsics_group[c];
      const auto g = static_cast<std::size_t>(f.group);
      std::copy_n(&groups.tie_w[static_cast<std::size_t>(np) * g], np, f.w);
      add_factor(ties, {{intr_off[c], -1}, {group_off[g], -1}}, Loss{});
      if (intr_off[c] >= 0) ties.slot_mask[ties.slot_mask.size() - 2] = static_cast<std::uint8_t>(intr_fixed_cols[c]);
      ties.slot_mask.back() = static_cast<std::uint8_t>(group_fixed_cols[g]);
      group_ties.push_back(f);
    }
    FactorSet targets = make_set(FactorKind::GroupTarget, np, {np});
    targets.grain = std::size_t(1) << 20;
    for (std::size_t g = 0; g < groups.groups; ++g) {
      if (group_off[g] < 0) continue;
      IntrinsicsPrior f;
      f.block = static_cast<int>(g);
      bool any = false;
      for (int a = 0; a < np; ++a) {
        const std::size_t at = static_cast<std::size_t>(np) * g + static_cast<std::size_t>(a);
        f.w[a] = groups.anchor_w[at];
        f.k0[a] = f.w[a] != 0 ? groups.target[at] : 0.0;
        if (f.w[a] != 0) group_target_written[at] = f.k0[a];
        any = any || f.w[a] != 0;
      }
      if (!any) continue;
      add_factor(targets, {{group_off[g], -1}}, Loss{});
      targets.slot_mask.back() = static_cast<std::uint8_t>(group_fixed_cols[g]);
      group_targets.push_back(f);
    }
    if (ties.count > 0) fsets.push_back(std::move(ties));
    if (targets.count > 0) fsets.push_back(std::move(targets));
    if (group_targets.empty()) group_target_written.clear();
  }

  if (frame_dof > 0) {
    // Intrinsics are a slot of their own: a segment informs the focal length and the
    // principal point through its vanishing point (D163). A block is a variable only
    // when observations use it, so pose-only solves keep it fixed.
    FactorSet line_set = make_set(FactorKind::Line, 1, {6, frame_dof, np});
    for (std::size_t k = 0; k < p.num_lines(); ++k) {
      if (p.line_axis[k] < 0) continue;
      LineFactor f;
      f.image = p.line_image[k];
      f.axis = p.line_axis[k];
      f.block = image_block[static_cast<std::size_t>(f.image)];
      const auto blk = static_cast<std::size_t>(f.block);
      f.k_free = intr_off[blk] >= 0;
      std::copy_n(&p.line_xy[4 * k], 4, f.xy);
      // An endpoint outside the camera model at the start: left out.
      if (!line_uv(model, &p.intrinsics[static_cast<std::size_t>(np) * blk], f.xy, f.uv)) continue;
      f.w = 1 / line_residual_sigma(p, k, o.prior_defaults);
      add_factor(line_set, {{pose_off[static_cast<std::size_t>(f.image)], f.image}, {frame_off, -1}, {intr_off[blk], -1}},
                 o.line_loss);
      if (f.k_free) line_set.slot_mask.back() = static_cast<std::uint8_t>(intr_fixed_cols[blk]);
      lines.push_back(f);
    }
    if (line_set.count > 0) fsets.push_back(std::move(line_set));
  }

  for (FactorSet& fs : fsets) finish_set(fs);
}

void System::finish_set(FactorSet& fs) {
  const std::size_t rows = static_cast<std::size_t>(fs.rows);
  fs.r.assign(rows * fs.count, 0);
  fs.J.assign(rows * static_cast<std::size_t>(fs.cols) * fs.count, 0);
  fs.m.assign(rows * fs.count, 0);
  std::vector<int> block_of(nf + 1, -1);
  for (std::size_t b = 0; b < fblocks.size(); ++b) block_of[static_cast<std::size_t>(fblocks[b].offset)] = static_cast<int>(b);
  std::vector<std::vector<std::array<int, 2>>> per(fblocks.size());
  for (std::size_t f = 0; f < fs.count; ++f) {
    for (int slot = 0; slot < fs.nslots; ++slot) {
      const int so = fs.off(f, slot);
      if (so >= 0) per[static_cast<std::size_t>(block_of[static_cast<std::size_t>(so)])].push_back({static_cast<int>(f), slot});
    }
  }
  fs.begin.assign(fblocks.size() + 1, 0);
  for (std::size_t b = 0; b < fblocks.size(); ++b) {
    fs.begin[b + 1] = fs.begin[b] + per[b].size();
    fs.list.insert(fs.list.end(), per[b].begin(), per[b].end());
  }
}

void System::eval_factor(const FactorSet& fs, std::size_t f, const Params& P, double* rw, double* J) const {
  switch (fs.kind) {
    case FactorKind::PosePrior: return eval_pose_prior(pose_priors[f], P.poses, P.sets, rw, J);
    case FactorKind::Gravity: {
      const GravityPrior& g = gravity_priors[f];
      gravity_prior<double>(&P.poses[7 * static_cast<std::size_t>(g.image)], g_world, g.g, rw, J);
      for (int a = 0; a < 3; ++a) rw[a] *= g.w;
      if (J)
        for (int a = 0; a < 18; ++a) J[a] *= g.w;
      return;
    }
    case FactorKind::DepthPrior: {
      const DepthPrior& d = depth_priors[f];
      const double* c = &P.depth[2 * static_cast<std::size_t>(d.image)];
      rw[0] = (c[0] - 1) * d.w[0];
      rw[1] = c[1] * d.w[1];
      if (J) {
        J[0] = d.w[0];
        J[1] = 0;
        J[2] = 0;
        J[3] = d.w[1];
      }
      return;
    }
    case FactorKind::IntrinsicsPrior: {
      const IntrinsicsPrior& q = intrinsics_priors[f];
      const double* k = &P.intr[static_cast<std::size_t>(np) * static_cast<std::size_t>(q.block)];
      for (int a = 0; a < np; ++a) rw[a] = (k[a] - q.k0[a]) * q.w[a];
      if (J) {
        for (int a = 0; a < np * np; ++a) J[a] = 0;
        for (int a = 0; a < np; ++a) J[a * np + a] = q.w[a];
      }
      return;
    }
    case FactorKind::GroupTie: {
      const GroupTie& t = group_ties[f];
      const double* k = &P.intr[static_cast<std::size_t>(np) * static_cast<std::size_t>(t.block)];
      const double* mu = &P.groups[static_cast<std::size_t>(np) * static_cast<std::size_t>(t.group)];
      for (int a = 0; a < np; ++a) rw[a] = (k[a] - mu[a]) * t.w[a];
      if (J) {
        const int cols = 2 * np;
        for (int a = 0; a < np * cols; ++a) J[a] = 0;
        for (int a = 0; a < np; ++a) {
          J[a * cols + a] = t.w[a];
          J[a * cols + np + a] = -t.w[a];
        }
      }
      return;
    }
    case FactorKind::GroupTarget: {
      const IntrinsicsPrior& q = group_targets[f];
      const double* mu = &P.groups[static_cast<std::size_t>(np) * static_cast<std::size_t>(q.block)];
      for (int a = 0; a < np; ++a) rw[a] = (mu[a] - q.k0[a]) * q.w[a];
      if (J) {
        for (int a = 0; a < np * np; ++a) J[a] = 0;
        for (int a = 0; a < np; ++a) J[a * np + a] = q.w[a];
      }
      return;
    }
    case FactorKind::Line: {
      const LineFactor& l = lines[f];
      const double* K = &P.intr[static_cast<std::size_t>(np) * static_cast<std::size_t>(l.block)];
      double uv_now[4];
      const double* uv = l.uv;
      if (l.k_free) {
        // The intrinsics moved: undistort again. Where an endpoint no longer can be
        // (the distortion folds back inside the image), the candidate is invalid: a
        // non-finite cost rejects the step, as Ceres does. Dropping the segment
        // would lower the cost by its share and reward the step. The current point
        // always undistorts: it was accepted, or it is the start, where segments that
        // fail are left out.
        if (!line_uv(model, K, l.xy, uv_now)) {
          rw[0] = std::numeric_limits<double>::quiet_NaN();
          if (J)
            for (int c = 0; c < fs.cols; ++c) J[c] = 0;
          return;
        }
        uv = uv_now;
      }
      double axis[3] = {0, 0, 0}, Jl[6], Jm[3], Jk[kMaxParams];
      axis[l.axis] = 1;
      manhattan_line<double>(model, &P.poses[7 * static_cast<std::size_t>(l.image)], P.frame, axis, uv, uv + 2, K, rw,
                             J ? Jl : nullptr, J ? Jm : nullptr, J && l.k_free ? Jk : nullptr);
      rw[0] *= l.w;
      if (J) {
        for (int c = 0; c < 6; ++c) J[c] = Jl[c] * l.w;
        if (frame_dof == 3) {
          for (int c = 0; c < 3; ++c) J[6 + c] = Jm[c] * l.w;
        } else {
          J[6] = (Jm[0] * up[0] + Jm[1] * up[1] + Jm[2] * up[2]) * l.w;
        }
        const int base = fs.slot_base[2];
        for (int c = 0; c < np; ++c) J[base + c] = l.k_free ? Jk[c] * l.w : 0.0;
      }
      return;
    }
  }
}

void System::apply_basis(const double* B, double* J, int rows, int stride) {
  for (int row = 0; row < rows; ++row) {
    double* j = J + row * stride;
    const double j0 = j[0], j1 = j[1], j2 = j[2];
    for (int c = 0; c < 3; ++c) j[c] = j0 * B[c] + j1 * B[3 + c] + j2 * B[6 + c];
  }
}

void System::eval_pose_prior(const PosePrior& f, const std::vector<double>& P, const std::vector<double>& S, double* rw,
                     double* J) const {
  double res[6];
  double Jraw[kPriorRows * kPriorCols];
  if (f.relative) {
    double Ja[36], Jb[36];
    relative_prior<double>(&P[7 * static_cast<std::size_t>(f.a)], &P[7 * static_cast<std::size_t>(f.b)], f.prior, res,
                           J ? Ja : nullptr, J ? Jb : nullptr);
    if (J) {
      for (int row = 0; row < 6; ++row) {
        for (int c = 0; c < 6; ++c) {
          Jraw[row * kPriorCols + c] = Ja[row * 6 + c];
          Jraw[row * kPriorCols + 6 + c] = Jb[row * 6 + c];
        }
        Jraw[row * kPriorCols + 12] = 0;
      }
    }
  } else {
    const double* T = f.set_identity ? kIdentityTransform : &S[8 * static_cast<std::size_t>(f.set)];
    double Jpose[36], Jrigid[36], Jscale[6];
    global_prior<double>(&P[7 * static_cast<std::size_t>(f.a)], T, f.prior, f.prior + 4, res, J ? Jpose : nullptr,
                         J ? Jrigid : nullptr, J ? Jscale : nullptr);
    if (J) {
      for (int row = 0; row < 6; ++row) {
        for (int c = 0; c < 6; ++c) {
          Jraw[row * kPriorCols + c] = Jpose[row * 6 + c];
          Jraw[row * kPriorCols + 6 + c] = Jrigid[row * 6 + c];
        }
        Jraw[row * kPriorCols + 12] = Jscale[row];
      }
    }
  }
  for (int row = 0; row < 6; ++row) {
    double v = 0;
    for (int c = 0; c < 6; ++c) v += f.W[row * 6 + c] * res[c];
    rw[row] = v;
  }
  if (J) {
    for (int row = 0; row < 6; ++row) {
      for (int c = 0; c < kPriorCols; ++c) {
        double v = 0;
        for (int a = 0; a < 6; ++a) v += f.W[row * 6 + a] * Jraw[a * kPriorCols + c];
        J[row * kPriorCols + c] = v;
      }
    }
  }
}

template <typename Key>
void System::build_csr(std::size_t n, Key key, std::vector<std::size_t>& begin, std::vector<std::size_t>& list) {
  begin.assign(n + 1, 0);
  for (std::size_t k = 0; k < nobs; ++k) ++begin[key(k) + 1];
  for (std::size_t i = 0; i < n; ++i) begin[i + 1] += begin[i];
  list.resize(nobs);
  std::vector<std::size_t> fill(begin.begin(), begin.end() - 1);
  for (std::size_t k = 0; k < nobs; ++k) list[fill[key(k)]++] = k;
}

double System::cost_of(const Params& P) {
  double c = reprojection_cost(P.poses, P.points, P.intr);
  if (has_depth) c += depth_cost(P);
  for (const FactorSet& fs : fsets) c += factor_cost(fs, P);
  return c;
}

double System::depth_cost(const Params& P) {
  return 0.5 * pool.sum(nobs, [&](std::size_t k) {
    if (o_dw[k] == 0) return 0.0;
    const std::size_t i = static_cast<std::size_t>(o_img[k]);
    double res;
    depth_residual<double>(depth_kind, depth_form, &P.poses[7 * i], &P.points[3 * static_cast<std::size_t>(o_pt[k])], o_d[k],
                           &P.depth[2 * i], &res, nullptr, nullptr, nullptr);
    res *= o_dw[k];
    double rho[3];
    evaluate_loss(depth_loss, res * res, rho);
    return rho[0];
  });
}

void System::cost_by_type(double* out) {
  for (std::size_t ty = 0; ty < kNumLossTypes; ++ty) out[ty] = 0;
  out[kLossReprojection] = reprojection_cost(cur.poses, cur.points, cur.intr);
  if (has_depth) out[kLossDepth] += depth_cost(cur);
  for (const FactorSet& fs : fsets) {
    for (std::size_t f = 0; f < fs.count; ++f) {
      double rw[kMaxFactorRows];
      eval_factor(fs, f, cur, rw, nullptr);
      double sq = 0;
      for (int a = 0; a < fs.rows; ++a) sq += rw[a] * rw[a];
      double rho[3];
      evaluate_loss(fs.loss[f], sq, rho);
      std::size_t ty = kLossDepth;
      switch (fs.kind) {
        case FactorKind::PosePrior: ty = pose_priors[f].relative ? kLossRelativePrior : kLossGlobalPrior; break;
        case FactorKind::Gravity: ty = kLossGravity; break;
        case FactorKind::Line: ty = kLossLine; break;
        case FactorKind::DepthPrior: ty = kLossDepth; break;
        case FactorKind::IntrinsicsPrior:
        case FactorKind::GroupTie:
        case FactorKind::GroupTarget: ty = kLossReprojection; break;
      }
      out[ty] += 0.5 * rho[0];
    }
  }
}

double System::factor_cost(const FactorSet& fs, const Params& P) {
  return 0.5 * pool.sum(fs.count, [&](std::size_t f) {
    double rw[kMaxFactorRows];
    eval_factor(fs, f, P, rw, nullptr);
    double sq = 0;
    for (int a = 0; a < fs.rows; ++a) sq += rw[a] * rw[a];
    double rho[3];
    evaluate_loss(fs.loss[f], sq, rho);
    return rho[0];
  });
}

double System::reprojection_cost(const std::vector<double>& P, const std::vector<double>& X, const std::vector<double>& K) {
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

void System::update_bases() {
  for (std::size_t b = 0; b < basis_image.size(); ++b)
    heading_basis(&cur.poses[7 * static_cast<std::size_t>(basis_image[b])], up, &basis[9 * b]);
}

void System::linearize_depth_row(std::size_t k, std::uint8_t pmask, double* r, double* jp, double* jx, double* jc) const {
  const std::size_t i = static_cast<std::size_t>(o_img[k]);
  depth_residual<double>(depth_kind, depth_form, &cur.poses[7 * i], &cur.points[3 * static_cast<std::size_t>(o_pt[k])],
                         o_d[k], &cur.depth[2 * i], r, jp, jx, jc);
  if (!basis_image.empty() && basis_of[i] >= 0) apply_basis(&basis[9 * static_cast<std::size_t>(basis_of[i])], jp, 1, 6);
  const double w = o_dw[k];
  *r *= w;
  for (int a = 0; a < 6; ++a) jp[a] *= w;
  for (int a = 0; a < 3; ++a) jx[a] *= w;
  for (int a = 0; a < 2; ++a) jc[a] *= w;
  if (depth_loss.type != LossType::None) {
    const double sq = *r * *r;
    double rho[3];
    evaluate_loss(depth_loss, sq, rho);
    const Corrector corr(sq, rho);
    corr.jacobian_rows(r, jp, 1, 6);
    corr.jacobian_rows(r, jx, 1, 3);
    corr.jacobian_rows(r, jc, 1, 2);
    *r *= corr.residual_scaling;
  }
  for (int c = 0; c < 6; ++c)
    if (pmask & (1u << c)) jp[c] = 0;
  if (point_off[o_pt[k]] < 0) jx[0] = jx[1] = jx[2] = 0;
  const std::uint8_t cmask = o_dep[k] < 0 ? 3 : depth_fixed_cols[i];
  for (int c = 0; c < 2; ++c)
    if (cmask & (1u << c)) jc[c] = 0;
}

void System::linearize_factors() {
  for (FactorSet& fs : fsets) {
    pool.parallel_for(fs.count, [&](std::size_t f0, std::size_t f1) {
      for (std::size_t f = f0; f < f1; ++f) {
        double* rw = fs.rf(f);
        double* J = fs.Jf(f);
        eval_factor(fs, f, cur, rw, J);
        if (!basis_image.empty()) {
          for (int slot = 0; slot < fs.nslots; ++slot) {
            const int i = fs.image(f, slot);
            if (i >= 0 && basis_of[static_cast<std::size_t>(i)] >= 0)
              apply_basis(&basis[9 * static_cast<std::size_t>(basis_of[static_cast<std::size_t>(i)])],
                          J + fs.slot_base[slot], fs.rows, fs.cols);
          }
        }
        if (fs.loss[f].type != LossType::None) {
          double sq = 0;
          for (int a = 0; a < fs.rows; ++a) sq += rw[a] * rw[a];
          double rho[3];
          evaluate_loss(fs.loss[f], sq, rho);
          const Corrector corr(sq, rho);
          corr.jacobian_rows(rw, J, fs.rows, fs.cols);
          for (int a = 0; a < fs.rows; ++a) rw[a] *= corr.residual_scaling;
        }
        // Columns that are not variables are zero.
        for (int slot = 0; slot < fs.nslots; ++slot) {
          const int n = fs.slot_size[slot], base = fs.slot_base[slot];
          const std::uint8_t mask = fs.mask(f, slot);
          for (int c = 0; c < n; ++c) {
            if (!(mask & (1u << c))) continue;
            for (int row = 0; row < fs.rows; ++row) J[row * fs.cols + base + c] = 0;
          }
        }
      }
    }, fs.grain);
  }
}

void System::scale_factors() {
  for (FactorSet& fs : fsets) {
    pool.parallel_for(fs.count, [&](std::size_t f0, std::size_t f1) {
      for (std::size_t f = f0; f < f1; ++f) {
        double* J = fs.Jf(f);
        for (int slot = 0; slot < fs.nslots; ++slot) {
          const int so = fs.off(f, slot);
          if (so < 0) continue;
          const int n = fs.slot_size[slot], base = fs.slot_base[slot];
          for (int row = 0; row < fs.rows; ++row)
            for (int c = 0; c < n; ++c) J[row * fs.cols + base + c] *= f_scale[so + c];
        }
      }
    }, fs.grain);
  }
}

void System::obs_range(const FBlock& fb, std::size_t& q0, std::size_t& q1, const std::vector<std::size_t>*& list) const {
  q0 = q1 = 0;
  list = nullptr;
  if (fb.kind == BlockKind::Pose || fb.kind == BlockKind::Depth) {
    list = &img_list;
    q0 = img_begin[static_cast<std::size_t>(fb.index)];
    q1 = img_begin[static_cast<std::size_t>(fb.index) + 1];
  } else if (fb.kind == BlockKind::Intrinsics) {
    list = &ib_list;
    q0 = ib_begin[static_cast<std::size_t>(fb.index)];
    q1 = ib_begin[static_cast<std::size_t>(fb.index) + 1];
  }
}

template <typename T>
double System::retract(const T* dx_f, const T* dx_e) {
  // Candidate: unscale the step and retract.
  cand = cur;
  // The step norm is taken in stored coordinates (candidate minus current, the
  // quaternion included), as Ceres does for its parameter tolerance (D62).
  double sq = 0;
  auto add_difference = [&sq](const double* a, const double* b, int n) {
    for (int c = 0; c < n; ++c) sq += (a[c] - b[c]) * (a[c] - b[c]);
  };
  for (const FBlock& fb : fblocks) {
    double delta[kMaxParams];
    for (int c = 0; c < fb.size; ++c) delta[c] = f_scale[fb.offset + c] * dx_f[fb.offset + c];
    const auto idx = static_cast<std::size_t>(fb.index);
    switch (fb.kind) {
      case BlockKind::Pose:
        if (!basis_image.empty() && basis_of[idx] >= 0) {
          const double* B = &basis[9 * static_cast<std::size_t>(basis_of[idx])];
          const double d0 = delta[0], d1 = delta[1], d2 = delta[2];
          for (int c = 0; c < 3; ++c) delta[c] = B[3 * c] * d0 + B[3 * c + 1] * d1 + B[3 * c + 2] * d2;
        }
        pose_retract(&cur.poses[7 * idx], delta, &cand.poses[7 * idx]);
        add_difference(&cand.poses[7 * idx], &cur.poses[7 * idx], 7);
        break;
      case BlockKind::Intrinsics: {
        const std::size_t at = static_cast<std::size_t>(np) * idx;
        double* k = &cand.intr[at];
        for (int c = 0; c < fb.size; ++c) k[c] += delta[c];
        add_difference(k, &cur.intr[at], fb.size);
        break;
      }
      case BlockKind::IntrinsicsGroup: {
        const std::size_t at = static_cast<std::size_t>(np) * idx;
        double* mu = &cand.groups[at];
        for (int c = 0; c < fb.size; ++c) mu[c] += delta[c];
        add_difference(mu, &cur.groups[at], fb.size);
        break;
      }
      // A set's transform is two blocks, as in the Ceres harness: the rigid part
      // (7 stored values) and the scale (1).
      case BlockKind::SetRigid:
        pose_retract(&cur.sets[8 * idx], delta, &cand.sets[8 * idx]);
        add_difference(&cand.sets[8 * idx], &cur.sets[8 * idx], 7);
        break;
      case BlockKind::SetScale:
        cand.sets[8 * idx + 7] = cur.sets[8 * idx + 7] * std::exp(delta[0]);
        add_difference(&cand.sets[8 * idx + 7], &cur.sets[8 * idx + 7], 1);
        break;
      case BlockKind::Depth: {
        double* c = &cand.depth[2 * idx];
        for (int a = 0; a < 2; ++a) c[a] += delta[a];
        add_difference(c, &cur.depth[2 * idx], 2);
        break;
      }
      // The Manhattan frame: R_M <- Exp(dphi) R_M, or Exp(dtheta up) R_M.
      case BlockKind::Frame: {
        double v[3];
        if (frame_dof == 3) {
          std::copy_n(delta, 3, v);
        } else {
          for (int c = 0; c < 3; ++c) v[c] = delta[0] * up[c];
        }
        double dq[4];
        quat_exp(v, dq);
        quat_multiply(dq, cur.frame, cand.frame);
        quat_normalize(cand.frame);
        add_difference(cand.frame, cur.frame, 4);
        break;
      }
    }
  }
  for (std::size_t j = 0; j < point_off.size(); ++j) {
    const int e = point_off[j];
    if (e < 0) continue;
    for (int c = 0; c < 3; ++c) {
      // Two statements on purpose: one expression would let the compiler fuse the
      // multiply and add, which changes rounding and with it the iteration history.
      const double d = e_scale[e + c] * dx_e[e + c];
      cand.points[3 * j + c] += d;
    }
    add_difference(&cand.points[3 * j], &cur.points[3 * j], 3);
  }
  return std::sqrt(sq);
}

template double System::retract<double>(const double*, const double*);
template double System::retract<float>(const float*, const float*);

std::vector<char> System::free_columns() const {
  std::vector<char> free(nf, 1);
  for (const FBlock& fb : fblocks) {
    std::uint16_t mask = 0;
    if (fb.kind == BlockKind::Pose) mask = pose_fixed_cols[static_cast<std::size_t>(fb.index)];
    else if (fb.kind == BlockKind::Intrinsics) mask = intr_fixed_cols[static_cast<std::size_t>(fb.index)];
    else if (fb.kind == BlockKind::Depth) mask = depth_fixed_cols[static_cast<std::size_t>(fb.index)];
    else if (fb.kind == BlockKind::IntrinsicsGroup) mask = group_fixed_cols[static_cast<std::size_t>(fb.index)];
    for (int c = 0; c < fb.size; ++c)
      if (mask & (1u << c)) free[static_cast<std::size_t>(fb.offset + c)] = 0;
  }
  return free;
}

double System::parameter_norm() const {
  double s = 0;
  for (const FBlock& fb : fblocks) {
    const auto idx = static_cast<std::size_t>(fb.index);
    const double* v = nullptr;
    int n = 0;
    switch (fb.kind) {
      case BlockKind::Pose: v = &cur.poses[7 * idx]; n = 7; break;
      case BlockKind::Intrinsics: v = &cur.intr[static_cast<std::size_t>(np) * idx]; n = np; break;
      case BlockKind::SetRigid: v = &cur.sets[8 * idx]; n = 7; break;
      case BlockKind::SetScale: v = &cur.sets[8 * idx + 7]; n = 1; break;
      case BlockKind::Frame: v = cur.frame; n = 4; break;
      case BlockKind::Depth: v = &cur.depth[2 * idx]; n = 2; break;
      case BlockKind::IntrinsicsGroup: v = &cur.groups[static_cast<std::size_t>(np) * idx]; n = np; break;
    }
    for (int a = 0; a < n; ++a) s += v[a] * v[a];
  }
  for (std::size_t j = 0; j < point_off.size(); ++j) {
    if (point_off[j] < 0) continue;
    for (int a = 0; a < 3; ++a) s += cur.points[3 * j + a] * cur.points[3 * j + a];
  }
  return std::sqrt(s);
}

void System::write_back(Problem& problem) const {
  problem.poses = cur.poses;
  problem.points = cur.points;
  problem.intrinsics = cur.intr;
  const bool any_transform = std::any_of(problem.set_relation.begin(), problem.set_relation.end(),
                                         [](FrameRelation r) { return r != FrameRelation::Identity; });
  if (!problem.set_transform.empty() || any_transform) problem.set_transform = cur.sets;
  if (frame_dof > 0) problem.manhattan_frame.assign(cur.frame, cur.frame + 4);
  if (!problem.depth_correction.empty() || !problem.depth_correction_free.empty())
    problem.depth_correction = cur.depth;
  if (!intr_prior_target.empty()) problem.intrinsics_prior = intr_prior_target;
  // The group prior (D165): the means of the tied parameters, and the targets used;
  // other entries keep what they had.
  if (!group_mean_written.empty()) {
    std::vector<double> mean = group_mean_written;
    for (std::size_t g = 0; g < group_off.size(); ++g) {
      if (group_off[g] < 0) continue;
      for (int a = 0; a < np; ++a) {
        const std::size_t at = static_cast<std::size_t>(np) * g + static_cast<std::size_t>(a);
        if (!(group_fixed_cols[g] & (1u << a))) mean[at] = cur.groups[at];
      }
    }
    problem.intrinsics_group_mean = std::move(mean);
  }
  if (!group_target_written.empty()) problem.intrinsics_group_prior = group_target_written;
}

}  // namespace jba::cpu
