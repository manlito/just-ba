#include "jba/cpu/backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "jba/cpu/depth.h"
#include "jba/cpu/priors.h"
#include "jba/cpu/reprojection.h"
#include "jba/geometry.h"
#include "jba/loss.h"
#include "jba/manhattan.h"
#include "jba/priors.h"
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
  // J is `rows` x cols, row-major.
  void jacobian_rows(const double* r, double* J, int rows, int cols) const {
    if (alpha_sq_norm == 0) {
      for (int i = 0; i < rows * cols; ++i) J[i] *= sqrt_rho1;
      return;
    }
    for (int c = 0; c < cols; ++c) {
      double rJ = 0;
      for (int a = 0; a < rows; ++a) rJ += r[a] * J[a * cols + c];
      for (int a = 0; a < rows; ++a) J[a * cols + c] = sqrt_rho1 * (J[a * cols + c] - alpha_sq_norm * r[a] * rJ);
    }
  }
};

// Residuals that touch camera-side blocks only, and no point, are "factors" (D75):
// they add to J_f only. Each kind is a FactorSet with a fixed number of residual
// rows and up to three slots, each slot one camera-side block. Jacobians are rows x
// cols per factor, row-major, slot s in columns slot_base[s] .. + slot_size[s].
//
// Pose priors: 6 rows; slot 0 the pose (0..5), slot 1 the other pose or the set's
// rigid part (6..11), slot 2 the set's scale (12).
constexpr int kPriorRows = 6;
constexpr int kPriorCols = 13;
constexpr double kIdentityTransform[8] = {1, 0, 0, 0, 0, 0, 0, 1};

enum class BlockKind : std::uint8_t { Pose, Intrinsics, SetRigid, SetScale, Frame, Depth };

enum class FactorKind : std::uint8_t { PosePrior, Gravity, Line, DepthPrior };

struct PosePrior {
  bool relative = true;
  int a = -1, b = -1;   // images (b only for relative priors)
  int set = -1;         // global priors: prior set
  bool set_identity = true;
  double W[36];
  double prior[7];      // relative: T_j T_i^-1 (q, t); global: q_p, C_p
};

struct GravityPrior {
  int image = -1;
  double g[3];   // measured, camera axes
  double w = 1;  // 1 / sigma
};

struct LineFactor {
  int image = -1;
  int axis = 0;
  double n[3];   // plane normal, camera axes
  double w = 1;  // 1 / sigma
};

struct DepthPrior {
  int image = -1;
  double w[2];  // 1 / sigma of scale and offset; 0 leaves that part out
};

struct FactorSet {
  FactorKind kind{};
  int rows = 0, cols = 0, nslots = 0;
  int slot_base[3] = {0, 0, 0}, slot_size[3] = {0, 0, 0};
  std::size_t count = 0;
  std::vector<int> slot_off;            // nslots per factor: f offset, -1 when fixed or absent
  std::vector<std::uint8_t> slot_mask;  // nslots per factor: columns that are not variables
  std::vector<int> slot_image;          // nslots per factor: image of a pose slot, else -1
  std::vector<Loss> loss;               // per factor
  std::vector<double> r, J, m;          // rows, rows x cols and rows per factor
  std::vector<std::size_t> begin;       // per f-block, into `list`
  std::vector<std::array<int, 2>> list; // (factor, slot), in factor order per f-block

  int off(std::size_t f, int slot) const { return slot_off[static_cast<std::size_t>(nslots) * f + static_cast<std::size_t>(slot)]; }
  std::uint8_t mask(std::size_t f, int slot) const {
    return slot_mask[static_cast<std::size_t>(nslots) * f + static_cast<std::size_t>(slot)];
  }
  int image(std::size_t f, int slot) const {
    return slot_image[static_cast<std::size_t>(nslots) * f + static_cast<std::size_t>(slot)];
  }
  double* Jf(std::size_t f) { return &J[static_cast<std::size_t>(rows * cols) * f]; }
  const double* Jf(std::size_t f) const { return &J[static_cast<std::size_t>(rows * cols) * f]; }
  double* rf(std::size_t f) { return &r[static_cast<std::size_t>(rows) * f]; }
  const double* rf(std::size_t f) const { return &r[static_cast<std::size_t>(rows) * f]; }
  double* mf(std::size_t f) { return &m[static_cast<std::size_t>(rows) * f]; }
  const double* mf(std::size_t f) const { return &m[static_cast<std::size_t>(rows) * f]; }
};

// Parameter values: the current ones, or a candidate. `sets`: 8 per prior set
// (q, t, s); `frame`: the Manhattan frame (q); `depth`: 2 per image (a, b).
struct Params {
  std::vector<double> poses, points, intr, sets, depth;
  double frame[4] = {1, 0, 0, 0};
  void swap(Params& o) {
    poses.swap(o.poses);
    points.swap(o.points);
    intr.swap(o.intr);
    sets.swap(o.sets);
    depth.swap(o.depth);
    std::swap(frame, o.frame);
  }
};

}  // namespace

struct CpuBackend::Impl {
  CameraModel model;
  int np;
  Loss loss;
  ThreadPool pool;

  Params cur, cand;

  // Variables. f = cameras side (pose and intrinsics blocks), e = points.
  struct FBlock {
    int offset;
    int size;
    int pinv_offset;
    BlockKind kind;
    int index;  // image, intrinsics block or prior set
  };
  std::vector<FBlock> fblocks;
  std::vector<int> pose_off, intr_off, point_off, rigid_off, scale_off;  // -1 when fixed
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

  // Factors (camera-side residuals without a point), by kind; empty kinds are left out.
  std::vector<PosePrior> pose_priors;
  std::vector<GravityPrior> gravity_priors;
  double g_world[3] = {0, -1, 0};
  std::vector<LineFactor> lines;
  int frame_dof = 0, frame_off = -1;  // the Manhattan frame block, when there is one
  std::vector<FactorSet> fsets;

  // Depth at keypoints, one row per observation that has a reading, in the
  // observations' order (D87): reading, 1 / sigma (0 for none), the image's depth
  // block, and the row's residual and Jacobian blocks (pose 6, point 3, block 2).
  bool has_depth = false;
  DepthKind depth_kind = DepthKind::Z;
  DepthForm depth_form = DepthForm::Linear;
  Loss depth_loss;
  std::vector<double> o_d, o_dw;
  std::vector<int> o_dep;
  std::vector<int> depth_off;                   // per image, -1 when fixed or unused
  std::vector<std::uint8_t> depth_fixed_cols;   // per image: bit 0 scale, bit 1 offset
  std::vector<double> dr, dJp, dJx, dJc, td;
  std::vector<DepthPrior> depth_priors;

  // Poses whose heading is held by the gauge (D83): rotation tangent basis, from the
  // current pose at each linearization, and its index per image (-1 for none).
  double up[3] = {0, 1, 0};
  std::vector<int> basis_of, basis_image;
  std::vector<double> basis;  // 9 per entry of basis_image

  // Linearization.
  std::vector<double> r, Jp, Jx, Jk;
  std::vector<double> f_scale, e_scale, g_f, g_e, diag_f, diag_e;
  bool have_scaling = false;
  double gmax = 0;

  // Step.
  std::vector<double> d2_f, d2_e, Cinv, Pinv, t, dx_f, dx_e;

  Impl(const Problem& p, const SolverOptions& o, const Gauge& gauge)
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
        const auto blk = static_cast<std::size_t>(image_block[i]);
        if (!p.intrinsics_free.empty() && p.intrinsics_free[blk] != 0)
          throw std::invalid_argument("the Manhattan term needs fixed intrinsics for images with assigned segments");
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
      dr.assign(nobs, 0);
      dJp.assign(6 * nobs, 0);
      dJx.assign(3 * nobs, 0);
      dJc.assign(2 * nobs, 0);
      td.assign(nobs, 0);
    }
    pt_begin.assign(n_pts + 1, 0);
    for (std::size_t k = 0; k < nobs; ++k) ++pt_begin[o_pt[k] + 1];
    for (std::size_t j = 0; j < n_pts; ++j) pt_begin[j + 1] += pt_begin[j];
    build_csr(n_img, [&](std::size_t k) { return static_cast<std::size_t>(o_img[k]); }, img_begin, img_list);
    build_csr(n_blk, [&](std::size_t k) { return static_cast<std::size_t>(image_block[o_img[k]]); }, ib_begin,
              ib_list);

    build_factors(p, o);

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

  // A factor set of `rows` residual rows over slots of the given sizes.
  static FactorSet make_set(FactorKind kind, int rows, std::initializer_list<int> slot_sizes) {
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

  // Columns of a slot that are not variables: all of them when the block is fixed
  // or absent, otherwise the gauge's fixed coordinates of a pose.
  std::uint8_t slot_mask_of(int off, int image) const {
    if (off < 0) return 0x3f;
    return image >= 0 ? pose_fixed_cols[static_cast<std::size_t>(image)] : std::uint8_t(0);
  }

  void add_factor(FactorSet& fs, std::initializer_list<std::array<int, 2>> slots, const Loss& factor_loss) {
    for (const auto& [off, image] : slots) {
      fs.slot_off.push_back(off);
      fs.slot_mask.push_back(slot_mask_of(off, image));
      fs.slot_image.push_back(off >= 0 ? image : -1);
    }
    fs.loss.push_back(factor_loss);
    ++fs.count;
  }

  void build_factors(const Problem& p, const SolverOptions& o) {
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

    if (frame_dof > 0) {
      FactorSet line_set = make_set(FactorKind::Line, 1, {6, frame_dof});
      for (std::size_t k = 0; k < p.num_lines(); ++k) {
        if (p.line_axis[k] < 0) continue;
        LineFactor f;
        f.image = p.line_image[k];
        f.axis = p.line_axis[k];
        if (!line_normal(p, k, f.n)) continue;  // an endpoint outside the camera model
        f.w = 1 / line_residual_sigma(p, k, o.prior_defaults);
        add_factor(line_set, {{pose_off[static_cast<std::size_t>(f.image)], f.image}, {frame_off, -1}}, o.line_loss);
        lines.push_back(f);
      }
      if (line_set.count > 0) fsets.push_back(std::move(line_set));
    }

    for (FactorSet& fs : fsets) finish_set(fs);
  }

  // Storage and the per-f-block lists of a filled factor set.
  void finish_set(FactorSet& fs) {
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

  // Whitened residual of factor f of a set and, if J is set, its whitened Jacobian
  // (rows x cols).
  void eval_factor(const FactorSet& fs, std::size_t f, const Params& P, double* rw, double* J) const {
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
      case FactorKind::Line: {
        const LineFactor& l = lines[f];
        double axis[3] = {0, 0, 0}, Jl[6], Jm[3];
        axis[l.axis] = 1;
        manhattan_line<double>(&P.poses[7 * static_cast<std::size_t>(l.image)], l.n, P.frame, axis, rw,
                               J ? Jl : nullptr, J ? Jm : nullptr);
        rw[0] *= l.w;
        if (J) {
          for (int c = 0; c < 6; ++c) J[c] = Jl[c] * l.w;
          if (frame_dof == 3) {
            for (int c = 0; c < 3; ++c) J[6 + c] = Jm[c] * l.w;
          } else {
            J[6] = (Jm[0] * up[0] + Jm[1] * up[1] + Jm[2] * up[2]) * l.w;
          }
        }
        return;
      }
    }
  }

  // Rotation columns of a pose Jacobian (rows x stride, row-major, rotation in the
  // first three columns) in the heading basis B: J <- J B.
  static void apply_basis(const double* B, double* J, int rows, int stride) {
    for (int row = 0; row < rows; ++row) {
      double* j = J + row * stride;
      const double j0 = j[0], j1 = j[1], j2 = j[2];
      for (int c = 0; c < 3; ++c) j[c] = j0 * B[c] + j1 * B[3 + c] + j2 * B[6 + c];
    }
  }

  // Pose prior residual and, if J is set, its Jacobian (6 x 13), both whitened by W.
  void eval_pose_prior(const PosePrior& f, const std::vector<double>& P, const std::vector<double>& S, double* rw,
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

  double cost_of(const Params& P) {
    double c = reprojection_cost(P.poses, P.points, P.intr);
    if (has_depth) c += depth_cost(P);
    for (const FactorSet& fs : fsets) c += factor_cost(fs, P);
    return c;
  }

  double depth_cost(const Params& P) {
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

  void cost_by_type(double* out) {
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
        }
        out[ty] += 0.5 * rho[0];
      }
    }
  }

  static constexpr int kMaxFactorRows = 6;

  double factor_cost(const FactorSet& fs, const Params& P) {
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

  double reprojection_cost(const std::vector<double>& P, const std::vector<double>& X, const std::vector<double>& K) {
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
    for (std::size_t b = 0; b < basis_image.size(); ++b)
      heading_basis(&cur.poses[7 * static_cast<std::size_t>(basis_image[b])], up, &basis[9 * b]);
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
        if (has_depth && o_dw[k] != 0) linearize_depth(k, pmask);
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
        }, 16);
      }
    }
    column_norms_and_gradient(true);

    double m = 0;
    for (std::size_t c = 0; c < nf; ++c) m = std::max(m, std::abs(g_f[c] / f_scale[c]));
    for (std::size_t c = 0; c < ne; ++c) m = std::max(m, std::abs(g_e[c] / e_scale[c]));
    gmax = m;
  }

  // Depth row of observation k: whitened, robustified, fixed columns zeroed.
  void linearize_depth(std::size_t k, std::uint8_t pmask) {
    const std::size_t i = static_cast<std::size_t>(o_img[k]);
    double* jp = &dJp[6 * k];
    double* jx = &dJx[3 * k];
    double* jc = &dJc[2 * k];
    depth_residual<double>(depth_kind, depth_form, &cur.poses[7 * i], &cur.points[3 * static_cast<std::size_t>(o_pt[k])],
                           o_d[k], &cur.depth[2 * i], &dr[k], jp, jx, jc);
    if (!basis_image.empty() && basis_of[i] >= 0) apply_basis(&basis[9 * static_cast<std::size_t>(basis_of[i])], jp, 1, 6);
    const double w = o_dw[k];
    dr[k] *= w;
    for (int a = 0; a < 6; ++a) jp[a] *= w;
    for (int a = 0; a < 3; ++a) jx[a] *= w;
    for (int a = 0; a < 2; ++a) jc[a] *= w;
    if (depth_loss.type != LossType::None) {
      const double sq = dr[k] * dr[k];
      double rho[3];
      evaluate_loss(depth_loss, sq, rho);
      const Corrector corr(sq, rho);
      corr.jacobian_rows(&dr[k], jp, 1, 6);
      corr.jacobian_rows(&dr[k], jx, 1, 3);
      corr.jacobian_rows(&dr[k], jc, 1, 2);
      dr[k] *= corr.residual_scaling;
    }
    for (int c = 0; c < 6; ++c)
      if (pmask & (1u << c)) jp[c] = 0;
    if (point_off[o_pt[k]] < 0) jx[0] = jx[1] = jx[2] = 0;
    const std::uint8_t cmask = o_dep[k] < 0 ? 3 : depth_fixed_cols[i];
    for (int c = 0; c < 2; ++c)
      if (cmask & (1u << c)) jc[c] = 0;
  }

  void linearize_factors() {
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
      }, 16);
    }
  }

  // Observation list of a pose or intrinsics block; empty for the others.
  void obs_range(const FBlock& fb, std::size_t& q0, std::size_t& q1, const std::vector<std::size_t>*& list) const {
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
      }, 16);
    }
  }

  // diag = squared column norms of J; g = J^T r when `gradient` is set.
  void column_norms_and_gradient(bool gradient) {
    pool.parallel_for(
        fblocks.size(),
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const FBlock& fb = fblocks[b];
            double d[kMaxParams] = {}, g[kMaxParams] = {};
            const int n = fb.size;
            std::size_t q0, q1;
            const std::vector<std::size_t>* list;
            obs_range(fb, q0, q1, list);
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
        },
        1);
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
    pool.parallel_for(
        fblocks.size(),
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const FBlock& fb = fblocks[b];
            double acc[kMaxParams] = {};
            const int n = fb.size;
            std::size_t q0, q1;
            const std::vector<std::size_t>* list;
            obs_range(fb, q0, q1, list);
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
        },
        1);
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
    pool.parallel_for(
        fblocks.size(),
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const FBlock& fb = fblocks[b];
            const int n = fb.size;
            const bool depth_here = has_depth && fb.kind != BlockKind::Intrinsics;
            double P[kMaxParams * kMaxParams] = {};
            std::size_t q, qend;
            const std::vector<std::size_t>* lp;
            obs_range(fb, q, qend, lp);
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
    out.step_norm = std::sqrt(sq);
    out.valid = std::isfinite(out.model_cost_change) && out.model_cost_change > 0 && std::isfinite(out.step_norm);
    return out;
  }

  // ---------------------------------------------------------------------------
  // Covariance.

  // Free columns: those of variable blocks that are not held (masks).
  std::vector<char> free_columns() const {
    std::vector<char> free(nf, 1);
    for (const FBlock& fb : fblocks) {
      std::uint16_t mask = 0;
      if (fb.kind == BlockKind::Pose) mask = pose_fixed_cols[static_cast<std::size_t>(fb.index)];
      else if (fb.kind == BlockKind::Intrinsics) mask = intr_fixed_cols[static_cast<std::size_t>(fb.index)];
      else if (fb.kind == BlockKind::Depth) mask = depth_fixed_cols[static_cast<std::size_t>(fb.index)];
      for (int c = 0; c < fb.size; ++c)
        if (mask & (1u << c)) free[static_cast<std::size_t>(fb.offset + c)] = 0;
    }
    return free;
  }

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
        static constexpr const char* kKinds[] = {"pose of image", "intrinsics block", "prior set", "prior set scale",
                                                 "Manhattan frame", "depth correction of image"};
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
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
          pose_cov[36 * i + static_cast<std::size_t>(6 * a + b)] = S[static_cast<std::size_t>(o + a) * n + static_cast<std::size_t>(o + b)];
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

  double parameter_norm() const {
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
      }
      for (int a = 0; a < n; ++a) s += v[a] * v[a];
    }
    for (std::size_t j = 0; j < point_off.size(); ++j) {
      if (point_off[j] < 0) continue;
      for (int a = 0; a < 3; ++a) s += cur.points[3 * j + a] * cur.points[3 * j + a];
    }
    return std::sqrt(s);
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

void CpuBackend::write_back(Problem& problem) const {
  problem.poses = impl_->cur.poses;
  problem.points = impl_->cur.points;
  problem.intrinsics = impl_->cur.intr;
  const bool any_transform = std::any_of(problem.set_relation.begin(), problem.set_relation.end(),
                                         [](FrameRelation r) { return r != FrameRelation::Identity; });
  if (!problem.set_transform.empty() || any_transform) problem.set_transform = impl_->cur.sets;
  if (impl_->frame_dof > 0) problem.manhattan_frame.assign(impl_->cur.frame, impl_->cur.frame + 4);
  if (!problem.depth_correction.empty() || !problem.depth_correction_free.empty())
    problem.depth_correction = impl_->cur.depth;
}

}  // namespace jba::cpu
