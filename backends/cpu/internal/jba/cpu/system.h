#pragma once

// The host side of a solve, shared by every backend: the layout of the variables
// (offsets, held columns, the gauge's heading bases), the observations in
// point-major order, the camera-side factors (pose priors, gravity, segments, depth
// corrections) and their double-precision evaluation, the cost in double, and the
// retraction of a step. The CPU backend adds its linear algebra in double; the
// Metal backend moves the per-observation work to the GPU and keeps every decision
// here, in double (document 03, Precision).

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "jba/camera_model.h"
#include "jba/loss.h"
#include "jba/problem.h"
#include "jba/solver.h"
#include "parallel.h"

namespace jba::cpu {

constexpr int kMaxParams = 8;

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
constexpr int kMaxFactorRows = 6;
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

struct System {
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
  std::size_t pinv_size = 0;  // sum of the f-blocks' squared sizes

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
  // block.
  bool has_depth = false;
  DepthKind depth_kind = DepthKind::Z;
  DepthForm depth_form = DepthForm::Linear;
  Loss depth_loss;
  std::vector<double> o_d, o_dw;
  std::vector<int> o_dep;
  std::vector<int> depth_off;                   // per image, -1 when fixed or unused
  std::vector<std::uint8_t> depth_fixed_cols;   // per image: bit 0 scale, bit 1 offset
  std::vector<DepthPrior> depth_priors;

  // Poses whose heading is held by the gauge (D83): rotation tangent basis, from the
  // current pose at each linearization, and its index per image (-1 for none).
  double up[3] = {0, 1, 0};
  std::vector<int> basis_of, basis_image;
  std::vector<double> basis;  // 9 per entry of basis_image

  // Jacobian column scaling (Ceres: 1 / (1 + column norm), from the first Jacobian).
  std::vector<double> f_scale, e_scale;
  bool have_scaling = false;

  System(const Problem& p, const SolverOptions& o, const Gauge& gauge);

  // Costs in double: the total at P, and by residual type at the current parameters.
  double cost_of(const Params& P);
  void cost_by_type(double* out);
  double reprojection_cost(const std::vector<double>& P, const std::vector<double>& X, const std::vector<double>& K);
  double depth_cost(const Params& P);
  double factor_cost(const FactorSet& fs, const Params& P);

  // The gauge's heading bases at the current poses; call before linearizing.
  void update_bases();
  // Depth row of observation k at the current parameters: whitened, robustified,
  // the heading basis applied, columns that are not variables zeroed (pmask: the
  // pose's). Writes 1 residual, 6 pose, 3 point and 2 correction values.
  void linearize_depth_row(std::size_t k, std::uint8_t pmask, double* r, double* jp, double* jx, double* jc) const;
  // Residuals and Jacobians of every factor at the current parameters (unscaled).
  void linearize_factors();
  // Multiplies the factors' Jacobian columns by f_scale.
  void scale_factors();

  // Observation list of a pose or intrinsics block; empty for the others.
  void obs_range(const FBlock& fb, std::size_t& q0, std::size_t& q1, const std::vector<std::size_t>*& list) const;

  // Candidate = current retracted by the scaled step (dx_f, dx_e in scaled
  // tangent coordinates). Returns the step norm in stored coordinates (D62).
  template <typename T>
  double retract(const T* dx_f, const T* dx_e);

  double parameter_norm() const;
  // Free columns: those of variable blocks that are not held (masks).
  std::vector<char> free_columns() const;
  // Copies the current parameters into `problem`.
  void write_back(Problem& problem) const;

  // Whitened residual of factor f of a set and, if J is set, its whitened Jacobian
  // (rows x cols).
  void eval_factor(const FactorSet& fs, std::size_t f, const Params& P, double* rw, double* J) const;
  // Rotation columns of a pose Jacobian (rows x stride, row-major, rotation in the
  // first three columns) in the heading basis B: J <- J B.
  static void apply_basis(const double* B, double* J, int rows, int stride);

 private:
  static FactorSet make_set(FactorKind kind, int rows, std::initializer_list<int> slot_sizes);
  std::uint8_t slot_mask_of(int off, int image) const;
  void add_factor(FactorSet& fs, std::initializer_list<std::array<int, 2>> slots, const Loss& factor_loss);
  void build_factors(const Problem& p, const SolverOptions& o);
  void finish_set(FactorSet& fs);
  void eval_pose_prior(const PosePrior& f, const std::vector<double>& P, const std::vector<double>& S, double* rw,
                       double* J) const;
  template <typename Key>
  void build_csr(std::size_t n, Key key, std::vector<std::size_t>& begin, std::vector<std::size_t>& list);
};

extern template double System::retract<double>(const double*, const double*);
extern template double System::retract<float>(const float*, const float*);

}  // namespace jba::cpu
