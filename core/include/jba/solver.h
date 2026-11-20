#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "jba/loss.h"
#include "jba/priors.h"
#include "jba/problem.h"

namespace jba {

// How global freedoms (gauge) are removed when nothing else does it (document 02).
enum class GaugeMode {
  // When no image and no point is held fixed: in every group of connected images
  // whose frame nothing else defines, hold a reference pose, and the scale when
  // nothing metric sets it. See choose_gauge().
  Auto,
  // Leave the freedoms in; damping keeps the steps bounded.
  None,
};

// Images are connected when they share a point or a relative prior, or have global
// priors from the same set with a transform. Per group of connected images
// (document 02, Gauge):
// - a global prior from a set that is the working frame defines the frame: nothing
//   is held;
// - otherwise the reference image's pose is held: the one with the most
//   observations, then the most priors, then the lowest index. When gravity priors
//   in the group, or segments assigned to a one-degree-of-freedom Manhattan frame,
//   set the vertical, only its translation and its heading about the vertical are
//   held (Gauge::heading_images). The Manhattan frame is shared, so it ties the
//   rotations of every group whose rotation its segments set: normally the frame is
//   held and those groups hold their reference's translation alone
//   (Gauge::translation_images, manhattan_frame_held; D86 has the exceptions);
// - the scale is then set by relative priors, a rigid set or metric depth (readings
//   whose scale is not estimated) if there are any;
//   otherwise by holding the scale of a similarity set (the lowest index) if there
//   is one; otherwise, as COLMAP does, by holding one translation coordinate of the
//   image sharing the most points with the reference (its largest coordinate of
//   R2 (C2 - C1), D57).
struct Gauge {
  std::vector<int> fixed_images;                     // poses held fixed
  // Poses whose translation and heading about the vertical are held, their tilt left
  // free: the reference of a group whose vertical is set by gravity priors or by
  // segments assigned to a one-degree-of-freedom Manhattan frame (D83).
  std::vector<int> heading_images;
  // Poses whose translation alone is held: the reference of a group whose rotation
  // the shared Manhattan frame sets (D86).
  std::vector<int> translation_images;
  // The Manhattan frame is held at its starting value (D86).
  bool manhattan_frame_held = false;
  std::vector<std::array<int, 2>> fixed_translations;  // (image, coordinate 0..2)
  std::vector<int> fixed_set_scales;                 // prior sets whose scale is held
  bool empty() const {
    return fixed_images.empty() && heading_images.empty() && translation_images.empty() && fixed_translations.empty() &&
           fixed_set_scales.empty() && !manhattan_frame_held;
  }
};

// Where the solve runs (document 03). The CPU backend works in double and is the
// reference. The GPU backends, Metal (Apple GPUs) and CUDA (NVIDIA GPUs), do the
// per-observation work and the linear solve in single precision on the GPU, and
// every decision (costs, accepting a step) on the host in double; their answers
// agree with the CPU's within single-precision tolerance, not bit for bit.
enum class BackendKind {
  Cpu,
  Metal,
  Cuda,
  WebGpu,  // exploration: WGSL kernels on WebGPU (Dawn natively, or the browser)
};

std::string_view backend_name(BackendKind kind);  // "cpu", "metal", "cuda", "webgpu"
std::optional<BackendKind> backend_from_name(std::string_view name);
// Whether this build has the backend and the machine can run it.
bool backend_available(BackendKind kind);
// Whether this build has the backend, whatever the machine.
bool backend_compiled(BackendKind kind);

// The linear solver of each step (document 03, strategies A and B). Both eliminate
// the points by the Schur complement.
enum class LinearSolverType {
  // The backend's fastest for the problem (D92, D98, D102): on Metal, and on CUDA
  // when built with cuDSS, SparseSchur unless the reduced system would not fit the
  // device's memory (and if its factorization runs out of memory part way,
  // IterativeSchur from there on); IterativeSchur otherwise (the only one the CPU
  // has). SolverSummary::linear_solver says which ran.
  Auto,
  // A: the reduced camera system applied implicitly, preconditioned conjugate
  // gradients, preconditioner its camera blocks (Ceres's ITERATIVE_SCHUR with
  // SCHUR_JACOBI). Every backend.
  IterativeSchur,
  // B: the reduced camera system formed and factorized by sparse Cholesky (Ceres's
  // SPARSE_SCHUR): exact steps, so fewer and dearer iterations. GPU backends only:
  // formed on the GPU, factorized in double by Accelerate (Metal) or cuDSS (CUDA, when
  // built with it).
  SparseSchur,
};

std::string_view linear_solver_name(LinearSolverType type);  // "auto", "iterative_schur", "sparse_schur"
// The type Auto stands for on a backend before the problem is seen (its first choice);
// other types are returned as they are.
LinearSolverType resolve_linear_solver(LinearSolverType type, BackendKind backend);
std::optional<LinearSolverType> linear_solver_from_name(std::string_view name);

struct SolverOptions {
  // Trust region (Levenberg-Marquardt). Defaults are Ceres's, so the two solvers
  // can be compared iteration by iteration.
  int max_iterations = 50;
  double function_tolerance = 1e-6;   // stop when |cost change| <= tol * cost
  double gradient_tolerance = 1e-10;  // stop when max |gradient| <= tol
  double parameter_tolerance = 1e-8;  // stop when |step| <= tol * (|x| + tol)
  double initial_trust_region_radius = 1e4;
  double max_trust_region_radius = 1e16;
  double min_trust_region_radius = 1e-32;
  double min_relative_decrease = 1e-3;
  int max_consecutive_invalid_steps = 5;  // the solve fails on reaching this count

  LinearSolverType linear_solver = LinearSolverType::Auto;
  // Conjugate gradients (IterativeSchur only).
  int max_linear_iterations = 500;
  int min_linear_iterations = 0;
  double eta = 0.1;  // forcing parameter of the conjugate gradients stopping test

  Loss loss;  // applied to every reprojection residual
  // Pose priors are always robust (document 02). Scales are in standard deviations.
  // Provisional defaults, to be set from the synthetic jump tests and real captures.
  Loss relative_prior_loss{LossType::Cauchy, 3.0};
  Loss global_prior_loss{LossType::Cauchy, 3.0};
  Loss gravity_loss{LossType::Cauchy, 3.0};
  Loss line_loss{LossType::Cauchy, 3.0};
  Loss depth_loss{LossType::Cauchy, 3.0};
  PriorDefaults prior_defaults;  // weights for priors that come without any
  GaugeMode gauge = GaugeMode::Auto;

  BackendKind backend = BackendKind::Cpu;

  // Worker threads for the CPU backend, and for the host side of the others; 0
  // means all hardware threads. Results do not depend on the thread count.
  int num_threads = 0;
};

struct IterationSummary {
  int iteration = 0;
  double cost = 0;              // after this iteration
  double cost_change = 0;       // of the candidate step, accepted or not
  double gradient_max_norm = 0;
  double step_norm = 0;         // of candidate - current, in stored coordinates
  double relative_decrease = 0;  // actual over predicted decrease
  double trust_region_radius = 0;
  int linear_iterations = 0;
  bool step_valid = false;
  bool step_accepted = false;
  double time_seconds = 0;      // wall time since solve() was called
};

enum class Termination {
  Convergence,    // a tolerance was met
  NoConvergence,  // iteration limit reached
  Failure,        // numerical failure; parameters are the last accepted ones
  Cancelled,      // the callback returned false
};

std::string_view termination_name(Termination t);

struct SolverSummary {
  Termination termination = Termination::Failure;
  std::string message;
  double initial_cost = 0;
  double final_cost = 0;
  // The same by residual type, indexed like Problem::loss_type (kLossReprojection,
  // relative prior, global prior, gravity, line, depth); each sums to the total up to
  // rounding. The depth entry includes the priors on free depth corrections, the
  // reprojection entry the priors on free intrinsics (D161).
  std::array<double, kNumLossTypes> initial_cost_by_type{}, final_cost_by_type{};
  int iterations = 0;           // not counting iteration 0
  int accepted_steps = 0;
  int linear_iterations = 0;    // total
  // Wall time of the whole call to solve(), from its entry to its return: checks,
  // the backend's set-up (on a GPU, the device's start-up in the first solve of a
  // process, every allocation and upload, and for SparseSchur the reduced system's
  // structure and its analysis), the iterations and the write-back (document 05).
  double time_seconds = 0;
  double setup_seconds = 0;     // the part before the trust-region loop starts
  // The linear solver the steps used: what Auto resolved to for this problem and
  // device, or the type asked for. A SparseSchur solve that ran out of device memory
  // part way continues with IterativeSchur and reports it here and in the message.
  LinearSolverType linear_solver = LinearSolverType::IterativeSchur;
  // SparseSchur steps whose factorization failed (a reduced system not positive
  // definite in single precision, or a solver failure) and that were solved by
  // conjugate gradients instead (D98).
  int linear_solver_fallbacks = 0;
  std::vector<IterationSummary> history;  // iteration 0 first
  Gauge gauge;  // freedoms actually held fixed
};

// Called after every iteration, including iteration 0. Return false to stop; the
// solve then ends with Termination::Cancelled and the last accepted parameters.
// No printing happens inside the solver; this is the logging hook (document 01).
using IterationCallback = std::function<bool(const IterationSummary&)>;


// The gauge GaugeMode::Auto would apply to this problem, or an empty Gauge when
// none applies (any image or point already fixed). Exposed so reference solvers can
// apply the same one.
Gauge choose_gauge(const Problem& problem);

// The loss of residual type `type` (kLossReprojection, ...) in the options, or
// nullptr past the last type.
Loss* loss_of(SolverOptions& options, std::size_t type);
const Loss* loss_of(const SolverOptions& options, std::size_t type);
// Name of a residual type, as in command-line flags with '-' for '_':
// "reprojection", "relative_prior", "global_prior", "gravity", "line", "depth".
std::string_view residual_type_name(std::size_t type);

// The losses stored with a problem (Problem::loss_type, loss.* sections; D80) in
// place of the options' own, for the residual types the problem names.
SolverOptions with_problem_losses(SolverOptions options, const Problem& problem);
// Stores the options' losses in the problem, one per residual type.
void store_losses(Problem& problem, const SolverOptions& options);

// Solves in place on options.backend. Throws std::invalid_argument for an invalid
// problem or a backend this build or machine does not have.
SolverSummary solve(Problem& problem, const SolverOptions& options = {},
                    const IterationCallback& callback = {});

}  // namespace jba
