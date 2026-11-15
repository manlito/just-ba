#pragma once

// The device interface under the shared GPU host logic (D96): buffers, dispatches of
// the shared kernels, and the optional direct factorization of strategy B. Metal
// (backends/metal) and CUDA (backends/cuda) implement it; everything above it is in
// backends/gpu/src/backend.cpp.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/gpu/layout.h"

namespace jba::gpu {

using jba_gpu::Args;
using jba_gpu::Op;
using jba_gpu::State;

// The kernels of backends/gpu/kernels/solver.inc, in this order; kKernelNames spells
// them as they are named there.
enum class Kernel : int {
  LinearizeSimplePinhole,
  LinearizePinhole,
  LinearizeSimpleRadial,
  LinearizeRadial,
  LinearizeOpenCV,
  LinearizeOpenCVFisheye,
  ScaleObservations,
  ScaleFactors,
  ChunkPartial,
  BlockFinish,
  BlockGather,
  PointColnorm,
  Damping,
  PointBlocks,
  SchurPoints,
  FactorsTimes,
  RhsPoints,
  BackSubstitute,
  FactorModel,
  Reduce,
  PcgVectorsInit,
  PcgInit,
  PcgPrecondition,
  PcgRho,
  PcgDirection,
  PcgAlpha,
  PcgUpdate,
  PcgReset,
  PcgTest,
  SchurTiles,
  TileFinish,
  ObsSchurE,
  ObsSchurG,
  Count,
};

inline constexpr const char* kKernelNames[] = {
    "linearize_simple_pinhole", "linearize_pinhole", "linearize_simple_radial", "linearize_radial",
    "linearize_opencv", "linearize_opencv_fisheye", "scale_observations", "scale_factors", "chunk_partial",
    "block_finish", "block_gather", "point_colnorm", "damping", "point_blocks", "schur_points",
    "factors_times", "rhs_points", "back_substitute", "factor_model", "reduce", "pcg_vectors_init",
    "pcg_init", "pcg_precondition", "pcg_rho", "pcg_direction", "pcg_alpha", "pcg_update", "pcg_reset",
    "pcg_test", "schur_tiles", "tile_finish", "obs_schur_e", "obs_schur_g"};
static_assert(sizeof(kKernelNames) / sizeof(kKernelNames[0]) == static_cast<std::size_t>(Kernel::Count));

// A device buffer and its host side: on Metal the same shared memory, on CUDA a host
// mirror (only for buffers allocated host-visible; `host` is null otherwise).
// `address` is the device address the kernels see.
struct Buffer {
  void* host = nullptr;
  std::uint64_t address = 0;
  std::size_t bytes = 0;
};

// Thrown where the device's memory, or a 32-bit index of strategy B, does not reach:
// LinearSolverType::Auto then takes the iterative solver (D102).
struct DoesNotFit : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class FactorStatus {
  Solved,
  NotPositiveDefinite,  // the step falls back to conjugate gradients
  Failed,               // the solver failed for another reason: the step falls back; repeated, an error
  OutOfMemory,          // the factorization could not get its memory
};

// Factorization of the reduced camera system in double (strategy B), lower triangle
// in compressed columns. analyze() once per solve (throws DoesNotFit when the
// factorization cannot fit); factor_solve() per step.
class DirectSolver {
 public:
  virtual ~DirectSolver() = default;
  virtual void analyze(std::size_t n, const std::vector<std::int64_t>& column_starts,
                       const std::vector<int>& rows) = 0;
  // Factorizes the matrix with these values and solves in place.
  virtual FactorStatus factor_solve(const std::vector<double>& values, std::vector<double>& rhs) = 0;
};

class Device {
 public:
  virtual ~Device() = default;
  virtual std::string name() const = 0;

  // `bytes` (at least 16), copied from `init` (which holds that many) or zeroed. A
  // host-visible buffer has a host side the host may read and write (with to_device /
  // to_host). Throws DoesNotFit when the memory is not there.
  virtual Buffer alloc(std::size_t bytes, const void* init, bool host_visible) = 0;
  // Device memory still free for this process's allocations, as far as it is known.
  virtual std::size_t free_memory() const { return std::numeric_limits<std::size_t>::max(); }
  // Host side to device after the host wrote it, and back before the host reads it.
  // No-ops where the memory is shared. to_device may return before the copy is done:
  // the host writes that host side again only after a finish() or a to_host().
  virtual void to_device(const Buffer& b) = 0;
  virtual void to_host(const Buffer& b) = 0;

  // The argument structure and the state every dispatch sees; once, after setup.
  virtual void bind(const Args& args, const Buffer& state) = 0;

  // A batch of dispatches, run in order; finish() waits for it and brings the state
  // to the host.
  virtual void begin() = 0;
  // n threads, each with its own index (gid), in groups of the device's choosing.
  virtual void dispatch_threads(Kernel k, std::size_t n, const Op& op) = 0;
  // `groups` groups of `threads` threads each.
  virtual void dispatch_groups(Kernel k, std::size_t groups, std::size_t threads, const Op& op) = 0;
  virtual void finish() = 0;

  // Runs f with whatever per-call scope the device needs (an autorelease pool on Metal).
  virtual void scoped(const std::function<void()>& f) { f(); }

  // Strategy B's factorization, or null where this device has none.
  virtual std::unique_ptr<DirectSolver> direct_solver() { return nullptr; }
};

}  // namespace jba::gpu
