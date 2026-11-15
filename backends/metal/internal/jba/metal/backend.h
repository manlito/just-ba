#pragma once

#include <memory>

#include "jba/gpu/backend.h"
#include "jba/gpu/device.h"

namespace jba::metal {

// Metal backend (milestone 4): the shared GPU host logic (jba/gpu/backend.h, D96)
// on a Metal device, with Accelerate's sparse Cholesky for sparse_schur. Kernels are
// compiled at run time from embedded source (D50).
//
// Throws std::runtime_error when no Metal device is present or a kernel fails.
class MetalBackend final : public gpu::GpuBackend {
 public:
  MetalBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge);
};

// The Metal device under the shared host logic.
std::unique_ptr<gpu::Device> make_device();

}  // namespace jba::metal
