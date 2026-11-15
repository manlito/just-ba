#pragma once

#include <memory>
#include <string>

#include "jba/gpu/backend.h"
#include "jba/gpu/device.h"

namespace jba::cuda {

// Whether a CUDA device is present (and the driver can run this build's kernels).
bool available();

// Name of the CUDA device used (device 0), or an empty string.
std::string device_name();

// CUDA backend (milestone 5): the shared GPU host logic (jba/gpu/backend.h, D96) on
// a CUDA device. The kernels are the shared source compiled by nvcc. Needs only the
// CUDA runtime, linked statically.
//
// Throws std::runtime_error when no CUDA device is present or a kernel fails.
class CudaBackend final : public gpu::GpuBackend {
 public:
  CudaBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge);
};

// The CUDA device under the shared host logic.
std::unique_ptr<gpu::Device> make_device();

}  // namespace jba::cuda
