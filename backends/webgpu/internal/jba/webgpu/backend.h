#pragma once

#include <memory>
#include <string>

#include "jba/gpu/backend.h"
#include "jba/gpu/device.h"

namespace jba::webgpu {

// Whether a WebGPU adapter is present and the kernels compile on it (once per
// process).
bool available();

// The adapter's description, for reports, or an empty string.
std::string device_name();

// WebGPU backend (exploration): the shared GPU host logic (jba/gpu/backend.h, D96) on
// a WebGPU device, natively through Dawn (Metal, Vulkan or Direct3D underneath) or in
// the browser through Emscripten's WebGPU port. The kernels are a WGSL port of the
// shared source (backends/webgpu/kernels), compiled at run time.
//
// Every buffer is a range of one storage buffer, so the device's addresses are offsets
// into it and the host logic is unchanged. Single precision; strategy B only where the
// host has a factorization (Accelerate on Apple, natively); elsewhere iterative_schur.
//
// Environment, native only: JBA_WEBGPU_TOGGLES, a comma-separated list of Dawn toggles
// to enable (for example disable_robustness), and JBA_WEBGPU_NO_DIRECT=1 to leave
// strategy B out.
class WebGpuBackend final : public gpu::GpuBackend {
 public:
  WebGpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge);
};

// Whether this build has a direct solver for strategy B on the host.
bool has_direct_solver();

std::unique_ptr<gpu::Device> make_device();

}  // namespace jba::webgpu
