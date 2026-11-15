#pragma once

#include <memory>
#include <string>

#include "jba/gpu/backend.h"
#include "jba/gpu/device.h"

namespace jba::cuda {

// Whether a CUDA device is present and runs this build's kernels (a test kernel, once
// per process).
bool available();

// Name of the CUDA device used (the calling thread's current device), or an empty
// string.
std::string device_name();

// CUDA backend (milestone 5): the shared GPU host logic (jba/gpu/backend.h, D96) on
// the calling thread's current CUDA device (cudaSetDevice before the solve picks
// another; the thread's device is restored after each call). The kernels are the
// shared source compiled by nvcc. Needs the CUDA runtime, linked statically, and with
// sparse_schur cuDSS, which loads cuBLAS and NVRTC.
//
// Throws std::runtime_error when no CUDA device is present or a kernel fails.
class CudaBackend final : public gpu::GpuBackend {
 public:
  CudaBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge);
};

// The CUDA device under the shared host logic.
std::unique_ptr<gpu::Device> make_device();

}  // namespace jba::cuda
