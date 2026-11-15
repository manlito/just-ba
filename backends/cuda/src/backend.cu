// CUDA backend (milestone 5, D96): the shared kernels compiled by nvcc, and the
// Device the shared host logic runs on.

#include "jba/cuda/backend.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// The shared kernels behind the CUDA prelude.
#include "prelude.cuh"
#include "jba/gpu/layout.h"
#include "../../gpu/kernels/solver.inc"

namespace jba::cuda {
namespace {

using gpu::Buffer;
using gpu::Kernel;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

using KernelFn = void (*)(const jba_gpu::Args, jba_gpu::State*, const jba_gpu::Op);

// In the order of gpu::Kernel.
constexpr KernelFn kKernels[] = {
    linearize_simple_pinhole, linearize_pinhole, linearize_simple_radial, linearize_radial, linearize_opencv,
    linearize_opencv_fisheye, scale_observations, scale_factors, chunk_partial, block_finish, block_gather,
    point_colnorm, damping, point_blocks, schur_points, factors_times, rhs_points, back_substitute, factor_model,
    reduce, pcg_vectors_init, pcg_init, pcg_precondition, pcg_rho, pcg_direction, pcg_alpha, pcg_update, pcg_reset,
    pcg_test, schur_tiles, tile_finish};
static_assert(sizeof(kKernels) / sizeof(kKernels[0]) == static_cast<std::size_t>(Kernel::Count));

constexpr unsigned kThreads = 256;  // per block for one-thread-per-item dispatches

// Device memory, with a pinned host mirror for the buffers the host reads or writes.
// One stream; finish() waits for it and copies the state back.
class CudaDevice final : public gpu::Device {
 public:
  CudaDevice() {
    check(cudaSetDevice(0), "no device");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  }

  ~CudaDevice() override {
    for (const Buffer& b : buffers_) {
      cudaFree(reinterpret_cast<void*>(static_cast<std::uintptr_t>(b.address)));
      if (b.host != nullptr) cudaFreeHost(b.host);
    }
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  std::string name() const override { return "cuda"; }

  Buffer alloc(std::size_t bytes, const void* init, bool host_visible) override {
    void* d = nullptr;
    check(cudaMalloc(&d, bytes), ("allocation of " + std::to_string(bytes) + " bytes").c_str());
    Buffer b{nullptr, reinterpret_cast<std::uintptr_t>(d), bytes};
    buffers_.push_back(b);
    if (host_visible) {
      check(cudaMallocHost(&b.host, bytes), "pinned allocation");
      buffers_.back().host = b.host;
      if (init != nullptr) std::memcpy(b.host, init, bytes);
      else std::memset(b.host, 0, bytes);
    }
    // On this device's stream, and waited for: the stream does not synchronize with
    // the legacy default stream, and cudaMemset there returns before it is done, so a
    // late memset could otherwise land on a kernel's output.
    if (init != nullptr) check(cudaMemcpyAsync(d, init, bytes, cudaMemcpyHostToDevice, stream_), "upload");
    else check(cudaMemsetAsync(d, 0, bytes, stream_), "memset");
    check(cudaStreamSynchronize(stream_), "upload");
    return b;
  }

  void to_device(const Buffer& b) override {
    check(cudaMemcpyAsync(device_ptr(b), b.host, b.bytes, cudaMemcpyHostToDevice, stream_), "upload");
  }

  void to_host(const Buffer& b) override {
    check(cudaMemcpyAsync(b.host, device_ptr(b), b.bytes, cudaMemcpyDeviceToHost, stream_), "download");
    check(cudaStreamSynchronize(stream_), "download");
  }

  void bind(const gpu::Args& args, const Buffer& state) override {
    args_ = args;
    state_ = state;
  }

  void begin() override {}

  void dispatch_threads(Kernel k, std::size_t n, const gpu::Op& op) override {
    const auto blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    kKernels[static_cast<int>(k)]<<<blocks, kThreads, 0, stream_>>>(args_, state_ptr(), op);
  }

  void dispatch_groups(Kernel k, std::size_t groups, std::size_t threads, const gpu::Op& op) override {
    kKernels[static_cast<int>(k)]<<<static_cast<unsigned>(groups), static_cast<unsigned>(threads), 0, stream_>>>(
        args_, state_ptr(), op);
  }

  void finish() override {
    check(cudaGetLastError(), "kernel launch");
    to_host(state_);
  }

 private:
  static void* device_ptr(const Buffer& b) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(b.address)); }
  jba_gpu::State* state_ptr() const { return static_cast<jba_gpu::State*>(device_ptr(state_)); }

  cudaStream_t stream_ = nullptr;
  std::vector<Buffer> buffers_;
  gpu::Args args_{};
  Buffer state_{};
};

}  // namespace

bool available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string device_name() {
  cudaDeviceProp prop{};
  if (!available() || cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return {};
  return prop.name;
}

std::unique_ptr<gpu::Device> make_device() { return std::make_unique<CudaDevice>(); }

CudaBackend::CudaBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : gpu::GpuBackend(problem, options, gauge, make_device()) {}

}  // namespace jba::cuda
