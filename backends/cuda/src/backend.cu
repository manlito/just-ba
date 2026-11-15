// CUDA backend (milestone 5, D96): the shared kernels compiled by nvcc, and the
// Device the shared host logic runs on.

#include "jba/cuda/backend.h"

#include <cuda_runtime.h>
#ifdef JBA_HAVE_CUDSS
#include <cudss.h>
#endif

#include <algorithm>
#include <cmath>
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
    pcg_test, schur_tiles, tile_finish, obs_schur_e, obs_schur_g};
static_assert(sizeof(kKernels) / sizeof(kKernels[0]) == static_cast<std::size_t>(Kernel::Count));

constexpr unsigned kThreads = 256;  // per block for one-thread-per-item dispatches

#ifdef JBA_HAVE_CUDSS
void check_cudss(cudssStatus_t s, const char* what) {
  if (s != CUDSS_STATUS_SUCCESS) throw std::runtime_error(std::string("cudss: ") + what + " failed (status " +
                                                          std::to_string(static_cast<int>(s)) + ")");
}

// Strategy B's factorization on the GPU: cuDSS's sparse Cholesky in double, the
// analysis (reordering and symbolic factorization) once per solve, deterministic. The
// lower triangle in compressed columns is the upper triangle in compressed rows.
class CudssSolver final : public gpu::DirectSolver {
 public:
  explicit CudssSolver(cudaStream_t stream) : stream_(stream) {
    check_cudss(cudssCreate(&handle_), "create");
    check_cudss(cudssSetStream(handle_, stream_), "stream");
    check_cudss(cudssConfigCreate(&config_), "config");
    const int deterministic = 1;
    check_cudss(cudssConfigSet(config_, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic, sizeof(deterministic)),
                "deterministic mode");
    check_cudss(cudssDataCreate(handle_, &data_), "data");
  }

  ~CudssSolver() override {
    if (A_ != nullptr) cudssMatrixDestroy(A_);
    if (X_ != nullptr) cudssMatrixDestroy(X_);
    if (B_ != nullptr) cudssMatrixDestroy(B_);
    if (data_ != nullptr) cudssDataDestroy(handle_, data_);
    if (config_ != nullptr) cudssConfigDestroy(config_);
    if (handle_ != nullptr) cudssDestroy(handle_);
    for (void* p : {static_cast<void*>(offsets_), static_cast<void*>(cols_), static_cast<void*>(vals_),
                    static_cast<void*>(x_), static_cast<void*>(b_)})
      if (p != nullptr) cudaFree(p);
  }

  void analyze(std::size_t n, const std::vector<long>& column_starts, const std::vector<int>& rows) override {
    n_ = n;
    nnz_ = rows.size();
    std::vector<int> offsets(column_starts.begin(), column_starts.end());
    check(cudaMalloc(&offsets_, offsets.size() * sizeof(int)), "cudss offsets");
    check(cudaMalloc(&cols_, std::max<std::size_t>(nnz_, 1) * sizeof(int)), "cudss indices");
    check(cudaMalloc(&vals_, std::max<std::size_t>(nnz_, 1) * sizeof(double)), "cudss values");
    check(cudaMalloc(&x_, n * sizeof(double)), "cudss solution");
    check(cudaMalloc(&b_, n * sizeof(double)), "cudss right-hand side");
    check(cudaMemcpyAsync(offsets_, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream_),
          "cudss upload");
    check(cudaMemcpyAsync(cols_, rows.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice, stream_), "cudss upload");
    // The analysis looks at the pattern only; the values must be finite.
    check(cudaMemsetAsync(vals_, 0, nnz_ * sizeof(double), stream_), "cudss memset");
    const auto ni = static_cast<std::int64_t>(n);
    check_cudss(cudssMatrixCreateCsr(&A_, ni, ni, static_cast<std::int64_t>(nnz_), offsets_, nullptr, cols_, vals_,
                                     CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER,
                                     CUDSS_BASE_ZERO),
                "matrix");
    check_cudss(cudssMatrixCreateDn(&X_, ni, 1, ni, x_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR), "solution");
    check_cudss(cudssMatrixCreateDn(&B_, ni, 1, ni, b_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR), "right-hand side");
    check_cudss(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_, A_, X_, B_), "analysis");
    check(cudaStreamSynchronize(stream_), "cudss analysis");
  }

  bool factor_solve(const std::vector<double>& values, std::vector<double>& rhs) override {
    check(cudaMemcpyAsync(vals_, values.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, stream_),
          "cudss upload");
    check(cudaMemcpyAsync(b_, rhs.data(), n_ * sizeof(double), cudaMemcpyHostToDevice, stream_), "cudss upload");
    int info = 0;
    check_cudss(cudssDataSet(handle_, data_, CUDSS_DATA_INFO, &info, sizeof(info)), "reset info");
    check_cudss(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, data_, A_, X_, B_), "factorization");
    // Not positive definite: a zero or negative pivot (info), or pivots cuDSS had to
    // replace; the step then falls back to conjugate gradients, as on Metal.
    std::size_t written = 0;
    check_cudss(cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &info, sizeof(info), &written), "info");
    int npivots = 0;
    check_cudss(cudssDataGet(handle_, data_, CUDSS_DATA_NPIVOTS, &npivots, sizeof(npivots), &written), "pivots");
    if (info != 0 || npivots != 0) return false;
    check_cudss(cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, A_, X_, B_), "solve");
    check(cudaMemcpyAsync(rhs.data(), x_, n_ * sizeof(double), cudaMemcpyDeviceToHost, stream_), "cudss download");
    check(cudaStreamSynchronize(stream_), "cudss solve");
    for (double v : rhs)
      if (!std::isfinite(v)) return false;
    return true;
  }

 private:
  cudaStream_t stream_;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t A_ = nullptr, X_ = nullptr, B_ = nullptr;
  int *offsets_ = nullptr, *cols_ = nullptr;
  double *vals_ = nullptr, *x_ = nullptr, *b_ = nullptr;
  std::size_t n_ = 0, nnz_ = 0;
};
#endif

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

#ifdef JBA_HAVE_CUDSS
  std::unique_ptr<gpu::DirectSolver> direct_solver() override { return std::make_unique<CudssSolver>(stream_); }
#endif

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
