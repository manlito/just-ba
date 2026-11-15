// CUDA backend (milestone 5, D96): the shared kernels compiled by nvcc, and the
// Device the shared host logic runs on.

#include "jba/cuda/backend.h"

#include <cuda_runtime.h>
#ifdef JBA_HAVE_CUDSS
#include <cudss.h>
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
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

// The same, for an allocation: out of memory is DoesNotFit (LinearSolverType::Auto
// can then take the iterative solver, D102).
void check_alloc(cudaError_t e, std::size_t bytes) {
  if (e == cudaErrorMemoryAllocation) {
    cudaGetLastError();  // not sticky; clear it
    throw gpu::DoesNotFit("cuda: out of device memory (allocation of " + std::to_string(bytes) + " bytes)");
  }
  check(e, "allocation");
}

// Runs the kernels on `device` and gives the calling thread its own device back
// (review of milestone 5, finding 10).
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    check(cudaGetDevice(&previous_), "current device");
    if (previous_ != device) check(cudaSetDevice(device), "device");
  }
  ~DeviceScope() {
    int now = previous_;
    if (cudaGetDevice(&now) == cudaSuccess && now != previous_) cudaSetDevice(previous_);
  }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int previous_ = 0;
};

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
  if (s == CUDSS_STATUS_ALLOC_FAILED)
    throw gpu::DoesNotFit(std::string("cudss: ") + what + ": out of device memory");
  if (s != CUDSS_STATUS_SUCCESS) throw std::runtime_error(std::string("cudss: ") + what + " failed (status " +
                                                          std::to_string(static_cast<int>(s)) + ")");
}

// Strategy B's factorization on the GPU: cuDSS's sparse Cholesky in double, the
// analysis (reordering and symbolic factorization) once per solve, deterministic. The
// lower triangle in compressed columns is the upper triangle in compressed rows.
class CudssSolver final : public gpu::DirectSolver {
 public:
  explicit CudssSolver(cudaStream_t stream) : stream_(stream) {
    // A constructor that throws runs no destructor: release what was made so far.
    try {
      init();
    } catch (...) {
      release();
      throw;
    }
  }

  ~CudssSolver() override { release(); }

 private:
  void init() {
    check_cudss(cudssCreate(&handle_), "create");
    check_cudss(cudssSetStream(handle_, stream_), "stream");
    check_cudss(cudssConfigCreate(&config_), "config");
    const int deterministic = 1;
    check_cudss(cudssConfigSet(config_, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic, sizeof(deterministic)),
                "deterministic mode");
    check_cudss(cudssDataCreate(handle_, &data_), "data");
  }

  void release() {
    if (A_ != nullptr) cudssMatrixDestroy(A_);
    if (X_ != nullptr) cudssMatrixDestroy(X_);
    if (B_ != nullptr) cudssMatrixDestroy(B_);
    if (data_ != nullptr) cudssDataDestroy(handle_, data_);
    if (config_ != nullptr) cudssConfigDestroy(config_);
    if (handle_ != nullptr) cudssDestroy(handle_);
    for (void* p : {static_cast<void*>(offsets_), static_cast<void*>(cols_), static_cast<void*>(vals_),
                    static_cast<void*>(x_), static_cast<void*>(b_)})
      if (p != nullptr) cudaFree(p);
    A_ = X_ = B_ = nullptr;
    data_ = nullptr;
    config_ = nullptr;
    handle_ = nullptr;
    offsets_ = cols_ = nullptr;
    vals_ = x_ = b_ = nullptr;
  }

 public:
  void analyze(std::size_t n, const std::vector<std::int64_t>& column_starts, const std::vector<int>& rows) override {
    n_ = n;
    nnz_ = rows.size();
    // cuDSS's indices here are 32-bit (the host checks the count first; D102).
    if (nnz_ > static_cast<std::size_t>(INT_MAX) || n > static_cast<std::size_t>(INT_MAX))
      throw gpu::DoesNotFit("cudss: the reduced camera system passes 32-bit indices");
    std::vector<int> offsets(column_starts.begin(), column_starts.end());
    auto alloc = [](auto** p, std::size_t bytes) { check_alloc(cudaMalloc(p, bytes), bytes); };
    alloc(&offsets_, offsets.size() * sizeof(int));
    alloc(&cols_, std::max<std::size_t>(nnz_, 1) * sizeof(int));
    alloc(&vals_, std::max<std::size_t>(nnz_, 1) * sizeof(double));
    alloc(&x_, n * sizeof(double));
    alloc(&b_, n * sizeof(double));
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
    // The factorization's peak device memory, known after the analysis, against
    // what is free (finding 4 of the review of milestone 5).
    std::int64_t est[16] = {};
    std::size_t written = 0;
    if (cudssDataGet(handle_, data_, CUDSS_DATA_MEMORY_ESTIMATES, est, sizeof(est), &written) ==
        CUDSS_STATUS_SUCCESS) {
      std::size_t free = 0, total = 0;
      check(cudaMemGetInfo(&free, &total), "memory");
      const auto peak = static_cast<std::size_t>(std::max<std::int64_t>(est[1], 0));
      if (peak > free / 10 * 9)
        throw gpu::DoesNotFit("cudss: the factorization needs about " + std::to_string(peak >> 20) +
                              " MiB of device memory, " + std::to_string(free >> 20) + " MiB are free");
    }
  }

  gpu::FactorStatus factor_solve(const std::vector<double>& values, std::vector<double>& rhs) override {
    check(cudaMemcpyAsync(vals_, values.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, stream_),
          "cudss upload");
    check(cudaMemcpyAsync(b_, rhs.data(), n_ * sizeof(double), cudaMemcpyHostToDevice, stream_), "cudss upload");
    int info = 0;
    check_cudss(cudssDataSet(handle_, data_, CUDSS_DATA_INFO, &info, sizeof(info)), "reset info");
    // A factorization that does not succeed falls back to conjugate gradients for
    // this step, as a matrix that is not positive definite does (D98); one that could
    // not get its memory says so.
    const cudssStatus_t fs = cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, data_, A_, X_, B_);
    if (fs == CUDSS_STATUS_ALLOC_FAILED) return gpu::FactorStatus::OutOfMemory;
    if (fs != CUDSS_STATUS_SUCCESS) return gpu::FactorStatus::Failed;
    // Not positive definite: a zero or negative pivot (info), or pivots cuDSS had to
    // replace; the step then falls back to conjugate gradients, as on Metal.
    std::size_t written = 0;
    check_cudss(cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &info, sizeof(info), &written), "info");
    int npivots = 0;
    check_cudss(cudssDataGet(handle_, data_, CUDSS_DATA_NPIVOTS, &npivots, sizeof(npivots), &written), "pivots");
    if (info != 0 || npivots != 0) return gpu::FactorStatus::NotPositiveDefinite;
    const cudssStatus_t ss = cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, A_, X_, B_);
    if (ss == CUDSS_STATUS_ALLOC_FAILED) return gpu::FactorStatus::OutOfMemory;
    if (ss != CUDSS_STATUS_SUCCESS) return gpu::FactorStatus::Failed;
    check(cudaMemcpyAsync(rhs.data(), x_, n_ * sizeof(double), cudaMemcpyDeviceToHost, stream_), "cudss download");
    check(cudaStreamSynchronize(stream_), "cudss solve");
    for (double v : rhs)
      if (!std::isfinite(v)) return gpu::FactorStatus::NotPositiveDefinite;
    return gpu::FactorStatus::Solved;
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
  // The calling thread's current device (0 unless the caller chose another).
  CudaDevice() {
    check(cudaGetDevice(&device_), "no device");
    // An error the caller left unread would otherwise be reported as ours.
    cudaGetLastError();
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
    check_alloc(cudaMalloc(&d, bytes), bytes);
    Buffer b{nullptr, reinterpret_cast<std::uintptr_t>(d), bytes};
    buffers_.push_back(b);
    if (host_visible) {
      check_alloc(cudaMallocHost(&b.host, bytes), bytes);
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

  std::size_t free_memory() const override {
    std::size_t free = 0, total = 0;
    return cudaMemGetInfo(&free, &total) == cudaSuccess ? free : 0;
  }

  void scoped(const std::function<void()>& f) override {
    DeviceScope scope(device_);
    f();
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

  int device_ = 0;
  cudaStream_t stream_ = nullptr;
  std::vector<Buffer> buffers_;
  gpu::Args args_{};
  Buffer state_{};
};

}  // namespace

namespace {

__global__ void probe(int* out) { *out = 1; }

}  // namespace

// A device is present and runs this build's kernels: a one-thread kernel on the
// calling thread's current device, once per process (a GPU of an architecture the
// build has no code for fails here rather than at the first solve).
bool available() {
  static bool ok = false;
  static std::once_flag once;
  std::call_once(once, [] {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
      cudaGetLastError();
      return;
    }
    int* d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) == cudaSuccess) {
      probe<<<1, 1>>>(d);
      ok = cudaGetLastError() == cudaSuccess && cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
           h == 1;
      cudaFree(d);
    }
    cudaGetLastError();
  });
  return ok;
}

std::string device_name() {
  cudaDeviceProp prop{};
  int device = 0;
  if (!available() || cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
    return {};
  return prop.name;
}

std::unique_ptr<gpu::Device> make_device() { return std::make_unique<CudaDevice>(); }

CudaBackend::CudaBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : gpu::GpuBackend(problem, options, gauge, make_device) {}

}  // namespace jba::cuda
