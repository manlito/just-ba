#include "jba/metal/backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/gpu/device.h"

// Embedded at build time: the generated Metal reprojection code, the Metal prelude,
// the shared structures and kernels (D96), and layout_sizes (backends/metal/CMakeLists.txt).
extern const char* const kJbaMetalSolverSource;

namespace jba::metal {
namespace {

using gpu::Buffer;
using gpu::Kernel;
using gpu::kKernelNames;

constexpr std::size_t kKernelCount = static_cast<std::size_t>(Kernel::Count);

[[noreturn]] void fail(const std::string& what, NSError* error) {
  std::string msg = "metal: " + what;
  if (error != nil) msg += ": " + std::string(error.localizedDescription.UTF8String);
  throw std::runtime_error(msg);
}

// The compiled kernels, once per process (D50: compiled at run time from embedded
// source; pipelines are cached so a second solve pays nothing).
struct Library {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> pipelines[kKernelCount] = {};
};

const Library& library() {
  static Library lib;
  static std::once_flag once;
  static std::string error;
  std::call_once(once, [] {
    @autoreleasepool {
      try {
        lib.device = MTLCreateSystemDefaultDevice();
        if (lib.device == nil) fail("no Metal device", nil);
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;  // device pointers in argument structures
        // No fast math: the generated arithmetic must match the CPU emission.
        if (@available(macOS 15.0, *)) {
          options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* err = nil;
        id<MTLLibrary> mtl = [lib.device newLibraryWithSource:@(kJbaMetalSolverSource) options:options error:&err];
        if (mtl == nil) fail("kernel compilation failed", err);
        auto pipeline = [&](const char* name) {
          id<MTLFunction> fn = [mtl newFunctionWithName:@(name)];
          if (fn == nil) fail(std::string("kernel not found: ") + name, nil);
          id<MTLComputePipelineState> pso = [lib.device newComputePipelineStateWithFunction:fn error:&err];
          if (pso == nil) fail(std::string("pipeline creation failed: ") + name, err);
          return pso;
        };
        for (std::size_t k = 0; k < kKernelCount; ++k) lib.pipelines[k] = pipeline(kKernelNames[k]);
        // The kernels that work in groups need their group size (review of milestones
        // 3 and 4, finding 11).
        const std::pair<Kernel, NSUInteger> groups[] = {{Kernel::Reduce, jba_gpu::kReduceThreads},
                                                        {Kernel::BlockGather, jba_gpu::kGatherThreads},
                                                        {Kernel::SchurTiles, jba_gpu::kTileThreads}};
        for (const auto& [k, n] : groups)
          if (lib.pipelines[static_cast<int>(k)].maxTotalThreadsPerThreadgroup < n)
            fail(std::string(kKernelNames[static_cast<int>(k)]) + " needs " + std::to_string(n) +
                     " threads per threadgroup",
                 nil);
        // The host's structure layouts must match the kernels'.
        id<MTLComputePipelineState> sizes_pso = pipeline("layout_sizes");
        id<MTLBuffer> sizes = [lib.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [lib.device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sizes_pso];
        [enc setBuffer:sizes offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const auto* s = static_cast<const std::uint32_t*>(sizes.contents);
        const std::uint32_t want[] = {sizeof(jba_gpu::Args),      sizeof(jba_gpu::State),     sizeof(jba_gpu::Op),
                                      sizeof(jba_gpu::Factor),    sizeof(jba_gpu::Chunk),     sizeof(jba_gpu::FBlockDesc),
                                      sizeof(jba_gpu::Tile),      sizeof(jba_gpu::TileEntry), sizeof(jba_gpu::TileChunk)};
        for (int i = 0; i < 9; ++i)
          if (s[i] != want[i])
            fail("structure " + std::to_string(i) + " is " + std::to_string(s[i]) + " bytes in the kernels, " +
                     std::to_string(want[i]) + " on the host",
                 nil);
      } catch (const std::exception& e) {
        error = e.what();
        lib.device = nil;
      }
    }
  });
  if (lib.device == nil) throw std::runtime_error(error.empty() ? "metal: no Metal device" : error);
  return lib;
}

// Strategy B's factorization: Accelerate's sparse Cholesky in double, the symbolic
// factorization once per solve.
class AccelerateSolver final : public gpu::DirectSolver {
 public:
  ~AccelerateSolver() override {
    if (have_symbolic_) SparseCleanup(symbolic_);
  }

  void analyze(std::size_t n, const std::vector<std::int64_t>& column_starts, const std::vector<int>& rows) override {
    column_starts_.assign(column_starts.begin(), column_starts.end());
    structure_ = {};
    structure_.rowCount = static_cast<int>(n);
    structure_.columnCount = static_cast<int>(n);
    structure_.columnStarts = column_starts_.data();
    structure_.rowIndices = const_cast<int*>(rows.data());
    structure_.attributes.kind = SparseSymmetric;
    structure_.attributes.triangle = SparseLowerTriangle;
    structure_.blockSize = 1;
    n_ = n;
    SparseSymbolicFactorOptions opts = _SparseDefaultSymbolicFactorOptions;
    symbolic_ = SparseFactor(SparseFactorizationCholesky, structure_, opts);
    if (symbolic_.status != SparseStatusOK) fail("Accelerate symbolic factorization failed", nil);
    have_symbolic_ = true;
  }

  gpu::FactorStatus factor_solve(const std::vector<double>& values, std::vector<double>& rhs) override {
    SparseMatrix_Double S{};
    S.structure = structure_;
    S.data = const_cast<double*>(values.data());
    SparseOpaqueFactorization_Double f = SparseFactor(symbolic_, S);
    const bool ok = f.status == SparseStatusOK;
    if (ok) {
      DenseVector_Double v{};
      v.count = static_cast<int>(n_);
      v.data = rhs.data();
      SparseSolve(f, v);
    }
    SparseCleanup(f);
    return ok ? gpu::FactorStatus::Solved : gpu::FactorStatus::NotPositiveDefinite;
  }

 private:
  std::vector<long> column_starts_;
  SparseMatrixStructure structure_{};
  SparseOpaqueSymbolicFactorization symbolic_{};
  bool have_symbolic_ = false;
  std::size_t n_ = 0;
};

// Shared memory: the host side of a buffer is the buffer itself.
class MetalDevice final : public gpu::Device {
 public:
  MetalDevice() : lib_(library()) {
    queue_ = [lib_.device newCommandQueue];
    if (queue_ == nil) fail("no command queue", nil);
  }

  std::string name() const override { return "metal"; }

  Buffer alloc(std::size_t bytes, const void* init, bool) override {
    id<MTLBuffer> b = [lib_.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (b == nil) throw gpu::DoesNotFit("metal: buffer allocation failed (" + std::to_string(bytes) + " bytes)");
    if (init != nullptr) std::memcpy(b.contents, init, bytes);
    else std::memset(b.contents, 0, bytes);
    resources_.push_back(b);
    return {b.contents, b.gpuAddress, bytes};
  }

  // Unified memory: what the device recommends for its working set, less what is
  // allocated.
  std::size_t free_memory() const override {
    const std::uint64_t limit = lib_.device.recommendedMaxWorkingSetSize, used = lib_.device.currentAllocatedSize;
    return limit > used ? static_cast<std::size_t>(limit - used) : 0;
  }

  void to_device(const Buffer&) override {}
  void to_host(const Buffer&) override {}

  void bind(const gpu::Args& args, const Buffer& state) override {
    args_buf_ = [lib_.device newBufferWithBytes:&args length:sizeof(args) options:MTLResourceStorageModeShared];
    if (args_buf_ == nil) fail("buffer allocation failed", nil);
    for (id<MTLResource> r : resources_) {
      id<MTLBuffer> b = (id<MTLBuffer>)r;
      if (b.contents == state.host) state_buf_ = b;
    }
    if (state_buf_ == nil) fail("state buffer not found", nil);
  }

  void begin() override {
    cb_ = [queue_ commandBuffer];
    enc_ = [cb_ computeCommandEncoder];
    [enc_ useResources:resources_.data()
                 count:resources_.size()
                 usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc_ setBuffer:args_buf_ offset:0 atIndex:0];
    [enc_ setBuffer:state_buf_ offset:0 atIndex:1];
  }

  void dispatch_threads(Kernel k, std::size_t n, const gpu::Op& op) override {
    id<MTLComputePipelineState> pso = lib_.pipelines[static_cast<int>(k)];
    [enc_ setComputePipelineState:pso];
    [enc_ setBytes:&op length:sizeof(op) atIndex:2];
    const NSUInteger group = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
    [enc_ dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
  }

  void dispatch_groups(Kernel k, std::size_t groups, std::size_t threads, const gpu::Op& op) override {
    [enc_ setComputePipelineState:lib_.pipelines[static_cast<int>(k)]];
    [enc_ setBytes:&op length:sizeof(op) atIndex:2];
    [enc_ dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  }

  void finish() override {
    [enc_ endEncoding];
    [cb_ commit];
    [cb_ waitUntilCompleted];
    if (cb_.status != MTLCommandBufferStatusCompleted) fail("command buffer failed", cb_.error);
    enc_ = nil;
    cb_ = nil;
  }

  void scoped(const std::function<void()>& f) override {
    @autoreleasepool {
      f();
    }
  }

  std::unique_ptr<gpu::DirectSolver> direct_solver() override { return std::make_unique<AccelerateSolver>(); }

 private:
  const Library& lib_;
  id<MTLCommandQueue> queue_ = nil;
  std::vector<id<MTLResource>> resources_;
  id<MTLBuffer> args_buf_ = nil, state_buf_ = nil;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
};

}  // namespace

std::unique_ptr<gpu::Device> make_device() { return std::make_unique<MetalDevice>(); }

MetalBackend::MetalBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : gpu::GpuBackend(problem, options, gauge, make_device) {}

}  // namespace jba::metal
