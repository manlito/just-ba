#include "jba/webgpu/backend.h"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
#define JBA_WEBGPU_ACCELERATE 1
#include <Accelerate/Accelerate.h>
#endif

#include "jba/gpu/device.h"

// Embedded at build time (backends/webgpu/CMakeLists.txt): the generated WGSL
// reprojection code, the WGSL port of the shared kernels, and the linearize template.
extern const char* const kJbaWgslReprojection;
extern const char* const kJbaWgslSolver;
extern const char* const kJbaWgslLinearize;

namespace jba::webgpu {
namespace {

using gpu::Buffer;
using gpu::Kernel;
using gpu::kKernelNames;

constexpr std::size_t kKernelCount = static_cast<std::size_t>(Kernel::Count);
constexpr std::uint32_t kThreads = 256;        // group size of the kernels dispatched by thread
constexpr std::uint64_t kAlign = 256;          // sub-allocations of the heap
constexpr std::uint32_t kOpStride = 256;       // minUniformBufferOffsetAlignment at most
constexpr std::uint32_t kOpSlots = 4096;       // dispatches per submission before a flush
constexpr std::uint32_t kMaxGroupsX = 65535;   // WebGPU's guaranteed groups per dimension

// The Args of solver.wgsl: layout.h's pointers as word indices, then its scalars, then
// the State's word index.
constexpr std::size_t kArgPointers = offsetof(jba_gpu::Args, nobs) / sizeof(void*);
static_assert(offsetof(jba_gpu::Args, nobs) % sizeof(void*) == 0);
static_assert(kArgPointers == 69, "layout.h's Args changed: update solver.wgsl's Args and this count");
constexpr std::size_t kArgScalars = 17;  // nobs .. pivot_floor
static_assert(offsetof(jba_gpu::Args, pivot_floor) + 4 == offsetof(jba_gpu::Args, nobs) + 4 * kArgScalars);
static_assert(sizeof(jba_gpu::State) == 100, "layout.h's State changed: update solver.wgsl's S_ constants");

struct WgslOp {
  std::uint32_t x, y, n;
  std::int32_t mode, slot, iter;
  float radius;
  std::uint32_t groups;  // groups in the dispatch; the grid may hold more
};
static_assert(sizeof(WgslOp) == 32);

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("webgpu: " + what); }

WGPUStringView sv(const char* s) { return WGPUStringView{s, WGPU_STRLEN}; }
WGPUStringView sv(const std::string& s) { return WGPUStringView{s.data(), s.size()}; }
std::string str(WGPUStringView v) {
  if (v.data == nullptr) return {};
  return v.length == WGPU_STRLEN ? std::string(v.data) : std::string(v.data, v.length);
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  for (std::size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size()))
    s.replace(at, from.size(), to);
  return s;
}

// Group sizes of the kernels dispatched by groups (gpu/device.h, solver.wgsl).
std::uint32_t group_size(Kernel k) {
  if (k == Kernel::SchurTiles) return jba_gpu::kTileThreads;
  return kThreads;  // Reduce and BlockGather are 256 as well
}

// The whole WGSL module: generated residuals, the kernels, one linearize per camera
// model, and an entry point per kernel.
std::string module_source() {
  std::string src = kJbaWgslReprojection;
  src += kJbaWgslSolver;
  const std::pair<const char*, int> models[] = {{"simple_pinhole", 3}, {"pinhole", 4}, {"simple_radial", 4},
                                                {"radial", 5},         {"opencv", 8},  {"opencv_fisheye", 8}};
  for (const auto& [name, np] : models)
    src += replace_all(replace_all(kJbaWgslLinearize, "$MODEL", name), "$NP", std::to_string(np));
  for (std::size_t k = 0; k < kKernelCount; ++k) {
    const std::string n = kKernelNames[k];
    const std::string w = std::to_string(group_size(static_cast<Kernel>(k)));
    src += "\n@compute @workgroup_size(" + w + ") fn " + n +
           "(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u, "
           "@builtin(local_invocation_index) li: u32) {\n"
           "  let g = wg.y * nw.x + wg.x;\n"
           "  if (g >= op.groups) { return; }\n"
           "  " + n + "_body(g * " + w + "u + li, g, li);\n}\n";
  }
  // solver.wgsl's Op ends with `pad`; the device puts the group count there.
  return replace_all(src, "  radius: f32,\n  pad: f32,\n}", "  radius: f32,\n  groups: u32,\n}");
}

// ---------------------------------------------------------------------------
// Process-wide: instance, adapter, device, pipelines. Created once (like Metal's
// library, D50); a solve gets its own heap and bindings.

struct Context {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  WGPULimits limits = WGPU_LIMITS_INIT;
  std::string name;
  WGPUBindGroupLayout bgl = nullptr;
  WGPUPipelineLayout layout = nullptr;
  WGPUComputePipeline pipelines[kKernelCount] = {};
  std::string error;       // why there is no device
  std::string device_error;  // an uncaptured error since the last check
  double compile_seconds = 0;
  // One solve at a time on the shared device (Dawn's device is not thread-safe by
  // default). Recursive: a solve started from another solve's iteration callback, on
  // the same thread, runs between that solve's batches and must not deadlock.
  std::recursive_mutex mutex;
};

Context& ctx() {
  static Context c;
  return c;
}

void wait(WGPUFuture f) {
  WGPUFutureWaitInfo info{f, false};
  const WGPUWaitStatus st = wgpuInstanceWaitAny(ctx().instance, 1, &info, UINT64_MAX);
  if (st != WGPUWaitStatus_Success || !info.completed) fail("waiting for the device failed");
}

void check_device_error() {
  Context& c = ctx();
  if (!c.device_error.empty()) {
    const std::string e = c.device_error;
    c.device_error.clear();
    fail(e);
  }
}

// Validation errors of what f creates, as an exception.
template <typename F>
void checked(const std::string& what, F f) {
  Context& c = ctx();
  wgpuDevicePushErrorScope(c.device, WGPUErrorFilter_Validation);
  f();
  std::string msg;
  WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
  cb.mode = WGPUCallbackMode_WaitAnyOnly;
  cb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* u1, void*) {
    if (type != WGPUErrorType_NoError) *static_cast<std::string*>(u1) = str(message);
  };
  cb.userdata1 = &msg;
  wait(wgpuDevicePopErrorScope(c.device, cb));
  if (!msg.empty()) fail(what + ": " + msg);
}

void create_context() {
  Context& c = ctx();
  WGPUInstanceDescriptor idesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
  const WGPUInstanceFeatureName feature = WGPUInstanceFeatureName_TimedWaitAny;
  idesc.requiredFeatureCount = 1;
  idesc.requiredFeatures = &feature;
  c.instance = wgpuCreateInstance(&idesc);
  if (c.instance == nullptr) fail("no WebGPU instance");

  WGPURequestAdapterOptions aopt = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
  aopt.powerPreference = WGPUPowerPreference_HighPerformance;
  WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  acb.mode = WGPUCallbackMode_WaitAnyOnly;
  std::string amsg;
  acb.callback = [](WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView m, void* u1, void* u2) {
    if (st == WGPURequestAdapterStatus_Success) *static_cast<WGPUAdapter*>(u1) = a;
    else *static_cast<std::string*>(u2) = str(m);
  };
  acb.userdata1 = &c.adapter;
  acb.userdata2 = &amsg;
  wait(wgpuInstanceRequestAdapter(c.instance, &aopt, acb));
  if (c.adapter == nullptr) fail("no adapter" + (amsg.empty() ? "" : ": " + amsg));

  WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
  bool not_a_gpu = false;
  if (wgpuAdapterGetInfo(c.adapter, &info) == WGPUStatus_Success) {
    c.name = str(info.device);
    if (const std::string d = str(info.description); !d.empty() && d != c.name) c.name += " (" + d + ")";
    // Dawn's null backend runs no shaders, and a CPU adapter (a software renderer)
    // is slower than the CPU backend: neither is a GPU. On a machine without a GPU
    // driver Dawn offers one of them, where the solves came out wrong (CI, D128).
    not_a_gpu = info.backendType == WGPUBackendType_Null || info.adapterType == WGPUAdapterType_CPU;
    wgpuAdapterInfoFreeMembers(info);
  }
  if (not_a_gpu) fail("no GPU adapter (only " + (c.name.empty() ? std::string("a software one") : c.name) + ")");
  WGPULimits have = WGPU_LIMITS_INIT;
  if (wgpuAdapterGetLimits(c.adapter, &have) != WGPUStatus_Success) fail("adapter limits unavailable");

  // Ask for what the kernels need and as much buffer as the adapter offers.
  WGPULimits want = WGPU_LIMITS_INIT;
  want.maxStorageBufferBindingSize = have.maxStorageBufferBindingSize;
  want.maxBufferSize = have.maxBufferSize;
  want.maxComputeWorkgroupStorageSize = 16384;
  if (have.maxComputeWorkgroupStorageSize < 16384) fail("the kernels need 16 KiB of group memory");

  WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
  ddesc.requiredLimits = &want;
  ddesc.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType, WGPUStringView m, void*, void*) {
    ctx().device_error = str(m);
  };
  ddesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  ddesc.deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason r, WGPUStringView m, void*,
                                             void*) {
    if (r != WGPUDeviceLostReason_Destroyed && r != WGPUDeviceLostReason_CallbackCancelled)
      ctx().device_error = "device lost: " + str(m);
  };
#ifndef __EMSCRIPTEN__
  // Dawn toggles for experiments (JBA_WEBGPU_TOGGLES=disable_robustness,...).
  std::vector<std::string> toggle_names;
  std::vector<const char*> toggles;
  if (const char* env = std::getenv("JBA_WEBGPU_TOGGLES")) {
    std::string s = env;
    for (std::size_t at = 0; at <= s.size();) {
      const std::size_t end = std::min(s.find(',', at), s.size());
      if (end > at) toggle_names.push_back(s.substr(at, end - at));
      at = end + 1;
    }
  }
  for (const auto& t : toggle_names) toggles.push_back(t.c_str());
  WGPUDawnTogglesDescriptor tdesc = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
  tdesc.enabledToggleCount = toggles.size();
  tdesc.enabledToggles = toggles.data();
  if (!toggles.empty()) ddesc.nextInChain = &tdesc.chain;
#endif
  WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  dcb.mode = WGPUCallbackMode_WaitAnyOnly;
  std::string dmsg;
  dcb.callback = [](WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView m, void* u1, void* u2) {
    if (st == WGPURequestDeviceStatus_Success) *static_cast<WGPUDevice*>(u1) = d;
    else *static_cast<std::string*>(u2) = str(m);
  };
  dcb.userdata1 = &c.device;
  dcb.userdata2 = &dmsg;
  wait(wgpuAdapterRequestDevice(c.adapter, &ddesc, dcb));
  if (c.device == nullptr) fail("no device" + (dmsg.empty() ? "" : ": " + dmsg));
  c.queue = wgpuDeviceGetQueue(c.device);
  if (wgpuDeviceGetLimits(c.device, &c.limits) != WGPUStatus_Success) fail("device limits unavailable");

  // One layout for every kernel: the heap, Args, and Op at a dynamic offset.
  WGPUBindGroupLayoutEntry e[3] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                   WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
  for (int i = 0; i < 3; ++i) {
    e[i].binding = static_cast<std::uint32_t>(i);
    e[i].visibility = WGPUShaderStage_Compute;
  }
  e[0].buffer.type = WGPUBufferBindingType_Storage;
  e[1].buffer.type = WGPUBufferBindingType_Uniform;
  e[2].buffer.type = WGPUBufferBindingType_Uniform;
  e[2].buffer.hasDynamicOffset = true;
  e[2].buffer.minBindingSize = sizeof(WgslOp);
  WGPUBindGroupLayoutDescriptor bdesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  bdesc.entryCount = 3;
  bdesc.entries = e;
  c.bgl = wgpuDeviceCreateBindGroupLayout(c.device, &bdesc);
  WGPUPipelineLayoutDescriptor pdesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  pdesc.bindGroupLayoutCount = 1;
  pdesc.bindGroupLayouts = &c.bgl;
  c.layout = wgpuDeviceCreatePipelineLayout(c.device, &pdesc);

  const auto t0 = std::chrono::steady_clock::now();
  const std::string src = module_source();
  WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
  wgsl.code = sv(src);
  WGPUShaderModuleDescriptor mdesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
  mdesc.nextInChain = &wgsl.chain;
  WGPUShaderModule module = nullptr;
  checked("kernel compilation failed", [&] { module = wgpuDeviceCreateShaderModule(c.device, &mdesc); });
  for (std::size_t k = 0; k < kKernelCount; ++k) {
    WGPUComputePipelineDescriptor cdesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    cdesc.layout = c.layout;
    cdesc.compute.module = module;
    cdesc.compute.entryPoint = sv(kKernelNames[k]);
    checked(std::string("pipeline creation failed: ") + kKernelNames[k],
            [&] { c.pipelines[k] = wgpuDeviceCreateComputePipeline(c.device, &cdesc); });
  }
  wgpuShaderModuleRelease(module);
  c.compile_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (std::getenv("JBA_WEBGPU_VERBOSE"))
    std::fprintf(stderr, "webgpu: %s, kernels compiled in %.3f s, storage binding up to %llu MiB\n", c.name.c_str(),
                 c.compile_seconds, static_cast<unsigned long long>(c.limits.maxStorageBufferBindingSize >> 20));
}

Context& context() {
  static std::once_flag once;
  std::call_once(once, [] {
    try {
      create_context();
    } catch (const std::exception& e) {
      ctx().error = e.what();
      ctx().device = nullptr;
    }
  });
  Context& c = ctx();
  if (c.device == nullptr) throw std::runtime_error(c.error.empty() ? "webgpu: no device" : c.error);
  return c;
}

#ifdef JBA_WEBGPU_ACCELERATE
// Strategy B's factorization on the host: Accelerate's sparse Cholesky in double, as
// on Metal (backends/metal/src/backend.mm; duplicated here for the exploration).
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
    if (symbolic_.status != SparseStatusOK) fail("Accelerate symbolic factorization failed");
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
#endif

// ---------------------------------------------------------------------------
// A solve's device: every buffer a range of one heap, laid out by alloc() and created
// at bind(), when the host logic has allocated everything.

class WebGpuDevice final : public gpu::Device {
 public:
  WebGpuDevice() : c_(context()), lock_(c_.mutex) {}

  ~WebGpuDevice() override {
    // A batch left open by an exception between begin() and finish().
    if (pass_) wgpuComputePassEncoderRelease(pass_);
    if (enc_) wgpuCommandEncoderRelease(enc_);
    if (bind_group_) wgpuBindGroupRelease(bind_group_);
    report_repeats();
    if (compare_group_) wgpuBindGroupRelease(compare_group_);
    if (compare_) wgpuComputePipelineRelease(compare_);
    for (WGPUBuffer b : {heap_, args_buf_, op_buf_, readback_, backup_, first_, compare_out_})
      if (b) wgpuBufferRelease(b);
  }

  std::string name() const override { return "webgpu"; }

  Buffer alloc(std::size_t bytes, const void* init, bool host_visible) override {
    if (heap_ != nullptr) fail("allocation after the heap was created");
    const std::uint64_t offset = (cursor_ + kAlign - 1) / kAlign * kAlign;
    const std::uint64_t end = offset + (bytes + 3) / 4 * 4;
    if (end > limit()) {
      throw gpu::DoesNotFit("webgpu: the solve needs more than one storage binding holds (" +
                            std::to_string(end >> 20) + " MiB of " + std::to_string(limit() >> 20) + " MiB)");
    }
    Range r;
    r.offset = offset;
    r.bytes = (bytes + 3) / 4 * 4;
    if (host_visible) r.mirror.assign(r.bytes, 0);
    if (init != nullptr) {
      if (host_visible) std::memcpy(r.mirror.data(), init, bytes);
      else r.init.assign(static_cast<const unsigned char*>(init), static_cast<const unsigned char*>(init) + bytes);
    }
    cursor_ = end;
    ranges_.push_back(std::move(r));
    Range& last = ranges_.back();
    return {host_visible ? last.mirror.data() : nullptr, offset, bytes};
  }

  std::size_t free_memory() const override {
    return cursor_ < limit() ? static_cast<std::size_t>(limit() - cursor_) : 0;
  }

  void to_device(const Buffer& b) override {
    if (b.host == nullptr || b.bytes == 0) return;
    wgpuQueueWriteBuffer(c_.queue, heap_, b.address, b.host, (b.bytes + 3) / 4 * 4);
  }

  void to_host(const Buffer& b) override {
    if (b.host == nullptr || b.bytes == 0) return;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c_.device, nullptr);
    read_back(enc, b.address, b.host, b.bytes);
  }

  void bind(const gpu::Args& args, const Buffer& state) override {
    // The heap, written while mapped at creation: every initial value at once.
    WGPUBufferDescriptor hd = WGPU_BUFFER_DESCRIPTOR_INIT;
    hd.size = std::max<std::uint64_t>(cursor_, 256);
    hd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    hd.mappedAtCreation = true;
    checked("heap allocation failed (" + std::to_string(hd.size >> 20) + " MiB)",
            [&] { heap_ = wgpuDeviceCreateBuffer(c_.device, &hd); });
    if (heap_ == nullptr) throw gpu::DoesNotFit("webgpu: heap allocation failed");
    heap_size_ = hd.size;
    auto* mapped = static_cast<unsigned char*>(wgpuBufferGetMappedRange(heap_, 0, hd.size));
    if (mapped == nullptr) fail("heap not mapped");
    for (Range& r : ranges_) {
      if (!r.init.empty()) std::memcpy(mapped + r.offset, r.init.data(), r.init.size());
      else if (!r.mirror.empty()) std::memcpy(mapped + r.offset, r.mirror.data(), r.mirror.size());
      std::vector<unsigned char>().swap(r.init);
    }
    wgpuBufferUnmap(heap_);

    // Args: layout.h's pointers become word indices.
    std::uint32_t w[kArgPointers + kArgScalars + 1];
    const auto* bytes = reinterpret_cast<const unsigned char*>(&args);
    for (std::size_t i = 0; i < kArgPointers; ++i) {
      std::uintptr_t p;
      std::memcpy(&p, bytes + i * sizeof(void*), sizeof(p));
      w[i] = word(p);
    }
    std::memcpy(w + kArgPointers, bytes + offsetof(jba_gpu::Args, nobs), 4 * kArgScalars);
    w[kArgPointers + kArgScalars] = word(state.address);
    state_ = state;
    args_buf_ = make_buffer(sizeof(w) + 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(c_.queue, args_buf_, 0, w, sizeof(w));
    op_buf_ = make_buffer(std::uint64_t{kOpStride} * kOpSlots, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    ops_.assign(std::size_t{kOpStride} * kOpSlots, 0);

    WGPUBindGroupEntry be[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    be[0].binding = 0;
    be[0].buffer = heap_;
    be[0].size = hd.size;
    be[1].binding = 1;
    be[1].buffer = args_buf_;
    be[1].size = sizeof(w) + 16;
    be[2].binding = 2;
    be[2].buffer = op_buf_;
    be[2].size = sizeof(WgslOp);
    WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg.layout = c_.bgl;
    bg.entryCount = 3;
    bg.entries = be;
    checked("bind group creation failed", [&] { bind_group_ = wgpuDeviceCreateBindGroup(c_.device, &bg); });
  }

  void begin() override {
    enc_ = wgpuDeviceCreateCommandEncoder(c_.device, nullptr);
    pass_ = wgpuCommandEncoderBeginComputePass(enc_, nullptr);
    nops_ = 0;
  }

  void dispatch_threads(Kernel k, std::size_t n, const gpu::Op& op) override {
    dispatch(k, (n + kThreads - 1) / kThreads, op);
  }

  void dispatch_groups(Kernel k, std::size_t groups, std::size_t threads, const gpu::Op& op) override {
    if (threads != group_size(k)) fail(std::string(kKernelNames[static_cast<int>(k)]) + ": unexpected group size");
    dispatch(k, groups, op);
  }

  void finish() override {
    wgpuComputePassEncoderEnd(pass_);
    wgpuComputePassEncoderRelease(pass_);
    pass_ = nullptr;
    flush_ops();
    WGPUCommandEncoder enc = enc_;
    enc_ = nullptr;
    read_back(enc, state_.address, state_.host, state_.bytes);
  }

  std::unique_ptr<gpu::DirectSolver> direct_solver() override {
#ifdef JBA_WEBGPU_ACCELERATE
    if (!std::getenv("JBA_WEBGPU_NO_DIRECT")) return std::make_unique<AccelerateSolver>();
#endif
    return nullptr;
  }

 private:
  struct Range {
    std::uint64_t offset = 0, bytes = 0;
    std::vector<unsigned char> mirror;  // host side, for host-visible buffers
    std::vector<unsigned char> init;    // initial values until the heap exists
  };

  std::uint64_t limit() const {
    return std::min<std::uint64_t>(c_.limits.maxStorageBufferBindingSize, c_.limits.maxBufferSize);
  }

  static std::uint32_t word(std::uint64_t address) {
    if (address % 4 != 0) fail("unaligned device address");
    return static_cast<std::uint32_t>(address / 4);
  }

  WGPUBuffer make_buffer(std::uint64_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
    d.size = size;
    d.usage = usage;
    WGPUBuffer b = nullptr;
    checked("buffer allocation failed", [&] { b = wgpuDeviceCreateBuffer(c_.device, &d); });
    return b;
  }

  void dispatch(Kernel k, std::size_t groups, const gpu::Op& op) {
    if (groups == 0) return;
    if (nops_ == kOpSlots) {  // the Op ring is full: submit what is encoded, go on
      wgpuComputePassEncoderEnd(pass_);
      wgpuComputePassEncoderRelease(pass_);
      flush_ops();
      WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc_, nullptr);
      wgpuQueueSubmit(c_.queue, 1, &cb);
      wgpuCommandBufferRelease(cb);
      wgpuCommandEncoderRelease(enc_);
      begin();
    }
    WgslOp o{};
    o.x = word(reinterpret_cast<std::uintptr_t>(op.x));
    o.y = word(reinterpret_cast<std::uintptr_t>(op.y));
    o.n = op.n;
    o.mode = op.mode;
    o.slot = op.slot;
    o.iter = op.iter;
    o.radius = op.radius;
    if (groups > std::uint64_t{kMaxGroupsX} * kMaxGroupsX) fail("dispatch too large");
    o.groups = static_cast<std::uint32_t>(groups);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min<std::size_t>(groups, kMaxGroupsX));
    const std::uint32_t gy = static_cast<std::uint32_t>((groups + gx - 1) / gx);
    // Experiment (document 16): JBA_WEBGPU_REPEAT=n runs every dispatch n more times on
    // the memory it started from and counts the dispatches whose results differ.
    static const int repeat = std::getenv("JBA_WEBGPU_REPEAT") ? std::atoi(std::getenv("JBA_WEBGPU_REPEAT")) : 0;
    if (repeat > 0) {
      repeat_dispatch(k, gx, gy, o, repeat);
      return;
    }
    std::memcpy(ops_.data() + std::size_t{kOpStride} * nops_, &o, sizeof(o));
    const std::uint32_t dyn = kOpStride * nops_;
    ++nops_;
    wgpuComputePassEncoderSetPipeline(pass_, c_.pipelines[static_cast<int>(k)]);
    wgpuComputePassEncoderSetBindGroup(pass_, 0, bind_group_, 1, &dyn);
    wgpuComputePassEncoderDispatchWorkgroups(pass_, gx, gy, 1);
    // Experiment (exploration notes): JBA_WEBGPU_SYNC=1 waits after every dispatch.
    static const bool sync = std::getenv("JBA_WEBGPU_SYNC") != nullptr;
    if (sync) {
      finish();
      begin();
    }
  }

  // JBA_WEBGPU_REPEAT: submits what is encoded, then runs the dispatch once from a copy
  // of the heap and `times` more from the same copy, comparing the whole heap after
  // each run with the first, bit for bit, on the device (kCompareWgsl); only the count
  // of differing words and the first of them come back. The solve goes on from the
  // last run.
  void repeat_dispatch(Kernel k, std::uint32_t gx, std::uint32_t gy, const WgslOp& o, int times) {
    wgpuComputePassEncoderEnd(pass_);
    wgpuComputePassEncoderRelease(pass_);
    pass_ = nullptr;
    flush_ops();
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc_, nullptr);
    wgpuQueueSubmit(c_.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc_);
    enc_ = nullptr;
    if (compare_ == nullptr) make_compare();
    wgpuQueueWriteBuffer(c_.queue, op_buf_, 0, &o, sizeof(o));
    RepeatStat& s = repeat_stats_[{static_cast<int>(k), o.mode}];
    const std::uint64_t words = heap_size_ / 4;
    const std::uint64_t groups = (words + 255) / 256;
    const std::uint32_t cx = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, kMaxGroupsX));
    const std::uint32_t cy = static_cast<std::uint32_t>((groups + cx - 1) / cx);
    bool differed = false;
    for (int r = 0; r <= times; ++r) {
      WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c_.device, nullptr);
      if (r == 0) wgpuCommandEncoderCopyBufferToBuffer(enc, heap_, 0, backup_, 0, heap_size_);
      else wgpuCommandEncoderCopyBufferToBuffer(enc, backup_, 0, heap_, 0, heap_size_);
      WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
      const std::uint32_t dyn = 0;
      wgpuComputePassEncoderSetPipeline(pass, c_.pipelines[static_cast<int>(k)]);
      wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group_, 1, &dyn);
      wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
      if (r > 0) {
        wgpuComputePassEncoderSetPipeline(pass, compare_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, compare_group_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, cx, cy, 1);
      }
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
      if (r == 0) {
        wgpuCommandEncoderCopyBufferToBuffer(enc, heap_, 0, first_, 0, heap_size_);
        WGPUCommandBuffer b = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(c_.queue, 1, &b);
        wgpuCommandBufferRelease(b);
        wgpuCommandEncoderRelease(enc);
        continue;
      }
      const std::uint32_t reset[2] = {0, 0xffffffffu};
      wgpuQueueWriteBuffer(c_.queue, compare_out_, 0, reset, sizeof(reset));
      std::uint32_t got[2] = {};
      read_back(enc, 0, got, sizeof(got), compare_out_);
      if (got[0] > 0) {
        if (s.differed == 0 && !differed) s.first_word = got[1];
        differed = true;
        s.max_words = std::max<std::size_t>(s.max_words, got[0]);
      }
    }
    ++s.dispatches;
    if (differed) ++s.differed;
    begin();
  }

  // The heap's copies and the comparison kernel for JBA_WEBGPU_REPEAT.
  void make_compare() {
    constexpr WGPUBufferUsage copy = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    backup_ = make_buffer(heap_size_, copy);
    first_ = make_buffer(heap_size_, WGPUBufferUsage_Storage | copy);
    compare_out_ = make_buffer(256, WGPUBufferUsage_Storage | copy);
    static const char* const kCompareWgsl = R"(
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>, 2>;  // count, first word
@compute @workgroup_size(256) fn compare(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u,
                                         @builtin(local_invocation_index) li: u32) {
  let i = (wg.y * nw.x + wg.x) * 256u + li;
  if (i >= arrayLength(&a)) { return; }
  if (a[i] != b[i]) {
    atomicAdd(&out[0], 1u);
    atomicMin(&out[1], i);
  }
}
)";
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = sv(kCompareWgsl);
    WGPUShaderModuleDescriptor mdesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    mdesc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(c_.device, &mdesc);
    WGPUComputePipelineDescriptor cdesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    cdesc.compute.module = module;
    cdesc.compute.entryPoint = sv("compare");
    checked("compare pipeline", [&] { compare_ = wgpuDeviceCreateComputePipeline(c_.device, &cdesc); });
    wgpuShaderModuleRelease(module);
    WGPUBindGroupEntry be[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    const WGPUBuffer bufs[3] = {heap_, first_, compare_out_};
    for (std::uint32_t i = 0; i < 3; ++i) {
      be[i].binding = i;
      be[i].buffer = bufs[i];
      be[i].size = i < 2 ? heap_size_ : 8;
    }
    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(compare_, 0);
    WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg.layout = bgl;
    bg.entryCount = 3;
    bg.entries = be;
    checked("compare bind group", [&] { compare_group_ = wgpuDeviceCreateBindGroup(c_.device, &bg); });
    wgpuBindGroupLayoutRelease(bgl);
  }

  // The Ops of the dispatches encoded so far, before their submission.
  void flush_ops() {
    if (nops_ > 0) wgpuQueueWriteBuffer(c_.queue, op_buf_, 0, ops_.data(), std::size_t{kOpStride} * nops_);
    nops_ = 0;
  }

  // Ends `enc` with a copy of heap bytes [address, address + bytes) to the read-back
  // buffer, submits it and waits, then copies into `host`.
  void read_back(WGPUCommandEncoder enc, std::uint64_t address, void* host, std::size_t bytes,
                 WGPUBuffer from = nullptr) {
    const std::uint64_t size = (bytes + 3) / 4 * 4;
    if (readback_size_ < size) {
      if (readback_) wgpuBufferRelease(readback_);
      readback_size_ = std::max<std::uint64_t>(size, 1 << 16);
      readback_ = make_buffer(readback_size_, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }
    wgpuCommandEncoderCopyBufferToBuffer(enc, from ? from : heap_, address, readback_, 0, size);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(c_.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    WGPUBufferMapCallbackInfo mcb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    std::string msg;
    mcb.callback = [](WGPUMapAsyncStatus st, WGPUStringView m, void* u1, void* u2) {
      *static_cast<WGPUMapAsyncStatus*>(u1) = st;
      *static_cast<std::string*>(u2) = str(m);
    };
    mcb.userdata1 = &status;
    mcb.userdata2 = &msg;
    wait(wgpuBufferMapAsync(readback_, WGPUMapMode_Read, 0, size, mcb));
    // Unmapped before anything can throw, so the next read-back can map it again.
    if (status == WGPUMapAsyncStatus_Success) {
      std::memcpy(host, wgpuBufferGetConstMappedRange(readback_, 0, size), bytes);
      wgpuBufferUnmap(readback_);
    }
    // Every batch ends here, waiting on the device, which is where Dawn reports
    // errors it did not raise at once (validation inside a pass, a lost device).
    check_device_error();
    if (status != WGPUMapAsyncStatus_Success) fail("read-back failed: " + msg);
  }

  struct RepeatStat {
    std::size_t dispatches = 0, differed = 0, max_words = 0, first_word = 0;
  };

  void report_repeats() const {
    for (const auto& [key, s] : repeat_stats_) {
      std::fprintf(stderr, "webgpu repeat: %s mode %d: %zu of %zu dispatches differed", kKernelNames[key.first],
                   key.second, s.differed, s.dispatches);
      if (s.differed > 0) std::fprintf(stderr, " (up to %zu words; first at word %zu)", s.max_words, s.first_word);
      std::fprintf(stderr, "\n");
    }
  }

  Context& c_;
  std::unique_lock<std::recursive_mutex> lock_;
  std::vector<Range> ranges_;  // std::vector of vectors: mirrors keep their addresses on growth
  std::uint64_t cursor_ = kAlign;  // nothing at address 0: it stays the null pointer
  std::uint64_t heap_size_ = 0;
  // JBA_WEBGPU_REPEAT: the heap before the dispatch and after its first run, the
  // comparison's result, its kernel and binding.
  WGPUBuffer backup_ = nullptr, first_ = nullptr, compare_out_ = nullptr;
  WGPUComputePipeline compare_ = nullptr;
  WGPUBindGroup compare_group_ = nullptr;
  std::map<std::pair<int, int>, RepeatStat> repeat_stats_;
  WGPUBuffer heap_ = nullptr, args_buf_ = nullptr, op_buf_ = nullptr, readback_ = nullptr;
  std::uint64_t readback_size_ = 0;
  WGPUBindGroup bind_group_ = nullptr;
  Buffer state_;
  std::vector<unsigned char> ops_;
  std::uint32_t nops_ = 0;
  WGPUCommandEncoder enc_ = nullptr;
  WGPUComputePassEncoder pass_ = nullptr;
};

}  // namespace

std::string wgsl_module() { return module_source(); }

bool available() {
  try {
    context();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

std::string device_name() {
  try {
    return context().name;
  } catch (const std::exception&) {
    return {};
  }
}

bool has_direct_solver() {
#ifdef JBA_WEBGPU_ACCELERATE
  return !std::getenv("JBA_WEBGPU_NO_DIRECT");
#else
  return false;
#endif
}

std::unique_ptr<gpu::Device> make_device() { return std::make_unique<WebGpuDevice>(); }

WebGpuBackend::WebGpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : gpu::GpuBackend(problem, options, gauge, make_device) {}

}  // namespace jba::webgpu
