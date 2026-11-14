#include "jba/metal/reprojection.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <stdexcept>

// Embedded at build time from residuals/generated/metal/reprojection.metal.h and
// backends/metal/kernels/reprojection.metal (see backends/metal/CMakeLists.txt).
extern const char* const kJbaMetalReprojectionSource;

namespace jba::metal {
namespace {

const char* kernel_name(CameraModel model) {
  switch (model) {
    case CameraModel::SimplePinhole: return "reprojection_simple_pinhole";
    case CameraModel::Pinhole: return "reprojection_pinhole";
    case CameraModel::SimpleRadial: return "reprojection_simple_radial";
    case CameraModel::Radial: return "reprojection_radial";
    default: throw std::invalid_argument("metal reprojection: camera model not implemented yet");
  }
}

[[noreturn]] void fail(const std::string& what, NSError* error) {
  std::string msg = "metal: " + what;
  if (error != nil) msg += ": " + std::string(error.localizedDescription.UTF8String);
  throw std::runtime_error(msg);
}

template <typename T>
id<MTLBuffer> upload(id<MTLDevice> device, const std::vector<T>& v) {
  const std::size_t bytes = std::max<std::size_t>(v.size() * sizeof(T), 16);
  id<MTLBuffer> b = v.empty() ? [device newBufferWithLength:bytes options:MTLResourceStorageModeShared]
                              : [device newBufferWithBytes:v.data()
                                                    length:v.size() * sizeof(T)
                                                   options:MTLResourceStorageModeShared];
  if (b == nil) fail("buffer allocation failed", nil);
  return b;
}

id<MTLBuffer> output(id<MTLDevice> device, std::size_t floats) {
  id<MTLBuffer> b = [device newBufferWithLength:std::max<std::size_t>(floats * sizeof(float), 16)
                                        options:MTLResourceStorageModeShared];
  if (b == nil) fail("buffer allocation failed", nil);
  return b;
}

std::vector<float> download(id<MTLBuffer> b, std::size_t floats) {
  const float* p = static_cast<const float*>(b.contents);
  return std::vector<float>(p, p + floats);
}

std::vector<float> to_float(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

}  // namespace

bool available() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}

std::string device_name() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device == nil ? std::string() : std::string(device.name.UTF8String);
  }
}

ReprojectionBatch evaluate_reprojection(const Problem& p) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) fail("no Metal device", nil);

    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    // No fast math: the generated arithmetic must match the CPU emission.
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
    }
    id<MTLLibrary> library = [device newLibraryWithSource:@(kJbaMetalReprojectionSource)
                                                  options:options
                                                    error:&error];
    if (library == nil) fail("kernel compilation failed", error);
    id<MTLFunction> function = [library newFunctionWithName:@(kernel_name(p.model))];
    if (function == nil) fail("kernel not found", nil);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) fail("pipeline creation failed", error);

    const std::size_t n = p.num_observations();
    const std::size_t np = static_cast<std::size_t>(num_params(p.model));
    std::vector<std::int32_t> image_intrinsics = p.image_intrinsics;
    if (image_intrinsics.empty()) image_intrinsics.assign(p.num_images(), 0);

    id<MTLBuffer> buffers[] = {
        upload(device, to_float(p.poses)),
        upload(device, to_float(p.points)),
        upload(device, to_float(p.intrinsics)),
        upload(device, image_intrinsics),
        upload(device, p.obs_image),
        upload(device, p.obs_point),
        upload(device, to_float(p.obs_xy)),
        output(device, 2 * n),
        output(device, 12 * n),
        output(device, 6 * n),
        output(device, 2 * np * n),
    };
    const std::uint32_t num_obs = static_cast<std::uint32_t>(n);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger b = 0; b < 11; ++b) [encoder setBuffer:buffers[b] offset:0 atIndex:b];
    [encoder setBytes:&num_obs length:sizeof(num_obs) atIndex:11];
    const NSUInteger group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    if (n > 0) {
      [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) fail("command buffer failed", command.error);

    ReprojectionBatch out;
    out.residual = download(buffers[7], 2 * n);
    out.J_pose = download(buffers[8], 12 * n);
    out.J_point = download(buffers[9], 6 * n);
    out.J_intrinsics = download(buffers[10], 2 * np * n);
    return out;
  }
}

}  // namespace jba::metal
