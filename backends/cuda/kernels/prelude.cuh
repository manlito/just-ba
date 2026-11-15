#pragma once

// CUDA prelude of the shared GPU kernels (backends/gpu/kernels/solver.inc, D96): the
// generated reprojection code for CUDA and the macros that source expects.

#include <cuda_runtime.h>

#include "reprojection.cuh"

namespace jba_generated = jba::cuda_generated;

#define JBA_DEVICE
#define JBA_THREAD
#define JBA_CONSTANT_DATA constexpr
#define JBA_ATOMIC_INT int
#define JBA_FUNC __device__ inline
#define JBA_ARGS_REF const Args&
#define JBA_SHARED __shared__
#define JBA_BARRIER() __syncthreads()
#define JBA_SET_BAD(s) atomicExch(&(s).bad, 1)
#define JBA_GET_BAD(s) atomicAdd(&(s).bad, 0)

// Metal's clamp: fmin(fmax(x, lo), hi).
__device__ inline float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

// A kernel takes Args and Op by value (kernel parameters, in constant memory) and the
// State in device memory; its body is a device function with the indices as Metal
// names them.
#define JBA_KERNEL(name)                                                                              \
  __device__ inline void name##_body(const Args& a, State& s, const Op& op, unsigned gid, unsigned group, \
                                     unsigned lid);                                                   \
  __global__ void name(const Args a, State* s, const Op op) {                                         \
    name##_body(a, *s, op, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.x, threadIdx.x);           \
  }                                                                                                   \
  __device__ inline void name##_body(const Args& a, State& s, const Op& op, [[maybe_unused]] unsigned gid, \
                                     [[maybe_unused]] unsigned group, [[maybe_unused]] unsigned lid)
