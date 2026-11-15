// Metal prelude of the shared GPU kernels (backends/gpu/kernels/solver.inc, D96):
// the macros that source expects, in Metal Shading Language. Embedded after the
// generated reprojection code and before backends/gpu/internal/jba/gpu/layout.h.

#include <metal_stdlib>
using namespace metal;

#define JBA_DEVICE device
#define JBA_THREAD thread
#define JBA_CONSTANT_DATA constant
#define JBA_ATOMIC_INT atomic_int
#define JBA_FUNC
#define JBA_ARGS_REF constant Args&
#define JBA_SHARED threadgroup
#define JBA_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define JBA_SET_BAD(s) atomic_store_explicit(&(s).bad, 1, memory_order_relaxed)
#define JBA_GET_BAD(s) atomic_load_explicit(&(s).bad, memory_order_relaxed)

// Every kernel binds Args at buffer 0, State at 1 and Op at 2; the host sets all three
// for every dispatch.
#define JBA_KERNEL(name)                                                                        \
  kernel void name(constant Args& a [[buffer(0)]], device State& s [[buffer(1)]],               \
                   constant Op& op [[buffer(2)]], uint gid [[thread_position_in_grid]],         \
                   uint group [[threadgroup_position_in_grid]],                                 \
                   uint lid [[thread_position_in_threadgroup]])
