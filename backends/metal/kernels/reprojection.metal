// Reprojection kernels for the Metal backend (milestone 0 spike).
//
// At build time CMake prepends residuals/generated/metal/reprojection.metal.h to this
// file and embeds the result in the library; Metal compiles it at run time. One
// thread per observation. Inputs are already in single precision.

#include <metal_stdlib>
using namespace metal;

#define JBA_REPROJECTION_KERNEL(NAME, NP)                                              \
  kernel void reprojection_##NAME(                                                     \
      device const float* poses [[buffer(0)]], device const float* points [[buffer(1)]], \
      device const float* intrinsics [[buffer(2)]],                                    \
      device const int* image_intrinsics [[buffer(3)]],                                \
      device const int* obs_image [[buffer(4)]], device const int* obs_point [[buffer(5)]], \
      device const float* obs_xy [[buffer(6)]], device float* out_residual [[buffer(7)]], \
      device float* out_J_pose [[buffer(8)]], device float* out_J_point [[buffer(9)]],  \
      device float* out_J_intrinsics [[buffer(10)]], constant uint& num_obs [[buffer(11)]], \
      uint k [[thread_position_in_grid]]) {                                            \
    if (k >= num_obs) return;                                                          \
    const int i = obs_image[k];                                                        \
    const int j = obs_point[k];                                                        \
    const int c = image_intrinsics[i];                                                 \
    float q[4], t[3], X[3], K[NP], o[2];                                               \
    for (int a = 0; a < 4; ++a) q[a] = poses[7 * i + a];                               \
    for (int a = 0; a < 3; ++a) t[a] = poses[7 * i + 4 + a];                           \
    for (int a = 0; a < 3; ++a) X[a] = points[3 * j + a];                              \
    for (int a = 0; a < NP; ++a) K[a] = intrinsics[NP * c + a];                        \
    o[0] = obs_xy[2 * k];                                                              \
    o[1] = obs_xy[2 * k + 1];                                                          \
    float r[2], jp[12], jx[6], jk[2 * NP];                                             \
    jba_generated::reprojection_##NAME<float>(q, t, X, K, o, r, jp, jx, jk);           \
    for (int a = 0; a < 2; ++a) out_residual[2 * k + a] = r[a];                        \
    for (int a = 0; a < 12; ++a) out_J_pose[12 * k + a] = jp[a];                       \
    for (int a = 0; a < 6; ++a) out_J_point[6 * k + a] = jx[a];                        \
    for (int a = 0; a < 2 * NP; ++a) out_J_intrinsics[2 * NP * k + a] = jk[a];         \
  }

JBA_REPROJECTION_KERNEL(simple_pinhole, 3)
JBA_REPROJECTION_KERNEL(pinhole, 4)
JBA_REPROJECTION_KERNEL(simple_radial, 4)
JBA_REPROJECTION_KERNEL(radial, 5)
