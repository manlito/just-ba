#pragma once

#include <initializer_list>

#include "directions_generated.h"  // residuals/generated/cpu
#include "jba/camera_model.h"
#include "jba/geometry.h"
#include "priors_generated.h"

namespace jba::cpu {

// Pose prior residuals (6 values, unwhitened) and Jacobians. Conventions are those
// of residuals/symbolic/priors.py. Jacobian pointers may be null.

// Relative prior: poses i and j (q then t), prior T_j T_i^-1 (q, t).
template <typename Scalar>
inline void relative_prior(const Scalar* pose_i, const Scalar* pose_j, const Scalar* prior, Scalar* residual,
                           Scalar* J_i, Scalar* J_j) {
  generated::relative_pose_prior(pose_i, pose_i + 4, pose_j, pose_j + 4, prior, prior + 4, residual, J_i, J_j);
}

// Global prior: pose, set transform (q, t, s), prior rotation q_p and centre C_p in
// the set's frame. J_rigid is 6 x 6, J_scale 6 x 1.
template <typename Scalar>
inline void global_prior(const Scalar* pose, const Scalar* transform, const Scalar* q_prior,
                         const Scalar* c_prior, Scalar* residual, Scalar* J_pose, Scalar* J_rigid,
                         Scalar* J_scale) {
  generated::global_pose_prior(pose, pose + 4, transform, transform + 4, transform[7], q_prior, c_prior, residual,
                               J_pose, J_rigid, J_scale);
}

// Gravity prior (3 values, unwhitened): pose (q, t), gravity in the working frame,
// gravity measured in camera axes. J_pose is 3 x 6.
template <typename Scalar>
inline void gravity_prior(const Scalar* pose, const Scalar* g_world, const Scalar* g_camera, Scalar* residual,
                          Scalar* J_pose) {
  generated::gravity_prior(pose, g_world, g_camera, residual, J_pose);
}

// Manhattan line residual (1 value, pixels, unwhitened; D163): pose (q, t),
// Manhattan frame (q), building axis, the segment's endpoints in normalized
// coordinates (undistorted with K), intrinsics K. J_pose is 1 x 6, J_frame 1 x 3
// (left perturbation of the frame in the working frame's axes), J_intrinsics 1 x
// num_params(model), the total derivative with the endpoints' pixels held.
// An endpoint whose bearing is the axis direction itself (the endpoint at the
// vanishing point) gives 0 / 0 in the circle's tangent there; the residual tends to
// zero, so it is written as zero, with zero Jacobians.
template <typename Scalar>
inline bool line_endpoint_at_vanishing_point(const Scalar* pose, const Scalar* frame, const Scalar* axis,
                                             const Scalar* uv0, const Scalar* uv1) {
  Scalar dm[3], d[3];
  quat_rotate(frame, axis, dm);
  quat_rotate(pose, dm, d);
  for (const Scalar* uv : {uv0, uv1}) {
    const Scalar b[3] = {uv[0], uv[1], Scalar(1)};
    const Scalar c[3] = {d[1] * b[2] - d[2] * b[1], d[2] * b[0] - d[0] * b[2], d[0] * b[1] - d[1] * b[0]};
    const Scalar bb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    if (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] < Scalar(1e-20) * bb) return true;
  }
  return false;
}

template <typename Scalar>
inline void manhattan_line(CameraModel model, const Scalar* pose, const Scalar* frame, const Scalar* axis,
                           const Scalar* uv0, const Scalar* uv1, const Scalar* K, Scalar* residual, Scalar* J_pose,
                           Scalar* J_frame, Scalar* J_intrinsics) {
  if (line_endpoint_at_vanishing_point(pose, frame, axis, uv0, uv1)) {
    residual[0] = Scalar(0);
    if (J_pose)
      for (int c = 0; c < 6; ++c) J_pose[c] = Scalar(0);
    if (J_frame)
      for (int c = 0; c < 3; ++c) J_frame[c] = Scalar(0);
    if (J_intrinsics)
      for (int c = 0; c < num_params(model); ++c) J_intrinsics[c] = Scalar(0);
    return;
  }
  switch (model) {
    case CameraModel::SimplePinhole:
      return generated::manhattan_line_simple_pinhole(pose, frame, axis, uv0, uv1, K, residual, J_pose, J_frame,
                                                      J_intrinsics);
    case CameraModel::Pinhole:
      return generated::manhattan_line_pinhole(pose, frame, axis, uv0, uv1, K, residual, J_pose, J_frame, J_intrinsics);
    case CameraModel::SimpleRadial:
      return generated::manhattan_line_simple_radial(pose, frame, axis, uv0, uv1, K, residual, J_pose, J_frame,
                                                     J_intrinsics);
    case CameraModel::Radial:
      return generated::manhattan_line_radial(pose, frame, axis, uv0, uv1, K, residual, J_pose, J_frame, J_intrinsics);
    case CameraModel::OpenCV:
      return generated::manhattan_line_opencv(pose, frame, axis, uv0, uv1, K, residual, J_pose, J_frame, J_intrinsics);
    case CameraModel::OpenCVFisheye:
      return generated::manhattan_line_opencv_fisheye(pose, frame, axis, uv0, uv1, K, residual, J_pose, J_frame,
                                                      J_intrinsics);
  }
}

}  // namespace jba::cpu
