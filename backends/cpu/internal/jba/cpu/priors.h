#pragma once

#include "directions_generated.h"  // residuals/generated/cpu
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

// Manhattan line residual (1 value, unwhitened): pose (q, t), plane normal in camera
// axes, Manhattan frame (q), building axis. J_pose is 1 x 6, J_frame 1 x 3 (left
// perturbation of the frame in the working frame's axes).
template <typename Scalar>
inline void manhattan_line(const Scalar* pose, const Scalar* normal, const Scalar* frame, const Scalar* axis,
                           Scalar* residual, Scalar* J_pose, Scalar* J_frame) {
  generated::manhattan_line(pose, normal, frame, axis, residual, J_pose, J_frame);
}

}  // namespace jba::cpu
