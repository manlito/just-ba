#pragma once

#include "depth_generated.h"  // residuals/generated/cpu
#include "jba/problem.h"

namespace jba::cpu {

// Depth residual at a keypoint (1 value, unwhitened), dispatched on what the reading
// measures and the residual's form: pose (q, t), point, reading, correction (a, b).
// Conventions of residuals/symbolic/depth.py. Jacobian pointers may be null.
template <typename Scalar>
inline void depth_residual(DepthKind kind, DepthForm form, const Scalar* pose, const Scalar* point, Scalar reading,
                           const Scalar* correction, Scalar* residual, Scalar* J_pose, Scalar* J_point,
                           Scalar* J_correction) {
  namespace g = jba::generated;
  const bool range = kind == DepthKind::Range, inverse = form == DepthForm::Inverse;
  if (!range && !inverse)
    return g::depth_z(pose, pose + 4, point, reading, correction, residual, J_pose, J_point, J_correction);
  if (range && !inverse)
    return g::depth_range(pose, pose + 4, point, reading, correction, residual, J_pose, J_point, J_correction);
  if (!range)
    return g::depth_z_inverse(pose, pose + 4, point, reading, correction, residual, J_pose, J_point, J_correction);
  return g::depth_range_inverse(pose, pose + 4, point, reading, correction, residual, J_pose, J_point, J_correction);
}

}  // namespace jba::cpu
