#pragma once

#include <limits>

#include "jba/camera_model.h"
#include "reprojection_generated.h"  // residuals/generated/cpu

namespace jba::cpu {

// Reprojection residual and Jacobians for one observation, dispatched on the camera
// model. Any Jacobian pointer may be null. Layout and tangent conventions are those
// of residuals/symbolic/reprojection.py.
template <typename Scalar>
inline void reprojection(CameraModel model, const Scalar* q, const Scalar* t, const Scalar* point,
                         const Scalar* intrinsics, const Scalar* observed, Scalar* residual,
                         Scalar* J_pose, Scalar* J_point, Scalar* J_intrinsics) {
  namespace g = jba::generated;
  switch (model) {
    case CameraModel::SimplePinhole:
      return g::reprojection_simple_pinhole(q, t, point, intrinsics, observed, residual, J_pose, J_point, J_intrinsics);
    case CameraModel::Pinhole:
      return g::reprojection_pinhole(q, t, point, intrinsics, observed, residual, J_pose, J_point, J_intrinsics);
    case CameraModel::SimpleRadial:
      return g::reprojection_simple_radial(q, t, point, intrinsics, observed, residual, J_pose, J_point, J_intrinsics);
    case CameraModel::Radial:
      return g::reprojection_radial(q, t, point, intrinsics, observed, residual, J_pose, J_point, J_intrinsics);
    case CameraModel::OpenCV:
      return g::reprojection_opencv(q, t, point, intrinsics, observed, residual, J_pose, J_point, J_intrinsics);
    case CameraModel::OpenCVFisheye:
      return g::reprojection_opencv_fisheye(q, t, point, intrinsics, observed, residual, J_pose, J_point,
                                            J_intrinsics);
  }
  // Not a camera model (validate() refuses it): no residual.
  residual[0] = residual[1] = std::numeric_limits<Scalar>::quiet_NaN();
}

}  // namespace jba::cpu
