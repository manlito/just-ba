#include "jba/camera_model.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace jba {
namespace {

struct ModelInfo {
  CameraModel model;
  std::string_view name;
  int num_params;
  bool has_reprojection;
};

constexpr std::array<ModelInfo, 6> kModels = {{
    {CameraModel::SimplePinhole, "SIMPLE_PINHOLE", 3, true},
    {CameraModel::Pinhole, "PINHOLE", 4, true},
    {CameraModel::SimpleRadial, "SIMPLE_RADIAL", 4, true},
    {CameraModel::Radial, "RADIAL", 5, true},
    {CameraModel::OpenCV, "OPENCV", 8, true},
    {CameraModel::OpenCVFisheye, "OPENCV_FISHEYE", 8, true},
}};

const ModelInfo& info(CameraModel model) { return kModels.at(static_cast<std::size_t>(model)); }

}  // namespace

int num_params(CameraModel model) { return info(model).num_params; }

std::string_view model_name(CameraModel model) { return info(model).name; }

bool has_reprojection(CameraModel model) { return info(model).has_reprojection; }

IntrinsicsGroup intrinsics_group(CameraModel model, int index) {
  switch (model) {
    case CameraModel::SimplePinhole:
    case CameraModel::SimpleRadial:
    case CameraModel::Radial:
      // f, cx, cy, then distortion
      return index == 0 ? kFocal : index <= 2 ? kPrincipalPoint : kDistortion;
    case CameraModel::Pinhole:
    case CameraModel::OpenCV:
    case CameraModel::OpenCVFisheye:
      // fx, fy, cx, cy, then distortion
      return index <= 1 ? kFocal : index <= 3 ? kPrincipalPoint : kDistortion;
  }
  return kDistortion;
}

namespace {

// Focal lengths and principal point of a model.
void linear_part(CameraModel model, const double* k, double& fx, double& fy, double& cx, double& cy) {
  switch (model) {
    case CameraModel::SimplePinhole:
    case CameraModel::SimpleRadial:
    case CameraModel::Radial:
      fx = fy = k[0];
      cx = k[1];
      cy = k[2];
      return;
    case CameraModel::Pinhole:
    case CameraModel::OpenCV:
    case CameraModel::OpenCVFisheye:
      fx = k[0];
      fy = k[1];
      cx = k[2];
      cy = k[3];
      return;
  }
}

// Distortion of normalized coordinates (u, v) -> (ud, vd), pinhole-type models.
void distort(CameraModel model, const double* k, double u, double v, double& ud, double& vd) {
  const double u2 = u * u, v2 = v * v, uv = u * v, r2 = u2 + v2;
  switch (model) {
    case CameraModel::SimplePinhole:
    case CameraModel::Pinhole:
    case CameraModel::OpenCVFisheye:  // handled separately
      ud = u;
      vd = v;
      return;
    case CameraModel::SimpleRadial: {
      const double f = 1 + k[3] * r2;
      ud = u * f;
      vd = v * f;
      return;
    }
    case CameraModel::Radial: {
      const double f = 1 + k[3] * r2 + k[4] * r2 * r2;
      ud = u * f;
      vd = v * f;
      return;
    }
    case CameraModel::OpenCV: {
      const double radial = k[4] * r2 + k[5] * r2 * r2;
      ud = u + u * radial + 2 * k[6] * uv + k[7] * (r2 + 2 * u2);
      vd = v + v * radial + 2 * k[7] * uv + k[6] * (r2 + 2 * v2);
      return;
    }
  }
}

double fisheye_theta_d(const double* k, double theta) {
  const double t2 = theta * theta;
  return theta * (1 + t2 * (k[4] + t2 * (k[5] + t2 * (k[6] + t2 * k[7]))));
}

}  // namespace

bool project(CameraModel model, const double* k, const double* p, double* x, double* y) {
  if (!(p[2] > 0)) return false;
  double fx, fy, cx, cy;
  linear_part(model, k, fx, fy, cx, cy);
  const double u = p[0] / p[2], v = p[1] / p[2];
  double ud, vd;
  if (model == CameraModel::OpenCVFisheye) {
    const double r = std::sqrt(u * u + v * v);
    const double f = r > 1e-12 ? fisheye_theta_d(k, std::atan(r)) / r : 1.0;
    ud = u * f;
    vd = v * f;
  } else {
    distort(model, k, u, v, ud, vd);
  }
  *x = fx * ud + cx;
  *y = fy * vd + cy;
  return true;
}

bool unproject(CameraModel model, const double* k, double x, double y, double* bearing) {
  double fx, fy, cx, cy;
  linear_part(model, k, fx, fy, cx, cy);
  const double ud = (x - cx) / fx, vd = (y - cy) / fy;
  double u = ud, v = vd;
  if (model == CameraModel::OpenCVFisheye) {
    // theta_d(theta) = r_d, solved by Newton's method for theta in [0, pi/2).
    const double rd = std::sqrt(ud * ud + vd * vd);
    if (rd > 1e-12) {
      double theta = rd;
      bool ok = false;
      for (int it = 0; it < 100; ++it) {
        const double t2 = theta * theta;
        const double f = fisheye_theta_d(k, theta) - rd;
        const double df = 1 + t2 * (3 * k[4] + t2 * (5 * k[5] + t2 * (7 * k[6] + t2 * 9 * k[7])));
        if (!(df > 0)) return false;
        const double step = f / df;
        theta -= step;
        if (std::abs(step) < 1e-14 * std::max(1.0, theta)) {
          ok = true;
          break;
        }
      }
      if (!ok || !(theta > 0) || !(theta < 1.5707963267948966)) return false;
      const double r = std::tan(theta);
      u = ud * r / rd;
      v = vd * r / rd;
    }
  } else if (model != CameraModel::SimplePinhole && model != CameraModel::Pinhole) {
    // Newton's method on distort(u, v) = (ud, vd), with a numerical Jacobian, as
    // COLMAP's IterativeUndistortion does.
    bool ok = false;
    for (int it = 0; it < 100; ++it) {
      double fu, fv;
      distort(model, k, u, v, fu, fv);
      const double ru = fu - ud, rv = fv - vd;
      const double h = 1e-7 * std::max(1.0, std::abs(u) + std::abs(v));
      double a1, a2, b1, b2;
      distort(model, k, u + h, v, a1, a2);
      distort(model, k, u, v + h, b1, b2);
      const double J00 = (a1 - fu) / h, J10 = (a2 - fv) / h, J01 = (b1 - fu) / h, J11 = (b2 - fv) / h;
      const double det = J00 * J11 - J01 * J10;
      if (!(std::abs(det) > 1e-12)) return false;
      const double du = (J11 * ru - J01 * rv) / det, dv = (-J10 * ru + J00 * rv) / det;
      u -= du;
      v -= dv;
      if (std::abs(du) + std::abs(dv) < 1e-14 * std::max(1.0, std::abs(u) + std::abs(v))) {
        ok = true;
        break;
      }
    }
    if (!ok) return false;
    // Reject a converged root on the wrong branch (distortion folding back).
    double fu, fv;
    distort(model, k, u, v, fu, fv);
    if (!(std::abs(fu - ud) + std::abs(fv - vd) < 1e-9 * std::max(1.0, std::abs(ud) + std::abs(vd)))) return false;
  }
  const double n = std::sqrt(u * u + v * v + 1);
  bearing[0] = u / n;
  bearing[1] = v / n;
  bearing[2] = 1 / n;
  return std::isfinite(bearing[0]) && std::isfinite(bearing[1]);
}

std::optional<CameraModel> model_from_id(std::int32_t id) {
  if (id < 0 || id >= static_cast<std::int32_t>(kModels.size())) return std::nullopt;
  return kModels[static_cast<std::size_t>(id)].model;
}

std::optional<CameraModel> model_from_name(std::string_view name) {
  for (const auto& m : kModels) {
    if (m.name == name) return m.model;
  }
  return std::nullopt;
}

}  // namespace jba
